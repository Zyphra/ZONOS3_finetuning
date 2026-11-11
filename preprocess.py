#!/usr/bin/env python3
"""Encode a text/audio dataset for ZONOS3-Mini fine-tuning.

Every example becomes: the dual-codec codes (``int16 [frames, 10]``, 15 Hz, the codec the model was trained with), the
192-d MECT speaker embedding of the clip, the SuperBPE text ids and the three condition buckets (integrated loudness,
60 dB bandwidth, speaking rate) the release checkpoint reads. Needs a CUDA GPU and the ``zonos3`` package
(``bash setup.sh nvidia --preprocess``).

Input, one of:

- ``--dataset`` / ``--parquet-file``: a Hugging Face Parquet file with a ``text`` column and an embedded ``audio``
  column (``{"bytes", "path"}``), optionally a ``speaker`` column;
- ``--manifest``: a JSONL file, one ``{"audio": path, "text": ..., "speaker": optional}`` per line (paths relative to
  the manifest's directory);
- ``--audio-dir``: a directory of audio files, each with a sibling ``.txt`` transcript; the speaker is ``--speaker``,
  or the parent directory name with ``--speaker-from-dir``.

Text is normalized by the ZONOS3-Mini normalizer for ``--language`` (a server language code; other languages only get
their line breaks collapsed, as the servers do) unless ``--no-normalize``.
"""

from __future__ import annotations

import argparse
import bisect
import os
import json
import math
import sys
from io import BytesIO
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".m4a")
CODEC_RATE = 44100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_argument_group("input")
    source.add_argument("--dataset", help="Hugging Face dataset repo of a Parquet file (with --parquet-file)")
    source.add_argument("--parquet-file", default="data/train-00000-of-00001.parquet")
    source.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Download directory of --dataset")
    source.add_argument("--manifest", type=Path, help="JSONL manifest: audio, text, optional speaker")
    source.add_argument("--audio-dir", type=Path, help="Directory of audio files with sibling .txt transcripts")
    source.add_argument("--speaker", default=None, help="Speaker id for every example without one")
    source.add_argument("--speaker-from-dir", action="store_true", help="--audio-dir: the parent folder is the speaker")
    parser.add_argument("--base-model", type=Path, default=Path("artifacts/base_model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="en_us", help="Text normalization language (server code)")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-examples", type=int, default=0, help="0 encodes everything")
    parser.add_argument("--shard", default=None, help="INDEX/COUNT: encode every COUNT-th input starting at INDEX")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=2000,
        help="Write the examples so far to OUTPUT.partial.pt every N clips and resume from it on restart; 0 disables",
    )
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=120.0)
    parser.add_argument("--conditions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--codec-batch-size",
        type=int,
        default=4,
        help="Clips (20 s lanes) the codec encoder batches per GPU pass; the training data was encoded with 4. "
        "Lanes are independent, but bf16 kernels differ by batch shape, so near-tie codes move with the batch size.",
    )
    parser.add_argument(
        "--codec-chunk-seconds",
        type=float,
        default=20.0,
        help="Lane length of the codec batcher (the training data used 20 s); shorter lanes waste less GPU time on "
        "short clips, longer clips are split with 10 s of context",
    )
    args = parser.parse_args()
    if sum(x is not None for x in (args.dataset, args.manifest, args.audio_dir)) != 1:
        parser.error("Give exactly one of --dataset, --manifest, --audio-dir")
    return args


# ------------------------------------------------------------------------------------------------- inputs


def _decode(data) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(data, dtype="float32", always_2d=True)
    return samples.T.copy(), int(sample_rate)  # [channels, samples]


def rows_from_parquet(args) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = args.data_dir / args.parquet_file
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path = Path(
            hf_hub_download(args.dataset, filename=args.parquet_file, repo_type="dataset", local_dir=args.data_dir)
        )
    rows = []
    parquet = pq.ParquetFile(path)
    columns = [c for c in ("text", "audio", "speaker") if c in parquet.schema.names]
    for batch in parquet.iter_batches(columns=columns, batch_size=64):
        for index, row in enumerate(batch.to_pylist()):
            audio = row.get("audio")
            if not isinstance(audio, dict) or audio.get("bytes") is None:
                raise RuntimeError("Expected the Parquet audio column to contain embedded {'bytes', 'path'} data")
            rows.append(
                {
                    "text": row.get("text"),
                    "speaker": row.get("speaker") or args.speaker,
                    "load": (lambda data=audio["bytes"]: _decode(BytesIO(data))),
                    "name": audio.get("path") or f"row-{len(rows)}",
                }
            )
    return rows


def rows_from_manifest(args) -> list[dict]:
    root = args.manifest.parent
    rows = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        path = Path(item["audio"])
        if not path.is_absolute():
            path = root / path
        rows.append(
            {
                "text": item.get("text"),
                "speaker": item.get("speaker") or args.speaker,
                "load": (lambda p=path: _decode(str(p))),
                "name": str(path),
            }
        )
    return rows


def rows_from_audio_dir(args) -> list[dict]:
    rows = []
    for path in sorted(p for p in args.audio_dir.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES):
        transcript = path.with_suffix(".txt")
        if not transcript.is_file():
            print(f"skipping {path}: no transcript", file=sys.stderr)
            continue
        speaker = path.parent.name if args.speaker_from_dir else args.speaker
        rows.append(
            {
                "text": transcript.read_text(encoding="utf-8").strip(),
                "speaker": speaker,
                "load": (lambda p=path: _decode(str(p))),
                "name": str(path),
            }
        )
    return rows


# ------------------------------------------------------------------------------------------------- conditions


def _rbj(kind, gain_db, q, f0, fs):
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    if kind == "high_shelf":
        b0 = a * ((a + 1) + (a - 1) * cw + 2 * math.sqrt(a) * alpha)
        b1 = -2 * a * ((a - 1) + (a + 1) * cw)
        b2 = a * ((a + 1) + (a - 1) * cw - 2 * math.sqrt(a) * alpha)
        a0 = (a + 1) - (a - 1) * cw + 2 * math.sqrt(a) * alpha
        a1 = 2 * ((a - 1) - (a + 1) * cw)
        a2 = (a + 1) - (a - 1) * cw - 2 * math.sqrt(a) * alpha
    else:  # high_pass
        b0 = (1 + cw) / 2
        b1 = -(1 + cw)
        b2 = (1 + cw) / 2
        a0 = 1 + alpha
        a1 = -2 * cw
        a2 = 1 - alpha
    return np.array([b0, b1, b2]) / a0, np.array([a0, a1, a2]) / a0


def integrated_loudness(x: np.ndarray, fs: int) -> float:
    """ITU-R BS.1770-4 integrated loudness (K-weighting, 400 ms blocks, -70 LUFS absolute and -10 LU relative gates);
    the ungated value when the clip is shorter than a block or no block passes the gate."""
    from scipy.signal import lfilter

    (b1, a1) = _rbj("high_shelf", 4.0, 1 / math.sqrt(2), 1500.0, fs)
    (b2, a2) = _rbj("high_pass", 0.0, 0.5, 38.0, fs)
    y = lfilter(b2, a2, lfilter(b1, a1, x.astype(np.float64)))
    sq = y * y
    ms_all = float(sq.mean()) if sq.size else 0.0
    ungated = -0.691 + 10 * math.log10(ms_all) if ms_all > 0 else -120.0
    block, hop = int(round(0.4 * fs)), int(round(0.1 * fs))
    if sq.size < block:
        return ungated
    c = np.concatenate([[0.0], np.cumsum(sq, dtype=np.float64)])
    starts = np.arange(0, sq.size - block + 1, hop)
    ms = (c[starts + block] - c[starts]) / block
    with np.errstate(divide="ignore"):
        lb = -0.691 + 10 * np.log10(ms)
    keep = lb > -70.0
    if not keep.any():
        return ungated
    relative = -0.691 + 10 * math.log10(ms[keep].mean()) - 10.0
    keep2 = keep & (lb > relative)
    if not keep2.any():
        return ungated
    return -0.691 + 10 * math.log10(ms[keep2].mean())


def bandwidth_60db(x: np.ndarray, fs: int, n_fft: int = 2048, hop: int = 1024) -> float:
    """Highest frequency (Hz) whose bin of the time-averaged Hann power spectrum lies within 60 dB of the peak."""
    window = np.hanning(n_fft).astype(np.float32)
    n = x.size
    if n < n_fft:
        frames = np.zeros((1, n_fft), dtype=np.float32)
        frames[0, :n] = x
    else:
        count = 1 + (n - n_fft) // hop
        frames = np.lib.stride_tricks.as_strided(
            x, shape=(count, n_fft), strides=(x.strides[0] * hop, x.strides[0]), writeable=False
        )
    psum = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    for i in range(0, frames.shape[0], 2048):
        spec = np.fft.rfft(frames[i : i + 2048] * window, axis=1)
        psum += (spec.real**2 + spec.imag**2).sum(axis=0)
    peak = psum.max()
    if peak <= 0:
        return 0.0
    with np.errstate(divide="ignore"):
        db = 10 * np.log10(psum / peak)
    idx = np.nonzero(db >= -60.0)[0]
    return float(np.fft.rfftfreq(n_fft, 1.0 / fs)[idx[-1]]) if idx.size else 0.0


def condition_buckets(measurements: dict[str, float], params: dict) -> list[int]:
    """Bucket ids of the measured conditions under the checkpoint's edges (bucket i = [edge_{i-1}, edge_i))."""
    model = params.get("model", params)
    names = model["condition_names"]
    edges = model["condition_edges"]
    counts = model.get("condition_bucket_counts") or [model["condition_n_buckets"]] * len(names)
    ids = []
    for name, count in zip(names, counts):
        value = measurements.get(name)
        if value is None or not math.isfinite(value):
            ids.append(int(count))  # unknown
            continue
        bucket = bisect.bisect_right(list(edges[name]), float(value))
        ids.append(min(bucket, int(count) - 1))
    return ids


# ------------------------------------------------------------------------------------------------- encoders


@dataclass
class Clip:
    """One input of the codec's streaming batcher (``audio`` / ``sample_rate`` / ``error`` are its contract) with the
    fields the example needs afterwards."""

    audio: np.ndarray
    sample_rate: int
    error: str | None = None
    index: int = 0
    name: str = ""
    text: str = ""
    text_norm: str = ""
    text_ids: list = field(default_factory=list)
    speaker: str = ""
    mono: np.ndarray | None = None
    native_rate: int = 0
    duration: float = 0.0
    measurements: dict | None = None


def build_encoder(device: str, batch_size: int, chunk_seconds: float = 20.0):
    """The release dual codec's encoder (``zonos3.codec.codec_adapter.MossAutoencoder``) batching ``batch_size``
    20-second lanes per GPU pass; lanes are independent, so the codes equal the one-clip-at-a-time encoding."""
    from types import SimpleNamespace

    from zonos3.tokenizer import matched_codec
    from zonos3.utils import release

    saved = sys.modules.get("kernels")  # the codec's Transformers teacher must not pick up the pinned kernels module
    sys.modules["kernels"] = None  # type: ignore[assignment]
    try:
        from transformers import Wav2Vec2BertModel  # noqa: F401
    finally:
        if saved is None:
            sys.modules.pop("kernels", None)
        else:
            sys.modules["kernels"] = saved
    from zonos3.codec.codec_adapter import MossAutoencoder

    entries = [e for e in matched_codec.registry().values() if e.get("family") == "dual"]
    if len(entries) != 1:
        raise RuntimeError("expected exactly one dual-family codec in the registry")
    entry = entries[0]
    torch.cuda.set_device(torch.device(device).index or 0)
    return MossAutoencoder(
        SimpleNamespace(
            hybrid_batch_size=int(batch_size),
            hybrid_chunk_seconds=float(chunk_seconds),
            hybrid_context_seconds=min(10.0, float(chunk_seconds) / 2),
            hybrid_cuda_graph=True,
        ),
        generator_path=release.resolve_file(entry["checkpoint"]),
        teacher_dir=release.resolve_dir_or_local(entry["teacher"]),
        step=int(entry["codec_step"]),
        codebook_sizes=entry["codebook_sizes"],
        name=entry["codec"],
        checkpoint_sha256=entry["checkpoint_sha256"],
        device=device,
    )


def load_tokenizer(base_model: Path, params: dict):
    import hashlib

    from tokenizers import Tokenizer

    spec = params.get("model", params)["text_tokenizer_spec"]
    path = Path(spec["path"])
    if not path.is_absolute():
        path = base_model / path
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError(f"SuperBPE tokenizer {path} does not match the checkpoint's sha256")
    tokenizer = Tokenizer.from_str(raw.decode())
    if tokenizer.get_vocab_size(with_added_tokens=True) != int(spec["vocab_size"]):
        raise ValueError("SuperBPE vocabulary size mismatch")
    return tokenizer, int(spec["offset"])


def make_normalizer(language: str, enabled: bool):
    if not enabled:
        return lambda text: " ".join(text.split())
    try:
        from zonos3.tokenizer.textnorm import TTSTextNormalizer, normalize_request_text
    except Exception as exc:  # the zonos3 package or its Rust normalizer is missing
        print(f"text normalization unavailable ({exc}); collapsing whitespace only", file=sys.stderr)
        return lambda text: " ".join(text.split())
    normalizer = TTSTextNormalizer()
    return lambda text: normalize_request_text(text, language, normalizer)


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()
    params = json.loads((args.base_model / "params.json").read_text(encoding="utf-8"))
    model = params.get("model", params)
    tokenizer, offset = load_tokenizer(args.base_model, params)
    normalize = make_normalizer(args.language, args.normalize)

    if args.dataset:
        rows = rows_from_parquet(args)
    elif args.manifest:
        rows = rows_from_manifest(args)
    else:
        rows = rows_from_audio_dir(args)
    if args.max_examples:
        rows = rows[: args.max_examples]
    if args.shard:
        index, count = (int(v) for v in args.shard.split("/"))
        if not 0 <= index < count:
            raise SystemExit("--shard must be INDEX/COUNT with 0 <= INDEX < COUNT")
        rows = rows[index::count]
        print(f"shard {index}/{count}: {len(rows)} inputs", flush=True)
    if not rows:
        raise RuntimeError("no input examples")

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required: the codec encoder runs on the GPU (bash setup.sh nvidia --preprocess)")
    from zonos3.models.mect_speaker import MECTFbankSpeakerEmbedding, registry_entry

    device = args.device
    print(f"preprocessing on {torch.cuda.get_device_name(torch.device(device))}", flush=True)
    codec = build_encoder(device, args.codec_batch_size, args.codec_chunk_seconds)
    entry = registry_entry(model["speaker_embedding_codec"])
    if entry is None:
        raise RuntimeError(f"no speaker encoder registered for {model['speaker_embedding_codec']!r}")
    speaker_encoder = MECTFbankSpeakerEmbedding(entry, device=device)
    frame_hop = 2940
    stats = {"skipped": 0}

    def clips():
        """Decoded, resampled clips for the codec's streaming batcher (it reads them from its own thread)."""
        for index, row in enumerate(rows, start=1):
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                stats["skipped"] += 1
                continue
            try:
                wav, sample_rate = row["load"]()
            except Exception as exc:
                print(f"skipping {row['name']}: {exc}", file=sys.stderr)
                stats["skipped"] += 1
                continue
            mono = wav.mean(axis=0).astype(np.float32)
            duration = mono.shape[0] / sample_rate
            if duration < args.min_duration or duration > args.max_duration:
                stats["skipped"] += 1
                continue
            text_norm = normalize(text.strip())
            text_ids = [t + offset for t in tokenizer.encode(text_norm, add_special_tokens=False).ids]
            if not text_ids:
                stats["skipped"] += 1
                continue
            # codec input: mono at 44.1 kHz (libsoxr: any input rate at the same cost, unlike the GCD-based torchaudio
            # resampler, which crawls on odd rates such as 22056 Hz), peak-normalized only above 1.0
            audio44 = soxr.resample(mono, sample_rate, CODEC_RATE, quality="HQ") if sample_rate != CODEC_RATE else mono
            audio = torch.from_numpy(np.ascontiguousarray(audio44, dtype=np.float32))[None]
            peak = float(audio.abs().amax())
            if peak > 1.0:
                audio = audio / peak
            measurements = None
            if args.conditions:
                measurements = {
                    "loudness": integrated_loudness(mono, sample_rate),
                    "fmax": bandwidth_60db(mono, sample_rate),
                    "speaking_rate": len(text_norm.encode("utf-8")) / duration,
                }
            yield Clip(
                audio=audio.numpy(),
                sample_rate=CODEC_RATE,
                index=index,
                name=row["name"],
                text=text.strip(),
                text_norm=text_norm,
                text_ids=text_ids,
                speaker=str(row.get("speaker") or "speaker"),
                mono=mono,
                native_rate=sample_rate,
                duration=duration,
                measurements=measurements,
            )

    examples = []
    partial = args.output.with_name(args.output.name + ".partial.pt")
    if args.checkpoint_every and partial.is_file():
        examples = torch.load(partial, map_location="cpu", weights_only=False)["examples"]
        done_sources = {e["source"] for e in examples}
        print(f"resuming from {partial}: {len(examples)} examples", flush=True)
        rows = [row for row in rows if row["name"] not in done_sources]

    def write_partial():
        tmp = partial.with_suffix(".tmp")
        torch.save({"examples": examples, "metadata": {"partial": True}}, tmp)
        tmp.replace(partial)

    with torch.inference_mode():
        for clip, codes in codec.encode_stream(clips()):
            if clip.error is not None or codes is None:
                print(f"skipping {clip.name}: {clip.error}", file=sys.stderr)
                stats["skipped"] += 1
                continue
            codes = np.asarray(codes)
            frames = math.ceil(clip.audio.shape[-1] / frame_hop)
            if codes.shape != (frames, 10):
                raise RuntimeError(f"codec returned {codes.shape}, expected {(frames, 10)} for {clip.name}")
            embedding = speaker_encoder(torch.from_numpy(clip.audio[0]), clip.sample_rate).reshape(-1).cpu().float()
            measurements = clip.measurements
            conditions = condition_buckets(measurements, params) if measurements is not None else None
            examples.append(
                {
                    "codes": torch.from_numpy(codes.astype(np.int16)).contiguous(),
                    "text": clip.text_norm,
                    "text_raw": clip.text,
                    "text_ids": clip.text_ids,
                    "speaker": clip.speaker,
                    "speaker_embedding": embedding,
                    "conditions": conditions,
                    "measurements": measurements,
                    "duration": clip.duration,
                    "segment_id": len(examples),
                    "source": clip.name,
                }
            )
            if len(examples) % 200 == 0:
                allocated = torch.cuda.memory_allocated() / 2**30
                reserved = torch.cuda.memory_reserved() / 2**30
                print(
                    f"encoded {len(examples)}/{clip.index} (skipped {stats['skipped']}) "
                    f"gpu allocated {allocated:.1f} GiB reserved {reserved:.1f} GiB",
                    flush=True,
                )
                torch.cuda.empty_cache()
            if args.checkpoint_every and len(examples) % args.checkpoint_every == 0:
                write_partial()
    skipped = stats["skipped"]
    print(f"encoded {len(examples)}/{len(rows)} (skipped {skipped})", flush=True)

    if not examples:
        raise RuntimeError("no usable examples")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    speakers = sorted({e["speaker"] for e in examples})
    torch.save(
        {
            "examples": examples,
            "metadata": {
                "source": args.dataset or str(args.manifest or args.audio_dir),
                "base_model": str(args.base_model),
                "audio_codec": model["audio_codec"],
                "speaker_embedding_codec": model["speaker_embedding_codec"],
                "language": args.language,
                "normalized": bool(args.normalize),
                "sample_rate": CODEC_RATE,
                "frame_rate": CODEC_RATE / frame_hop,
                "hours": sum(e["duration"] for e in examples) / 3600,
                "speakers": len(speakers),
            },
        },
        args.output,
    )
    hours = sum(e["duration"] for e in examples) / 3600
    print(f"wrote {len(examples)} examples ({hours:.2f} h, {len(speakers)} speakers) to {args.output}")
    if partial.is_file():
        partial.unlink()


if __name__ == "__main__":
    main()
