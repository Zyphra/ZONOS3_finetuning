#!/usr/bin/env python3
"""Generate a WAV from a fine-tuned checkpoint with the public ZONOS3-Mini runtimes.

``--backend sgl`` (default) uses the CUDA offline API of the ``zonos3-sgl`` package (``bash setup.sh nvidia --inference``);
``--backend reference`` runs the plain-PyTorch reference of the ``zonos3`` package (``python -m zonos3.reference``), slow
but CPU-capable.

The request ``language`` selects the text normalizer; it must be one of the server codes (``en_us``, ``en_gb``,
``fr_fr``, ``de``, ``es``, ``it``, ``pt_br``, ``ja``, ``cmn``, ``ko``). For a language outside that list (a new-language
fine-tune), pass any code and disable normalization (``--no-text-normalization``): the model then reads the text as typed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint directory (model.pth, params.json, tokenizer/)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="en_us")
    parser.add_argument("--speaker-audio", type=Path, help="Reference clip of the voice (wav/mp3/flac/ogg)")
    parser.add_argument("--speaker-embedding", type=Path, help="Saved 192-d speaker embedding (.npy)")
    parser.add_argument(
        "--speaker-mode",
        choices=("embedding", "reference"),
        default="embedding",
        help="How a reference clip conditions the model: its MECT embedding row or the raw codec prefix",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=None, help="Speaker CFG scale (runtime default when unset)")
    parser.add_argument("--text-normalization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend", choices=("sgl", "reference"), default="sgl")
    parser.add_argument("--device", default=None, help="reference backend device (cpu / cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speaker_audio and args.speaker_embedding:
        raise SystemExit("Provide either --speaker-audio or --speaker-embedding, not both")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.text_normalization:
        os.environ["ZONOS3_TTS_NORM"] = "0"

    if args.backend == "sgl":
        from zonos3_sgl.tts import TTSLLM, TTSSamplingParams

        tts = TTSLLM(model_path=str(args.model), attention_backend="fa", cache_type="naive", memory_ratio=0.5)
        sampling = {"seed": args.seed}
        if args.cfg_scale is not None:
            sampling["speaker_cfg_scale"] = args.cfg_scale
        params = TTSSamplingParams(**sampling)
        extra = {}
        if args.speaker_embedding:
            import numpy as np
            import torch

            extra["speaker_embedding"] = torch.from_numpy(np.load(args.speaker_embedding).astype("float32").reshape(-1))
        elif args.speaker_audio and args.speaker_mode == "embedding":
            extra["speaker_embedding"] = tts.embed_speaker_file(str(args.speaker_audio))
        elif args.speaker_audio:
            extra["speaker_reference_audio"] = str(args.speaker_audio)
        (result,) = tts.generate(
            [args.text], params, language=args.language, text_normalization=args.text_normalization, **extra
        )
        audio = result["audio"]
        if not getattr(audio, "numel", lambda: len(audio))():
            raise RuntimeError("The checkpoint emitted no audio")
        tts.save_audio(audio, str(args.output))
        print(f"wrote {args.output}")
        return

    request = {"text": args.text, "language": args.language, "seed": args.seed, "text_normalization": args.text_normalization}
    if args.cfg_scale is not None:
        request["speaker_cfg_scale"] = args.cfg_scale
    if args.speaker_embedding:
        request["speaker_embedding_base64"] = base64.b64encode(args.speaker_embedding.read_bytes()).decode()
        request["speaker_embedding_name"] = args.speaker_embedding.name
    elif args.speaker_audio:
        request["speaker_audio_base64"] = base64.b64encode(args.speaker_audio.read_bytes()).decode()
        request["speaker_mode"] = args.speaker_mode
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(request, handle)
        request_path = handle.name
    command = [sys.executable, "-m", "zonos3.reference", "--model", str(args.model), "--request", request_path, "-o", str(args.output)]
    if args.device:
        command += ["--device", args.device]
    try:
        subprocess.run(command, check=True)
    finally:
        os.unlink(request_path)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
