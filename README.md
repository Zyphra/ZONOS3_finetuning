# ZONOS3-Mini fine-tuning

A hackable, PyTorch-only trainer for the public
[`Zyphra/ZONOS3-Mini`](https://huggingface.co/Zyphra/ZONOS3-Mini) checkpoint, the port of
[ZONOS2_finetuning](https://github.com/Zyphra/ZONOS2_finetuning) to the ZONOS3 Mini architecture.

It supports two fine-tuning approaches:

- LoRA, best for small in-domain adaptations such as 15 minutes of one voice;
- full-model Muon, best for large distribution shifts backed by much larger datasets, such as 20+ hours in a new
  language.

Training uses BF16 FSDP2, PyTorch causal SDPA for the two global-attention layers, the FLA Triton kernels for the
gated-DeltaNet layers (the kernels the model was trained with) and a plain PyTorch expert loop for the MoE. The same
code runs on NVIDIA CUDA and AMD ROCm (both ship Triton).

GPU memory (measured on one H100 with four 3-10 s clips per step): LoRA 2.7 GiB, full fine-tuning 5.1 GiB with the
CPU-offloaded Muon momentum. A **24 GB GPU** runs both; longer packed batches (`--max-seq-len`, `--batch-size`) scale
the activations.

On NVIDIA Hopper GPUs (H100/H200) FLA's Triton backward of the gated delta rule is broken with Triton 3.4-3.7.0, so
`setup.sh nvidia` also installs TileLang (the `hopper` extra); its kernels compile with `nvcc`, so `CUDA_HOME` must
point at a CUDA 12.8 toolkit. Ampere and ROCm GPUs use the Triton kernels.

A fine-tuned checkpoint is written in the public format (`model.pth`, `params.json`, `tokenizer/`) and serves with
every ZONOS3-Mini runtime: the CUDA server, the MLX and ggml servers after GGUF conversion, and the plain-PyTorch
reference.

## What the model reads

ZONOS3-Mini conditions every utterance on three things the trainer reproduces from the release training loader
(`data.py`):

| Rows before the text | Content | Source in this repo |
| --- | --- | --- |
| raw reference prefix | the codec codes of **another clip of the same speaker**, delayed like audio | `--p-reference` of the draws |
| speaker row | one row the model replaces by the projected 192-d MECT speaker embedding | `--p-embedding` of the draws |
| condition rows | loudness (LUFS), bandwidth (Hz) and speaking rate (UTF-8 bytes/s) bucket ids | measured in `preprocess.py`, dropped to "unknown" with `--condition-dropout` |

The remaining draws are unconditioned. The defaults are 90 % embedding draws that carry the clip's **own** embedding,
10 % unconditioned draws (for CFG) and 0.2 condition dropout.

**Speaker conditioning draws.** The release recipe mixed 45 % reference-prefix draws and 45 % embedding draws, half of
the embeddings taken from another clip of the same speaker. On a fine-tune that mix teaches the model to follow the
speaker row less, and the loss shows up in every language. Rank-32 LoRA on 92 h of Urdu (six voices, 1500 steps of
16 clips, 24 held-out lines with a same-voice reference, Whisper-v3 WER, ReDimNet similarity x100):

| draws | Urdu WER | Urdu sim | English WER | English sim |
|---|---|---|---|---|
| release base (no fine-tune) | 8.8 | 73.0 | 1.5 | 68.4 |
| release mix (0.45 / 0.45 / other clip 0.5) | 9.1 | 69.8 | 1.5 | 54.7 |
| no reference prefix, other clip kept | 8.2 | 70.3 | 1.1 | 56.0 |
| reference prefix kept, no other clip | 11.3 | 73.3 | 3.6 | 65.2 |
| own embedding only (default) | 6.3 | 73.3 | 1.8 | 67.5 |

The other-clip embedding rows cost the similarity; the cross-clip reference prefixes cost intelligibility. Keep
`--p-reference` at 0 unless the fine-tuned model must clone from raw reference audio, and then expect a trade-off.

The loss is the release loss: cross entropy over every codebook's codes plus the EOA token, codebook 0 (the semantic
codebook) weighted 3x (`--semantic-loss-weight`). The MoE router balancing biases stay at their release values: the
release trainer re-estimated them after every step (quantile balancing) over ~12 M tokens, and on a fine-tune's
small, single-domain batches that per-step estimate drifts the biases far from the release routing (a 3000-step Urdu
run doubled them and lost 6 WER points and most of its speaker similarity until the release biases were restored).
`--qb-update` re-enables it, smoothed with `--qb-ema` (0.99).

## Install

Python 3.12, a current `uv` with `--torch-backend` support, and a CUDA- or ROCm-capable GPU are required.

```bash
# NVIDIA: PyTorch 2.9.1 + CUDA 12.8, plus the ZONOS3-Mini codec / speaker encoder for preprocessing
bash setup.sh nvidia --preprocess

# AMD: PyTorch 2.9.1 + ROCm 6.4
bash setup.sh amd --preprocess
```

Setup recreates `.venv`, installs the selected PyTorch build with `uv`, downloads the language model, the speaker
encoder and the registries from `Zyphra/ZONOS3-Mini` into `artifacts/` and links `artifacts/base_model` to the LM
folder. Activate it before running the commands below:

```bash
source .venv/bin/activate
```

`--preprocess` installs the `zonos3` package of the ZONOS3-Mini repo (the dual codec encoder with its SSL teacher, the
MECT speaker encoder and the Rust text normalizer); `preprocess.py` downloads the codec weights (about 9 GB) on first
use. `--inference` additionally installs the CUDA runtime for `infer.py --backend sgl` (NVIDIA only).

## Prepare data

`preprocess.py` accepts a Hugging Face Parquet file (a `text` column and an embedded `audio` column, as the
ZONOS2 trainer), a JSONL manifest or a directory of audio files with sibling `.txt` transcripts. A `speaker` column /
field groups the clips of one voice; it only matters for the optional reference-prefix and other-clip draws, which
need at least two clips per speaker.

```bash
# Hugging Face Parquet
python preprocess.py \
  --dataset gabrielclark3330/expresso_ex01_ftext_audio \
  --parquet-file data/train-00000-of-00001.parquet \
  --data-dir data/raw/expresso_ex01 \
  --output data/processed/expresso_ex01.pt

# JSONL manifest: {"audio": "clips/0001.wav", "text": "...", "speaker": "narrator"} per line
python preprocess.py --manifest data/raw/farsi/manifest.jsonl --language en_us --no-normalize \
  --output data/processed/farsi.pt

# A folder of wav + txt pairs, one sub-folder per speaker
python preprocess.py --audio-dir data/raw/voices --speaker-from-dir --output data/processed/voices.pt
```

Preprocessing converts audio to mono, drops clips shorter than 0.5 s or longer than `--max-duration` (120 s), encodes
the clip with the 44.1 kHz dual codec (15 frames/s, 10 codebooks), embeds it with the MECT speaker encoder, measures
the three conditions, normalizes the text (`--language`, a server code: `en_us`, `en_gb`, `fr_fr`, `de`, `es`, `it`,
`pt_br`, `ja`, `cmn`, `ko`; other languages only get line breaks collapsed) and tokenizes it with the release SuperBPE
tokenizer. Write transcripts in the form you will send at inference (spelled-out numbers, the model's punctuation).

### A new language

For a language that the base model does not fully support (Farsi, say), use a full fine-tune with 20+ hours of multi-speaker data and
`configs/full-new-language.toml`. The text normalizer has no grammar for such a language, so preprocess with
`--no-normalize` and serve with text normalization disabled (`ZONOS3_TTS_NORM=0` for the Python servers, or
`"text_normalization": false` per request) and any server language code: the code only selects the normalizer and
the alignment heads. Validate the languages you want to keep with held-out prompts; a long run moves them.

## Train

Use `torchrun` directly. Values passed after `--config` override the TOML recipe, so every run only needs its data and
output paths.

### Choose LoRA or a full fine-tune

Use LoRA when you have a small amount of clean, single-speaker data. It is the recommended starting point for voice
adaptation: it trains only a small set of adapter weights, is less prone to overfitting, and preserves more of the
base model's language and generation behavior. Start with `configs/lora-adamw.toml`.

Use a full fine-tune when you have substantially more varied data and need a more dramatic distribution shift, for
example a new language, speaking style, recording domain, or acoustic profile that LoRA does not learn sufficiently.
A full fine-tune updates the complete model, requires more compute and data, and has a greater risk of overfitting
or degrading behavior outside the target distribution. Use `configs/full.toml` or `configs/full-new-language.toml`
and validate both the target voice and any languages or capabilities you need to preserve.

```bash
torchrun --standalone --nproc-per-node=1 \
  finetune.py \
  --config configs/lora-adamw.toml \
  --data data/processed/expresso_ex01.pt \
  --output outputs/voice/checkpoint
```

| Config | Recipe | Use |
|---|---|---|
| `configs/lora-adamw.toml` | rank-32 attention/GDN/output LoRA, AdamW + LoRA+ | single voice, the transferable default |
| `configs/lora-muon.toml` | rank-16 attention/GDN LoRA, manifold LoRA-Muon | single voice, alternative optimizer |
| `configs/lora-all.toml` | rank-16 LoRA on attention, GDN, FFN, experts and the head | style / accent shifts |
| `configs/full.toml` | full-model Muon, release learning rates x 0.05 | larger datasets |
| `configs/full-new-language.toml` | full-model Muon, longer schedule, release conditioning mix | a new language |

LoRA recipes write an adapter checkpoint and periodic adapter snapshots. `metrics.jsonl` is written beside the
checkpoint. The full recipes write a public-inference `model.pth` directly. `--data` takes several processed files
(the shards of `preprocess.py --shard INDEX/COUNT`).

Full fine-tuning on one GPU keeps Muon's momentum in CPU memory and streams one matrix at a time through the GPU;
on multiple GPUs the full recipe uses Dion's distributed Muon path without CPU offload.

Draw mix and data options (`python finetune.py --help` lists everything):

| Option | Default | Effect |
|---|---|---|
| `--p-embedding` / `--p-reference` | 0.9 / 0 | share of draws with the speaker row / the reference prefix of another clip |
| `--embedding-other-clip` | 0 | share of speaker rows whose embedding comes from another clip of the speaker |
| `--condition-dropout` | 0.2 | per condition row, replace the bucket by "unknown" |
| `--reference-max-seconds` | 30 | longest reference prefix (the model accepts 60) |
| `--max-seq-len` | 8192 | longest packed example in rows (about 9 minutes of audio) |
| `--qb-update` / `--qb-ema` | off / 0.99 | per-step quantile-balancing update of the router biases (EMA-smoothed); off keeps the release routing |
| `--save-every` / `--resume` | 0 / off | full fine-tunes: a resumable training state (model, optimizer, loader position) every N steps and at the last step under the run directory; `--resume` continues from the latest one, for preemptible or time-limited jobs. A chained job (`sbatch --dependency=afterany:<id>`) with `--resume` exits at once when the run is already complete |
| `--gdn-backend` | auto | `fla` (Triton kernels) or the slow `torch` recurrence (tests) |

## Merge a LoRA adapter

```bash
python merge_adapter.py \
  --base-model artifacts/base_model \
  --adapter outputs/voice/checkpoint \
  --adapter-scale 1.0 \
  --output outputs/voice/merged
```

The merged directory contains `model.pth`, `params.json` and `tokenizer/` in the unmodified public ZONOS3-Mini
format.

## Generate audio

```bash
# the CUDA offline API (bash setup.sh nvidia --inference)
python infer.py --model outputs/voice/merged --text "This checkpoint is ready for a listening test." \
  --speaker-audio data/raw/voices/narrator/0001.wav --output outputs/voice/sample.wav

# the plain-PyTorch reference (slow, CPU-capable; bash setup.sh ... --preprocess)
python infer.py --backend reference --device cpu --model outputs/voice/merged --text "..." --output sample.wav
```

A fine-tuned voice still needs a speaker: `--speaker-audio` (its MECT embedding by default, `--speaker-mode
reference` for the raw prefix) or `--speaker-embedding emb.npy`. Without one the model speaks unconditioned.

To serve the checkpoint with the ZONOS3-Mini servers and the Zonos Voice Studio:

```bash
cd ZONOS3-Mini && SERVE_MODEL=/path/to/outputs/voice/merged servers/sgl/scripts/serve.sh
```

For the MLX, ggml and in-browser servers convert the checkpoint with `tools/convert_gguf.py lm` of the ZONOS3-Mini
repo.

## Tests

```bash
pytest tests/test_layout.py tests/test_model_tiny.py          # CPU, seconds
pytest tests/test_gdn_backends.py tests/test_gdn_backward.py  # CUDA: FLA kernels against the plain recurrence
ZONOS3_BASE_MODEL=artifacts/base_model pytest tests/test_release_checkpoint.py   # strict load of the release weights and
                                                               # logits parity with the zonos3 reference (--preprocess)
bash tests/gpu_smoke.sh 0                                      # everything plus short LoRA / full runs on synthetic data
```

The parity test reproduces the reference's float32 logits exactly (max difference 0.0 on the release weights).

## Layout

| File | Role |
|---|---|
| `model.py` | the trainable ZONOS3-Mini LM in the release checkpoint's parameter layout |
| `data.py` | examples, draw policy and packed sequences in the training loader's layout |
| `preprocess.py` | audio/text -> codec codes, speaker embeddings, text ids, condition buckets |
| `finetune.py` | FSDP2 trainer: LoRA (AdamW / LoRA-Muon) or full Muon, export |
| `lora.py`, `lora_muon.py` | adapters and the manifold optimizer |
| `merge_adapter.py`, `infer.py` | merge and listen |
