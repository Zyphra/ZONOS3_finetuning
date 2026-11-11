#!/bin/bash
# GPU smoke test: kernel parity, the release checkpoint, a LoRA run and a full run on synthetic data, merge, export.
#   bash tests/gpu_smoke.sh [GPU index]   (run from the repo root with .venv and artifacts/base_model in place)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${1:-0}"
export ZONOS3_BASE_MODEL=artifacts/base_model
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"   # TileLang JIT needs nvcc on Hopper
export PATH="${CUDA_HOME}/bin:${PATH}"
source .venv/bin/activate
echo "== python $(python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))')"

echo "== unit tests + kernels + release checkpoint"
python -m pytest tests/test_layout.py tests/test_model_tiny.py tests/test_gdn_backends.py tests/test_release_checkpoint.py -q

echo "== synthetic data"
python tests/make_synthetic_dataset.py --base-model artifacts/base_model --output data/processed/synthetic.pt --examples 24

echo "== LoRA run (eager)"
torchrun --standalone --nproc-per-node=1 finetune.py --config configs/lora-adamw.toml \
  --data data/processed/synthetic.pt --output outputs/smoke_lora/checkpoint \
  --steps 4 --eval-every 2 --save-every 2 --num-workers 2 --no-compile
python merge_adapter.py --base-model artifacts/base_model --adapter outputs/smoke_lora/checkpoint --output outputs/smoke_lora/merged
ZONOS3_BASE_MODEL=outputs/smoke_lora/merged python -m pytest tests/test_release_checkpoint.py -q -k strict_load

echo "== full run (compiled, quantile balancing on)"
torchrun --standalone --nproc-per-node=1 finetune.py --config configs/full.toml \
  --data data/processed/synthetic.pt --output outputs/smoke_full/checkpoint \
  --steps 3 --batch-size 2 --grad-accum 2 --warmup-steps 1 --num-workers 2
ZONOS3_BASE_MODEL=outputs/smoke_full/checkpoint python -m pytest tests/test_release_checkpoint.py -q -k strict_load

echo "== LoRA 'all' preset, compiled"
torchrun --standalone --nproc-per-node=1 finetune.py --config configs/lora-all.toml \
  --data data/processed/synthetic.pt --output outputs/smoke_lora_all/checkpoint \
  --steps 2 --eval-every 1 --num-workers 2 --export-merged
ZONOS3_BASE_MODEL=outputs/smoke_lora_all/checkpoint python -m pytest tests/test_release_checkpoint.py -q -k strict_load
echo "== smoke test complete"
