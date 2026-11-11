#!/bin/bash
# Gradient-norm diagnostic with the plain-PyTorch GDN recurrence (slow) and per-parameter gradient norms.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${1:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
source .venv/bin/activate
export ZONOS3_GRAD_DEBUG=1
for backend in fla torch; do
  echo "== gdn_$backend"
  torchrun --standalone --nproc-per-node=1 finetune.py --config configs/full.toml \
    --data data/processed/synthetic.pt --output "outputs/diag_gdn_$backend/checkpoint" \
    --steps 6 --batch-size 2 --grad-accum 2 --warmup-steps 1 --lr-scale 0 --num-workers 0 --no-export-merged \
    --no-compile --no-qb-update --gdn-backend "$backend" 2>&1 | grep -E "^step=|^grad norms|Traceback|Error" | cut -c1-400
done
echo "== diagnostic complete"
