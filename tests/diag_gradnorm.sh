#!/bin/bash
# Gradient-norm diagnostic: the full recipe with the learning rate at zero (weights never move) over the same
# synthetic batches, eager vs compiled, with and without the quantile-balancing update.
#   bash tests/diag_gradnorm.sh [GPU index]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${1:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
source .venv/bin/activate
run() {
  local name="$1"; shift
  echo "== $name"
  torchrun --standalone --nproc-per-node=1 finetune.py --config configs/full.toml \
    --data data/processed/synthetic.pt --output "outputs/diag_$name/checkpoint" \
    --steps 6 --batch-size 2 --grad-accum 2 --warmup-steps 1 --lr-scale 0 --num-workers 0 --no-export-merged "$@" \
    2>&1 | grep -E "^step=|Traceback|Error" | cut -c1-160
}
run eager_noqb --no-compile --no-qb-update
run eager_qb --no-compile --qb-update
run compiled_noqb --compile --no-qb-update
echo "== diagnostic complete"
