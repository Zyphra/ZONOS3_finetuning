#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

if ! uv help pip install | grep -q -- "--torch-backend"; then
    echo "This setup requires uv with --torch-backend support. Update uv and try again." >&2
    exit 2
fi

usage() {
    cat <<'USAGE'
Usage: bash setup.sh [auto|nvidia|amd] [--preprocess] [--inference]

  auto          Detect the installed GPU driver (default).
  nvidia        Install PyTorch 2.9.1 for CUDA 12.8.
  amd           Install PyTorch 2.9.1 for ROCm 6.4.
  --preprocess  Also install the ZONOS3-Mini model package (codec encoder, speaker encoder, text normalizer)
                that preprocess.py and `infer.py --backend reference` use.
  --inference   Also install the ZONOS3-Mini CUDA runtime (zonos3-sgl) for `infer.py --backend sgl` (NVIDIA only).

The environment is created at .venv (ZONOS3_VENV overrides the directory). The base checkpoint is downloaded from
the public Hugging Face repo Zyphra/ZONOS3-Mini into artifacts/ (ZONOS3_REVISION pins the revision).
USAGE
}

GPU_BACKEND="auto"
INSTALL_PREPROCESS=0
INSTALL_INFERENCE=0
for argument in "$@"; do
    case "${argument}" in
        auto|nvidia|amd) GPU_BACKEND="${argument}" ;;
        --preprocess) INSTALL_PREPROCESS=1 ;;
        --inference) INSTALL_INFERENCE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

case "${GPU_BACKEND}" in
    auto) TORCH_BACKEND="auto" ;;
    nvidia) TORCH_BACKEND="cu128" ;;
    amd) TORCH_BACKEND="rocm6.4" ;;
esac

if [[ "${INSTALL_INFERENCE}" == "1" && "${GPU_BACKEND}" != "nvidia" ]]; then
    echo "The ZONOS3-Mini CUDA runtime requires the explicit nvidia backend." >&2
    exit 2
fi

VENV_PATH="$(realpath -m "${ZONOS3_VENV:-${PROJECT_ROOT}/.venv}")"
if [[ "${VENV_PATH}" == "/" || "${VENV_PATH}" == "${PROJECT_ROOT}" || "${VENV_PATH}" == "${HOME:-}" ]]; then
    echo "Refusing to clear unsafe environment path: ${VENV_PATH}" >&2
    exit 2
fi
if [[ -e "${VENV_PATH}" && ! -f "${VENV_PATH}/pyvenv.cfg" ]]; then
    echo "Refusing to clear an existing directory that is not a Python virtual environment: ${VENV_PATH}" >&2
    exit 2
fi
mkdir -p artifacts
export HF_HOME="${PROJECT_ROOT}/artifacts/hf_cache"
export UV_CACHE_DIR="${PROJECT_ROOT}/artifacts/uv_cache"
ZONOS3_REVISION="${ZONOS3_REVISION:-main}"

uv venv --clear --python 3.12 "${VENV_PATH}"
INSTALL_ARGUMENTS=(
    --python "${VENV_PATH}/bin/python"
    --torch-backend "${TORCH_BACKEND}"
    --requirements pyproject.toml
    --group dev
)
if [[ "${GPU_BACKEND}" == "nvidia" ]]; then
    INSTALL_ARGUMENTS+=(--extra hopper)   # TileLang: the GDN backward on H100-class GPUs (needs nvcc / CUDA_HOME)
fi
if [[ "${INSTALL_PREPROCESS}" == "1" ]]; then
    INSTALL_ARGUMENTS+=(--extra preprocess)
fi
if [[ "${INSTALL_INFERENCE}" == "1" ]]; then
    INSTALL_ARGUMENTS+=(--extra inference-nvidia)
fi
uv pip install "${INSTALL_ARGUMENTS[@]}"

"${VENV_PATH}/bin/python" - "${GPU_BACKEND}" <<'PY'
import sys

import torch

requested = sys.argv[1]
if torch.version.hip:
    detected, runtime = "amd", f"ROCm {torch.version.hip}"
elif torch.version.cuda:
    detected, runtime = "nvidia", f"CUDA {torch.version.cuda}"
else:
    detected, runtime = "cpu", "CPU-only PyTorch"
print(f"Installed torch {torch.__version__}: {runtime}")
if requested != "auto" and detected != requested:
    raise SystemExit(f"Requested {requested}, but uv installed the {detected} PyTorch build")
if requested == "auto" and detected == "cpu":
    raise SystemExit(
        "Automatic detection installed CPU-only PyTorch. "
        "Run `bash setup.sh nvidia` or `bash setup.sh amd` explicitly on GPU-less login nodes."
    )
PY

# The language model (model.pth, params.json, tokenizer/) and the speaker encoder + registries the preprocessing uses.
"${VENV_PATH}/bin/hf" download Zyphra/ZONOS3-Mini \
  --include 'lm/zonos3_mini/*' --include 'speaker_encoders/*' --include 'registries/*' \
  --revision "${ZONOS3_REVISION}" \
  --local-dir "${PROJECT_ROOT}/artifacts/zonos3_mini"
ln -sfn "${PROJECT_ROOT}/artifacts/zonos3_mini/lm/zonos3_mini" "${PROJECT_ROOT}/artifacts/base_model"
"${VENV_PATH}/bin/python" -c "import model; print('PyTorch training model import succeeded')"

echo "Setup complete."
echo "Backend: ${GPU_BACKEND} (${TORCH_BACKEND})"
echo "Environment: ${VENV_PATH}"
echo "Base checkpoint: ${PROJECT_ROOT}/artifacts/base_model"
