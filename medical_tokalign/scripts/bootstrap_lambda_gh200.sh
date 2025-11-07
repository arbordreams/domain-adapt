#!/usr/bin/env bash
set -euo pipefail

# Bootstrap environment for Lambda Labs GH200 (ARM64 + H100 96GB)
# - Creates Python venv
# - Installs PyTorch (CUDA 12.x, aarch64)
# - Installs project requirements
# - Attempts to build flash-attn for H100 (optional)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.."
cd "$REPO_ROOT"

echo "[GH200] Detected arch: $(uname -m) | NVIDIA: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"

# Prefer /data for caches if available
CACHE_ROOT_DEFAULT="/data/medtokalign_cache"
if [[ -d "/data" ]]; then
  CACHE_ROOT="$CACHE_ROOT_DEFAULT"
else
  CACHE_ROOT="$HOME/.cache/medtokalign"
fi
mkdir -p "$CACHE_ROOT" "$CACHE_ROOT/hf" "$CACHE_ROOT/hf/transformers" "$CACHE_ROOT/hf/datasets" "$CACHE_ROOT/torch_extensions"

export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_ROOT/hf/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/hf/datasets}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE_ROOT/torch_extensions}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# System build tooling (requires sudo). Skip if not available.
if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update -y || true
  sudo apt-get install -y build-essential ninja-build cmake git pkg-config || true
fi


PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -d "$VENV_DIR" ]]; then
  $PYTHON_BIN -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo "[GH200] Installing PyTorch for CUDA 12.x on aarch64..."
# Use the official PyTorch CUDA 12.4 index (works for aarch64 server builds)
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

echo "[GH200] Installing project requirements..."
pip install -r medical_tokalign/requirements.txt || true

echo "[GH200] Attempting flash-attn build for H100 (optional)..."
# Build from source targeting Hopper (sm_90). If this fails, SDPA will be used instead.
export MAX_JOBS=${MAX_JOBS:-64}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-90}
export FLASH_ATTENTION_FORCE_BUILD=1
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
(
  pip install --no-build-isolation --no-binary :all: flash-attn &&
  python -c 'import flash_attn; print("[GH200] flash-attn available:", flash_attn.__version__)'
) || echo "[GH200] flash-attn build failed; proceeding with SDPA attention."

echo "[GH200] Done. Activate venv with: source $VENV_DIR/bin/activate"
echo "[GH200] Run eval with: ./medical_tokalign/scripts/eval_medical.sh --config medical_tokalign/configs/eval_medical_gh200.yaml"


