#!/usr/bin/env bash
# Bootstrap Python environment with pinned deps and runtime preflight
set -euo pipefail
IFS=$'\n\t'

RDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ_ROOT="$(cd "${RDIR}/.." && pwd)"

usage() {
  cat <<EOF
Usage: $0 [--gpu-cu12] [--no-install] [--print-env]
  --gpu-cu12     Use requirements-gpu-cu12.txt (CUDA 12.x systems)
  --no-install   Do not run pip install; only print preflight
  --print-env    Print environment exports to eval in shell
EOF
}

GPU=0
DO_INSTALL=1
PRINT_ENV=0
for arg in "$@"; do
  case "${arg}" in
    --gpu-cu12) GPU=1 ;;
    --no-install) DO_INSTALL=0 ;;
    --print-env) PRINT_ENV=1 ;;
    -h|--help) usage; exit 0 ;;
    *) ;;
  esac
done

# Detect CUDA if not explicitly requested
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU=1
fi

REQ_BASE="${PROJ_ROOT}/requirements.txt"
REQ_GPU="${PROJ_ROOT}/requirements-gpu-cu12.txt"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
# Avoid vision imports unless explicitly needed
export TRANSFORMERS_NO_TORCHVISION="${TRANSFORMERS_NO_TORCHVISION:-1}"

if [ "${DO_INSTALL}" = "1" ]; then
  export PIP_BREAK_SYSTEM_PACKAGES=1
  python -m pip install -U pip >/dev/null 2>&1 || true
  if [ "${GPU}" = "1" ]; then
    echo "[bootstrap] Installing GPU requirements: ${REQ_GPU}"
    python -m pip install -r "${REQ_GPU}"
  else
    echo "[bootstrap] Installing CPU requirements: ${REQ_BASE}"
    python -m pip install -r "${REQ_BASE}"
  fi
fi

preflight_py=$(cat <<'PY'
import os, sys
def getver(mod):
    try:
        m=__import__(mod)
        return getattr(m, "__version__", "unknown")
    except Exception as e:
        return f"missing ({e.__class__.__name__})"
res={}
for m in ("torch","torchvision","torchaudio","transformers","datasets","huggingface_hub","hf_transfer","gensim","numpy"):
    res[m]=getver(m)
print("[versions]", " ".join(f"{k}={v}" for k,v in res.items()))
print("[env] HF_HUB_ENABLE_HF_TRANSFER=%s TRANSFORMERS_NO_TORCHVISION=%s" % (
    os.environ.get("HF_HUB_ENABLE_HF_TRANSFER",""), os.environ.get("TRANSFORMERS_NO_TORCHVISION","")))
PY
)
python - <<PY
${preflight_py}
PY

if [ "${PRINT_ENV}" = "1" ]; then
  cat <<EOF
export HF_HOME="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export HF_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER}"
export TRANSFORMERS_NO_TORCHVISION="${TRANSFORMERS_NO_TORCHVISION}"
EOF
fi

echo "[bootstrap] Done."

