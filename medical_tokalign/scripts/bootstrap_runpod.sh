#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[MedTokAlign] Bootstrapping RunPod environment..."

# Accelerate Hugging Face downloads via hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

# Recommended env for memory behavior
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if command -v apt-get >/dev/null 2>&1; then
  echo "[MedTokAlign] Installing OS build tools (requires root)..."
  apt-get update -y
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential git ca-certificates curl
fi

echo "[MedTokAlign] Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

echo "[MedTokAlign] Installing Python dependencies (excluding torch which should come with the image)..."
# Install flash-attn from specific wheel
if ! python - <<'PY'
try:
    import flash_attn  # noqa: F401
    print("ok")
except Exception:
    raise SystemExit(1)
PY
then
  echo "[MedTokAlign] Installing flash-attn from prebuilt wheel..."
  python -m pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl || true
fi

# Install hf_transfer for accelerated dataset downloads when HF_HUB_ENABLE_HF_TRANSFER=1
if ! python - <<'PY'
try:
    import hf_transfer  # noqa: F401
    print("ok")
except Exception:
    raise SystemExit(1)
PY
then
  echo "[MedTokAlign] Installing hf_transfer for fast Hugging Face downloads..."
  python -m pip install --no-cache-dir hf_transfer || true
fi

python -m pip install -r "$ROOT_DIR/requirements.txt"

if [[ -n "${SCISPACY_MODEL_URL:-}" ]]; then
  echo "[MedTokAlign] Installing SciSpacy model from $SCISPACY_MODEL_URL"
  python -m pip install --no-cache-dir "$SCISPACY_MODEL_URL" || true
fi

TOOLS_DIR="$ROOT_DIR/tools"
GLOVE_DIR="$TOOLS_DIR/GloVe"
if [[ ! -d "$GLOVE_DIR" ]]; then
  echo "[MedTokAlign] Cloning Stanford GloVe..."
  mkdir -p "$TOOLS_DIR"
  git clone https://github.com/stanfordnlp/GloVe.git "$GLOVE_DIR"
fi

echo "[MedTokAlign] Building GloVe binaries..."
make -C "$GLOVE_DIR"

# Enforce flash-attn availability post-install
if ! python - <<'PY'
import sys
try:
    import flash_attn  # noqa: F401
except Exception:
    sys.stderr.write("\n[MedTokAlign] ERROR: flash-attn is required and failed to import. Install a CUDA-compatible wheel for your Torch/CUDA.\n")
    raise SystemExit(1)
PY
then
  exit 1
fi

echo "[MedTokAlign] Bootstrap complete. Next steps:"
echo "  1) bash $ROOT_DIR/scripts/prepare_medical_data.sh"
echo "  2) bash $ROOT_DIR/scripts/run_vocab_adaptation.sh --model_id Qwen/Qwen2-7B --top_k 8192"
echo "  3) bash $ROOT_DIR/scripts/eval_medical.sh --config $ROOT_DIR/configs/eval_medical.yaml"


