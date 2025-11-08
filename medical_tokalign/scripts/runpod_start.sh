#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
echo "This script is deprecated."
echo "Use medical_tokalign/scripts/autorun_demo.sh or medical_tokalign/scripts/autorun_prod.sh instead."
echo "Example:"
echo "  bash medical_tokalign/scripts/autorun_prod.sh --model_id <MODEL_ID>"
exit 2

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[RunPod] Starting MedTokAlign wrapper from $ROOT_DIR"

# Set recommended environment variables for RunPod H100
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

# Execute the Python wrapper with strict validations and artifact checks.
# Customize flags via environment variables if needed.
PY_WRAP_TIMEOUT_SECONDS=${PY_WRAP_TIMEOUT_SECONDS:-14400}

# We are now in medical_tokalign/. Run the wrapper relative to this directory.
python -u ./runpod_wrapper.py \
  --timeout "$PY_WRAP_TIMEOUT_SECONDS" \
  --require-flash-attn \
  --require-hf-token \
  --verify-glove-tools


