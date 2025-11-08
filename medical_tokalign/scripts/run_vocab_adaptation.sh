#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
echo "This script is deprecated."
echo "Use medical_tokalign/scripts/autorun_demo.sh or medical_tokalign/scripts/autorun_prod.sh instead."
echo "Example:"
echo "  EMBEDDING_BACKEND=glove bash medical_tokalign/scripts/autorun_prod.sh --model_id <MODEL_ID>"
exit 2

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="Qwen/Qwen2-7B"
TOP_K=8192
PIVOT=300
WARMUP_STEPS=0

usage() {
  echo "Usage: $0 [--model_id MODEL] [--top_k N] [--pivot N] [--warmup_steps N]" >&2
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --model_id) MODEL_ID="$2"; shift 2;;
    --top_k) TOP_K="$2"; shift 2;;
    --pivot) PIVOT="$2"; shift 2;;
    --warmup_steps) WARMUP_STEPS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

echo "[MedTokAlign] Running unified CLI adaptation (TokAlign faithful) ..."

python -m medical_tokalign.src.cli adapt \
  --model_id "$MODEL_ID" \
  --top_k "$TOP_K" \
  --pivot "$PIVOT" \
  --warmup_steps "$WARMUP_STEPS"


