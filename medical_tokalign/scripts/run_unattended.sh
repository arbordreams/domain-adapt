#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
echo "This script is deprecated."
echo "Use medical_tokalign/scripts/autorun_demo.sh or medical_tokalign/scripts/autorun_prod.sh instead."
echo "Example:"
echo "  bash medical_tokalign/scripts/autorun_demo.sh --model_id meta-llama/Llama-3-8b --quick"
exit 2

#!/usr/bin/env bash
set -euo pipefail

# Unattended MedTokAlign Orchestrator wrapper
# Usage examples:
#   bash medical_tokalign/scripts/run_unattended.sh \
#     --model_id Qwen/Qwen2-7B \
#     --corpus_config medical_tokalign/configs/corpus_biomed.yaml \
#     --eval_config medical_tokalign/configs/eval_medical.yaml \
#     --top_k 8192 --pivot 300 --warmup_steps 0
#
#   bash medical_tokalign/scripts/run_unattended.sh --tmux --step_timeout 28800 --max_retries 2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Recommended env for RunPod H100
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export HF_DATASETS_TRUST_REMOTE_CODE=${HF_DATASETS_TRUST_REMOTE_CODE:-1}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/workspace/.cache/huggingface/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/workspace/.cache/huggingface/transformers}
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# Auto-detect CPU cores and set GloVe threads if not provided (leave one core free)
if command -v nproc >/dev/null 2>&1; then
  __CORES=$(nproc)
elif command -v getconf >/dev/null 2>&1; then
  __CORES=$(getconf _NPROCESSORS_ONLN || echo 1)
else
  __CORES=$(python - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)
fi
if [[ -n "${__CORES:-}" ]]; then
  export GLOVE_THREADS=${GLOVE_THREADS:-$(( __CORES>1 ? __CORES-1 : 1 ))}
fi

TMUX_MODE=0
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tmux) TMUX_MODE=1; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

PY_CMD=("python" "-m" "medical_tokalign.tools.pipeline_runner" "${ARGS[@]}")

LOG_DIR="$ROOT_DIR/runs/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/pipeline_${TS}.log"

if [[ "$TMUX_MODE" -eq 1 ]]; then
  SESSION="medtok_${TS}"
  echo "[run_unattended] Launching in tmux session: $SESSION"
  tmux new-session -d -s "$SESSION" "${PY_CMD[@]} | tee -a '$LOG_FILE'"
  echo "Attach: tmux attach -t $SESSION"
  echo "Logs:   $LOG_FILE"
else
  echo "[run_unattended] Running orchestrator... logs -> $LOG_FILE"
  "${PY_CMD[@]}" | tee -a "$LOG_FILE"
fi


