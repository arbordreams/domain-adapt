#!/usr/bin/env bash
set -euo pipefail

# Defaults (override via flags)
MODEL_ID="Qwen/Qwen2-7B"
TOP_K=8192
PIVOT=300
WARMUP_STEPS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_id) MODEL_ID="$2"; shift 2;;
    --top_k) TOP_K="$2"; shift 2;;
    --pivot) PIVOT="$2"; shift 2;;
    --warmup_steps) WARMUP_STEPS="$2"; shift 2;;
    *) echo "[autorun_stable] Unknown flag $1"; exit 1;;
  esac
done

# Resolve paths
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"      # medical_tokalign
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"                           # repo root
cd "$REPO_ROOT"

# Env defaults
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"

LOGDIR="$ROOT_DIR/runs/logs"
mkdir -p "$LOGDIR"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE_LOG="$LOGDIR/autorun_stable_${TS}.log"

{
  echo "[autorun_stable] prepare-data --all"
  python -m medical_tokalign.src.cli prepare-data --all

  echo "[autorun_stable] corpus_stable.sh"
  bash "$ROOT_DIR/scripts/corpus_stable.sh"

  echo "[autorun_stable] adapt model_id=$MODEL_ID top_k=$TOP_K pivot=$PIVOT warmup_steps=$WARMUP_STEPS"
  python -m medical_tokalign.src.cli adapt \
    --model_id "$MODEL_ID" \
    --top_k "$TOP_K" \
    --pivot "$PIVOT" \
    --warmup_steps "$WARMUP_STEPS"

  echo "[autorun_stable] eval"
  python -m medical_tokalign.src.cli eval \
    --config "$ROOT_DIR/configs/eval_medical.yaml"

  echo "[autorun_stable] DONE"
} 2>&1 | tee -a "$PIPE_LOG"

echo "[autorun_stable] Logs: $PIPE_LOG"


