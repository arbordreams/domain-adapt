#!/usr/bin/env bash
set -euo pipefail

MODEL_ID=${MODEL_ID:-${1:-Qwen/Qwen2-7B}}
TOP_K=${TOP_K:-8192}
PIVOT=${PIVOT:-300}
WARMUP_STEPS=${WARMUP_STEPS:-0}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

# Conservative defaults for GloVe and threads
RAM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 65536)
: "${GLOVE_MEMORY_MB:=$(( RAM_MB / 3 ))}"
if [ "$GLOVE_MEMORY_MB" -lt 8192 ]; then GLOVE_MEMORY_MB=8192; fi
if [ "$GLOVE_MEMORY_MB" -gt 65536 ]; then GLOVE_MEMORY_MB=65536; fi
: "${GLOVE_THREADS:=$(( $(nproc 2>/dev/null || echo 32) - 1 ))}"
if [ "$GLOVE_THREADS" -lt 1 ]; then GLOVE_THREADS=1; fi
export GLOVE_MEMORY_MB GLOVE_THREADS

ROOT=/workspace/domain-adapt
cd "$ROOT"
LOG_DIR=$ROOT/medical_tokalign/runs/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/autorun_stable_${TS}.log"
EVAL_CFG=medical_tokalign/configs/eval_medical.yaml

{
  echo "[env] RAM_MB=$RAM_MB GLOVE_MEMORY_MB=$GLOVE_MEMORY_MB GLOVE_THREADS=$GLOVE_THREADS"
  # Optional prep if present
  if [ -x medical_tokalign/scripts/prepare_medical_data.sh ]; then
    echo "[prepare]"; bash medical_tokalign/scripts/prepare_medical_data.sh || true
  fi
  echo "[corpus]"; bash medical_tokalign/scripts/corpus_stable.sh
  echo "[adapt]";
  set +e
  python -m medical_tokalign.src.cli adapt \
    --model_id "$MODEL_ID" \
    --top_k "$TOP_K" \
    --pivot "$PIVOT" \
    --warmup_steps "$WARMUP_STEPS"
  ADAPT_RC=$?
  set -e
  if [ "$ADAPT_RC" -ne 0 ]; then
    echo "[adapt] failed rc=$ADAPT_RC; retrying with higher memory"
    GLOVE_MEMORY_MB=$(( GLOVE_MEMORY_MB * 2 ))
    if [ "$GLOVE_MEMORY_MB" -gt 65536 ]; then GLOVE_MEMORY_MB=65536; fi
    export GLOVE_MEMORY_MB
    python -m medical_tokalign.src.cli adapt \
      --model_id "$MODEL_ID" \
      --top_k "$TOP_K" \
      --pivot "$PIVOT" \
      --warmup_steps "$WARMUP_STEPS"
    ADAPT_RC=$?
  fi
  if [ "$ADAPT_RC" -ne 0 ]; then
    echo "[autorun] adapt failed after retry; exiting before eval"
    exit 1
  fi
  echo "[eval]"; python -m medical_tokalign.src.cli eval --config "$EVAL_CFG"
} 2>&1 | tee -a "$LOG"
