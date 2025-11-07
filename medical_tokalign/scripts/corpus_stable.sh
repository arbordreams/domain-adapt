#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

ROOT=/workspace/domain-adapt
cd "$ROOT"
LOG_DIR=$ROOT/medical_tokalign/runs/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/corpus_stable_${TS}.log"
CFG=medical_tokalign/configs/corpus_biomed.yaml

{
  echo "[preflight]"
  python -m medical_tokalign.src.cli build-corpus \
    --config "$CFG" \
    --preflight_only \
    --logdir "$LOG_DIR"
  echo "[build]"
  python -m medical_tokalign.src.cli build-corpus \
    --config "$CFG" \
    --strict_sources \
    --backfill_store \
    --logdir "$LOG_DIR"
} 2>&1 | tee -a "$LOG"
