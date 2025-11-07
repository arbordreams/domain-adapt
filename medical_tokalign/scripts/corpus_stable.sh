#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"      # medical_tokalign
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"                           # repo root
cd "$REPO_ROOT"

# Env defaults (non-invasive)
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"

CONFIG_PATH="$ROOT_DIR/configs/corpus_biomed.yaml"
LOGDIR="${LOGDIR:-$ROOT_DIR/runs/logs}"
mkdir -p "$LOGDIR"

TS="$(date +%Y%m%d_%H%M%S)"
TEXT_LOG="$LOGDIR/corpus_stable_${TS}.log"

echo "[corpus_stable] Preflight only..." | tee -a "$TEXT_LOG"
python -m medical_tokalign.src.cli build-corpus \
  --config "$CONFIG_PATH" \
  --preflight_only \
  --logdir "$LOGDIR" 2>&1 | tee -a "$TEXT_LOG"

echo "[corpus_stable] Building corpus (strict, resumable)..." | tee -a "$TEXT_LOG"
python -m medical_tokalign.src.cli build-corpus \
  --config "$CONFIG_PATH" \
  --strict_sources \
  --logdir "$LOGDIR" 2>&1 | tee -a "$TEXT_LOG"

echo "[corpus_stable] Done. Logs: $TEXT_LOG" | tee -a "$TEXT_LOG"


