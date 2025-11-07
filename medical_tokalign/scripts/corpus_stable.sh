#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"      # medical_tokalign
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"                           # repo root
cd "$REPO_ROOT"

# Env defaults (non-invasive)
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-1}"

# Allow override via CORPUS_CONFIG; default to standard biomed config
CONFIG_PATH="${CORPUS_CONFIG:-$ROOT_DIR/configs/corpus_biomed.yaml}"
LOGDIR="${LOGDIR:-$ROOT_DIR/runs/logs}"
mkdir -p "$LOGDIR"

TS="$(date +%Y%m%d_%H%M%S)"
TEXT_LOG="$LOGDIR/corpus_stable_${TS}.log"

# Skip if corpus already exists (check for any .jsonl files in biomed_corpus)
BIOMED_DIR="$ROOT_DIR/data/biomed_corpus"
if [[ -d "$BIOMED_DIR" ]] && [[ -n "$(find "$BIOMED_DIR" -maxdepth 1 -name "*.jsonl" 2>/dev/null | head -n1)" ]]; then
  echo "[corpus_stable] Corpus already exists, skipping build" | tee -a "$TEXT_LOG"
  echo "[corpus_stable] Found files:" | tee -a "$TEXT_LOG"
  ls -lh "$BIOMED_DIR"/*.jsonl 2>/dev/null | tee -a "$TEXT_LOG" || true
  exit 0
fi

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
