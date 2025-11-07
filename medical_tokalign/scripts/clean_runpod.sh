#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"/..  # repo root

FRESH_CORPUS=0
KEEP_LATEST_LOG=1
for arg in "$@"; do
  case "$arg" in
    --fresh-corpus) FRESH_CORPUS=1 ;;
    --keep-all-logs) KEEP_LATEST_LOG=0 ;;
  esac
done

echo "[clean] Killing tmux sessions and pipeline processes..."
if command -v tmux >/dev/null 2>&1; then
  tmux kill-server || true
fi
pkill -9 -f "medical_tokalign\.src\.cli build-corpus" 2>/dev/null || true
pkill -9 -f "medical_tokalign\.src\.cli adapt" 2>/dev/null || true
pkill -9 -f "medical_tokalign\.src\.cli eval" 2>/dev/null || true
pkill -9 -f "medical_tokalign\.tools\.pipeline_runner" 2>/dev/null || true
pkill -9 -f "run_unattended\.sh" 2>/dev/null || true
pkill -9 -f "runpod_start\.sh" 2>/dev/null || true
pkill -9 -f "vllm" 2>/dev/null || true

RUNS_DIR="$ROOT_DIR/runs"
LOGDIR="$RUNS_DIR/logs"
BIO_DIR="$ROOT_DIR/data/biomed_corpus"

echo "[clean] Removing vestigial logs, tmp and zero-byte files..."
mkdir -p "$LOGDIR"
find "$LOGDIR" -type f -name "._*" -delete || true
if [ $KEEP_LATEST_LOG -eq 1 ]; then
  latest=$(ls -1t "$LOGDIR"/corpus_*.log 2>/dev/null | head -n1 || true)
  for f in "$LOGDIR"/corpus_*.log; do
    [ "$f" = "$latest" ] || rm -f "$f" || true
  done
  latestc=$(ls -1t "$LOGDIR"/corpus_*_captured.log 2>/dev/null | head -n1 || true)
  for f in "$LOGDIR"/corpus_*_captured.log; do
    [ "$f" = "$latestc" ] || rm -f "$f" || true
  done
else
  rm -f "$LOGDIR"/corpus_*.log "$LOGDIR"/corpus_*_captured.log 2>/dev/null || true
fi
find "$LOGDIR" -type f -size 0 -delete || true

echo "[clean] Removing incomplete runs..."
if [ -d "$RUNS_DIR/tokenizer_adapt" ]; then
  find "$RUNS_DIR/tokenizer_adapt" -mindepth 1 -maxdepth 1 -type d | while read -r d; do
    [ -d "$d/model" ] && [ -d "$d/tokenizer" ] && [ -f "$d/align_matrix.json" ] && continue || rm -rf "$d" || true
  done
fi
if [ -d "$RUNS_DIR/medical_eval" ]; then
  find "$RUNS_DIR/medical_eval" -mindepth 1 -maxdepth 1 -type d | while read -r d; do
    [ -f "$d/metrics.json" ] || rm -rf "$d" || true
  done
fi

echo "[clean] Cleaning corpus leftovers..."
if [ -d "$BIO_DIR" ]; then
  find "$BIO_DIR" -maxdepth 1 -type f -name "*.tmp" -delete || true
  find "$BIO_DIR" -maxdepth 1 -type f -size 0 -delete || true
  rm -f "$BIO_DIR/summary.json" || true
  if [ $FRESH_CORPUS -eq 1 ]; then
    rm -rf "$BIO_DIR" || true
  fi
fi

echo "[clean] Done."


