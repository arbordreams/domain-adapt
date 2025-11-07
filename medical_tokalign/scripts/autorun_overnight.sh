#!/usr/bin/env bash
set -euo pipefail

# Highest-quality overnight autorun:
# - Adapt (TokAlign-style) with strong settings
# - Eval
# - Always archive logs/artifacts, push to GitHub (LFS for tarball)
# - Then shutdown the machine (even on partial failure)

MODEL_ID="Qwen/Qwen2-7B"
TOP_K=8192
PIVOT=300
WARMUP_STEPS=3000
# Auto-shutdown guards
IDLE_MINUTES=120       # shut down if no log activity for N minutes (default 2h)
WALL_MINUTES=720       # hard wall-time in minutes before shutdown (default 12h)
SHUTDOWN_DELAY=600     # seconds to wait before shutdown to allow pushes to finish (default 10m)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_id) MODEL_ID="$2"; shift 2;;
    --top_k) TOP_K="$2"; shift 2;;
    --pivot) PIVOT="$2"; shift 2;;
    --warmup_steps) WARMUP_STEPS="$2"; shift 2;;
    --idle_minutes) IDLE_MINUTES="$2"; shift 2;;
    --wall_minutes) WALL_MINUTES="$2"; shift 2;;
    --shutdown_delay) SHUTDOWN_DELAY="$2"; shift 2;;
    *) echo "[overnight] Unknown flag: $1"; exit 1;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Environment (safe defaults)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export HF_DATASETS_TRUST_REMOTE_CODE=${HF_DATASETS_TRUST_REMOTE_CODE:-1}

LOGDIR="$ROOT_DIR/runs/logs"
mkdir -p "$LOGDIR"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE_LOG="$LOGDIR/overnight_${TS}.log"

on_exit() {
  set +e
  echo "[overnight] archiving artifacts..." | tee -a "$PIPE_LOG"
  mkdir -p artifacts
  ART="artifacts/run_${TS}.tar.gz"
  ARTIFACTS_DIR="$REPO_ROOT/artifacts"
  MODELS_DIR="$ARTIFACTS_DIR/models/run_${TS}"
  EVALS_DIR="$ARTIFACTS_DIR/evals/run_${TS}"
  LOGS_DIR_OUT="$ARTIFACTS_DIR/logs/run_${TS}"
  CORPUS_DIR_OUT="$ARTIFACTS_DIR/corpus"
  # Find latest adapt/eval dirs (best-effort)
  ADAPT_DIR=$(ls -td "$ROOT_DIR/runs/tokenizer_adapt"/* 2>/dev/null | head -n1 || true)
  EVAL_DIR=$(ls -td "$ROOT_DIR/runs/medical_eval"/* 2>/dev/null | head -n1 || true)
  tar -C "$REPO_ROOT" -czf "$ART" \
    medical_tokalign/runs/logs \
    ${ADAPT_DIR#"$REPO_ROOT/"} \
    ${EVAL_DIR#"$REPO_ROOT/"} \
    medical_tokalign/data/biomed_corpus/summary.json 2>/dev/null || true
  ls -lh "$ART" || true

  # Also copy raw directories to workspace (non-tar) for permanence
  mkdir -p "$MODELS_DIR" "$EVALS_DIR" "$LOGS_DIR_OUT" "$CORPUS_DIR_OUT"
  if [ -n "$ADAPT_DIR" ] && [ -d "$ADAPT_DIR" ]; then
    cp -a "$ADAPT_DIR"/. "$MODELS_DIR"/ || true
  fi
  if [ -n "$EVAL_DIR" ] && [ -d "$EVAL_DIR" ]; then
    cp -a "$EVAL_DIR"/. "$EVALS_DIR"/ || true
  fi
  if [ -d "$ROOT_DIR/runs/logs" ]; then
    cp -a "$ROOT_DIR/runs/logs"/. "$LOGS_DIR_OUT"/ || true
  fi
  if [ -f "$ROOT_DIR/data/biomed_corpus/summary.json" ]; then
    cp -a "$ROOT_DIR/data/biomed_corpus/summary.json" "$CORPUS_DIR_OUT/summary_${TS}.json" || true
  fi
  sync || true

  # Reproducibility manifest
  MAN="$ARTIFACTS_DIR/run_${TS}_manifest.json"
  GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
  HOSTNAME=$(hostname 2>/dev/null || echo "unknown")
  {
    echo "{"
    echo "  \"timestamp\": \"$TS\","
    echo "  \"model_id\": \"$MODEL_ID\","
    echo "  \"top_k\": $TOP_K,"
    echo "  \"pivot\": $PIVOT,"
    echo "  \"warmup_steps\": $WARMUP_STEPS,"
    echo "  \"idle_minutes\": $IDLE_MINUTES,"
    echo "  \"wall_minutes\": $WALL_MINUTES,"
    echo "  \"shutdown_delay\": $SHUTDOWN_DELAY,"
    echo "  \"adapt_rc\": ${ADAPT_RC:--1},"
    echo "  \"eval_rc\": ${EVAL_RC:--1},"
    echo "  \"git_commit\": \"$GIT_COMMIT\","
    echo "  \"hostname\": \"$HOSTNAME\""
    echo "}"
  } > "$MAN" || true

  # Push tarball via Git LFS (best-effort)
  if command -v git >/dev/null 2>&1; then
    git lfs install || true
    git lfs track "artifacts/*.tar.gz"
    git add .gitattributes "$ART" || true
    git commit -m "overnight(${TS}): archive logs/adapt/eval" || true
    # push git objects and LFS with retries
    for i in 1 2 3; do
      git push origin main && git lfs push origin main && sync && break || true
      echo "[overnight] push retry $i..." | tee -a "$PIPE_LOG"; sleep 10
    done
  fi

  echo "[overnight] shutting down in ${SHUTDOWN_DELAY}s..." | tee -a "$PIPE_LOG"
  (sleep "$SHUTDOWN_DELAY" && shutdown -h now) || true
}
# Ensure we run cleanup on any termination
trap on_exit EXIT INT TERM

# Idle/watchdog: if no log activity for IDLE_MINUTES, terminate main to trigger trap
(
  set +e
  idle_sec=$(( IDLE_MINUTES * 60 ))
  while true; do
    sleep 60
    now=$(date +%s)
    mt=$(stat -c %Y "$PIPE_LOG" 2>/dev/null || echo "$now")
    diff=$(( now - mt ))
    if [ "$diff" -ge "$idle_sec" ]; then
      echo "[overnight] idle ${IDLE_MINUTES}m exceeded; initiating shutdown..." | tee -a "$PIPE_LOG"
      kill -TERM $$ || true
      exit 0
    fi
  done
) &
WATCHDOG_PID=$!

# Wall-time guard: terminate main after WALL_MINUTES to trigger trap
(
  set +e
  sleep $(( WALL_MINUTES * 60 ))
  echo "[overnight] wall ${WALL_MINUTES}m reached; initiating shutdown..." | tee -a "$PIPE_LOG"
  kill -TERM $$ || true
  exit 0
) &
WALL_PID=$!

{
  echo "[overnight] START model_id=$MODEL_ID top_k=$TOP_K pivot=$PIVOT warmup=$WARMUP_STEPS"

  # Adapt
  echo "[overnight] adapt..."
  set +e
  python -m medical_tokalign.src.cli adapt \
    --model_id "$MODEL_ID" \
    --top_k "$TOP_K" \
    --pivot "$PIVOT" \
    --warmup_steps "$WARMUP_STEPS"
  ADAPT_RC=$?
  set -e
  echo "[overnight] adapt rc=$ADAPT_RC"

  # Eval (run even if adapt_rc!=0; logs/artifacts still saved)
  echo "[overnight] eval..."
  set +e
  python -m medical_tokalign.src.cli eval \
    --config "$ROOT_DIR/configs/eval_medical.yaml"
  EVAL_RC=$?
  set -e
  echo "[overnight] eval rc=$EVAL_RC"

  echo "[overnight] DONE adapt_rc=$ADAPT_RC eval_rc=$EVAL_RC"
} 2>&1 | tee -a "$PIPE_LOG"



# Conservative defaults for GloVe and threads
RAM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 65536)
: "${GLOVE_MEMORY_MB:=$(( RAM_MB / 5 ))}"
if [ "$GLOVE_MEMORY_MB" -lt 4096 ]; then GLOVE_MEMORY_MB=4096; fi
if [ "$GLOVE_MEMORY_MB" -gt 65536 ]; then GLOVE_MEMORY_MB=65536; fi
: "${GLOVE_THREADS:=$(( $(nproc 2>/dev/null || echo 32) - 1 ))}"
if [ "$GLOVE_THREADS" -lt 1 ]; then GLOVE_THREADS=1; fi
export GLOVE_MEMORY_MB GLOVE_THREADS
