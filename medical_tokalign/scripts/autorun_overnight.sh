#!/usr/bin/env bash
set -euo pipefail

# Enhanced overnight autorun: best of both worlds
# - Prepare data + corpus (from autorun_stable.sh)
# - Adapt with retry logic (escalating memory for GloVe failures)
# - Eval (gated on adapt success)
# - Always archive logs/artifacts, push to GitHub (LFS for tarball)
# - Auto-shutdown guards (idle + wall-time)
# - Then shutdown the machine (even on partial failure)

MODEL_ID="Qwen/Qwen2-7B"
TOP_K=8192
PIVOT=300
WARMUP_STEPS=3000
# Retry configuration
MAX_ADAPT_RETRIES=2      # max retries for adapt step
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
    --max_retries) MAX_ADAPT_RETRIES="$2"; shift 2;;
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

# Conservative defaults for GloVe and threads
# Use allocated vCPU count (31 for RunPod H100) or override via env
# Default to 31 vCPU if not specified (RunPod H100 instance)
: "${VCPU_COUNT:=31}"
RAM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 65536)

# GloVe memory: 20% of RAM (conservative), clamped 4-64 GB
: "${GLOVE_MEMORY_MB:=$(( RAM_MB / 5 ))}"
if [ "$GLOVE_MEMORY_MB" -lt 4096 ]; then GLOVE_MEMORY_MB=4096; fi
if [ "$GLOVE_MEMORY_MB" -gt 65536 ]; then GLOVE_MEMORY_MB=65536; fi

# GloVe threads: vCPU - 1 (leave one core free for system)
: "${GLOVE_THREADS:=$(( VCPU_COUNT - 1 ))}"
if [ "$GLOVE_THREADS" -lt 1 ]; then GLOVE_THREADS=1; fi

export GLOVE_MEMORY_MB GLOVE_THREADS
echo "[overnight] VCPU_COUNT=$VCPU_COUNT GLOVE_MEMORY_MB=$GLOVE_MEMORY_MB GLOVE_THREADS=$GLOVE_THREADS"

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
    echo "  \"prepare_rc\": ${PREPARE_RC:--1},"
    echo "  \"corpus_rc\": ${CORPUS_RC:--1},"
    echo "  \"adapt_rc\": ${ADAPT_RC:--1},"
    echo "  \"eval_rc\": ${EVAL_RC:--1},"
    echo "  \"glove_memory_mb\": ${GLOVE_MEMORY_MB:-unknown},"
    echo "  \"glove_threads\": ${GLOVE_THREADS:-unknown},"
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
  echo "[overnight] GLOVE_MEMORY_MB=$GLOVE_MEMORY_MB GLOVE_THREADS=$GLOVE_THREADS"

  # 1) Prepare data (from autorun_stable.sh)
  echo "[overnight] prepare-data --all"
  set +e
  python -m medical_tokalign.src.cli prepare-data --all
  PREPARE_RC=$?
  set -e
  echo "[overnight] prepare-data rc=$PREPARE_RC"

  # 2) Build corpus (from autorun_stable.sh, skips if exists)
  echo "[overnight] corpus_stable.sh"
  set +e
  bash "$ROOT_DIR/scripts/corpus_stable.sh"
  CORPUS_RC=$?
  set -e
  echo "[overnight] corpus_stable rc=$CORPUS_RC"

  # 3) Adapt with retry logic (escalating memory on failure)
  ADAPT_RC=1
  for retry in $(seq 0 $MAX_ADAPT_RETRIES); do
    if [[ $retry -gt 0 ]]; then
      # Escalate memory: +50% each retry, up to 64GB
      OLD_MEM=$GLOVE_MEMORY_MB
      GLOVE_MEMORY_MB=$(( GLOVE_MEMORY_MB * 150 / 100 ))
      if [[ $GLOVE_MEMORY_MB -gt 65536 ]]; then
        GLOVE_MEMORY_MB=65536
      fi
      export GLOVE_MEMORY_MB
      echo "[overnight] adapt retry $retry/$MAX_ADAPT_RETRIES (memory: ${OLD_MEM}MB -> ${GLOVE_MEMORY_MB}MB)"
    else
      echo "[overnight] adapt (attempt 1)"
    fi

    set +e
    python -m medical_tokalign.src.cli adapt \
      --model_id "$MODEL_ID" \
      --top_k "$TOP_K" \
      --pivot "$PIVOT" \
      --warmup_steps "$WARMUP_STEPS"
    ADAPT_RC=$?
    set -e

    if [[ $ADAPT_RC -eq 0 ]]; then
      echo "[overnight] adapt succeeded (rc=0)"
      break
    else
      echo "[overnight] adapt failed (rc=$ADAPT_RC)"
      if [[ $retry -lt $MAX_ADAPT_RETRIES ]]; then
        echo "[overnight] will retry with higher memory..."
        sleep 10  # Brief pause before retry
      else
        echo "[overnight] max retries ($MAX_ADAPT_RETRIES) reached, giving up"
      fi
    fi
  done

  # 4) Eval (only if adapt succeeded)
  EVAL_RC=-1
  if [[ $ADAPT_RC -eq 0 ]]; then
    echo "[overnight] eval (adapt succeeded)"
    set +e
    python -m medical_tokalign.src.cli eval \
      --config "$ROOT_DIR/configs/eval_medical.yaml"
    EVAL_RC=$?
    set -e
    echo "[overnight] eval rc=$EVAL_RC"
  else
    echo "[overnight] skipping eval (adapt failed, rc=$ADAPT_RC)"
  fi

  echo "[overnight] DONE prepare_rc=$PREPARE_RC corpus_rc=$CORPUS_RC adapt_rc=$ADAPT_RC eval_rc=$EVAL_RC"
} 2>&1 | tee -a "$PIPE_LOG"



# Conservative defaults for GloVe and threads
RAM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 65536)
: "${GLOVE_MEMORY_MB:=$(( RAM_MB / 5 ))}"
if [ "$GLOVE_MEMORY_MB" -lt 8192 ]; then GLOVE_MEMORY_MB=4096; fi
if [ "$GLOVE_MEMORY_MB" -gt 65536 ]; then GLOVE_MEMORY_MB=65536; fi
: "${GLOVE_THREADS:=$(( $(nproc 2>/dev/null || echo 32) - 1 ))}"
if [ "$GLOVE_THREADS" -lt 1 ]; then GLOVE_THREADS=1; fi
export GLOVE_MEMORY_MB GLOVE_THREADS
echo "[overnight] GLOVE_MEMORY_MB=$GLOVE_MEMORY_MB GLOVE_THREADS=$GLOVE_THREADS"
