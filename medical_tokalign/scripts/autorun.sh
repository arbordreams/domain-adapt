#!/usr/bin/env bash
set -euo pipefail

# Unified unattended autorun for MedTokAlign (TokAlign‑faithful)
# - Prepares data and corpus (skips if present)
# - Runs adapt with retries
# - Gates eval on adapt success
# - Archives artifacts and emits a manifest
# - Optional tmux orchestration (default on)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
LOGDIR="$ROOT_DIR/runs/logs"
mkdir -p "$LOGDIR"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE_LOG="$LOGDIR/autorun_${TS}.log"

# -------- args --------
MODEL_ID=""
TOP_K="8192"
PIVOT="300"
WARMUP_STEPS="3000"
STAGE1_LR="${STAGE1_LR:-5e-4}"
STAGE2_STEPS="${STAGE2_STEPS:-0}"
STAGE2_LR="${STAGE2_LR:-5e-5}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-fasttext}"
MAX_RETRIES="2"
EVAL_CONFIG="$ROOT_DIR/configs/eval_medical.yaml"
USE_TMUX="${USE_TMUX:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_id) MODEL_ID="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --pivot) PIVOT="$2"; shift 2 ;;
    --warmup_steps) WARMUP_STEPS="$2"; shift 2 ;;
    --embedding_backend) EMBEDDING_BACKEND="$2"; shift 2 ;;
    --stage1_lr) STAGE1_LR="$2"; shift 2 ;;
    --stage2_steps) STAGE2_STEPS="$2"; shift 2 ;;
    --stage2_lr) STAGE2_LR="$2"; shift 2 ;;
    --max_retries) MAX_RETRIES="$2"; shift 2 ;;
    --eval_config) EVAL_CONFIG="$2"; shift 2 ;;
    --no-tmux) USE_TMUX=0; shift 1 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL_ID" ]]; then
  echo "Usage: $0 --model_id <hf_model_id> [--top_k N] [--pivot N] [--warmup_steps N] [--embedding_backend fasttext|glove] [--stage1_lr LR] [--stage2_steps N] [--stage2_lr LR] [--max_retries N] [--no-tmux]" >&2
  exit 2
fi

# -------- tmux orchestration --------
if [[ -z "${TMUX:-}" && "${USE_TMUX}" != "0" ]]; then
  # Launch a tmux session named 'autorun'
  { tmux kill-session -t autorun 2>/dev/null || true; }
  echo "[autorun] starting tmux session 'autorun'..."
  tmux new -d -s autorun "bash '$0' --model_id '$MODEL_ID' --top_k '$TOP_K' --pivot '$PIVOT' --warmup_steps '$WARMUP_STEPS' --max_retries '$MAX_RETRIES' --eval_config '$EVAL_CONFIG' --no-tmux"
  echo "[autorun] attached log: $PIPE_LOG"
  exit 0
fi

exec &> >(tee -a "$PIPE_LOG")
echo "[autorun] ts=$TS model_id=$MODEL_ID top_k=$TOP_K pivot=$PIVOT warmup_steps=$WARMUP_STEPS emb_backend=$EMBEDDING_BACKEND stage1_lr=$STAGE1_LR stage2_steps=$STAGE2_STEPS stage2_lr=$STAGE2_LR max_retries=$MAX_RETRIES"

# -------- resources --------
RAM_MB="$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 131072)"

# Detect actual allocated vCPUs (prefer cgroup quota over nproc logical cores)
# This is critical for cloud instances where nproc shows all logical cores but quota is limited
ALLOCATED_VCPUS=20  # Default fallback
if [[ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]] && [[ -f /sys/fs/cgroup/cpu/cpu.cfs_period_us ]]; then
  QUOTA=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo "-1")
  PERIOD=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo "100000")
  if [[ "$QUOTA" != "-1" ]] && [[ "$PERIOD" -gt 0 ]]; then
    ALLOCATED_VCPUS=$((QUOTA / PERIOD))
  fi
fi
# Fallback to nproc if cgroup unavailable, but clamp to reasonable max for cloud instances
if [[ "$ALLOCATED_VCPUS" -lt 1 ]] || [[ "$ALLOCATED_VCPUS" -gt 200 ]]; then
  NPROC_LOGICAL="$(nproc 2>/dev/null || echo 20)"
  # If nproc shows many cores but we're on a cloud instance, be conservative
  if [[ "$NPROC_LOGICAL" -gt 100 ]]; then
    ALLOCATED_VCPUS=20  # Conservative default for cloud
  else
    ALLOCATED_VCPUS="$NPROC_LOGICAL"
  fi
fi
NPROC="$ALLOCATED_VCPUS"

# Cooccur memory: 30% of RAM (increased from 25% for better performance), clamped 4–64 GB
: "${GLOVE_MEMORY_MB:=$(( RAM_MB * 3 / 10 ))}"
if [[ "$GLOVE_MEMORY_MB" -lt 4096 ]]; then GLOVE_MEMORY_MB=4096; fi
if [[ "$GLOVE_MEMORY_MB" -gt 65536 ]]; then GLOVE_MEMORY_MB=65536; fi

# TokAlign-faithful cap for per-run GloVe corpora (combined src+tgt bytes).
# Default ~1 GiB total; override via GLOVE_CORPUS_MAX_BYTES if desired.
: "${GLOVE_CORPUS_MAX_BYTES:=1000000000}"

# Shuffle memory: conservative 8 GB, allow env override
: "${GLOVE_SHUFFLE_MEMORY_MB:=8192}"

# GloVe threads: allocated vCPUs - 1 (leave one core for system)
: "${GLOVE_THREADS:=$(( NPROC>1 ? NPROC-1 : 1 ))}"
if [[ "$GLOVE_THREADS" -lt 1 ]]; then GLOVE_THREADS=1; fi
# Cap at 64 threads to avoid excessive context switching
if [[ "$GLOVE_THREADS" -gt 64 ]]; then GLOVE_THREADS=64; fi

export GLOVE_MEMORY_MB GLOVE_THREADS GLOVE_SHUFFLE_MEMORY_MB GLOVE_CORPUS_MAX_BYTES
echo "[autorun] RAM_MB=$RAM_MB ALLOCATED_VCPUS=$NPROC GLOVE_MEMORY_MB=$GLOVE_MEMORY_MB GLOVE_SHUFFLE_MEMORY_MB=$GLOVE_SHUFFLE_MEMORY_MB GLOVE_THREADS=$GLOVE_THREADS GLOVE_CORPUS_MAX_BYTES=$GLOVE_CORPUS_MAX_BYTES"

# Alignment BLAS threading (modest speedup, safe defaults)
# Derive ALIGN_THREADS ~= allocated_vCPUs/2, clamped [4,16], overridable via env
if [[ -z "${ALIGN_THREADS:-}" ]]; then
  _align_tmp=$(( NPROC/2 ))
  if [[ "$_align_tmp" -lt 4 ]]; then _align_tmp=4; fi
  if [[ "$_align_tmp" -gt 16 ]]; then _align_tmp=16; fi
  ALIGN_THREADS="$_align_tmp"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$ALIGN_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$ALIGN_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$ALIGN_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$ALIGN_THREADS}"
echo "[autorun] ALIGN_THREADS=$ALIGN_THREADS (OMP/MKL/OPENBLAS/NUMEXPR)"

# (Removed fast-run overrides; run with stable defaults)

# -------- cleanup vestiges (best‑effort) --------
for s in "$ROOT_DIR/scripts/autorun_overnight.sh" "$ROOT_DIR/scripts/autorun_stable.sh"; do
  if [[ -f "$s" ]]; then
    echo "[autorun] removing vestige script: ${s##*/}"
    rm -f "$s" || true
  fi
done

# -------- archiving on exit --------
on_exit() {
  set +e
  echo "[autorun] archiving artifacts..." | tee -a "$PIPE_LOG"
  ARTIFACTS_DIR="$REPO_ROOT/artifacts"
  mkdir -p "$ARTIFACTS_DIR/logs/run_${TS}" "$ARTIFACTS_DIR/models/run_${TS}" "$ARTIFACTS_DIR/evals/run_${TS}"
  ADAPT_DIR=$(ls -td "$ROOT_DIR/runs/tokenizer_adapt"/* 2>/dev/null | head -n1 || true)
  EVAL_DIR=$(ls -td "$ROOT_DIR/runs/medical_eval"/* 2>/dev/null | head -n1 || true)
  if [[ -n "$ADAPT_DIR" && -d "$ADAPT_DIR" ]]; then cp -a "$ADAPT_DIR"/. "$ARTIFACTS_DIR/models/run_${TS}"/ || true; fi
  if [[ -n "$EVAL_DIR" && -d "$EVAL_DIR" ]]; then cp -a "$EVAL_DIR"/. "$ARTIFACTS_DIR/evals/run_${TS}"/ || true; fi
  if [[ -d "$ROOT_DIR/runs/logs" ]]; then cp -a "$ROOT_DIR/runs/logs"/. "$ARTIFACTS_DIR/logs/run_${TS}"/ || true; fi
  (cd "$REPO_ROOT" && tar -czf "$ARTIFACTS_DIR/run_${TS}.tar.gz" medical_tokalign/runs/logs 2>/dev/null || true)
  ls -lh "$ARTIFACTS_DIR/run_${TS}.tar.gz" 2>/dev/null || true

  # Best‑effort LFS push (non‑fatal)
  if command -v git >/dev/null 2>&1; then
    (cd "$REPO_ROOT" && git lfs install && git add -A && git commit -m "autorun artifacts ${TS}" 2>/dev/null || true && git push 2>/dev/null || true)
  fi
}
trap on_exit EXIT

# -------- bootstrap tools --------
if [[ ! -d "$ROOT_DIR/tools/GloVe" ]]; then
  echo "[autorun] bootstrap GloVe..."
  git clone https://github.com/stanfordnlp/GloVe.git "$ROOT_DIR/tools/GloVe"
  make -C "$ROOT_DIR/tools/GloVe"
else
  make -C "$ROOT_DIR/tools/GloVe" || true
fi

# -------- pipeline --------
set +e
PREP_RC=0
echo "[autorun] prepare-data --all"
python -m medical_tokalign.src.cli prepare-data --all || PREP_RC=$?
echo "[autorun] prepare rc=$PREP_RC"

echo "[autorun] corpus_stable.sh (skip if exists)"
# Allow custom corpus config (defaults to 3GB TokAlign medical if present)
if [[ -z "${CORPUS_CONFIG:-}" ]]; then
  if [[ -f "$ROOT_DIR/configs/corpus_med_tokalign_3gb.yaml" ]]; then
    export CORPUS_CONFIG="$ROOT_DIR/configs/corpus_med_tokalign_3gb.yaml"
  fi
fi
if [[ -n "${CORPUS_CONFIG:-}" ]]; then
  echo "[autorun] CORPUS_CONFIG=${CORPUS_CONFIG}"
fi
bash "$ROOT_DIR/scripts/corpus_stable.sh" || true

ADAPT_RC=1
for retry in $(seq 0 "$MAX_RETRIES"); do
  if [[ "$retry" -gt 0 ]]; then
    OLD_MEM=$GLOVE_MEMORY_MB
    GLOVE_MEMORY_MB=$(( GLOVE_MEMORY_MB * 150 / 100 ))
    if [[ "$GLOVE_MEMORY_MB" -gt 65536 ]]; then GLOVE_MEMORY_MB=65536; fi
    export GLOVE_MEMORY_MB
    echo "[autorun] adapt retry $retry/$MAX_RETRIES (memory: ${OLD_MEM}MB -> ${GLOVE_MEMORY_MB}MB)"
  else
    echo "[autorun] adapt (attempt 1)"
  fi
  python -m medical_tokalign.src.cli adapt \
    --model_id "$MODEL_ID" \
    --top_k "$TOP_K" \
    --pivot "$PIVOT" \
    --warmup_steps "$WARMUP_STEPS" \
    --embedding_backend "$EMBEDDING_BACKEND" \
    --stage1_lr "$STAGE1_LR" \
    --stage2_steps "$STAGE2_STEPS" \
    --stage2_lr "$STAGE2_LR"
  ADAPT_RC=$?
  echo "[autorun] adapt rc=$ADAPT_RC"
  if [[ "$ADAPT_RC" -eq 0 ]]; then break; fi
  sleep 10
done

EVAL_RC=-1
if [[ "$ADAPT_RC" -eq 0 ]]; then
  # Post-adapt pruning of heavy intermediates (keep vectors and models)
  echo "[autorun] pruning heavy GloVe intermediates and per-run corpora..."
  # Delete global cooccurrence binaries (safe after vectors are emitted)
  find "$ROOT_DIR/tools/GloVe" -maxdepth 1 -type f -name 'cooccurrence.*.bin' -delete 2>/dev/null || true
  find "$ROOT_DIR/tools/GloVe" -maxdepth 1 -type f -name 'cooccurrence.shuf.*.bin' -delete 2>/dev/null || true
  # Delete per-run glove corpora in the latest tokenizer_adapt dir (vectors are in tools/GloVe)
  LATEST_ADAPT_DIR="$(ls -td "$ROOT_DIR/runs/tokenizer_adapt"/* 2>/dev/null | head -n1 || true)"
  if [[ -n "$LATEST_ADAPT_DIR" && -d "$LATEST_ADAPT_DIR" ]]; then
    rm -f "$LATEST_ADAPT_DIR/glove_source.txt" "$LATEST_ADAPT_DIR/glove_target.txt" 2>/dev/null || true
  fi

  echo "[autorun] eval (adapt succeeded)"
  # Prefer vLLM by default; allow override via EVAL_BACKEND env
  if [[ -z "${EVAL_BACKEND:-}" ]]; then
    EVAL_BACKEND="vllm"
  fi
  export EVAL_BACKEND
  # Determine effective eval config (priority: override > EVAL_CONFIG)
  if [[ -n "${EVAL_CONFIG_OVERRIDE:-}" ]]; then
    EFF_EVAL_CONFIG="$EVAL_CONFIG_OVERRIDE"
  elif [[ -n "${EVAL_CONFIG:-}" ]]; then
    EFF_EVAL_CONFIG="$EVAL_CONFIG"
  fi
  python -m medical_tokalign.src.cli eval --config "$EFF_EVAL_CONFIG"
  EVAL_RC=$?
  echo "[autorun] eval rc=$EVAL_RC"
else
  echo "[autorun] skipping eval (adapt failed, rc=$ADAPT_RC)"
fi

# Manifest
MAN="$LOGDIR/run_${TS}_manifest.json"
{
  echo "{"
  echo "  \"ts\": \"${TS}\","
  echo "  \"model_id\": \"${MODEL_ID}\","
  echo "  \"top_k\": ${TOP_K},"
  echo "  \"pivot\": ${PIVOT},"
  echo "  \"warmup_steps\": ${WARMUP_STEPS},"
  echo "  \"glove_memory_mb\": ${GLOVE_MEMORY_MB},"
  echo "  \"glove_shuffle_memory_mb\": ${GLOVE_SHUFFLE_MEMORY_MB},"
  echo "  \"glove_threads\": ${GLOVE_THREADS},"
  echo "  \"eval_backend\": \"${EVAL_BACKEND:-unknown}\","
  echo "  \"prepare_rc\": ${PREP_RC},"
  echo "  \"adapt_rc\": ${ADAPT_RC},"
  echo "  \"eval_rc\": ${EVAL_RC}"
  echo "}"
} > "$MAN" || true
echo "[autorun] manifest: $MAN"

# Update artifacts index (append)
INDEX_DIR="$REPO_ROOT/artifacts/runs"
mkdir -p "$INDEX_DIR"
INDEX_PATH="$INDEX_DIR/index.jsonl"
{
  echo -n "{"
  echo -n "\"ts\":\"${TS}\",\"model_id\":\"${MODEL_ID}\",\"eval_backend\":\"${EVAL_BACKEND:-unknown}\","
  # capture sizes (best-effort)
  ADAPT_DIR=$(ls -td "$ROOT_DIR/runs/tokenizer_adapt"/* 2>/dev/null | head -n1 || true)
  EVAL_DIR=$(ls -td "$ROOT_DIR/runs/medical_eval"/* 2>/dev/null | head -n1 || true)
  ADAPT_SZ=$(du -s "$ADAPT_DIR" 2>/dev/null | awk '{print $1}' || echo 0)
  EVAL_SZ=$(du -s "$EVAL_DIR" 2>/dev/null | awk '{print $1}' || echo 0)
  echo -n "\"adapt_dir\":\"${ADAPT_DIR:-}\",\"adapt_kb\":${ADAPT_SZ},\"eval_dir\":\"${EVAL_DIR:-}\",\"eval_kb\":${EVAL_SZ}"
  echo "}"
} >> "$INDEX_PATH" 2>/dev/null || true

exit 0


