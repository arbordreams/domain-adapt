#!/usr/bin/env bash
# MedTokAlign shared shell library - common utilities
# Strict mode and safe IFS
set -euo pipefail
IFS=$'\n\t'

# Resolve repo root (three levels up from this file: scripts/lib → scripts → medical_tokalign → repo root)
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
PROJECT_DIR="${REPO_ROOT}/medical_tokalign"
RUNS_DIR="${PROJECT_DIR}/runs"
LOCKS_DIR="${RUNS_DIR}/locks"
LOGS_DIR="${RUNS_DIR}/logs"
mkdir -p "${LOCKS_DIR}" "${LOGS_DIR}"

# Timestamp helpers
_ts() { date +"%Y-%m-%dT%H:%M:%S%z"; }

# Logging helpers
log()   { printf "[%s] %s\n" "$(_ts)" "${*}"; }
info()  { printf "[%s] [INFO] %s\n" "$(_ts)" "${*}"; }
warn()  { printf "[%s] [WARN] %s\n" "$(_ts)" "${*}" >&2; }
error() { printf "[%s] [ERROR] %s\n" "$(_ts)" "${*}" >&2; }

# Compact JSONL event logger: log_json key=value ... (values auto-quoted)
log_json() {
  local kvs=()
  kvs+=("\"ts\":\"$(_ts)\"")
  while [ $# -gt 0 ]; do
    local kv="${1}"; shift
    local k="${kv%%=*}"
    local v="${kv#*=}"
    # escape backslashes and double quotes
    v="${v//\\/\\\\}"
    v="${v//\"/\\\"}"
    kvs+=("\"${k}\":\"${v}\"")
  done
  printf "{%s}\n" "$(IFS=,; echo "${kvs[*]}")"
}

# Defaults (may be overridden by parse_args)
MODEL_ID="${MODEL_ID:-}"
CORPUS_CONFIG="${CORPUS_CONFIG:-}"
EVAL_CONFIG="${EVAL_CONFIG:-}"
TOP_K="${TOP_K:-}"
PIVOT="${PIVOT:-}"
WARMUP_STEPS="${WARMUP_STEPS:-}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-fasttext}" # fasttext|glove
MAX_RETRIES="${MAX_RETRIES:-2}"
STEP_TIMEOUT="${STEP_TIMEOUT:-0}" # seconds; 0 means no timeout
RESUME_FLAG="${RESUME_FLAG:-0}"
FORCE_FLAG="${FORCE_FLAG:-0}"
QUICK_FLAG="${QUICK_FLAG:-0}"     # demo-only
NO_TMUX_FLAG="${NO_TMUX_FLAG:-0}" # prod-only

print_usage() {
  cat <<EOF
Usage: $0 [options]
  --model_id ID
  --corpus_config PATH
  --eval_config PATH
  --top_k INT
  --pivot INT
  --warmup_steps INT
  --embedding_backend fasttext|glove (default: fasttext)
  --max_retries INT (default: ${MAX_RETRIES})
  --step_timeout SECONDS (default: ${STEP_TIMEOUT})
  --resume
  --force
  --quick          (demo only)
  --no-tmux        (prod only)
EOF
}

# parse_args populates globals; unknown flags cause exit 2
parse_args() {
  while [ $# -gt 0 ]; do
    case "${1}" in
      --model_id) MODEL_ID="${2:-}"; shift 2;;
      --corpus_config) CORPUS_CONFIG="${2:-}"; shift 2;;
      --eval_config) EVAL_CONFIG="${2:-}"; shift 2;;
      --top_k) TOP_K="${2:-}"; shift 2;;
      --pivot) PIVOT="${2:-}"; shift 2;;
      --warmup_steps) WARMUP_STEPS="${2:-}"; shift 2;;
      --embedding_backend) EMBEDDING_BACKEND="${2:-}"; shift 2;;
      --max_retries) MAX_RETRIES="${2:-}"; shift 2;;
      --step_timeout) STEP_TIMEOUT="${2:-}"; shift 2;;
      --resume) RESUME_FLAG=1; shift ;;
      --force) FORCE_FLAG=1; shift ;;
      --quick) QUICK_FLAG=1; shift ;;
      --no-tmux) NO_TMUX_FLAG=1; shift ;;
      -h|--help) print_usage; exit 0;;
      *) error "Unknown argument: ${1}"; print_usage; exit 2;;
    esac
  done
}

# Ensure baseline env - CUDA, HF caches, PyTorch allocator
ensure_env() {
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
  # Prefer HuggingFace accelerated transfer when available
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
  # Avoid importing torchvision in transformers unless strictly required
  export TRANSFORMERS_NO_TORCHVISION="${TRANSFORMERS_NO_TORCHVISION:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  # Propagate embedding backend to environment for helpers
  export EMBEDDING_BACKEND="${EMBEDDING_BACKEND}"
  # Best-effort detection (log only) of hf_transfer presence
  if command -v python >/dev/null 2>&1; then
    if python - <<'PY' 2>/dev/null; then
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("hf_transfer") is not None else 1)
PY
      info "hf_transfer available; fast transfer enabled"
    else
      warn "hf_transfer not installed; fast transfer may be disabled by runtime"
    fi
  fi
}

# Acquire single-run lock
LOCK_PATH="${LOCKS_DIR}/pipeline.lock"
_locked=0
_cleanup_lock() {
  if [ "${_locked}" -eq 1 ] && [ -f "${LOCK_PATH}" ]; then
    rm -f "${LOCK_PATH}" || true
  fi
}
trap _cleanup_lock EXIT INT TERM

acquire_lock() {
  if [ -f "${LOCK_PATH}" ]; then
    local lp pid ts
    lp="$(cat "${LOCK_PATH}" 2>/dev/null || true)"
    pid="$(echo "${lp}" | awk -F'[ =]' '/pid=/{print $2}')"
    ts="$(echo "${lp}" | awk -F'[ =]' '/ts=/{print $2}')"
    if [ -n "${pid:-}" ] && ps -p "${pid}" >/dev/null 2>&1; then
      warn "Existing lock at ${LOCK_PATH} held by pid ${pid} (since ${ts:-unknown}); use --force to override."
      if [ "${FORCE_FLAG}" -eq 0 ]; then
        exit 9
      fi
      warn "Force flag set; removing active lock (pid=${pid})."
    else
      warn "Found stale lock at ${LOCK_PATH} (pid=${pid:-?} ts=${ts:-?}); cleaning up."
    fi
    rm -f "${LOCK_PATH}" || true
  fi
  printf "pid=%s ts=%s\n" "$$" "$(_ts)" > "${LOCK_PATH}"
  _locked=1
}

# Timeout wrapper (portable: prefer GNU timeout; fallback to no-timeout)
_have_timeout=0
if command -v timeout >/dev/null 2>&1; then
  _have_timeout=1
fi

# with_retry name cmd timeout_s retries backoff_s
# Exposes LAST_STEP_RC, LAST_STEP_SECS and writes structured event to step log
LAST_STEP_RC=0
LAST_STEP_SECS=0
with_retry() {
  local name="$1"; shift
  local cmd="$1"; shift
  local timeout_s="$1"; shift
  local retries="$1"; shift
  local backoff_s="$1"; shift

  local attempt=0
  local start=
  local end=

  local step_log="${LOGS_DIR}/${name}.$(date +%Y%m%d_%H%M%S).log"
  info "Starting step '${name}' (log: ${step_log})"
  while :; do
    attempt=$((attempt + 1))
    start=$(date +%s)
    set +e
    if [ "${timeout_s}" -gt 0 ] && [ "${_have_timeout}" -eq 1 ]; then
      # shellcheck disable=SC2086
      timeout --preserve-status "${timeout_s}" bash -lc "${cmd}" 2>&1 | tee -a "${step_log}"
      LAST_STEP_RC="${PIPESTATUS[0]}"
    else
      # shellcheck disable=SC2086
      bash -lc "${cmd}" 2>&1 | tee -a "${step_log}"
      LAST_STEP_RC="${PIPESTATUS[0]}"
    fi
    set -e
    end=$(date +%s)
    LAST_STEP_SECS=$((end - start))
    log_json step="${name}" attempt="${attempt}" rc="${LAST_STEP_RC}" duration_s="${LAST_STEP_SECS}" log="${step_log}" >> "${LOGS_DIR}/events.jsonl"
    if [ "${LAST_STEP_RC}" -eq 0 ]; then
      info "Step '${name}' succeeded in ${LAST_STEP_SECS}s (log: ${step_log})"
      echo "${step_log}"
      return 0
    fi
    if [ "${attempt}" -ge "$((retries + 1))" ]; then
      error "Step '${name}' failed after ${attempt} attempt(s). See log: ${step_log}"
      return "${LAST_STEP_RC}"
    fi
    warn "Step '${name}' failed (rc=${LAST_STEP_RC}); retrying in ${backoff_s}s (attempt ${attempt}/${retries})"
    sleep "${backoff_s}"
  done
}

# Utility: discover latest adapted/eval dirs
latest_dir() {
  local parent="$1"
  [ -d "${parent}" ] || { return 0; }
  # Take last entry by name (stable enough for timestamped dirs) and ensure it's a directory.
  # Avoid subshell 'while ... |' which can lead to extra newlines; emit at most one line.
  local last
  last="$(ls -1 "${parent}" 2>/dev/null | sort | tail -n 1)"
  if [ -n "${last}" ] && [ -d "${parent}/${last}" ]; then
    printf "%s" "${parent}/${last}"
  fi
}

# End of common.sh

