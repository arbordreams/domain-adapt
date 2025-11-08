#!/usr/bin/env bash
# MedTokAlign shared shell library - evaluation wrapper
set -euo pipefail
IFS=$'\n\t'

# Requires: common.sh

_ev_log() { printf "[%s] [EVAL] %s\n" "$(date +"%Y-%m-%dT%H:%M:%S%z")" "${*}"; }

LAST_EVAL_DIR=""
LAST_EVAL_BACKEND=""

run_eval() {
  local cfg_path="${EVAL_CONFIG:?--eval_config required}"
  local timeout_s="${STEP_TIMEOUT:-0}"
  local retries="${MAX_RETRIES:-1}"
  # Prefer vLLM unless overridden
  if [ -z "${EVAL_BACKEND:-}" ]; then
    if python - <<'PY' 2>/dev/null; then
import importlib, sys
sys.exit(0 if importlib.util.find_spec("vllm") is not None else 1)
PY
      export EVAL_BACKEND="vllm"
    else
      export EVAL_BACKEND="hf"
    fi
  fi
  LAST_EVAL_BACKEND="${EVAL_BACKEND}"
  _ev_log "Selected backend=${EVAL_BACKEND}"
  local cmd="cd '${REPO_ROOT}' && python -m medical_tokalign.src.cli eval --config '${cfg_path}'"
  local step_log
  set +e
  step_log="$(with_retry "eval" "${cmd}" "${timeout_s}" "${retries}" 15)"
  local rc=$?
  # If vLLM failed, try HF once as an automatic fallback
  if [ "${rc}" -ne 0 ] && [ "${EVAL_BACKEND}" = "vllm" ]; then
    _ev_log "vLLM eval failed (rc=${rc}); falling back to HF generate once"
    export EVAL_BACKEND="hf"
    LAST_EVAL_BACKEND="${EVAL_BACKEND}"
    step_log="$(with_retry "eval" "${cmd}" "${timeout_s}" 0 5)"
    rc=$?
  fi
  set -e
  if [ "${rc}" -ne 0 ]; then
    return "${rc}"
  fi
  # Derive eval dir as latest run
  LAST_EVAL_DIR="$(latest_dir "${PROJECT_DIR}/runs/medical_eval")"
  if [ -n "${step_log}" ] && [ -f "${step_log}" ]; then
    # Surface throughput lines for convenience
    grep -E '\[(MedTokAlign)\]\[(vLLM|HF)\] gen: ' "${step_log}" || true
  fi
  if [ -n "${LAST_EVAL_DIR}" ]; then
    _ev_log "Eval outputs at ${LAST_EVAL_DIR}"
    echo "${LAST_EVAL_DIR}"
  fi
}

# End of eval.sh

