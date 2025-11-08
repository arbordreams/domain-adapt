#!/usr/bin/env bash
# MedTokAlign shared shell library - adaptation orchestration
set -euo pipefail
IFS=$'\n\t'

# Requires: common.sh, resources.sh, glove.sh sourced

_ad_log() { printf "[%s] [ADAPT] %s\n" "$(date +"%Y-%m-%dT%H:%M:%S%z")" "${*}"; }

# run_adapt uses with_retry per attempt and manages inter-attempt adjustments
# Exports LAST_RUN_DIR with the successful tokenizer_adapt run directory
LAST_RUN_DIR=""
run_adapt() {
  local model_id="${MODEL_ID:?--model_id required}"
  local top_k="${TOP_K:-8192}"
  local pivot="${PIVOT:-300}"
  local warm="${WARMUP_STEPS:-0}"
  local backend="${EMBEDDING_BACKEND:-fasttext}"
  local timeout_s="${STEP_TIMEOUT:-0}"
  local retries="${MAX_RETRIES:-2}"
  local backoff_s=10

  # Optional tuning via env
  local s1lr_arg=""; local s2st_arg=""; local s2lr_arg=""
  if [ -n "${STAGE1_LR:-}" ]; then s1lr_arg="--stage1_lr ${STAGE1_LR}"; fi
  if [ -n "${STAGE2_STEPS:-}" ]; then s2st_arg="--stage2_steps ${STAGE2_STEPS}"; fi
  if [ -n "${STAGE2_LR:-}" ]; then s2lr_arg="--stage2_lr ${STAGE2_LR}"; fi

  # Prepare GloVe if needed
  if [ "${backend}" = "glove" ]; then
    bootstrap_glove || true
  fi

  local attempt=0
  while :; do
    attempt=$((attempt + 1))
    _ad_log "Attempt ${attempt} for adapt (backend=${backend})"
    # Adjust GloVe memory on later attempts (best-effort; underlying Python may not honor env)
    if [ "${backend}" = "glove" ] && [ "${attempt}" -gt 1 ]; then
      if [ -n "${GLOVE_MEMORY_MB:-}" ]; then
        local bump=$(( GLOVE_MEMORY_MB + 1024 ))
        export GLOVE_MEMORY_MB="${bump}"
        _ad_log "Bumping GLOVE_MEMORY_MB to ${GLOVE_MEMORY_MB} (best-effort)"
      fi
      if [ -n "${GLOVE_SHUFFLE_MEMORY_MB:-}" ]; then
        local sbump=$(( GLOVE_SHUFFLE_MEMORY_MB + 512 ))
        export GLOVE_SHUFFLE_MEMORY_MB="${sbump}"
        _ad_log "Bumping GLOVE_SHUFFLE_MEMORY_MB to ${GLOVE_SHUFFLE_MEMORY_MB} (best-effort)"
      fi
    fi
    local cmd="cd '${REPO_ROOT}' && python -m medical_tokalign.src.cli adapt --model_id '${model_id}' --top_k ${top_k} --pivot ${pivot} --warmup_steps ${warm} --embedding_backend '${backend}' ${s1lr_arg} ${s2st_arg} ${s2lr_arg}"
    # Single attempt via with_retry (no inner retries; we handle outer attempts)
    local step_log
    set +e
    step_log="$(with_retry "adapt" "${cmd}" "${timeout_s}" 0 "${backoff_s}")"
    local rc=$?
    set -e
    if [ "${rc}" -eq 0 ]; then
      # Extract run_dir from the step log output (last path matching tokenizer_adapt)
      if [ -n "${step_log}" ] && [ -f "${step_log}" ]; then
        local rd
        rd="$(grep -E '/medical_tokalign/runs/tokenizer_adapt/' "${step_log}" | tail -n 1 || true)"
        # If grep failed, try reading last non-empty line
        if [ -z "${rd}" ]; then
          rd="$(tac "${step_log}" | sed '/^[[:space:]]*$/d' | head -n1 || true)"
        fi
        if [ -n "${rd}" ] && [ -d "${rd}" ]; then
          LAST_RUN_DIR="${rd}"
        fi
      fi
      # Fallback: find latest tokenizer_adapt dir
      if [ -z "${LAST_RUN_DIR}" ]; then
        LAST_RUN_DIR="$(latest_dir "${PROJECT_DIR}/runs/tokenizer_adapt")"
      fi
      if [ -n "${LAST_RUN_DIR}" ]; then
        _ad_log "Adapt succeeded; run_dir=${LAST_RUN_DIR}"
        if [ "${backend}" = "glove" ]; then
          cleanup_glove_intermediates "${LAST_RUN_DIR}" || true
        fi
        echo "${LAST_RUN_DIR}"
        return 0
      fi
      _ad_log "Adapt appears successful but run_dir not found; treating as failure for safety"
      return 1
    fi
    if [ "${attempt}" -ge "$((retries + 1))" ]; then
      _ad_log "Adapt failed after ${attempt} attempt(s)."
      return "${rc}"
    fi
    _ad_log "Adapt failed (rc=${rc}); sleeping ${backoff_s}s before retry"
    sleep "${backoff_s}"
  done
}

# End of adapt.sh

