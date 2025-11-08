#!/usr/bin/env bash
# MedTokAlign production runner - robust, resumable, tmux by default
set -euo pipefail
IFS=$'\n\t'

_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="${_THIS_DIR}/lib"
# shellcheck source=lib/common.sh
. "${LIB_DIR}/common.sh"
# shellcheck source=lib/resources.sh
. "${LIB_DIR}/resources.sh"
# shellcheck source=lib/glove.sh
. "${LIB_DIR}/glove.sh"
# shellcheck source=lib/tmux.sh
. "${LIB_DIR}/tmux.sh"
# shellcheck source=lib/data.sh
. "${LIB_DIR}/data.sh"
# shellcheck source=lib/corpus.sh
. "${LIB_DIR}/corpus.sh"
# shellcheck source=lib/adapt.sh
. "${LIB_DIR}/adapt.sh"
# shellcheck source=lib/eval.sh
. "${LIB_DIR}/eval.sh"
# shellcheck source=lib/artifacts.sh
. "${LIB_DIR}/artifacts.sh"

# Defaults for prod
TOP_K="${TOP_K:-8192}"
PIVOT="${PIVOT:-300}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-fasttext}"
EVAL_CONFIG="${EVAL_CONFIG:-${PROJECT_DIR}/configs/eval_medical.yaml}"
CORPUS_CONFIG="${CORPUS_CONFIG:-${PROJECT_DIR}/configs/corpus_biomed.yaml}"
MAX_RETRIES="${MAX_RETRIES:-2}"
STEP_TIMEOUT="${STEP_TIMEOUT:-0}"
RESUME_FLAG="${RESUME_FLAG:-1}"
NO_TMUX_FLAG="${NO_TMUX_FLAG:-0}"

parse_args "$@"
ensure_env

# If tmux is enabled, bootstrap a session and re-run inline within it
if [ "${NO_TMUX_FLAG}" -eq 0 ] && tmux_available && [ -z "${IN_TMUX_CHILD:-}" ]; then
  info "Starting tmux session '${TMUX_SESSION:-medtokalign}' for production run"
  tmux_ensure_session
  # Tail logs window
  tmux_run "logs" "bash -lc 'mkdir -p \"${RUNS_DIR}/logs\"; tail -F \"${RUNS_DIR}/logs\"/*.log 2>/dev/null || sleep 36000'"
  # Main orchestrator
  tmux_run "prod" "bash -lc 'IN_TMUX_CHILD=1 \"${_THIS_DIR}/autorun_prod.sh\" --no-tmux $(printf "%q " "$@")'"
  tmux_attach
  exit 0
fi

acquire_lock
setup_resources

fail_step="" ; fail_log=""

# Step: prepare-data (all)
set +e
prep_cmd="cd '${REPO_ROOT}' && python -m medical_tokalign.src.cli prepare-data --all"
prep_log="$(with_retry "prepare_data" "${prep_cmd}" "${STEP_TIMEOUT}" 0 5)"
rc=$?
set -e
if [ "${rc}" -ne 0 ]; then fail_step="prepare-data"; fail_log="${prep_log}"; fi

# Step: build-corpus (strict; resume-aware inside function if RESUME_FLAG=1)
if [ -z "${fail_step}" ]; then
  set +e
  build_cmd="source '${LIB_DIR}/common.sh'; PROJECT_DIR='${PROJECT_DIR}'; RUNS_DIR='${RUNS_DIR}'; REPO_ROOT='${REPO_ROOT}'; RESUME_FLAG='${RESUME_FLAG}'; source '${LIB_DIR}/corpus.sh'; build_corpus '${CORPUS_CONFIG}' '0'"
  corpus_log="$(with_retry "build_corpus" "${build_cmd}" "${STEP_TIMEOUT}" "${MAX_RETRIES}" 15)"
  rc=$?
  set -e
  if [ "${rc}" -ne 0 ]; then fail_step="build-corpus"; fail_log="${corpus_log}"; fi
fi

# Step: adapt (skip if resume and latest adapted model exists and not --force)
skip_adapt=0
if [ -z "${fail_step}" ] && [ "${RESUME_FLAG}" -eq 1 ] && [ "${FORCE_FLAG}" -eq 0 ]; then
  latest_adapt="$(latest_dir "${PROJECT_DIR}/runs/tokenizer_adapt")"
  if [ -n "${latest_adapt}" ] && [ -d "${latest_adapt}/model" ]; then
    info "Found existing adapted model at ${latest_adapt}; skipping adapt (use --force to rebuild)"
    LAST_RUN_DIR="${latest_adapt}"
    skip_adapt=1
  fi
fi
if [ -z "${fail_step}" ] && [ "${skip_adapt}" -eq 0 ]; then
  set +e
  run_dir="$(run_adapt)"
  rc=$?
  set -e
  if [ "${rc}" -ne 0 ]; then
    fail_step="adapt"; fail_log="$(ls -1t "${RUNS_DIR}/logs"/adapt.*.log 2>/dev/null | head -n1 || true)"
  fi
fi

# Step: eval (skip if resume and latest eval has metrics_summary.json)
if [ -z "${fail_step}" ]; then
  if [ "${RESUME_FLAG}" -eq 1 ] && [ "${FORCE_FLAG}" -eq 0 ]; then
    latest_eval="$(latest_dir "${PROJECT_DIR}/runs/medical_eval")"
    if [ -n "${latest_eval}" ] && [ -f "${latest_eval}/metrics_summary.json" ]; then
      info "Found existing eval results at ${latest_eval}; skipping eval (use --force to rerun)"
      LAST_EVAL_DIR="${latest_eval}"
    else
      set +e
      eval_dir="$(run_eval)"
      rc=$?
      set -e
      if [ "${rc}" -ne 0 ]; then
        fail_step="eval"; fail_log="$(ls -1t "${RUNS_DIR}/logs"/eval.*.log 2>/dev/null | head -n1 || true)"
      fi
    fi
  else
    set +e
    eval_dir="$(run_eval)"
    rc=$?
    set -e
    if [ "${rc}" -ne 0 ]; then
      fail_step="eval"; fail_log="$(ls -1t "${RUNS_DIR}/logs"/eval.*.log 2>/dev/null | head -n1 || true)"
    fi
  fi
fi

# Artifacts (best-effort)
manifest_path="$(collect_artifacts || true)"

if [ -n "${fail_step}" ]; then
  error "Failing step: ${fail_step}"
  if [ -n "${fail_log}" ]; then
    error "Attached log: ${fail_log}"
  fi
  exit 2
fi

info "Production run completed successfully."
if [ -n "${manifest_path}" ]; then
  info "Manifest: ${manifest_path}"
fi
echo "attached log: $(ls -1t "${RUNS_DIR}/logs"/*.log 2>/dev/null | head -n1 || true)"


