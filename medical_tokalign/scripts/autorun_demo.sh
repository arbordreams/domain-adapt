#!/usr/bin/env bash
# MedTokAlign demo runner - fast, stable smoke/demo/test profile
set -euo pipefail
IFS=$'\n\t'

# Load libs
_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="${_THIS_DIR}/lib"
# shellcheck source=lib/common.sh
. "${LIB_DIR}/common.sh"
# shellcheck source=lib/resources.sh
. "${LIB_DIR}/resources.sh"
# shellcheck source=lib/glove.sh
. "${LIB_DIR}/glove.sh"
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

# Defaults for demo
TOP_K="${TOP_K:-256}"
PIVOT="${PIVOT:-300}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-fasttext}"
EVAL_CONFIG="${EVAL_CONFIG:-${PROJECT_DIR}/configs/eval_medical_ultra_quick.yaml}"
CORPUS_CONFIG="${CORPUS_CONFIG:-${PROJECT_DIR}/configs/corpus_med_tokalign_3gb.yaml}"
MAX_RETRIES="${MAX_RETRIES:-1}"
STEP_TIMEOUT="${STEP_TIMEOUT:-0}"
QUICK_FLAG="${QUICK_FLAG:-0}" # enable with --quick

parse_args "$@"
ensure_env
acquire_lock
setup_resources

fail_step="" ; fail_log=""

# Step: prepare-data (minimal)
set +e
prep_log="$(with_retry "prepare_data" "cd '${REPO_ROOT}' && python -m medical_tokalign.src.cli prepare-data" "${STEP_TIMEOUT}" 0 5)"
rc=$?
set -e
if [ "${rc}" -ne 0 ]; then fail_step="prepare-data"; fail_log="${prep_log}"; fi

# Step: build-corpus (quick if --quick)
if [ -z "${fail_step}" ]; then
  set +e
  build_cmd="source '${LIB_DIR}/common.sh'; PROJECT_DIR='${PROJECT_DIR}'; RUNS_DIR='${RUNS_DIR}'; REPO_ROOT='${REPO_ROOT}'; source '${LIB_DIR}/corpus.sh'; build_corpus '${CORPUS_CONFIG}' '${QUICK_FLAG}'"
  corpus_log="$(with_retry "build_corpus" "${build_cmd}" "${STEP_TIMEOUT}" "${MAX_RETRIES}" 10)"
  rc=$?
  set -e
  if [ "${rc}" -ne 0 ]; then fail_step="build-corpus"; fail_log="${corpus_log}"; fi
fi

# Step: adapt
if [ -z "${fail_step}" ]; then
  set +e
  run_dir="$(run_adapt)"
  rc=$?
  set -e
  if [ "${rc}" -ne 0 ]; then
    fail_step="adapt"; fail_log="$(ls -1t "${RUNS_DIR}/logs"/adapt.*.log 2>/dev/null | head -n1 || true)"
  fi
fi

# Step: eval
if [ -z "${fail_step}" ]; then
  set +e
  eval_dir="$(run_eval)"
  rc=$?
  set -e
  if [ "${rc}" -ne 0 ]; then
    fail_step="eval"; fail_log="$(ls -1t "${RUNS_DIR}/logs"/eval.*.log 2>/dev/null | head -n1 || true)"
  fi
fi

# Artifacts (best-effort even if eval failed)
manifest_path="$(collect_artifacts || true)"

if [ -n "${fail_step}" ]; then
  error "Failing step: ${fail_step}"
  if [ -n "${fail_log}" ]; then
    error "Attached log: ${fail_log}"
  fi
  exit 2
fi

info "Demo run completed successfully."
if [ -n "${manifest_path}" ]; then
  info "Manifest: ${manifest_path}"
fi
echo "attached log: $(ls -1t "${RUNS_DIR}/logs"/*.log 2>/dev/null | head -n1 || true)"


