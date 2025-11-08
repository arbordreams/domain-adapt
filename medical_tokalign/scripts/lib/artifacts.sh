#!/usr/bin/env bash
# MedTokAlign shared shell library - artifacts and manifest
set -euo pipefail
IFS=$'\n\t'

# Requires: common.sh, resources.sh

_ar_log() { printf "[%s] [ART] %s\n" "$(date +"%Y-%m-%dT%H:%M:%S%z")" "${*}"; }

_json_escape() {
  local s="${1:-}"
  s="${s//\\/\\\\}"; s="${s//\"/\\\"}"
  printf "%s" "${s}"
}

_dir_size_kb() {
  local d="$1"
  if [ -d "${d}" ]; then
    du -sk "${d}" 2>/dev/null | awk '{print $1}'
  else
    echo 0
  fi
}

collect_artifacts() {
  local ts="$(date +%Y%m%d_%H%M%S)"
  local art_root="${REPO_ROOT}/artifacts"
  local models_dir="${art_root}/models/run_${ts}"
  local evals_dir="${art_root}/evals/run_${ts}"
  local logs_dir="${art_root}/logs/run_${ts}"
  mkdir -p "${models_dir}" "${evals_dir}" "${logs_dir}" "${art_root}/runs"

  local run_dir="${LAST_RUN_DIR:-$(latest_dir "${PROJECT_DIR}/runs/tokenizer_adapt")}"
  local eval_dir="${LAST_EVAL_DIR:-$(latest_dir "${PROJECT_DIR}/runs/medical_eval")}"
  local events="${RUNS_DIR}/logs/events.jsonl"

  # Copy artifacts (best-effort)
  if [ -n "${run_dir}" ] && [ -d "${run_dir}" ]; then
    _ar_log "Copying adapted artifacts to ${models_dir}"
    cp -a "${run_dir}/." "${models_dir}/" || true
  fi
  if [ -n "${eval_dir}" ] && [ -d "${eval_dir}" ]; then
    _ar_log "Copying eval artifacts to ${evals_dir}"
    cp -a "${eval_dir}/." "${evals_dir}/" || true
  fi
  # Copy logs
  if [ -d "${RUNS_DIR}/logs" ]; then
    cp -a "${RUNS_DIR}/logs/." "${logs_dir}/" || true
  fi

  # Manifest
  local vcpus="$({ type -t detect_vcpus >/dev/null 2>&1 && detect_vcpus; } || echo "")"
  local ram_mb="$({ type -t detect_ram_mb >/dev/null 2>&1 && detect_ram_mb; } || echo "")"
  local manifest="${art_root}/run_${ts}.manifest.json"
  {
    printf "{"
    printf '"ts":"%s",' "$(date +"%Y-%m-%dT%H:%M:%S%z")"
    printf '"model_id":"%s",' "$(_json_escape "${MODEL_ID:-}")"
    printf '"corpus_config":"%s",' "$(_json_escape "${CORPUS_CONFIG:-}")"
    printf '"eval_config":"%s",' "$(_json_escape "${EVAL_CONFIG:-}")"
    printf '"top_k":%s,' "${TOP_K:-0}"
    printf '"pivot":%s,' "${PIVOT:-0}"
    printf '"warmup_steps":%s,' "${WARMUP_STEPS:-0}"
    printf '"embedding_backend":"%s",' "$(_json_escape "${EMBEDDING_BACKEND:-fasttext}")"
    printf '"eval_backend":"%s",' "$(_json_escape "${EVAL_BACKEND:-}")"
    printf '"resources":{"vcpus":"%s","ram_mb":"%s"},' "$(_json_escape "${vcpus}")" "$(_json_escape "${ram_mb}")"
    printf '"paths":{"run_dir":"%s","eval_dir":"%s","events":"%s"},' "$(_json_escape "${run_dir}")" "$(_json_escape "${eval_dir}")" "$(_json_escape "${events}")"
    printf '"sizes_kb":{"run_dir":%s,"eval_dir":%s},' "$(_dir_size_kb "${run_dir}")" "$(_dir_size_kb "${eval_dir}")"
    # step events are file-backed; include path only
    printf '"notes":"%s"' "$(_json_escape "Best-effort manifest; step events in events.jsonl")"
    printf "}\n"
  } > "${manifest}" || true

  # Archive tar.gz
  local tarball="${art_root}/run_${ts}.tar.gz"
  ( cd "${art_root}" && tar -czf "${tarball}" "models/run_${ts}" "evals/run_${ts}" "logs/run_${ts}" "$(basename "${manifest}")" 2>/dev/null ) || true

  # Append index.jsonl
  local idx="${art_root}/runs/index.jsonl"
  {
    printf "{"
    printf '"ts":"%s",' "$(date +"%Y-%m-%dT%H:%M:%S%z")"
    printf '"model_id":"%s",' "$(_json_escape "${MODEL_ID:-}")"
    printf '"embedding_backend":"%s",' "$(_json_escape "${EMBEDDING_BACKEND:-fasttext}")"
    printf '"run_dir":"%s",' "$(_json_escape "${models_dir}")"
    printf '"eval_dir":"%s",' "$(_json_escape "${evals_dir}")"
    printf '"tarball":"%s",' "$(_json_escape "${tarball}")"
    printf '"sizes_kb":{"models":%s,"evals":%s}' "$(_dir_size_kb "${models_dir}")" "$(_dir_size_kb "${evals_dir}")"
    printf "}\n"
  } >> "${idx}" || true

  _ar_log "Artifacts collected. Manifest: ${manifest}"
  echo "${manifest}"
}

# End of artifacts.sh

