#!/usr/bin/env bash
# MedTokAlign shared shell library - data preparation
set -euo pipefail
IFS=$'\n\t'

# Requires common.sh for REPO_ROOT/PROJECT_DIR and logging

_data_bench_dir="${PROJECT_DIR}/data/medical/benchmarks"
_data_proc_dir="${PROJECT_DIR}/data/medical/processed"
mkdir -p "${_data_bench_dir}" "${_data_proc_dir}"

_have_file_with_prefix() {
  # $1: dir, $2: prefix
  local d="$1"; local pfx="$2"
  ls -1 "${d}/${pfx}"* 1>/dev/null 2>&1
}

_minimal_ready() {
  _have_file_with_prefix "${_data_bench_dir}" "pubmedqa_" && \
  _have_file_with_prefix "${_data_bench_dir}" "medqa_"
}

_all_ready() {
  local ok=0
  local count=0
  for p in "pubmedqa_" "medqa_" "medmcqa_" "mednli_" "ncbi_disease_" "bc5cdr_" "mmlu_"; do
    if _have_file_with_prefix "${_data_bench_dir}" "${p}"; then
      ok=$((ok + 1))
    fi
    count=$((count + 1))
  done
  # consider ready if at least 5 of the 7 major prefixes exist and perplexity corpus exists
  local rct="${_data_proc_dir}/pubmed_rct_test.jsonl"
  if [ "${ok}" -ge 5 ] && [ -f "${rct}" ]; then
    return 0
  fi
  return 1
}

_warm_hf_cache() {
  if [ "${WARM_HF_CACHE:-0}" = "1" ]; then
    info "Warming HF cache (best-effort)"
    set +e
    # shellcheck disable=SC2016
    python - <<'PY'
try:
    from transformers import AutoTokenizer
    for mid in ["distilbert-base-uncased", "bert-base-uncased"]:
        try:
            AutoTokenizer.from_pretrained(mid)
        except Exception:
            pass
except Exception:
    pass
PY
    set -e
  fi
}

# prepare_data minimal|all
prepare_data() {
  local mode="${1:-minimal}"
  if [ "${mode}" = "minimal" ]; then
    if _minimal_ready && [ "${FORCE_FLAG:-0}" -eq 0 ]; then
      info "Minimal datasets already present; skipping prepare-data"
      _warm_hf_cache
      return 0
    fi
    info "Preparing minimal datasets (PubMedQA + MedQA)"
    ( cd "${REPO_ROOT}" && python -m medical_tokalign.src.cli prepare-data )
    _warm_hf_cache
    return 0
  fi
  # all
  if _all_ready && [ "${FORCE_FLAG:-0}" -eq 0 ]; then
    info "All datasets already present; skipping prepare-data --all"
    _warm_hf_cache
    return 0
  fi
  info "Preparing all datasets"
  ( cd "${REPO_ROOT}" && python -m medical_tokalign.src.cli prepare-data --all )
  _warm_hf_cache
}

# End of data.sh

