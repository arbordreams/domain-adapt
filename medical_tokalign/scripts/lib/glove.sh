#!/usr/bin/env bash
# MedTokAlign shared shell library - GloVe gating and cleanup
set -euo pipefail
IFS=$'\n\t'

_gl_log() { printf "[%s] [GLOVE] %s\n" "$(date +"%Y-%m-%dT%H:%M:%S%z")" "${*}"; }

# Expect REPO_ROOT/PROJECT_DIR from common.sh if sourced
_resolve_glove_dir() {
  local proj="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/medical_tokalign}"
  echo "${proj}/tools/GloVe"
}

glove_binaries_ok() {
  local gdir="$(_resolve_glove_dir)"
  [ -x "${gdir}/build/vocab_count" ] && [ -x "${gdir}/build/cooccur" ] && \
  [ -x "${gdir}/build/shuffle" ] && [ -x "${gdir}/build/glove" ]
}

bootstrap_glove() {
  if [ "${EMBEDDING_BACKEND:-fasttext}" != "glove" ]; then
    return 0
  fi
  local gdir="$(_resolve_glove_dir)"
  if [ ! -d "${gdir}" ]; then
    if command -v git >/dev/null 2>&1; then
      _gl_log "Cloning Stanford GloVe into ${gdir}"
      mkdir -p "$(dirname "${gdir}")"
      git clone https://github.com/stanfordnlp/GloVe.git "${gdir}"
    else
      _gl_log "git not available; relying on Python helper to clone GloVe at runtime"
      return 0
    fi
  fi
  if ! glove_binaries_ok; then
    _gl_log "Building GloVe binaries in ${gdir}"
    make -C "${gdir}" >/dev/null
  fi
  if glove_binaries_ok; then
    _gl_log "GloVe binaries ready"
  else
    _gl_log "Failed to prepare GloVe binaries (best-effort)"
  fi
}

# Cleanup heavyweight intermediates after successful adapt
cleanup_glove_intermediates() {
  if [ "${EMBEDDING_BACKEND:-fasttext}" != "glove" ]; then
    return 0
  fi
  local run_dir="${1:-}"
  local gdir="$(_resolve_glove_dir)"
  # Remove per-run corpora
  if [ -n "${run_dir}" ] && [ -d "${run_dir}" ]; then
    rm -f "${run_dir}/glove_source.txt" "${run_dir}/glove_target.txt" \
          "${run_dir}/glove_source.stats.json" "${run_dir}/glove_target.stats.json" || true
  fi
  # Remove global cooccurrence binaries for stable bases used by CLI ("vec-source","vec-target")
  rm -f "${gdir}/cooccurrence.vec-source.bin" \
        "${gdir}/cooccurrence.shuf.vec-source.bin" \
        "${gdir}/cooccurrence.vec-target.bin" \
        "${gdir}/cooccurrence.shuf.vec-target.bin" || true
  _gl_log "Cleaned GloVe intermediates"
}

# End of glove.sh

