#!/usr/bin/env bash
# Remove run artifacts and temporary files to free space
set -euo pipefail
IFS=$'\n\t'

_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_THIS_DIR}/../.." && pwd)"
RUNS_DIR="${REPO_ROOT}/medical_tokalign/runs"

echo "[clean] Removing run artifacts under ${RUNS_DIR}"
rm -rf "${RUNS_DIR}/tokenizer_adapt"/* 2>/dev/null || true
rm -rf "${RUNS_DIR}/medical_eval"/* 2>/dev/null || true
rm -rf "${RUNS_DIR}/embeddings"/* 2>/dev/null || true
rm -f "${RUNS_DIR}/logs/"*.log 2>/dev/null || true

echo "[clean] Removing GloVe temp shuffle binaries (best-effort)"
rm -f "${REPO_ROOT}/medical_tokalign/tools/GloVe"/temp_shuffle_*.bin 2>/dev/null || true

echo "[clean] Done."


