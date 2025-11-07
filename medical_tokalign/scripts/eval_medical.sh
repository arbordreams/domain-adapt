#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="medical_tokalign/configs/eval_medical.yaml"

usage() {
  echo "Usage: $0 [--config PATH]" >&2
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG_PATH="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

echo "[MedTokAlign] Starting medical evaluation with config: $CONFIG_PATH"

# Single-run lock (repo-local, no hardcoded /workspace)
LOCK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.locks"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/eval.lock"
if ! flock -n 9; then
  echo "[MedTokAlign] Another eval is running (lock held). Exiting."
  exit 0
fi

# Ensure importing 'medical_tokalign' works regardless of cwd
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"    # medical_tokalign/
REPO_ROOT="$(cd "$PKG_ROOT/.." && pwd)"                         # repo root
cd "$REPO_ROOT"

# Fast local caches (use /data if present; otherwise ~/.cache)
CACHE_ROOT_DEFAULT="/data/medtokalign_cache"
if [[ -d "/data" ]]; then
  CACHE_ROOT="$CACHE_ROOT_DEFAULT"
else
  CACHE_ROOT="$HOME/.cache/medtokalign"
fi
mkdir -p "$CACHE_ROOT" "$CACHE_ROOT/hf" "$CACHE_ROOT/hf/transformers" "$CACHE_ROOT/hf/datasets" "$CACHE_ROOT/torch_extensions"

export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_ROOT/hf/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/hf/datasets}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE_ROOT/torch_extensions}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export HF_HUB_ENABLE_HF_TRANSFER=1

python -m medical_tokalign.src.cli eval --config "$CONFIG_PATH"


