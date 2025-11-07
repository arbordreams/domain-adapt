#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash medical_tokalign/scripts/export_hf_model.sh <hf_repo> [run_dir]
# Example:
#   bash medical_tokalign/scripts/export_hf_model.sh youruser/medtokalign-qwen2 latest

HF_REPO="${1:-}"
RUN_DIR_INPUT="${2:-}"

if [[ -z "$HF_REPO" ]]; then
  echo "Usage: $0 <hf_repo> [run_dir]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "$RUN_DIR_INPUT" || "$RUN_DIR_INPUT" == "latest" ]]; then
  RUN_DIR=$(ls -1d "$ROOT_DIR"/runs/tokenizer_adapt/* 2>/dev/null | tail -n1 || true)
else
  RUN_DIR="$RUN_DIR_INPUT"
fi

if [[ -z "${RUN_DIR:-}" || ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi

MODEL_DIR="$RUN_DIR/model"
TOK_DIR="$RUN_DIR/tokenizer"
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model dir missing: $MODEL_DIR" >&2
  exit 1
fi

python - <<PY
import os, sys
from huggingface_hub import HfApi, create_repo, upload_folder

repo_id = os.environ.get('HF_REPO')
run_dir = os.environ.get('RUN_DIR')
if not repo_id or not run_dir:
    print('Missing HF_REPO or RUN_DIR', file=sys.stderr)
    sys.exit(1)

api = HfApi()
try:
    create_repo(repo_id=repo_id, private=True, exist_ok=True, token=os.environ.get('HF_TOKEN'))
except Exception as e:
    print('create_repo warn:', e)

model_dir = os.path.join(run_dir, 'model')
tok_dir = os.path.join(run_dir, 'tokenizer')
upload_folder(repo_id=repo_id, folder_path=model_dir, path_in_repo='', commit_message=f'Upload model from {run_dir}')
if os.path.isdir(tok_dir):
    upload_folder(repo_id=repo_id, folder_path=tok_dir, path_in_repo='tokenizer', commit_message=f'Upload tokenizer from {run_dir}')
print('Uploaded to HF:', repo_id)
PY


