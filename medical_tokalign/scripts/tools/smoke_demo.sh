#!/usr/bin/env bash
# Lightweight smoke: env preflight + quick demo with small model (no heavy downloads)
set -euo pipefail
IFS=$'\n\t'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2-7B}"
EXTRA_ARGS=()
for a in "$@"; do EXTRA_ARGS+=("$a"); done

echo "[smoke] Bootstrap env (no install)..."
bash "${ROOT}/scripts/bootstrap_env.sh" --no-install || true

echo "[smoke] Preflight only..."
python - <<'PY'
import transformers, datasets, huggingface_hub, sys
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("ok")
PY

echo "[smoke] Running autorun_demo --quick"
exec bash "${ROOT}/scripts/autorun_demo.sh" --model_id "${MODEL_ID}" --quick "${EXTRA_ARGS[@]:-}"

