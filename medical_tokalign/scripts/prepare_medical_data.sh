#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
echo "This script is deprecated."
echo "Use medical_tokalign/scripts/autorun_demo.sh or medical_tokalign/scripts/autorun_prod.sh instead."
echo "Example:"
echo "  bash medical_tokalign/scripts/autorun_prod.sh --model_id <MODEL_ID>"
exit 2

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "[MedTokAlign] Preparing ALL medical datasets via unified CLI..."

# Accelerate HF downloads via hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

python -m medical_tokalign.src.cli prepare-data --all

echo "[MedTokAlign] Data preparation completed."


