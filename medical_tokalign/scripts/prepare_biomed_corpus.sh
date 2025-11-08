#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
echo "This script is deprecated."
echo "Use medical_tokalign/scripts/autorun_demo.sh or medical_tokalign/scripts/autorun_prod.sh instead."
echo "Example:"
echo "  bash medical_tokalign/scripts/autorun_prod.sh --corpus_config medical_tokalign/configs/corpus_biomed.yaml --model_id <MODEL_ID>"
exit 2

#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="medical_tokalign/configs/corpus_biomed.yaml"

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

echo "[MedTokAlign] Building biomedical corpus using: $CONFIG_PATH"
# Ensure module path resolves
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PKG_ROOT/.." && pwd)"
cd "$REPO_ROOT"

# Accelerate HF downloads if available
export HF_HUB_ENABLE_HF_TRANSFER=1

python -m medical_tokalign.src.biomed_corpus --config "$CONFIG_PATH"
echo "[MedTokAlign] Corpus build completed."


