#!/usr/bin/env bash
set -euo pipefail

# Benchmark TokAlign embedding backends (baseline GloVe vs FastText)
# Produces a compact JSON summary with timings and output directories.
#
# Usage:
#   bash medical_tokalign/scripts/benchmark_backends.sh --model_id <hf_model_id> \
#     --corpus_config medical_tokalign/configs/corpus_biomed.yaml \
#     --eval_config medical_tokalign/configs/eval_medical.yaml
#

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/runs/bench_${TS}"
mkdir -p "$OUT_DIR"

MODEL_ID=""
CORPUS_CONFIG="$ROOT_DIR/configs/corpus_biomed.yaml"
EVAL_CONFIG="$ROOT_DIR/configs/eval_medical.yaml"
STAGE1_STEPS="${STAGE1_STEPS:-1000}"
STAGE1_LR="${STAGE1_LR:-5e-4}"
STAGE2_STEPS="${STAGE2_STEPS:-0}"
STAGE2_LR="${STAGE2_LR:-5e-5}"
TOP_K="${TOP_K:-8192}"
PIVOT="${PIVOT:-300}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_id) MODEL_ID="$2"; shift 2 ;;
    --corpus_config) CORPUS_CONFIG="$2"; shift 2 ;;
    --eval_config) EVAL_CONFIG="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --pivot) PIVOT="$2"; shift 2 ;;
    --stage1_steps) STAGE1_STEPS="$2"; shift 2 ;;
    --stage1_lr) STAGE1_LR="$2"; shift 2 ;;
    --stage2_steps) STAGE2_STEPS="$2"; shift 2 ;;
    --stage2_lr) STAGE2_LR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL_ID" ]]; then
  echo "Usage: $0 --model_id <hf_model_id> [--corpus_config path] [--eval_config path]" >&2
  exit 2
fi

# 1) Build corpus (idempotent)
python -m medical_tokalign.src.cli build-corpus --config "$CORPUS_CONFIG" || true

summarize() {
  local NAME="$1"
  local ADAPT_DIR="$2"
  local EVAL_DIR="$3"
  local ALIGN_JSON="${ADAPT_DIR}/align_matrix.json"
  local ALIGN_METRICS="${EVAL_DIR}/alignment_metrics.json"
  local METRICS_JSON="${EVAL_DIR}/metrics_summary.json"
  echo "{"
  echo "  \"name\": \"${NAME}\","
  echo "  \"adapt_dir\": \"${ADAPT_DIR}\","
  echo "  \"eval_dir\": \"${EVAL_DIR}\","
  echo "  \"align_json_exists\": $( [[ -f \"$ALIGN_JSON\" ]] && echo true || echo false ),"
  echo "  \"alignment_metrics\": $( [[ -f \"$ALIGN_METRICS\" ]] && cat \"$ALIGN_METRICS\" || echo null ),"
  echo "  \"metrics\": $( [[ -f \"$METRICS_JSON\" ]] && cat \"$METRICS_JSON\" || echo null )"
  echo "}"
}

run_case() {
  local BACKEND="$1"
  local CASE_DIR="$OUT_DIR/$BACKEND"
  mkdir -p "$CASE_DIR"
  local T0=$(date +%s)
  python -m medical_tokalign.src.cli adapt \
    --model_id "$MODEL_ID" --top_k "$TOP_K" --pivot "$PIVOT" \
    --embedding_backend "$BACKEND" \
    --warmup_steps "$STAGE1_STEPS" --stage1_lr "$STAGE1_LR" \
    --stage2_steps "$STAGE2_STEPS" --stage2_lr "$STAGE2_LR"
  local ADAPT_DIR=$(ls -td "$ROOT_DIR/runs/tokenizer_adapt"/* 2>/dev/null | head -n1 || true)
  python -m medical_tokalign.src.cli eval --config "$EVAL_CONFIG"
  local EVAL_DIR=$(ls -td "$ROOT_DIR/runs/medical_eval"/* 2>/dev/null | head -n1 || true)
  local T1=$(date +%s)
  local DUR=$(( T1 - T0 ))
  summarize "$BACKEND" "$ADAPT_DIR" "$EVAL_DIR" > "$CASE_DIR/summary.json"
  echo "$DUR" > "$CASE_DIR/duration_seconds.txt"
}

run_case glove
run_case fasttext

# Write combined summary
{
  echo "{"
  echo "  \"timestamp\": \"${TS}\","
  echo "  \"model_id\": \"${MODEL_ID}\","
  echo "  \"top_k\": ${TOP_K},"
  echo "  \"pivot\": ${PIVOT},"
  echo "  \"stage1_steps\": ${STAGE1_STEPS},"
  echo "  \"stage2_steps\": ${STAGE2_STEPS},"
  echo "  \"cases\": ["
  paste -sd, "$OUT_DIR/glove/summary.json" "$OUT_DIR/fasttext/summary.json"
  echo "  ]"
  echo "}"
} > "$OUT_DIR/benchmark_summary.json"

echo "[benchmark] summary: $OUT_DIR/benchmark_summary.json"


