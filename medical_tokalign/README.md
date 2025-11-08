Medical TokAlign for Llama 3 on RunPod (H100, Single GPU)

Overview
- A self-contained tokenizer-alignment extension for medical domains, optimized for one NVIDIA H100 PCIe 80GB on RunPod.
- Implements a TokAlign‑style approach: builds a term↔token-bundle alignment matrix from robust medical corpora and lexicons, selects high-benefit medical terms to add to the tokenizer, resizes model embeddings, and warm‑initializes new rows.
- Default target: meta-llama/Meta-Llama-3.1-8B. Backend: vLLM by default; HF fallback; optional TensorRT-LLM stub.

Key Features
- Full adaptation: add medical tokens + resize model embeddings; optional short CLM warmup for new rows only.
- H100-tuned performance: BF16, FlashAttention-2, torch.compile, TF32 allowed, paged attention (vLLM), CUDA memory expandable segments.
- Robust datasets via Hugging Face standard splits (no full-corpus eval): PubMedQA, MedQA-USMLE, MedNLI, NCBI-Disease, BC5CDR; separate perplexity corpus from PubMed RCT.
- One-command evaluation producing metrics.json, per-dataset CSVs, and samples.md.

Directory Layout (no notebooks required)
```
medical_tokalign/
  README.md
  requirements.txt
  configs/
    eval_medical.yaml
    corpus_biomed.yaml
    pipeline.yaml
    runpod_env.example
  scripts/
    bootstrap_runpod.sh
    clean_runpod.sh
    export_hf_model.sh
    autorun_demo.sh
    autorun_prod.sh
    lib/
      common.sh  resources.sh  glove.sh  data.sh  corpus.sh
      adapt.sh   eval.sh       artifacts.sh       tmux.sh
  src/
    __init__.py
    cli.py
    datasets_medical.py
    metrics_medical.py
    llama_utils.py
    perf_utils.py
    medical_eval.py
    biomed_corpus.py
    alignment_core.py
    alignment_runner.py
    term_selector.py
    convert_strict.py
    tokalign_vendor/
      cal_trans_matrix.py
      convert.py
  runs/  # auto-created for artifacts
```

RunPod Quickstart
1) Launch container (recommended):
   - Image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
   - Hardware: One H100 PCIe (80GB)

2) Environment (set once per session) and bootstrap:
```
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/workspace/.cache/huggingface
# If the model is gated, set your token:
export HF_TOKEN=hf_...
# Prefer fast file transfers and avoid unnecessary vision deps:
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_NO_TORCHVISION=1

# Install pinned deps (GPU/CUDA 12.x) or CPU-only:
# GPU:
bash medical_tokalign/scripts/bootstrap_env.sh --gpu-cu12
# CPU:
bash medical_tokalign/scripts/bootstrap_env.sh
```

3) (Optional) Legacy bootstrap_runpod.sh retains data/tooling helpers but `bootstrap_env.sh` now manages Python deps.

4) Prepare data (downloads/caches standard HF splits locally)
```
python -m medical_tokalign.src.cli prepare-data --all
```

5) Run tokenizer adaptation (TokAlign‑style, adds tokens + resizes embeddings)
```
python -m medical_tokalign.src.cli adapt \
  --model_id Qwen/Qwen2-7B \
  --top_k 8192 \
  --pivot 300 \
  --embedding_backend fasttext \
  --warmup_steps 2000 \
  --stage1_lr 5e-4 \
  --stage2_steps 500 \
  --stage2_lr 5e-5
```
Artifacts are saved under `medical_tokalign/runs/tokenizer_adapt/<timestamp>/` and include:
- tokenizer/ (adapted tokenizer)
- model/ (adapted HF model dir)
- align_matrix.json (1-1 mapping, TokAlign)
- added_terms.txt, selector_report.json
- added_terms.txt, report.md

6) Evaluate (vLLM if available; auto‑fallback to HF)
```
python -m medical_tokalign.src.cli eval --config medical_tokalign/configs/eval_medical.yaml
```
Outputs under `medical_tokalign/runs/medical_eval/<timestamp>/`:
- metrics.json (aggregate)
- alignment_metrics.json (token BLEU‑1 and BERTScore for alignment quality)
- per-dataset CSVs (predictions, references)
- samples.md (qualitative samples)

### Runners
Use these consolidated runners for end-to-end flows. Default embeddings backend is fasttext; set `EMBEDDING_BACKEND=glove` to enable GloVe.

- Demo (fast smoke/test; no tmux):
  ```
  bash medical_tokalign/scripts/autorun_demo.sh \
    --model_id meta-llama/Meta-Llama-3.1-8B \
    --corpus_config medical_tokalign/configs/corpus_med_tokalign_3gb.yaml \
    --eval_config medical_tokalign/configs/eval_medical_ultra_quick.yaml \
    --top_k 256 --pivot 300 --warmup_steps 0 --quick
  ```
- Prod (robust, resumable; tmux default):
  ```
  bash medical_tokalign/scripts/autorun_prod.sh \
    --model_id meta-llama/Meta-Llama-3.1-8B \
  --corpus_config medical_tokalign/configs/corpus_biomed.yaml \
  --eval_config medical_tokalign/configs/eval_medical.yaml \
  --top_k 8192 --pivot 300 --warmup_steps 0
  # Run inline (no tmux)
  bash medical_tokalign/scripts/autorun_prod.sh --no-tmux [...]
  ```
Notes:
- GloVe is gated behind `EMBEDDING_BACKEND=glove`. The toolchain is built only when enabled and heavy intermediates are cleaned post-adapt.
- Evaluation prefers vLLM; falls back to HF automatically if vLLM is unavailable.
- Demo `--quick` temporarily reduces corpus budgets without modifying repo files.

Stable Corpus Build (preflight + deterministic, file-backed logging)
```
# Optional preflight-only (validates datasets/splits/fields; does not write)
python -m medical_tokalign.src.cli build-corpus \
  --config medical_tokalign/configs/corpus_biomed.yaml \
  --preflight_only \
  --logdir medical_tokalign/runs/logs

# Resumable strict build (fails fast if any source missing; no autofill)
python -m medical_tokalign.src.cli build-corpus \
  --config medical_tokalign/configs/corpus_biomed.yaml \
  --strict_sources \
  --logdir medical_tokalign/runs/logs

# Outputs
# - medical_tokalign/data/biomed_corpus/*.jsonl
# - medical_tokalign/data/biomed_corpus/summary.json (complete flag)
# - medical_tokalign/runs/logs/corpus_<ts>.{log,jsonl}
```

### Migration (deprecated → new)
| Old script | Replacement |
| --- | --- |
| scripts/run_unattended.sh | scripts/autorun_prod.sh |
| scripts/eval_medical.sh | scripts/autorun_demo.sh or scripts/autorun_prod.sh |
| scripts/prepare_medical_data.sh | python -m medical_tokalign.src.cli prepare-data (or runners) |
| scripts/run_vocab_adaptation.sh | scripts/autorun_* (uses CLI adapt under the hood) |
| scripts/corpus_stable.sh | python -m medical_tokalign.src.cli build-corpus (or runners) |
| scripts/prepare_biomed_corpus.sh | python -m medical_tokalign.src.cli build-corpus |
| scripts/runpod_start.sh | scripts/autorun_prod.sh |

Maintenance / Cleanup (RunPod)
```
# Kill sessions/processes and clean vestiges; add --fresh-corpus to delete corpus dir
bash medical_tokalign/scripts/clean_runpod.sh --fresh-corpus
```

TokAlign‑style Adaptation Summary
- Build alignment matrix A between medical terms and observed token bundles from the base tokenizer on medical corpora.
- Score each term by fragmentation reduction × TF‑IDF × boundary consistency (single‑pass selector).
- Select top‑K terms not already atomic; add to tokenizer; resize model embeddings.
- Apply alignment using TokAlign conversion semantics (vendored), with strict state_dict replacement to avoid silent mismatches.
- Fast embedding backend (default FastText; fallback GloVe) with caching under `runs/embeddings/`.
- Progressive adaptation: stage‑1 (embeddings + lm_head), optional stage‑2 (full model) on medical text.

H100‑Optimized Defaults
- precision: bf16
- allow_tf32: true, matmul_precision: high
- attn_implementation: flash_attention_2 (flash‑attn is required; enforced)
- torch.compile: true (inductor, autotune)
- Inference backend: vLLM with paged attention
- vLLM sizing (80GB): gpu_memory_utilization≈0.92, max_batch_tokens≈65536, max_model_len=8192

Datasets & Splits (standard HF sources)
- PubMedQA: `pubmed_qa` (config `pqa_labeled`) — accuracy (test/dev)
- MedQA-USMLE: `openlifescienceai/MedQA-USMLE-4-options` (fallback `bigbio/med_qa_usmle`) — accuracy (test)
- MedNLI: `bigbio/mednli` (task `pairs`) — accuracy (test)
- NCBI-Disease NER: `bigbio/ncbi_disease` (config `bigbio_ner`) — span‑level F1 micro/macro (test)
- BC5CDR NER: `bigbio/bc5cdr` (config `bigbio_ner`) — span‑level F1 (test)
- Perplexity corpus: PubMed RCT abstracts (prefer `bigbio/pubmed_rct`) — perplexity via chunked teacher‑forcing

Llama 3 Variants
- Default: `meta-llama/Meta-Llama-3.1-8B`
- You can switch models in `configs/eval_medical.yaml` via `model_id`.
  Ensure HF access is granted (set `HF_TOKEN` if gated).

Notes on FP8 (TransformerEngine)
- The stack defaults to BF16 for stability. FP8 (TransformerEngine) hooks can be enabled in advanced scenarios; requirements include matching CUDA/TE wheels.

Troubleshooting
- OOM: lower `max_batch_tokens` (vLLM) or `per_device_batch_size` (HF); reduce `max_model_len`.
- Gated models: export `HF_TOKEN` and ensure it has access to the Llama weights.
- SciSpacy model: if auto-download fails, set `SCISPACY_MODEL_URL` to a direct wheel URL and rerun the data script.
- vLLM wheels: on fresh Torch/CUDA combos vLLM may be unavailable. The pipeline will fall back to HF automatically. You can also set `eval_backend: hf` in `configs/eval_medical.yaml` to force HF.
- Dataset scripts errors on HF 4.x: we pin `datasets==2.21.x`. If you see “Dataset scripts are no longer supported”, run `bash scripts/bootstrap_env.sh` to install pins.
- Qwen2 import error / torchvision: export `TRANSFORMERS_NO_TORCHVISION=1` (default) or install a compatible torchvision; re-run `bootstrap_env.sh --gpu-cu12`.


One‑command End‑to‑End (CLI)
```
python -m medical_tokalign.src.cli pipeline \
  --model_id meta-llama/Meta-Llama-3.1-8B \
  --corpus_config medical_tokalign/configs/corpus_biomed.yaml \
  --eval_config medical_tokalign/configs/eval_medical.yaml \
  --top_k 8192 --pivot 300 --warmup_steps 0
```

Smoke Test (quick validation)
```
# Minimal top_k and no warmup to validate the path without heavy compute
python -m medical_tokalign.src.cli prepare-data --all
python -m medical_tokalign.src.cli build-corpus --config medical_tokalign/configs/corpus_biomed.yaml
python -m medical_tokalign.src.cli adapt \
  --model_id meta-llama/Meta-Llama-3.1-8B \
  --top_k 128 --pivot 100 --warmup_steps 0
# Optionally set small dataset limits in configs/eval_medical.yaml before running eval
python -m medical_tokalign.src.cli eval --config medical_tokalign/configs/eval_medical.yaml
```
Citation & Licenses
- Datasets and model licenses apply. Ensure you comply with source licenses (e.g., BigBio, PubMedQA, MedQA, MedNLI, NCBI, BC5CDR).



## Stable pipeline
Use the production runner or the CLI `pipeline` subcommand for fully unattended flows. Legacy orchestrators have been deprecated.

