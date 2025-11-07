import json
import os
from typing import Iterable, Optional

from datasets import load_dataset


def _save_jsonl(ds: Iterable[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def prepare_medical_benchmarks(bench_dir: str, proc_dir: str) -> None:
    """Download and prepare PubMedQA and MedQA datasets only."""
    os.makedirs(bench_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    cache_dir = os.path.join(proc_dir, ".hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # PubMedQA (pqa_labeled)
    print("[MedTokAlign] Downloading PubMedQA...")
    pubmedqa = load_dataset("pubmed_qa", "pqa_labeled", cache_dir=cache_dir)
    for split in pubmedqa:
        _save_jsonl(pubmedqa[split], os.path.join(bench_dir, f"pubmedqa_{split}.jsonl"))

    # MedQA-USMLE 4-options (packaged variants only, no BigBio fallback)
    print("[MedTokAlign] Downloading MedQA-USMLE 4-options...")
    medqa = None
    for ds_id in [
        "openlifescienceai/MedQA-USMLE-4-options-hf",
        "openlifescienceai/MedQA-USMLE-4-options",
    ]:
        try:
            medqa = load_dataset(ds_id, cache_dir=cache_dir)
            break
        except Exception:
            pass

    if medqa is None:
        raise SystemExit("Failed to load MedQA from openlifescienceai. Ensure internet access and HF credentials.")

    for split in medqa:
        _save_jsonl(medqa[split], os.path.join(bench_dir, f"medqa_{split}.jsonl"))


def prepare_medical_benchmarks_all(bench_dir: str, proc_dir: str) -> None:
    """Download and normalize all supported medical datasets to JSONL.

    Datasets: PubMedQA, MedQA-USMLE (4 options), MedMCQA, MedNLI, NCBI-Disease,
    BC5CDR, MMLU, PubMed RCT (perplexity corpus under processed/).
    """
    os.makedirs(bench_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    cache_dir = os.path.join(proc_dir, ".hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # PubMedQA
    print("[MedTokAlign] Downloading PubMedQA...")
    pubmedqa = load_dataset("pubmed_qa", "pqa_labeled", cache_dir=cache_dir)
    for split in pubmedqa:
        _save_jsonl(pubmedqa[split], os.path.join(bench_dir, f"pubmedqa_{split}.jsonl"))

    # MedQA-USMLE
    print("[MedTokAlign] Downloading MedQA-USMLE 4-options...")
    medqa = None
    for ds_id in [
        "openlifescienceai/MedQA-USMLE-4-options-hf",
        "openlifescienceai/MedQA-USMLE-4-options",
    ]:
        try:
            medqa = load_dataset(ds_id, cache_dir=cache_dir)
            break
        except Exception:
            pass
    if medqa is None:
        # Optional BigBio fallback (schema may differ; still useful for corpora)
        try:
            medqa = load_dataset("bigbio/med_qa_usmle", cache_dir=cache_dir)
        except Exception:
            medqa = None
    if medqa is not None:
        for split in medqa:
            _save_jsonl(medqa[split], os.path.join(bench_dir, f"medqa_{split}.jsonl"))

    # MedMCQA
    print("[MedTokAlign] Downloading MedMCQA...")
    medmcqa = None
    for ds_id in [
        "medmcqa",  # canonical id
        "openlifescienceai/MedMCQA",
    ]:
        try:
            medmcqa = load_dataset(ds_id, cache_dir=cache_dir)
            break
        except Exception:
            continue
    if medmcqa is not None:
        for split in medmcqa:
            _save_jsonl(medmcqa[split], os.path.join(bench_dir, f"medmcqa_{split}.jsonl"))

    # MedNLI (BigBio pairs)
    print("[MedTokAlign] Downloading MedNLI (bigbio/mednli:pairs)...")
    try:
        mednli = load_dataset("bigbio/mednli", "pairs", cache_dir=cache_dir, trust_remote_code=True)
        for split in mednli:
            _save_jsonl(mednli[split], os.path.join(bench_dir, f"mednli_{split}.jsonl"))
    except Exception:
        pass

    # NCBI-Disease NER (BigBio)
    print("[MedTokAlign] Downloading NCBI-Disease (bigbio_ner)...")
    try:
        ncbi = load_dataset("bigbio/ncbi_disease", "bigbio_ner", cache_dir=cache_dir, trust_remote_code=True)
        for split in ncbi:
            _save_jsonl(ncbi[split], os.path.join(bench_dir, f"ncbi_disease_{split}.jsonl"))
    except Exception:
        pass

    # BC5CDR NER (BigBio)
    print("[MedTokAlign] Downloading BC5CDR (bigbio_ner)...")
    try:
        bc5 = load_dataset("bigbio/bc5cdr", "bigbio_ner", cache_dir=cache_dir, trust_remote_code=True)
        for split in bc5:
            _save_jsonl(bc5[split], os.path.join(bench_dir, f"bc5cdr_{split}.jsonl"))
    except Exception:
        pass

    # MMLU (full; filtering happens at eval time via config subjects)
    print("[MedTokAlign] Downloading MMLU (cais/mmlu)...")
    try:
        mmlu = load_dataset("cais/mmlu", cache_dir=cache_dir)
        for split in mmlu:
            _save_jsonl(mmlu[split], os.path.join(bench_dir, f"mmlu_{split}.jsonl"))
    except Exception:
        pass

    # PubMed RCT (perplexity corpus) → processed/
    print("[MedTokAlign] Downloading PubMed RCT (for perplexity corpus)...")
    try:
        rct = load_dataset("bigbio/pubmed_rct", cache_dir=cache_dir, trust_remote_code=True)
        # Save only test split by default
        if "test" in rct:
            _save_jsonl(rct["test"], os.path.join(proc_dir, "pubmed_rct_test.jsonl"))
    except Exception:
        pass


