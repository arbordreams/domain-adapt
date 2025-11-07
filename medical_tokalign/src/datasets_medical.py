import json
import os
import warnings
from typing import Dict, List, Tuple

from datasets import load_dataset


def _load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_pubmedqa(bench_dir: str) -> Dict[str, List[dict]]:
    data = {}
    for split in ["train", "validation", "test"]:
        p = os.path.join(bench_dir, f"pubmedqa_{split}.jsonl")
        if os.path.isfile(p):
            data[split] = _load_jsonl(p)
    if not data:
        # HF fallback
        try:
            ds = load_dataset("pubmed_qa", "pqa_labeled")
            for split in ds:
                data[split] = list(ds[split])
        except Exception:
            warnings.warn(f"[datasets] PubMedQA files not found and HF fallback failed; skipping.")
    return data


def load_medqa(bench_dir: str) -> Dict[str, List[dict]]:
    data = {}
    for split in ["train", "validation", "test"]:
        p = os.path.join(bench_dir, f"medqa_{split}.jsonl")
        if os.path.isfile(p):
            data[split] = _load_jsonl(p)
    if not data:
        # HF fallback
        ds = None
        for ds_id in [
            "openlifescienceai/MedQA-USMLE-4-options-hf",
            "openlifescienceai/MedQA-USMLE-4-options",
        ]:
            try:
                ds = load_dataset(ds_id)
                break
            except Exception:
                continue
        if ds is None:
            try:
                ds = load_dataset("bigbio/med_qa_usmle")
            except Exception:
                ds = None
        if ds is not None:
            for split in ds:
                data[split] = list(ds[split])
        else:
            warnings.warn(f"[datasets] MedQA not found and HF fallback failed; skipping.")
    return data


def load_mednli(bench_dir: str) -> Dict[str, List[dict]]:
    data = {}
    for split in ["train", "validation", "test"]:
        p = os.path.join(bench_dir, f"mednli_{split}.jsonl")
        if os.path.isfile(p):
            data[split] = _load_jsonl(p)
    if not data:
        try:
            ds = load_dataset("bigbio/mednli", "pairs")
            for split in ds:
                data[split] = list(ds[split])
        except Exception:
            warnings.warn(f"[datasets] MedNLI files not found and HF fallback failed; skipping.")
    return data


def load_ncbi_disease(bench_dir: str) -> Dict[str, List[dict]]:
    data = {}
    for split in ["train", "validation", "test"]:
        p = os.path.join(bench_dir, f"ncbi_disease_{split}.jsonl")
        if os.path.isfile(p):
            data[split] = _load_jsonl(p)
    if not data:
        try:
            ds = load_dataset("bigbio/ncbi_disease", "bigbio_ner")
            for split in ds:
                data[split] = list(ds[split])
        except Exception:
            warnings.warn(f"[datasets] NCBI-Disease files not found and HF fallback failed; skipping.")
    return data


def load_bc5cdr(bench_dir: str) -> Dict[str, List[dict]]:
    data = {}
    for split in ["train", "validation", "test"]:
        p = os.path.join(bench_dir, f"bc5cdr_{split}.jsonl")
        if os.path.isfile(p):
            data[split] = _load_jsonl(p)
    if not data:
        try:
            ds = load_dataset("bigbio/bc5cdr", "bigbio_ner")
            for split in ds:
                data[split] = list(ds[split])
        except Exception:
            warnings.warn(f"[datasets] BC5CDR files not found and HF fallback failed; skipping.")
    return data


def load_medmcqa(bench_dir: str) -> Dict[str, List[dict]]:
    data = {}
    for split in ["train", "validation", "test"]:
        p = os.path.join(bench_dir, f"medmcqa_{split}.jsonl")
        if os.path.isfile(p):
            data[split] = _load_jsonl(p)
    if not data:
        ds = None
        for ds_id in [
            "medmcqa",
            "openlifescienceai/MedMCQA",
        ]:
            try:
                ds = load_dataset(ds_id)
                break
            except Exception:
                continue
        if ds is not None:
            for split in ds:
                data[split] = list(ds[split])
        else:
            warnings.warn(f"[datasets] MedMCQA files not found and HF fallback failed; skipping.")
    return data


def load_mmlu_medical(bench_dir: str, subjects: List[str] | None = None) -> Dict[str, List[dict]]:
    data = {}
    for split in ["train", "validation", "test"]:
        # Prefer pre-filtered medical subset if available
        p1 = os.path.join(bench_dir, f"mmlu_medical_{split}.jsonl")
        p2 = os.path.join(bench_dir, f"mmlu_{split}.jsonl")
        if os.path.isfile(p1):
            data[split] = _load_jsonl(p1)
        elif os.path.isfile(p2):
            exs = _load_jsonl(p2)
            if subjects:
                keep = set(s.lower() for s in subjects)
                exs = [e for e in exs if str(e.get("subject", "")).lower() in keep]
            data[split] = exs
    if not data:
        ds = None
        try:
            ds = load_dataset("cais/mmlu")
        except Exception:
            try:
                ds = load_dataset("openai_hendrycks_test")
            except Exception:
                ds = None
        if ds is not None:
            for split in ds:
                rows = list(ds[split])
                if subjects:
                    keep = set(s.lower() for s in subjects)
                    rows = [e for e in rows if str(e.get("subject", "")).lower() in keep]
                data[split] = rows
        else:
            warnings.warn(f"[datasets] MMLU files not found and HF fallback failed; skipping.")
    return data


def load_perplexity_corpus(proc_dir: str, source: str = "pubmed_rct") -> List[str]:
    if source == "pubmed_rct":
        p = os.path.join(proc_dir, "pubmed_rct_test.jsonl")
        if os.path.isfile(p):
            exs = _load_jsonl(p)
            texts = []
            for ex in exs:
                # bigbio/pubmed_rct uses fields like: abstract, title, sentences
                t = ex.get("text") or ex.get("abstract") or ex.get("sentence") or ""
                if isinstance(t, list):
                    t = "\n".join(t)
                texts.append(str(t))
            return texts
        # HF fallback for perplexity corpus
        try:
            ds = load_dataset("bigbio/pubmed_rct")
            exs = list(ds.get("test") or [])
            texts = []
            for ex in exs:
                t = ex.get("text") or ex.get("abstract") or ex.get("sentence") or ""
                if isinstance(t, list):
                    t = "\n".join(t)
                texts.append(str(t))
            return texts
        except Exception:
            return []
    # fallback empty
    return []


def format_pubmedqa_prompt(example: dict) -> Tuple[str, str]:
    # PubMedQA uses dict with fields: question, context(s), final_decision (yes/no/maybe)
    q = example.get("question") or example.get("question_concept") or ""
    ctx = example.get("context") or example.get("long_answer") or ""
    if isinstance(ctx, list):
        ctx = "\n".join(ctx)
    prompt = (
        "You are a medical QA assistant. Answer yes/no/maybe.\n"
        f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer: "
    )
    label = (example.get("final_decision") or example.get("answer") or "").strip().lower()
    return prompt, label


def format_medqa_prompt(example: dict) -> Tuple[str, str]:
    # Expect fields question + 4 options + correct answer letter or text
    # Support multiple schemas: (question, A-D), (question_text, option_a-d), (sent1, ending0-3)
    q = example.get("question") or example.get("question_text") or example.get("sent1") or ""
    opts = [
        example.get("A") or example.get("option_a") or example.get("ending0") or "",
        example.get("B") or example.get("option_b") or example.get("ending1") or "",
        example.get("C") or example.get("option_c") or example.get("ending2") or "",
        example.get("D") or example.get("option_d") or example.get("ending3") or "",
    ]
    prompt = (
        "You are a medical exam assistant. Choose A, B, C, or D and justify briefly.\n"
        f"Question: {q}\n"
        f"A) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}\n"
        "Answer (just the letter): "
    )
    # Accept answer in multiple formats: letter (A-D), index (0-3 or 1-4), or text
    raw_ans = example.get("answer")
    if raw_ans is None:
        raw_ans = example.get("label")
    if raw_ans is None:
        raw_ans = example.get("answer_idx")

    label = ""
    ans_str = str(raw_ans).strip().upper() if raw_ans is not None else ""

    # Case 1: direct letter
    if ans_str in {"A", "B", "C", "D"}:
        label = ans_str
    else:
        # Case 2: numeric index
        try:
            idx_val = int(ans_str)
        except (TypeError, ValueError):
            idx_val = None
        if idx_val is not None:
            if idx_val in {0, 1, 2, 3}:  # 0-based indices
                label = "ABCD"[idx_val]
            elif idx_val in {1, 2, 3, 4}:  # 1-based indices
                label = "ABCD"[idx_val - 1]
        
        # Case 3: answer text matching one of the options (normalize and compare)
        if not label and ans_str:
            def _norm(s: str) -> str:
                s = str(s).strip().lower()
                s = s.rstrip(". ")
                return " ".join(s.split())

            norm_opts = [_norm(o) for o in opts]
            norm_ans = _norm(ans_str)
            for i, no in enumerate(norm_opts):
                if no and no == norm_ans:
                    label = "ABCD"[i]
                    break
    return prompt, label


def format_mednli_prompt(example: dict) -> Tuple[str, str]:
    # Fields: premise, hypothesis, label in {entailment, contradiction, neutral}
    prem = example.get("premise") or ""
    hyp = example.get("hypothesis") or ""
    prompt = (
        "You are a clinical NLI assistant. Predict entailment, contradiction, or neutral.\n"
        f"Premise: {prem}\nHypothesis: {hyp}\nLabel: "
    )
    label = (example.get("label") or example.get("gold_label") or "").strip().lower()
    return prompt, label


def format_medmcqa_prompt(example: dict) -> Tuple[str, str]:
    # Support typical MedMCQA schemas: question + four options, label letter or index
    q = example.get("question") or example.get("question_text") or example.get("sent1") or ""
    opts = [
        example.get("A") or example.get("option_a") or example.get("opa") or example.get("ending0") or "",
        example.get("B") or example.get("option_b") or example.get("opb") or example.get("ending1") or "",
        example.get("C") or example.get("option_c") or example.get("opc") or example.get("ending2") or "",
        example.get("D") or example.get("option_d") or example.get("opd") or example.get("ending3") or "",
    ]
    prompt = (
        "You are a medical exam assistant. Choose A, B, C, or D and justify briefly.\n"
        f"Question: {q}\n"
        f"A) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}\n"
        "Answer (just the letter): "
    )
    raw_ans = example.get("answer")
    if raw_ans is None:
        raw_ans = example.get("label")
    if raw_ans is None:
        raw_ans = example.get("answer_idx")
    label = ""
    ans_str = str(raw_ans).strip().upper() if raw_ans is not None else ""
    if ans_str in {"A", "B", "C", "D"}:
        label = ans_str
    else:
        try:
            idx_val = int(ans_str)
        except (TypeError, ValueError):
            idx_val = None
        if idx_val is not None:
            if idx_val in {0, 1, 2, 3}:
                label = "ABCD"[idx_val]
            elif idx_val in {1, 2, 3, 4}:
                label = "ABCD"[idx_val - 1]
    return prompt, label


def format_mmlu_prompt(example: dict) -> Tuple[str, str]:
    q = example.get("question") or example.get("question_text") or ""
    # choices may be list under 'choices' or individual fields
    ch = example.get("choices")
    if isinstance(ch, list) and len(ch) >= 4:
        opts = [str(ch[i]) if i < len(ch) else "" for i in range(4)]
    else:
        opts = [
            example.get("A") or example.get("option_a") or example.get("ending0") or "",
            example.get("B") or example.get("option_b") or example.get("ending1") or "",
            example.get("C") or example.get("option_c") or example.get("ending2") or "",
            example.get("D") or example.get("option_d") or example.get("ending3") or "",
        ]
    prompt = (
        "You are a medical exam assistant. Choose A, B, C, or D and justify briefly.\n"
        f"Question: {q}\n"
        f"A) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}\n"
        "Answer (just the letter): "
    )
    raw_ans = example.get("answer")
    if raw_ans is None:
        raw_ans = example.get("label")
    label = ""
    ans_str = str(raw_ans).strip().upper() if raw_ans is not None else ""
    if ans_str in {"A", "B", "C", "D"}:
        label = ans_str
    else:
        try:
            idx_val = int(ans_str)
        except (TypeError, ValueError):
            idx_val = None
        if idx_val is not None:
            if idx_val in {0, 1, 2, 3}:
                label = "ABCD"[idx_val]
            elif idx_val in {1, 2, 3, 4}:
                label = "ABCD"[idx_val - 1]
    return prompt, label


def extract_ner_gold_spans(example: dict) -> List[str]:
    # bigbio_ner style: entities with text spans
    ents = example.get("entities") or []
    gold = []
    for e in ents:
        txt = e.get("text")
        if isinstance(txt, list):
            txt = " ".join(txt)
        if txt:
            gold.append(str(txt))
    return gold


def format_ner_prompt(text: str, entity_type: str = "Disease") -> str:
    return (
        f"Extract all {entity_type} entities from the text. One per line, no bullets, exact surface forms.\n"
        f"Text: {text}\nEntities:\n"
    )


