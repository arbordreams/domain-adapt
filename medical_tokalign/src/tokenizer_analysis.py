from __future__ import annotations

import argparse
import json
import os
import re
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

from transformers import AutoTokenizer


def _load_terms(path: Optional[str], limit: int = 2048) -> List[str]:
    if not path or not os.path.isfile(path):
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
            if len(out) >= limit:
                break
    return out


def _iter_sample_texts(bench_dir: str, proc_dir: str, biomed_dir: Optional[str] = None, cap: int = 2000) -> List[str]:
    from .term_selector import iter_med_texts
    texts: List[str] = []
    for t in iter_med_texts(bench_dir, proc_dir):
        texts.append(t)
        if len(texts) >= cap:
            break
    # optionally enrich with biomed corpus
    if biomed_dir and os.path.isdir(biomed_dir) and len(texts) < cap:
        try:
            for name in sorted(os.listdir(biomed_dir)):
                if not name.endswith(".jsonl"):
                    continue
                with open(os.path.join(biomed_dir, name), "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            import json as _json
                            ex = _json.loads(line)
                            v = ex.get("text")
                            if isinstance(v, list):
                                v = "\n".join(map(str, v))
                            if isinstance(v, str) and v:
                                texts.append(v)
                                if len(texts) >= cap:
                                    return texts
                        except Exception:
                            continue
        except Exception:
            pass
    return texts


def _tokens_per_word(tok, texts: List[str]) -> Dict[str, float]:
    lens: List[float] = []
    word_counts: List[int] = []
    for t in texts:
        words = len([w for w in re.split(r"\s+", t.strip()) if w])
        if words <= 0:
            continue
        ids = tok(t, add_special_tokens=False).input_ids
        lens.append(len(ids) / max(1, words))
        word_counts.append(words)
    if not lens:
        return {"avg": 0.0, "p50": 0.0, "p90": 0.0}
    s = sorted(lens)
    def pct(p: float) -> float:
        if not s:
            return 0.0
        k = int(max(0, min(len(s) - 1, round(p * (len(s) - 1)))))
        return float(s[k])
    return {"avg": float(mean(s)), "p50": pct(0.5), "p90": pct(0.9)}


def _fragmentation_for_terms(tok, terms: List[str]) -> Dict[str, int]:
    frag: Dict[str, int] = {}
    for w in terms:
        try:
            n = len(tok(w, add_special_tokens=False).input_ids)
            frag[w] = int(n)
        except Exception:
            continue
    return frag


def _naive_entities(t: str) -> List[str]:
    return re.findall(r"[A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)*", t)


def _atomic_entity_coverage(tok, texts: List[str], cap_per_doc: int = 16, doc_cap: int = 256) -> float:
    count = 0
    atomic = 0
    for i, t in enumerate(texts[:doc_cap]):
        ents = _naive_entities(t)[:cap_per_doc]
        for e in ents:
            ids = tok(e, add_special_tokens=False).input_ids
            count += 1
            if len(ids) == 1:
                atomic += 1
    return float(atomic / count) if count else 0.0


def analyze(
    base_tokenizer_path: str,
    adapted_tokenizer_path: Optional[str],
    bench_dir: str,
    proc_dir: str,
    biomed_dir: Optional[str] = None,
    added_terms_path: Optional[str] = None,
    sample_cap: int = 2000,
) -> Dict[str, object]:
    base_tok = AutoTokenizer.from_pretrained(base_tokenizer_path, use_fast=True)
    if adapted_tokenizer_path:
        adapted_tok = AutoTokenizer.from_pretrained(adapted_tokenizer_path, use_fast=True)
    else:
        adapted_tok = base_tok

    texts = _iter_sample_texts(bench_dir, proc_dir, biomed_dir=biomed_dir, cap=sample_cap)
    terms = _load_terms(added_terms_path, limit=2048)

    base_tpw = _tokens_per_word(base_tok, texts)
    adapt_tpw = _tokens_per_word(adapted_tok, texts)

    base_frag = _fragmentation_for_terms(base_tok, terms)
    adapt_frag = _fragmentation_for_terms(adapted_tok, terms)

    frag_delta: Dict[str, int] = {}
    for w in terms:
        b = base_frag.get(w)
        a = adapt_frag.get(w)
        if b is not None and a is not None:
            frag_delta[w] = int(b - a)

    base_entity_atomic = _atomic_entity_coverage(base_tok, texts)
    adapt_entity_atomic = _atomic_entity_coverage(adapted_tok, texts)

    return {
        "tokens_per_word_base": base_tpw,
        "tokens_per_word_adapted": adapt_tpw,
        "fragmentation_base_terms_count": int(len(base_frag)),
        "fragmentation_adapted_terms_count": int(len(adapt_frag)),
        "fragmentation_reduction_terms": frag_delta,
        "entity_atomic_coverage_base": float(base_entity_atomic),
        "entity_atomic_coverage_adapted": float(adapt_entity_atomic),
    }


def main():
    ap = argparse.ArgumentParser(description="Tokenizer analysis (base vs adapted)")
    ap.add_argument("--base_tokenizer", type=str, required=True)
    ap.add_argument("--adapted_tokenizer", type=str, default=None)
    ap.add_argument("--bench_dir", type=str, required=True)
    ap.add_argument("--proc_dir", type=str, required=True)
    ap.add_argument("--biomed_dir", type=str, default=None)
    ap.add_argument("--added_terms", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res = analyze(
        base_tokenizer_path=args.base_tokenizer,
        adapted_tokenizer_path=args.adapted_tokenizer,
        bench_dir=args.bench_dir,
        proc_dir=args.proc_dir,
        biomed_dir=args.biomed_dir,
        added_terms_path=args.added_terms,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


