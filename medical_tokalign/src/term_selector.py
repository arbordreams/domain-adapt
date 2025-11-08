import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

from transformers import AutoTokenizer


def load_jsonl(path: str) -> Iterable[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except FileNotFoundError:
        return []


def iter_med_texts(bench_dir: str, proc_dir: str) -> Iterable[str]:
    # 1) Benchmarks (PubMedQA, MedQA)
    files = [
        "pubmedqa_train.jsonl","pubmedqa_validation.jsonl","pubmedqa_test.jsonl",
        "medqa_train.jsonl","medqa_validation.jsonl","medqa_test.jsonl",
    ]
    for name in files:
        for ex in load_jsonl(os.path.join(bench_dir, name)):
            for k in ("text","document","passages","context","long_answer","question"):
                v = ex.get(k)
                if isinstance(v, list):
                    v = "\n".join(map(str, v))
                if isinstance(v, str) and v:
                    yield v

    # 2) Biomed corpus texts if present: medical_tokalign/data/biomed_corpus/*.jsonl
    biomed_dir = os.path.join(os.path.dirname(bench_dir), "biomed_corpus")
    try:
        for name in sorted(os.listdir(biomed_dir)):
            if not name.endswith(".jsonl"):
                continue
            for ex in load_jsonl(os.path.join(biomed_dir, name)) or []:
                t = ex.get("text")
                if isinstance(t, list):
                    t = "\n".join(map(str, t))
                if isinstance(t, str) and t:
                    yield t
    except Exception:
        # Optional corpus; ignore if unavailable
        pass


def select_terms(
    model_id: str,
    bench_dir: str,
    proc_dir: str,
    top_k: int,
    sample_per_term: int = 0,
    max_term_len: int = 64,
) -> Tuple[List[str], Dict[str, float]]:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    base_vocab = set(tok.get_vocab().keys())

    # 1) Stream texts once: collect term frequencies and document frequencies; estimate boundary consistency on the fly
    token_re = re.compile(r"[A-Za-z][A-Za-z0-9_\-/]{3,}")
    freq: Counter[str] = Counter()
    docfreq: Counter[str] = Counter()
    isolated_len: Dict[str, int] = {}
    samples_for_term: Dict[str, int] = defaultdict(int)
    equal_len_hits: Dict[str, int] = defaultdict(int)
    N_docs = 0
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    base_vocab = set(tok.get_vocab().keys())
    for t in iter_med_texts(bench_dir, proc_dir):
        N_docs += 1
        seen_in_doc: set[str] = set()
        for w in token_re.findall(t):
            if len(w) > max_term_len:
                continue
            if w in base_vocab:
                continue
            freq[w] += 1
            if w not in seen_in_doc:
                docfreq[w] += 1
                seen_in_doc.add(w)
            # compute isolated length once
            if w not in isolated_len:
                isolated_len[w] = len(tok(w, add_special_tokens=False).input_ids)
            # Optional boundary consistency sampling (disabled by default for speed)
            if sample_per_term and isolated_len.get(w, 0) > 1 and samples_for_term[w] < sample_per_term:
                span_tokens = len(tok(w, add_special_tokens=False).input_ids)
                samples_for_term[w] += 1
                if span_tokens == isolated_len.get(w, 0):
                    equal_len_hits[w] += 1

    if not freq:
        return [], {}

    # 2) Scoring: fragmentation gain × TF-IDF × boundary consistency
    scores: Dict[str, float] = {}
    def _idf(w: str) -> float:
        try:
            df = max(1, int(docfreq.get(w, 1)))
            return math.log(max(1, N_docs) / df)
        except Exception:
            return 0.0
    for w, tf in freq.most_common(top_k * 3):
        if isolated_len.get(w, 0) <= 1:
            continue
        # Use bc=1.0 when boundary sampling is disabled
        if sample_per_term:
            bc_denom = max(1, samples_for_term.get(w, 0))
            bc = (equal_len_hits.get(w, 0) / bc_denom) if bc_denom else 1.0
        else:
            bc = 1.0
        tfidf = float(tf) * _idf(w)
        score = max(0, isolated_len.get(w, 0) - 1) * tfidf * bc
        scores[w] = float(score)

    # sort by score
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    terms = [w for w, _ in ranked[:top_k]]
    return terms, scores


def main():
    ap = argparse.ArgumentParser(description="MedTokAlign term selector (fragmentation-based)")
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--bench_dir", type=str, required=True)
    ap.add_argument("--proc_dir", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=8192)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    terms, scores = select_terms(
        model_id=args.model_id,
        bench_dir=args.bench_dir,
        proc_dir=args.proc_dir,
        top_k=int(args.top_k),
    )

    added_path = os.path.join(args.out_dir, "added_terms.txt")
    with open(added_path, "w", encoding="utf-8") as f:
        for w in terms:
            f.write(w + "\n")

    report = {
        "num_selected": len(terms),
        "top_k": int(args.top_k),
        "notes": "score = (len(tokens)-1) * frequency * boundary_consistency",
    }
    with open(os.path.join(args.out_dir, "selector_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[MedTokAlign] Wrote {len(terms)} terms to {added_path}")


if __name__ == "__main__":
    main()


