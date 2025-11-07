from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

from . import data_prep as dp
from . import biomed_corpus as bc
from . import api as api


def _pkg_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_data_dirs() -> tuple[str, str, str, str, str]:
    pkg_root = _pkg_root()
    data_dir = os.path.join(pkg_root, "data", "medical")
    raw_dir = os.path.join(data_dir, "raw")
    proc_dir = os.path.join(data_dir, "processed")
    bench_dir = os.path.join(data_dir, "benchmarks")
    biomed_dir = os.path.join(pkg_root, "data", "biomed_corpus")
    return data_dir, raw_dir, proc_dir, bench_dir, biomed_dir


def cmd_prepare_data(args: argparse.Namespace) -> None:
    _, _raw, proc, bench, _ = _default_data_dirs()
    os.makedirs(proc, exist_ok=True)
    os.makedirs(bench, exist_ok=True)

    if getattr(args, "all", False):
        # Extended preparation: all medical datasets + perplexity corpus
        dp.prepare_medical_benchmarks_all(bench_dir=bench, proc_dir=proc)
    else:
        # Minimal preparation: PubMedQA + MedQA
        dp.prepare_medical_benchmarks(bench_dir=bench, proc_dir=proc)


def cmd_build_corpus(args: argparse.Namespace) -> None:
    # Build balanced biomedical corpus according to YAML config
    cfg_path = os.path.abspath(args.config)
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"Config not found: {cfg_path}")
    # Call library routine directly to avoid shelling out
    # Replicates biomed_corpus.main() without arg parsing
    import yaml

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("random_seed", 17))
    import random

    rng = random.Random(seed)
    out_root = cfg.get("output_dir", os.path.join(_pkg_root(), "data", "biomed_corpus"))
    defaults = cfg.get("defaults", {})
    min_chars = int(defaults.get("min_chars", 200))
    max_chars = int(defaults.get("max_chars", 20000))
    prevent_contam = bool(cfg.get("prevent_eval_contamination", False))
    near_dup_mode = str(cfg.get("near_dup", "none")).lower()

    sources_cfg: List[bc.SourceCfg] = []
    for name, sc in (cfg.get("sources") or {}).items():
        if not sc.get("enabled", True):
            continue
        sources_cfg.append(
            bc.SourceCfg(
                name=name,
                kind=str(sc.get("type", "hf")).strip(),
                enabled=True,
                target_bytes=int(sc.get("target_bytes", 0)),
                dataset=sc.get("dataset"),
                subset=sc.get("name"),
                splits=list(sc.get("splits", ["train"])),
                text_fields=list(sc.get("text_fields", ["text"])),
                urls=list(sc.get("urls", []) or []),
            )
        )

    os.makedirs(out_root, exist_ok=True)
    summary = {}
    # Optional contamination blocklist (best-effort)
    global_seen: set[str] = set()
    if prevent_contam:
        try:
            bench_dir = os.path.join(os.path.dirname(out_root), "..", "medical", "benchmarks")
            if os.path.isdir(bench_dir):
                for name in sorted(os.listdir(bench_dir)):
                    if not name.endswith(".jsonl"):
                        continue
                    pth = os.path.join(bench_dir, name)
                    with open(pth, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                rec = __import__("json").loads(line)
                                t = bc._norm_text(str(rec.get("text", "") or rec.get("context", "") or rec.get("question", "")))
                                if t:
                                    global_seen.add(bc._hash_3gram(t))
                            except Exception:
                                continue
        except Exception:
            pass

    near_dup_lsh = None
    if near_dup_mode == "minhash":
        try:
            from datasketch import MinHash, MinHashLSH
            def _mh(text: str) -> object:
                m = MinHash(num_perm=64)
                for w in set((text or "").lower().split()):
                    m.update(w.encode("utf-8"))
                return m
            near_dup_lsh = (_mh, MinHashLSH(threshold=0.9, num_perm=64))
        except Exception:
            near_dup_lsh = None

    for s in sources_cfg:
        if s.target_bytes <= 0:
            continue
        out_path = os.path.join(out_root, f"{s.name}.jsonl")
        stats = bc.build_source(
            out_path,
            s,
            min_chars=min_chars,
            max_chars=max_chars,
            rng=rng,
            seen_hashes=global_seen,
            near_dup_lsh=near_dup_lsh,
        )
        summary[s.name] = stats
    print(__import__("json").dumps(summary, indent=2))


def cmd_adapt(args: argparse.Namespace) -> None:
    data_dir, _raw, proc, bench, biomed_dir = _default_data_dirs()
    os.makedirs(bench, exist_ok=True)
    os.makedirs(proc, exist_ok=True)

    # Timestamped run directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(_pkg_root(), "runs", "tokenizer_adapt", ts)
    tok_dir = os.path.join(run_dir, "tokenizer")
    model_dir = os.path.join(run_dir, "model")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)

    model_id = args.model_id
    top_k = int(args.top_k)
    pivot = int(args.pivot)
    warmup_steps = int(getattr(args, "warmup_steps", 0) or 0)

    # 1) Term selection and tokenizer save
    api.select_and_save_tokenizer(
        model_id=model_id,
        bench_dir=bench,
        proc_dir=proc,
        top_k=top_k,
        out_dir=run_dir,
        tok_out_dir=tok_dir,
    )

    # 2) Build GloVe corpora (source=base tokenizer; target=adapted tokenizer)
    corpus_src = os.path.join(run_dir, "glove_source.txt")
    corpus_tgt = os.path.join(run_dir, "glove_target.txt")
    corpus_dirs: Optional[List[str]] = [biomed_dir] if os.path.isdir(biomed_dir) else None
    api.build_glove_corpora(
        model_id=model_id,
        tok_dir=tok_dir,
        bench_dir=bench,
        proc_dir=proc,
        corpus_src_path=corpus_src,
        corpus_tgt_path=corpus_tgt,
        corpus_dirs=corpus_dirs,
    )

    # 3) Train GloVe vectors
    vec_src = api.train_glove_vectors(corpus_path=corpus_src, save_name="vec-source")
    vec_tgt = api.train_glove_vectors(corpus_path=corpus_tgt, save_name="vec-target")

    # 4) Gold overlap and alignment mapping
    gold_json = os.path.join(run_dir, "gold.json")
    api._compute_gold_overlap_json(src_tok_path=model_id, tgt_tok_path=tok_dir, out_path=gold_json)
    mapping_json = os.path.join(run_dir, "align_matrix.json")
    vocab_src_size = len(__import__("transformers").AutoTokenizer.from_pretrained(model_id))
    vocab_tgt_size = len(__import__("transformers").AutoTokenizer.from_pretrained(tok_dir))
    api.compute_alignment(
        vec_src_path=vec_src,
        vec_tgt_path=vec_tgt,
        vocab_src_size=int(vocab_src_size),
        vocab_tgt_size=int(vocab_tgt_size),
        gold_json_path=gold_json,
        pivot=pivot,
        out_path=mapping_json,
        seed=int(getattr(args, "seed", 17)),
    )

    # 5) Strict apply + optional warmup
    api.apply_mapping_strict(
        mapping_json_path=mapping_json,
        src_model_id=model_id,
        tgt_tok_dir=tok_dir,
        out_model_dir=model_dir,
    )
    api.warmup_new_rows(
        out_model_dir=model_dir,
        base_model_id=model_id,
        tok_dir=tok_dir,
        bench_dir=bench,
        proc_dir=proc,
        steps=warmup_steps,
    )

    print(run_dir)


def cmd_eval(args: argparse.Namespace) -> None:
    # Delegate to module main (to reuse robust backend logic)
    from . import medical_eval as eval_mod
    sys.argv = [sys.argv[0], "--config", os.path.abspath(args.config)]
    eval_mod.main()


def cmd_pipeline(args: argparse.Namespace) -> None:
    # 1) Prepare data (all datasets)
    class A:  # simple namespace with attribute 'all'
        all = True
    cmd_prepare_data(A)

    # 2) Build corpus
    class B:
        config = os.path.abspath(args.corpus_config)
    cmd_build_corpus(B)

    # Optional sweep over adaptation hyperparameters
    sweep_top_k = str(getattr(args, "sweep_top_k", "")).strip()
    sweep_pivot = str(getattr(args, "sweep_pivot", "")).strip()
    sweep_warm = str(getattr(args, "sweep_warmup_steps", "")).strip()
    if sweep_top_k or sweep_pivot or sweep_warm:
        def _parse_list(s: str, fallback: int) -> list[int]:
            if not s:
                return [int(fallback)]
            try:
                return [int(x) for x in s.split(",") if str(x).strip()]
            except Exception:
                return [int(fallback)]
        topk_list = _parse_list(sweep_top_k, args.top_k)
        pivot_list = _parse_list(sweep_pivot, args.pivot)
        warm_list = _parse_list(sweep_warm, args.warmup_steps)
        grid_rows: list[str] = ["top_k,pivot,warmup_steps,eval_dir"]
        for tk in topk_list:
            for pv in pivot_list:
                for wm in warm_list:
                    class C:
                        model_id = args.model_id
                        top_k = tk
                        pivot = pv
                        warmup_steps = wm
                        seed = getattr(args, "seed", 17)
                    cmd_adapt(C)
                    class D:
                        config = os.path.abspath(args.eval_config)
                    cmd_eval(D)
                    grid_rows.append(f"{tk},{pv},{wm},{os.path.abspath(os.path.dirname(__file__))}")
        # Save a minimal sweep index next to eval outputs
        try:
            import time as _t
            ts = _t.strftime("%Y%m%d_%H%M%S")
            root = os.path.join(_pkg_root(), "runs", "sweeps")
            os.makedirs(root, exist_ok=True)
            with open(os.path.join(root, f"grid_{ts}.csv"), "w", encoding="utf-8") as f:
                f.write("\n".join(grid_rows))
        except Exception:
            pass
        return

    # 3) Adapt
    class C:
        model_id = args.model_id
        top_k = args.top_k
        pivot = args.pivot
        warmup_steps = args.warmup_steps
        seed = getattr(args, "seed", 17)
    cmd_adapt(C)

    # 4) Eval
    class D:
        config = os.path.abspath(args.eval_config)
    cmd_eval(D)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MedTokAlign unified CLI (RunPod-ready)")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p1 = sp.add_parser("prepare-data", help="Download and normalize medical datasets to JSONL")
    p1.add_argument("--all", action="store_true", help="Prepare all supported datasets (default minimal: PubMedQA+MedQA)")
    p1.set_defaults(func=cmd_prepare_data)

    p2 = sp.add_parser("build-corpus", help="Build balanced biomedical corpus from YAML config")
    p2.add_argument("--config", type=str, required=True)
    p2.set_defaults(func=cmd_build_corpus)

    p3 = sp.add_parser("adapt", help="Run TokAlign-style tokenizer adaptation end-to-end")
    p3.add_argument("--model_id", type=str, required=True)
    p3.add_argument("--top_k", type=int, default=8192)
    p3.add_argument("--pivot", type=int, default=300)
    p3.add_argument("--warmup_steps", type=int, default=0)
    p3.add_argument("--seed", type=int, default=17)
    p3.set_defaults(func=cmd_adapt)

    p4 = sp.add_parser("eval", help="Run medical evaluation")
    p4.add_argument("--config", type=str, required=True)
    p4.set_defaults(func=cmd_eval)

    p5 = sp.add_parser("pipeline", help="End-to-end: prepare-data + build-corpus + adapt + eval")
    p5.add_argument("--model_id", type=str, required=True)
    p5.add_argument("--corpus_config", type=str, required=True)
    p5.add_argument("--eval_config", type=str, required=True)
    p5.add_argument("--top_k", type=int, default=8192)
    p5.add_argument("--pivot", type=int, default=300)
    p5.add_argument("--warmup_steps", type=int, default=0)
    p5.add_argument("--seed", type=int, default=17)
    # optional sweeps (comma-separated integers)
    p5.add_argument("--sweep_top_k", type=str, default="")
    p5.add_argument("--sweep_pivot", type=str, default="")
    p5.add_argument("--sweep_warmup_steps", type=str, default="")
    p5.set_defaults(func=cmd_pipeline)

    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


