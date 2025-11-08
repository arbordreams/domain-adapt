from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

from . import data_prep as dp
from . import biomed_corpus as bc
try:
    from .dedup_store import SeenStore  # type: ignore
except Exception:
    SeenStore = None  # type: ignore
from . import dataset_preflight as preflight
from . import api as api


def _pkg_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ensure HF accelerated transfer unless explicitly disabled by env
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


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
    # Robust, file-backed corpus builder with optional auto-fill to target bytes
    cfg_path = os.path.abspath(args.config)
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"Config not found: {cfg_path}")

    import yaml, json, shutil, threading, time as _t, random

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("random_seed", 17))
    rng = random.Random(seed)
    out_root = cfg.get("output_dir", os.path.join(_pkg_root(), "data", "biomed_corpus"))
    defaults = cfg.get("defaults", {})
    min_chars = int(defaults.get("min_chars", 200))
    max_chars = int(defaults.get("max_chars", 20000))
    prevent_contam = bool(cfg.get("prevent_eval_contamination", False))
    near_dup_mode = str(cfg.get("near_dup", "none")).lower()
    target_total = int(cfg.get("target_total_bytes", 0))

    # Fresh start if requested
    if getattr(args, "fresh", False) and os.path.isdir(out_root):
        shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(out_root, exist_ok=True)

    # Logging (file-backed, no tmux/tee required)
    logdir = getattr(args, "logdir", None) or os.path.join(_pkg_root(), "runs", "logs")
    os.makedirs(logdir, exist_ok=True)
    ts = _t.strftime("%Y%m%d_%H%M%S")
    text_log = os.path.join(logdir, f"corpus_{ts}.log")
    jsonl_log = os.path.join(logdir, f"corpus_{ts}.jsonl")
    _fp = open(text_log, "a", encoding="utf-8", buffering=1)
    def _log(line: str) -> None:
        sys.stdout.write(line + "\n"); sys.stdout.flush(); _fp.write(line + "\n")
    def _event(ev: dict) -> None:
        try:
            with open(jsonl_log, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(ev) + "\n")
        except Exception:
            pass

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
                                rec = json.loads(line)
                                t = bc._norm_text(str(rec.get("text", "") or rec.get("context", "") or rec.get("question", "")))
                                if t:
                                    global_seen.add(bc._hash_3gram(t))
                            except Exception:
                                continue
        except Exception:
            pass

    # Near-dup option
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

    # Build list of sources
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

    # Per-source min/max overrides (fallback to defaults)
    per_source_minmax: dict[str, tuple[int, int]] = {}
    for name, sc in (cfg.get("sources") or {}).items():
        if not sc.get("enabled", True):
            continue
        s_min = int(sc.get("min_chars", min_chars))
        s_max = int(sc.get("max_chars", max_chars))
        per_source_minmax[name] = (s_min, s_max)

    # Preflight (fail fast if any source is missing/inaccessible)
    strict_sources = bool(getattr(args, "strict_sources", True))
    # Allow env override for quick/demo modes
    if os.environ.get("STRICT_SOURCES", "").lower() in ("0", "false", "no"):
        strict_sources = False
    preflight_specs: list[tuple[str, preflight.SourceSpec]] = []
    for s in sources_cfg:
        preflight_specs.append(
            (
                s.name,
                preflight.SourceSpec(
                    dataset=s.dataset or "",
                    subset=s.subset,
                    splits=s.splits or ["train"],
                    text_fields=s.text_fields or ["text"],
                ),
            )
        )
    ok, reports = preflight.preflight_sources(preflight_specs, strict=strict_sources)
    rep_txt = preflight.format_reports(reports)
    for line in rep_txt.splitlines():
        _log(f"[preflight] {line}")
    _event({"preflight": True, "ok": ok, "strict": strict_sources, "ts": _t.strftime("%Y-%m-%dT%H:%M:%SZ")})
    if getattr(args, "preflight_only", False):
        _log("[preflight] preflight-only requested; exiting after checks")
        try:
            stop_evt.set()  # may not exist yet
        except Exception:
            pass
        try:
            _fp.close()
        except Exception:
            pass
        return
    if strict_sources and not ok:
        try:
            stop_evt.set()
        except Exception:
            pass
        try:
            _fp.close()
        except Exception:
            pass
        raise SystemExit("Preflight failed for one or more sources (strict-sources). Aborting.")

    # Monitor thread to log progress
    stop_evt = threading.Event()
    def _monitor() -> None:
        last = ""
        while not stop_evt.is_set():
            try:
                total = 0
                for root, _d, files in os.walk(out_root):
                    for fn in files:
                        try:
                            total += os.path.getsize(os.path.join(root, fn))
                        except Exception:
                            pass
                pct = (total / target_total * 100.0) if target_total > 0 else 0.0
                line = f"[corpus] {total/1e9:.2f} GB" + (f" / {target_total/1e9:.2f} GB ({pct:.1f}%)" if target_total > 0 else "")
                if line != last:
                    _log(line)
                    _event({"progress": "corpus", "bytes": total, "target": target_total, "ts": _t.strftime("%Y-%m-%dT%H:%M:%SZ")})
                    last = line
            except Exception:
                pass
            stop_evt.wait(15.0)

    mon = threading.Thread(target=_monitor, daemon=True)
    mon.start()

    summary = {}
    global_written = 0
    # Prepare dedup store (default sqlite) under output_dir
    out_root = cfg.get("output_dir", os.path.join(_pkg_root(), "data", "biomed_corpus"))
    dedup_backend = getattr(args, "dedup_backend", "sqlite")
    dedup_db = getattr(args, "dedup_db", None) or os.path.join(out_root, "seen.sqlite")
    store = None
    if dedup_backend == "sqlite" and SeenStore is not None:
        try:
            store = SeenStore(dedup_db)
            # Optional backfill: populate store from existing JSONLs once
            if getattr(args, "backfill_store", False):
                for s in sources_cfg:
                    pth = os.path.join(out_root, f"{s.name}.jsonl")
                    if os.path.isfile(pth):
                        store.backfill_from_jsonl(pth, bc._hash_3gram)
        except Exception:
            store = None
    for s in sources_cfg:
        if s.target_bytes <= 0:
            continue
        out_path = os.path.join(out_root, f"{s.name}.jsonl")
        existing_bytes = os.path.getsize(out_path) if os.path.isfile(out_path) else 0
        stats = bc.build_source(
            out_path,
            s,
            min_chars=per_source_minmax.get(s.name, (min_chars, max_chars))[0],
            max_chars=per_source_minmax.get(s.name, (min_chars, max_chars))[1],
            rng=rng,
            seen_hashes=global_seen if dedup_backend != "sqlite" else None,
            near_dup_lsh=near_dup_lsh,
            seen_store=store,
        )
        inc = max(0, int(stats.get("bytes", 0)) - int(existing_bytes))
        global_written += inc
        summary[s.name] = stats
        _event({"source": s.name, "bytes": stats.get("bytes", 0), "kept": stats.get("kept", 0), "seen": stats.get("seen", 0), "ts": _t.strftime('%Y-%m-%dT%H%M%SZ')})
    # Deterministic completion: after iterating sources, decide whether to abort
    if target_total > 0 and strict_sources and global_written < target_total:
        remaining_capacity = 0
        for s in sources_cfg:
            p = os.path.join(out_root, f"{s.name}.jsonl")
            written = os.path.getsize(p) if os.path.isfile(p) else 0
            remaining_capacity += max(0, int(s.target_bytes) - int(written))
        if remaining_capacity > 0:
            _log(f"[warn] built {global_written} bytes < target {target_total} bytes; remaining per-source capacity {remaining_capacity}. Finishing without abort.")
        else:
            _log(f"[warn] built {global_written} bytes < target {target_total} bytes but all per-source budgets are exhausted. Finishing successfully.")

    # Write summary.json once
    try:
        with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as fsum:
            json.dump({
                "total_bytes": global_written,
                "target_total_bytes": target_total,
                "complete": bool(target_total > 0 and global_written >= target_total),
                "sources": summary
            }, fsum, ensure_ascii=False, indent=2)
    except Exception:
        pass
    print(json.dumps(summary, indent=2))
    # Cleanup resources and stop monitor thread
    try:
        if 'store' in locals() and store is not None and hasattr(store, 'close'):
            store.close()
    except Exception:
        pass
    stop_evt.set()
    _fp.close()


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
    stage1_lr = float(getattr(args, "stage1_lr", 5e-4))
    stage2_steps = int(getattr(args, "stage2_steps", 0))
    stage2_lr = float(getattr(args, "stage2_lr", 5e-5))
    embedding_backend = str(getattr(args, "embedding_backend", "fasttext")).strip().lower()

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

    # 3) Train embedding vectors (backend-selectable)
    if embedding_backend == "glove":
        vec_src = api.train_glove_vectors(corpus_path=corpus_src, save_name="vec-source")
        vec_tgt = api.train_glove_vectors(corpus_path=corpus_tgt, save_name="vec-target")
    elif embedding_backend == "fasttext":
        vec_src = api.train_fasttext_vectors(corpus_path=corpus_src, save_name="vec-source")
        vec_tgt = api.train_fasttext_vectors(corpus_path=corpus_tgt, save_name="vec-target")
    else:
        raise SystemExit(f"Unsupported embedding_backend: {embedding_backend} (choices: glove, fasttext)")

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
        avoid_unk_second_best=not bool(getattr(args, "no_unk_second_best", False)),
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
        stage1_lr=stage1_lr,
        stage2_steps=stage2_steps,
        stage2_lr=stage2_lr,
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
    p2.add_argument("--fresh", action="store_true", help="Delete existing corpus dir before building")
    p2.add_argument("--strict_sources", action="store_true", default=True, help="Fail fast if any source is missing/inaccessible")
    p2.add_argument("--preflight_only", action="store_true", help="Run dataset preflight only and exit")
    p2.add_argument("--dedup_backend", type=str, default="sqlite", choices=["sqlite", "memory"], help="Dedup backend (default: sqlite)")
    p2.add_argument("--dedup_db", type=str, default=None, help="Path to dedup sqlite db (default: <output_dir>/seen.sqlite)")
    p2.add_argument("--backfill_store", action="store_true", help="Backfill dedup store from existing JSONLs if missing")
    p2.add_argument("--logdir", type=str, default=os.path.join(_pkg_root(), "runs", "logs"))
    p2.set_defaults(func=cmd_build_corpus)

    p3 = sp.add_parser("adapt", help="Run TokAlign-style tokenizer adaptation end-to-end")
    p3.add_argument("--model_id", type=str, required=True)
    p3.add_argument("--top_k", type=int, default=8192)
    p3.add_argument("--pivot", type=int, default=300)
    p3.add_argument("--warmup_steps", type=int, default=0)
    p3.add_argument("--stage1_lr", type=float, default=5e-4, help="LR for stage-1 (embeddings + lm_head)")
    p3.add_argument("--stage2_steps", type=int, default=0, help="Optional stage-2 full-parameter finetuning steps")
    p3.add_argument("--stage2_lr", type=float, default=5e-5, help="LR for stage-2 full-parameter finetuning")
    p3.add_argument("--embedding_backend", type=str, default="fasttext", choices=["fasttext", "glove"], help="Embedding backend used to train token vectors (default: fasttext)")
    p3.add_argument("--seed", type=int, default=17)
    p3.add_argument("--no_unk_second_best", action="store_true", help="Disable second-best fallback when argmax is UNK (TokAlign ablation)")
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
    p5.add_argument("--embedding_backend", type=str, default="fasttext", choices=["fasttext", "glove"], help="Embedding backend used to train token vectors (default: fasttext)")
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


