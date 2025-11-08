import json
import os
import random
from typing import Dict, List, Tuple, Optional, Any

from transformers import AutoTokenizer

from .data_prep import prepare_medical_benchmarks
from .term_selector import select_terms
from .tokalign_vendor.cal_trans_matrix import (
    load_glove_model,
    convert2matrix,
    convert2relative_rep,
)
from .convert_strict import trans2switch_strict
import subprocess
import shutil


def prepare_data(bench_dir: str, proc_dir: str) -> None:
    prepare_medical_benchmarks(bench_dir=bench_dir, proc_dir=proc_dir)


def select_and_save_tokenizer(
    model_id: str,
    bench_dir: str,
    proc_dir: str,
    top_k: int,
    out_dir: str,
    tok_out_dir: str,
) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tok_out_dir, exist_ok=True)

    terms, _scores = select_terms(
        model_id=model_id,
        bench_dir=bench_dir,
        proc_dir=proc_dir,
        top_k=int(top_k),
    )
    added_path = os.path.join(out_dir, "added_terms.txt")
    with open(added_path, "w", encoding="utf-8") as f:
        for w in terms:
            f.write(w + "\n")

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.add_tokens(terms, special_tokens=False)
    tok.save_pretrained(tok_out_dir)
    return added_path, tok_out_dir


def build_glove_corpora(
    model_id: str,
    tok_dir: str,
    bench_dir: str,
    proc_dir: str,
    corpus_src_path: str,
    corpus_tgt_path: str,
    max_model_len: int = 8192,
    corpus_dirs: Optional[List[str]] = None,
) -> Tuple[str, str]:
    os.makedirs(os.path.dirname(corpus_src_path), exist_ok=True)

    tok_src = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok_tgt = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)

    # Optional TokAlign-faithful size cap for GloVe corpora (combined cap across src+tgt)
    # Set via env GLOVE_CORPUS_MAX_BYTES (e.g., 1073741824 for ~1 GiB total)
    _cap_total_bytes: Optional[int] = None
    try:
        _env_cap = os.environ.get("GLOVE_CORPUS_MAX_BYTES")
        if _env_cap:
            _cap_total_bytes = int(str(_env_cap).strip())
            if _cap_total_bytes <= 0:
                _cap_total_bytes = None
    except Exception:
        _cap_total_bytes = None

    def load_jsonl(path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
        except FileNotFoundError:
            return []

    def iter_texts_primary():
        if corpus_dirs:
            for d in corpus_dirs:
                try:
                    for name in sorted(os.listdir(d)):
                        if not name.endswith(".jsonl"):
                            continue
                        for ex in load_jsonl(os.path.join(d, name)) or []:
                            v = ex.get("text")
                            if isinstance(v, list):
                                v = "\n".join(map(str, v))
                            if isinstance(v, str):
                                yield v
                except Exception:
                    continue

    def iter_texts_fallback():
        files = [
            "pubmedqa_train.jsonl","pubmedqa_validation.jsonl","pubmedqa_test.jsonl",
            "medqa_train.jsonl","medqa_validation.jsonl","medqa_test.jsonl",
        ]
        for name in files:
            for ex in load_jsonl(os.path.join(bench_dir, name)) or []:
                for k in ("text","document","passages","context","long_answer","question"):
                    v = ex.get(k)
                    if isinstance(v, list):
                        v = "\n".join(map(str, v))
                    if isinstance(v, str):
                        yield v

    def _write_from(gen) -> Tuple[int, Dict[str, float], Dict[str, float]]:
        count = 0
        num_src = 0
        num_tgt = 0
        written_src = 0
        written_tgt = 0
        src_budget: Optional[int] = None
        tgt_budget: Optional[int] = None
        if _cap_total_bytes is not None:
            half = max(1, int(_cap_total_bytes // 2))
            src_budget = half
            tgt_budget = half
        sample_src: List[int] = []
        sample_tgt: List[int] = []
        sample_cap = 10000
        with open(corpus_src_path, "w") as fs, open(corpus_tgt_path, "w") as ft:
            for t in gen:
                ids_s = tok_src(t, add_special_tokens=False, truncation=True, max_length=max_model_len)["input_ids"]
                ids_t = tok_tgt(t, add_special_tokens=False, truncation=True, max_length=max_model_len)["input_ids"]
                if len(ids_s) >= 15:
                    if src_budget is None or written_src < src_budget:
                        _line_s = " ".join(map(str, ids_s)) + "\n"
                        fs.write(_line_s)
                        written_src += len(_line_s.encode("utf-8"))
                        num_src += 1
                        if len(sample_src) < sample_cap:
                            sample_src.append(len(ids_s))
                if len(ids_t) >= 15:
                    if tgt_budget is None or written_tgt < tgt_budget:
                        _line_t = " ".join(map(str, ids_t)) + "\n"
                        ft.write(_line_t)
                        written_tgt += len(_line_t.encode("utf-8"))
                        num_tgt += 1
                        if len(sample_tgt) < sample_cap:
                            sample_tgt.append(len(ids_t))
                if len(ids_s) >= 15 or len(ids_t) >= 15:
                    count += 1
                # Stop early once both budgets are exhausted
                if src_budget is not None and tgt_budget is not None:
                    if written_src >= src_budget and written_tgt >= tgt_budget:
                        break

        def _stats(n: int, sample: List[int]) -> Dict[str, float]:
            if not sample:
                return {"lines": float(n), "avg_len": 0.0, "p50": 0.0, "p90": 0.0}
            srt = sorted(sample)
            avg = float(sum(srt) / len(srt))
            def pct(p: float) -> float:
                if not srt:
                    return 0.0
                k = int(max(0, min(len(srt) - 1, round(p * (len(srt) - 1)))))
                return float(srt[k])
            return {"lines": float(n), "avg_len": avg, "p50": pct(0.5), "p90": pct(0.9)}

        return count, _stats(num_src, sample_src), _stats(num_tgt, sample_tgt)

    wrote, stats_src, stats_tgt = _write_from(iter_texts_primary() or [])
    if wrote == 0:
        wrote, stats_src, stats_tgt = _write_from(iter_texts_fallback())

    # Save stats sidecars next to corpora
    try:
        with open(os.path.splitext(corpus_src_path)[0] + ".stats.json", "w", encoding="utf-8") as fsrc:
            json.dump(stats_src, fsrc, indent=2)
        with open(os.path.splitext(corpus_tgt_path)[0] + ".stats.json", "w", encoding="utf-8") as ftgt:
            json.dump(stats_tgt, ftgt, indent=2)
    except Exception:
        pass
    return corpus_src_path, corpus_tgt_path


def _compute_gold_overlap_json(src_tok_path: str, tgt_tok_path: str, out_path: str) -> str:
    # Accept both remote model IDs and local paths
    src_tok = AutoTokenizer.from_pretrained(src_tok_path)
    tgt_tok = AutoTokenizer.from_pretrained(tgt_tok_path)
    src_vocab = src_tok.get_vocab()
    tgt_vocab = tgt_tok.get_vocab()

    # Standardize: gold keys are strings of target ids; values are source ids (ints)
    vocab_overlap: Dict[str, int] = {}
    tgt_vocab_sorted = dict(sorted(tgt_vocab.items(), key=lambda kv: kv[1]))
    for token, tgt_id in tgt_vocab_sorted.items():
        if token in src_vocab:
            vocab_overlap[str(int(tgt_id))] = int(src_vocab[token])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab_overlap, f, indent="\t", ensure_ascii=False)
    return out_path


def _normalize_gold(gold_raw: Any) -> Dict[str, int]:
    """Normalize a gold overlap dict to {str(target_id): int(source_id)}.

    - Keys become strings without leading/trailing spaces
    - Values coerced to int when possible; pairs with non-int values are dropped
    """
    norm: Dict[str, int] = {}
    if isinstance(gold_raw, dict):
        for k, v in gold_raw.items():
            try:
                ks = str(int(k)) if isinstance(k, (int, float)) else str(k).strip()
                vi = int(v)  # raises if not convertible
                norm[ks] = vi
            except Exception:
                continue
    return norm


def compute_alignment(
    vec_src_path: str,
    vec_tgt_path: str,
    vocab_src_size: int,
    vocab_tgt_size: int,
    gold_json_path: str,
    pivot: int,
    out_path: str,
    seed: int = 0,
    avoid_unk_second_best: bool = True,
) -> str:
    """Compute TokAlign-style 1:1 id mapping using relative GloVe representations.

    Notes on randomness:
    - convert2relative_rep(seed=seed) seeds the global RNG inside the vendor helper when
      shuffling pivot candidates.
    - This function also uses a local RNG for fallback assignments to ensure reproducibility.
    """
    with open(gold_json_path, "r", encoding="utf-8") as f:
        gold_loaded = json.load(f)
    gold = _normalize_gold(gold_loaded)

    embed1 = load_glove_model(vec_tgt_path)
    embed2 = load_glove_model(vec_src_path)

    ids1, rep1, ids2, rep2 = convert2relative_rep(
        embed1=embed1,
        embed2=embed2,
        gold=gold,
        num_pivot=int(pivot),
        seed=int(seed),
    )
    import numpy as np
    sim = np.matmul(rep1, rep2.T)

    # Fast lookup structures
    ids1_index: Dict[str, int] = {tid: i for i, tid in enumerate(ids1)}
    supl = set(gold.keys())
    rng = random.Random(int(seed))

    td: Dict[str, int] = {}
    report = {
        "pivot_requested": int(pivot),
        "gold_pairs": int(len(gold)),
        "vocab_src_size": int(vocab_src_size),
        "vocab_tgt_size": int(vocab_tgt_size),
        "random_supply_count": 0,
        "unk_second_best_count": 0,
        "oov_tgt_missing_count": 0,
    }
    for tid_int in range(int(vocab_tgt_size)):
        tid = str(tid_int)
        if tid in supl:
            td[tid] = int(gold[tid])
            continue
        id1_idx = ids1_index.get(tid)
        if id1_idx is None:
            # TokAlign-faithful fallback: random supply
            td[tid] = int(rng.randint(0, int(vocab_src_size) - 1))
            report["oov_tgt_missing_count"] += 1
            report["random_supply_count"] += 1
            continue
        lix = int(np.argmax(sim[id1_idx]))
        lid = ids2[lix]
        if avoid_unk_second_best and str(lid) in ("unk", "<unk>"):
            top2 = set(np.argpartition(sim[id1_idx], -2)[-2:])
            top1 = set(np.argpartition(sim[id1_idx], -1)[-1:])
            cand = list(top2 - top1)
            if cand:
                lid = ids2[int(cand[0])]
            report["unk_second_best_count"] += 1
        try:
            td[tid] = int(lid)
        except Exception:
            td[tid] = int(rng.randint(0, int(vocab_src_size) - 1))
            report["random_supply_count"] += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(td, f)
    # Save compact alignment report
    try:
        with open(out_path + ".report.json", "w", encoding="utf-8") as rf:
            json.dump(report, rf, indent=2)
    except Exception:
        pass
    return out_path


def apply_mapping_strict(
    mapping_json_path: str,
    src_model_id: str,
    tgt_tok_dir: str,
    out_model_dir: str,
    random_shuffle: float = -1.0,
    trust_remote_code: bool = False,
) -> str:
    os.makedirs(out_model_dir, exist_ok=True)
    trans2switch_strict(
        trans_path=mapping_json_path,
        src_clm_path=src_model_id,
        tgt_clm_path=out_model_dir,
        tgt_tok_path=tgt_tok_dir,
        random_shuffle=float(random_shuffle),
        trust_remote_code=bool(trust_remote_code),
    )
    return out_model_dir


def warmup_new_rows(
    out_model_dir: str,
    base_model_id: str,
    tok_dir: str,
    bench_dir: str,
    proc_dir: str,
    steps: int = 0,
) -> None:
    if steps and steps > 0:
        from .new_rows_warmup import warmup_rows
        warmup_rows(
            adapted_model_path=out_model_dir,
            base_model_id=base_model_id,
            tokenizer_dir=tok_dir,
            bench_dir=bench_dir,
            proc_dir=proc_dir,
            steps=int(steps),
        )


def run_vocab_adaptation(
    model_id: str,
    top_k: int,
    pivot: int,
    vec_src_path: str,
    vec_tgt_path: str,
    bench_dir: str,
    proc_dir: str,
    run_dir: str,
    warmup_steps: int = 0,
) -> Dict[str, str]:
    os.makedirs(run_dir, exist_ok=True)
    tok_dir = os.path.join(run_dir, "tokenizer")
    model_dir = os.path.join(run_dir, "model")
    gold_json = os.path.join(run_dir, "gold.json")
    mapping_json = os.path.join(run_dir, "align_matrix.json")

    # 1) Select terms and save tokenizer
    _, tok_dir = select_and_save_tokenizer(
        model_id=model_id,
        bench_dir=bench_dir,
        proc_dir=proc_dir,
        top_k=int(top_k),
        out_dir=run_dir,
        tok_out_dir=tok_dir,
    )

    # 2) Gold overlap dict
    _compute_gold_overlap_json(src_tok_path=model_id, tgt_tok_path=tok_dir, out_path=gold_json)

    # 3) Compute alignment mapping
    vocab_src_size = len(AutoTokenizer.from_pretrained(model_id))
    vocab_tgt_size = len(AutoTokenizer.from_pretrained(tok_dir))
    compute_alignment(
        vec_src_path=vec_src_path,
        vec_tgt_path=vec_tgt_path,
        vocab_src_size=int(vocab_src_size),
        vocab_tgt_size=int(vocab_tgt_size),
        gold_json_path=gold_json,
        pivot=int(pivot),
        out_path=mapping_json,
    )

    # 4) Apply mapping strictly
    apply_mapping_strict(
        mapping_json_path=mapping_json,
        src_model_id=model_id,
        tgt_tok_dir=tok_dir,
        out_model_dir=model_dir,
    )

    # 5) Optional warmup
    warmup_new_rows(
        out_model_dir=model_dir,
        base_model_id=model_id,
        tok_dir=tok_dir,
        bench_dir=bench_dir,
        proc_dir=proc_dir,
        steps=int(warmup_steps),
    )

    return {
        "run_dir": run_dir,
        "tokenizer_dir": tok_dir,
        "model_dir": model_dir,
        "gold_json": gold_json,
        "align_json": mapping_json,
    }


def train_glove_vectors(
    corpus_path: str,
    save_name: Optional[str] = None,
    glove_dir: Optional[str] = None,
    vocab_min_count: int = 5,
    vector_size: int = 300,
    max_iter: int = 15,
    window_size: int = 15,
    memory_mb: float = 1536.0,
    threads: Optional[int] = None,
    x_max: int = 10,
) -> str:
    """Train GloVe vectors from a token-id corpus (space-separated ids per line).

    Mirrors scripts/train_glove.sh using the Stanford GloVe repo.

    Returns the path to the created vector .txt file (save_file.txt).
    """
    if not os.path.isabs(corpus_path):
        # Auto-resolve to absolute using CWD
        corpus_path = os.path.abspath(corpus_path)
    if not os.path.isfile(corpus_path):
        raise FileNotFoundError(corpus_path)

    # Resolve project root -> tools/GloVe
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tools_dir = os.path.join(project_root, "tools")
    os.makedirs(tools_dir, exist_ok=True)
    glove_dir = glove_dir or os.path.join(tools_dir, "GloVe")

    # Clone GloVe if missing
    if not os.path.isdir(glove_dir):
        if shutil.which("git") is None:
            raise RuntimeError("git is required to clone Stanford GloVe repository")
        subprocess.run(["git", "clone", "https://github.com/stanfordnlp/GloVe.git", glove_dir], check=True)

    # Build GloVe binaries
    subprocess.run(["make"], cwd=glove_dir, check=True)

    build_dir = os.path.join(glove_dir, "build")
    vocab_bin = os.path.join(build_dir, "vocab_count")
    cooccur_bin = os.path.join(build_dir, "cooccur")
    shuffle_bin = os.path.join(build_dir, "shuffle")
    glove_bin = os.path.join(build_dir, "glove")
    for p in (vocab_bin, cooccur_bin, shuffle_bin, glove_bin):
        if not os.path.isfile(p):
            raise RuntimeError(f"GloVe binary missing: {p}")

    # Derive file names next to glove_dir to avoid large files in repo root
    base = save_name or os.path.splitext(os.path.basename(corpus_path))[0]
    vocab_file = os.path.join(glove_dir, f"vocab.{base}.txt")
    co_file = os.path.join(glove_dir, f"cooccurrence.{base}.bin")
    co_shuf_file = os.path.join(glove_dir, f"cooccurrence.shuf.{base}.bin")
    save_file = os.path.join(glove_dir, base)

    # Auto-adjust min_count for tiny corpora to avoid empty/small vocabularies
    try:
        file_bytes = os.path.getsize(corpus_path)
        # If corpus is very small, relax min_count aggressively
        if file_bytes < 10 * 1024 * 1024:  # < 10 MB
            vocab_min_count = min(vocab_min_count, 1)
        elif file_bytes < 50 * 1024 * 1024:  # < 50 MB
            vocab_min_count = min(vocab_min_count, 2)
    except Exception:
        pass

    # Allow overrides via environment for speed tuning
    try:
        memory_mb = float(os.environ.get("GLOVE_MEMORY_MB", memory_mb))
    except Exception:
        pass

    # Auto-size memory if not provided: use ~20% of system RAM, clamped [4096, 65536] MB
    if "GLOVE_MEMORY_MB" not in os.environ:
        try:
            with open("/proc/meminfo") as mf:
                for line in mf:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        mem_kb = int(parts[1])
                        auto_mb = max(8192, min(65536, int((mem_kb/1024) * 0.30)))
                        if auto_mb > memory_mb:
                            memory_mb = float(auto_mb)
                        break
        except Exception:
            pass

    # Determine threads: env override > explicit arg > auto from CPU (leave one core free)
    threads_val: int
    env_thr = os.environ.get("GLOVE_THREADS")
    if env_thr is not None:
        try:
            threads_val = max(1, int(env_thr))
        except Exception:
            threads_val = max(1, (os.cpu_count() or 1) - 1)
    elif threads is not None:
        threads_val = max(1, int(threads))
    else:
        threads_val = max(1, (os.cpu_count() or 1) - 1)

    # Log chosen GloVe resources
    try:
        print(f"[glove] memory_mb={memory_mb} threads={threads_val} min_count={vocab_min_count} window={window_size}", flush=True)
    except Exception:
        pass

    # 1) vocab_count
    with open(corpus_path, "rb") as fin, open(vocab_file, "wb") as fout:
        subprocess.run(
            [vocab_bin, "-min-count", str(vocab_min_count), "-verbose", "2"],
            stdin=fin,
            stdout=fout,
            check=True,
        )

    # 2) cooccur
    with open(corpus_path, "rb") as fin, open(co_file, "wb") as fout:
        subprocess.run(
            [
                cooccur_bin,
                "-memory",
                str(memory_mb),
                "-vocab-file",
                vocab_file,
                "-verbose",
                "2",
                "-window-size",
                str(window_size),
            ],
            stdin=fin,
            stdout=fout,
            check=True,
        )

    # Bound cooccurrence size; tighten params if oversized (check BEFORE shuffle to avoid crashes)
    # Shuffle binary is fragile with files > 1GB, so be very conservative
    # For fast runs, use even tighter threshold (800MB)
    try:
        _fast_run = os.environ.get("FAST_RUN", "0") == "1"
        threshold = (800 * 1024 * 1024) if _fast_run else (1536 * 1024 * 1024)  # 800 MB for fast, 1.5 GB otherwise

        def _regen_vocab_and_cooccur(_min_count: int, _window: int) -> int:
            with open(corpus_path, "rb") as fin, open(vocab_file, "wb") as fout:
                subprocess.run(
                    [vocab_bin, "-min-count", str(_min_count), "-verbose", "2"],
                    stdin=fin, stdout=fout, check=True,
                )
            with open(corpus_path, "rb") as fin, open(co_file, "wb") as fout:
                subprocess.run(
                    [
                        cooccur_bin, "-memory", str(memory_mb),
                        "-vocab-file", vocab_file, "-verbose", "2",
                        "-window-size", str(_window),
                    ],
                    stdin=fin, stdout=fout, check=True,
                )
            return os.path.getsize(co_file) if os.path.exists(co_file) else 0

        _co_size = os.path.getsize(co_file) if os.path.exists(co_file) else 0
        if _co_size > threshold:
            print(f"[glove] cooccur too large: {_co_size} bytes (>1.5GB); tightening and regenerating", flush=True)
            window_size = 5
            vocab_min_count = max(int(vocab_min_count), 20)
            _co_size = _regen_vocab_and_cooccur(vocab_min_count, window_size)
            print(f"[glove] cooccur regenerated: size={_co_size} bytes; window={window_size} min_count={vocab_min_count}", flush=True)
            if _co_size > threshold:
                print(f"[glove] cooccur still large: {_co_size} bytes; raising min_count further", flush=True)
                vocab_min_count = max(int(vocab_min_count), 30)
                _co_size = _regen_vocab_and_cooccur(vocab_min_count, window_size)
                print(f"[glove] cooccur final: size={_co_size} bytes; window={window_size} min_count={vocab_min_count}", flush=True)
    except Exception:
        pass
    # 3) shuffle with retry/backoff (use conservative memory for shuffle)
    _last_err = None
    # Allow independent override for shuffle memory
    try:
        _shuffle_mem_env = os.environ.get("GLOVE_SHUFFLE_MEMORY_MB")
        shuffle_memory_mb = float(_shuffle_mem_env) if _shuffle_mem_env else float(min(16384.0, float(memory_mb)))
    except Exception:
        shuffle_memory_mb = float(min(16384.0, float(memory_mb)))

    # Shuffle with conservative memory (params already tightened above if needed)
    # Try decreasing memory if first attempt fails
    shuffle_memory_plan: List[float] = [shuffle_memory_mb, 8192.0, 4096.0]
    for _attempt, mem_try in enumerate(shuffle_memory_plan):
        try:
            # Check cooccurrence size before each attempt
            _co_size_check = os.path.getsize(co_file) if os.path.exists(co_file) else 0
            if _co_size_check > 1536 * 1024 * 1024:
                print(f"[glove] WARNING: cooccur still {_co_size_check} bytes before shuffle attempt {_attempt}", flush=True)
            with open(co_file, 'rb') as fin, open(co_shuf_file, 'wb') as fout:
                print(f"[glove] shuffle attempt={_attempt} memory_mb={mem_try} cooccur_size={_co_size_check}", flush=True)
                subprocess.run([shuffle_bin, '-memory', str(mem_try), '-verbose', '2'], stdin=fin, stdout=fout, check=True)
            break
        except Exception as e:
            _last_err = e
            if _attempt < len(shuffle_memory_plan) - 1:
                print(f"[glove] shuffle attempt {_attempt} failed, retrying with lower memory...", flush=True)
            else:
                print(f"[glove] shuffle failed after all attempts", flush=True)
    else:
        raise _last_err or RuntimeError('GloVe shuffle failed after retries')


    # 4) glove (training)
    subprocess.run(
        [
            glove_bin,
            "-save-file",
            save_file,
            "-threads",
            str(threads_val),
            "-input-file",
            co_shuf_file,
            "-x-max",
            str(x_max),
            "-iter",
            str(max_iter),
            "-vector-size",
            str(vector_size),
            "-binary",
            str(2),
            "-vocab-file",
            vocab_file,
            "-verbose",
            "2",
        ],
        check=True,
        cwd=glove_dir,
    )

    # Emit resources.json for reproducibility
    try:
        _ram_mb = None
        try:
            with open('/proc/meminfo') as _mf:
                for _ln in _mf:
                    if _ln.startswith('MemTotal:'):
                        _ram_mb = int(_ln.split()[1]) // 1024
                        break
        except Exception:
            pass
        _res = {
            'ram_total_mb': _ram_mb,
            'memory_mb': float(memory_mb),
            'threads': int(threads_val),
            'window_size': int(window_size),
            'vocab_min_count': int(vocab_min_count),
        }
        with open(os.path.join(glove_dir, f'resources.{base}.json'), 'w', encoding='utf-8') as _rf:
            __import__('json').dump(_res, _rf)
    except Exception:
        pass

    vec_path = f"{save_file}.txt"
    if not os.path.isfile(vec_path):
        raise RuntimeError(f"Expected GloVe vectors not found: {vec_path}")
    # Emit simple coverage metadata (best-effort)
    try:
        meta = {
            "corpus_path": corpus_path,
            "vector_path": vec_path,
            "vocab_file": vocab_file,
            "cooccur_file": co_file,
        }
        try:
            with open(vocab_file, "r", encoding="utf-8") as f:
                meta["vocab_unique_tokens"] = sum(1 for _ in f)
        except Exception:
            meta["vocab_unique_tokens"] = 0
        try:
            with open(vec_path, "r", encoding="utf-8") as f:
                meta["vector_rows"] = sum(1 for _ in f)
        except Exception:
            meta["vector_rows"] = 0
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                meta["corpus_lines"] = sum(1 for _ in f)
        except Exception:
            meta["corpus_lines"] = 0
        cov = 0.0
        if meta.get("vocab_unique_tokens"):
            cov = float(meta.get("vector_rows", 0)) / float(max(1, meta.get("vocab_unique_tokens", 1)))
        meta["vector_vocab_coverage"] = cov
        # Threshold via env (optional), non-fatal
        try:
            min_cov = float(os.environ.get("GLOVE_MIN_COVERAGE", "0.0"))
        except Exception:
            min_cov = 0.0
        meta["meets_min_coverage"] = bool(cov >= min_cov)
        with open(f"{save_file}.meta.json", "w", encoding="utf-8") as fm:
            json.dump(meta, fm, indent=2)
    except Exception:
        pass
    return vec_path


