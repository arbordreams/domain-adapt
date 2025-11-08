from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional

import yaml
try:
    from .dedup_store import SeenStore  # type: ignore
except Exception:  # pragma: no cover
    SeenStore = None  # type: ignore


@dataclass
class SourceCfg:
    name: str
    kind: str
    enabled: bool
    target_bytes: int
    dataset: Optional[str] = None
    subset: Optional[str] = None
    splits: Optional[List[str]] = None
    text_fields: Optional[List[str]] = None
    urls: Optional[List[str]] = None


def _norm_text(t: str) -> str:
    # Normalize whitespace and drop stray control chars
    t = re.sub(r"\s+", " ", t or " ").strip()
    return t


def _hash_3gram(text: str) -> str:
    # Robust-ish content hash: 3-gram hash over lowercase ascii
    s = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    toks = s.split()
    if len(toks) < 3:
        base = " ".join(toks)
        return hashlib.sha1(base.encode("utf-8")).hexdigest()
    grams = [" ".join(toks[i : i + 3]) for i in range(len(toks) - 2)]
    h = hashlib.sha1()
    for g in grams:
        h.update(hashlib.md5(g.encode("utf-8")).digest())
    return h.hexdigest()


def _iter_hf_records(dataset: str, subset: Optional[str], splits: List[str]) -> Iterator[Dict]:
    try:
        from datasets import load_dataset, get_dataset_split_names
    except Exception as e:
        raise RuntimeError("Hugging Face datasets is required to stream HF sources") from e

    # Discover available splits to avoid hard failures
    try:
        available = set(get_dataset_split_names(dataset, subset))
    except Exception:
        available = set()

    req_splits = splits or ["train"]
    for sp in req_splits:
        if available and sp not in available:
            # Skip unknown split silently; we'll warn only on load failure
            continue
        try:
            # Prefer streaming; handle API changes where 'trust_remote_code' is removed.
            try:
                ds = load_dataset(dataset, subset, split=sp, streaming=True, trust_remote_code=True)  # type: ignore[call-arg]
            except TypeError:
                ds = load_dataset(dataset, subset, split=sp, streaming=True)
            except Exception:
                try:
                    ds = load_dataset(dataset, subset, split=sp, trust_remote_code=True)  # type: ignore[call-arg]
                except TypeError:
                    ds = load_dataset(dataset, subset, split=sp)
            for ex in ds:
                yield dict(ex)
        except Exception as e:
            # Skip this split if it fails (dataset might not have this split)
            import warnings
            warnings.warn(f"[biomed_corpus] Failed to load {dataset} split={sp}: {e}")
            continue


def _extract_text(ex: Dict, fields: List[str]) -> str:
    # Try fields in order; concatenate lists where needed
    if not fields:
        fields = ["text"]
    for k in fields:
        v = ex.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            # Join list items; if dicts, collect string-like values
            parts: List[str] = []
            for x in v:
                if isinstance(x, str):
                    parts.append(x)
                elif isinstance(x, dict):
                    for vv in x.values():
                        if isinstance(vv, str):
                            parts.append(vv)
                        elif isinstance(vv, list):
                            parts.extend(str(xx) for xx in vv if isinstance(xx, str))
                else:
                    try:
                        parts.append(str(x))
                    except Exception:
                        pass
            v = "\n".join(p for p in parts if p)
        if isinstance(v, dict):
            # Sometimes text nested under 'document'/'article'
            inner = []
            for vv in v.values():
                if isinstance(vv, str):
                    inner.append(vv)
                elif isinstance(vv, list):
                    inner.extend(str(x) for x in vv)
            v = "\n".join(inner)
        s = str(v).strip()
        if s:
            return s
    return ""


def _iter_http_texts(urls: List[str]) -> Iterator[str]:
    try:
        import requests
    except Exception as e:
        raise RuntimeError("requests is required for http_list sources") from e
    for u in urls or []:
        try:
            r = requests.get(u, timeout=60)
            r.raise_for_status()
            buf = io.StringIO(r.text)
            for line in buf:
                s = line.strip()
                if s:
                    yield s
        except Exception:
            continue


def _iter_parquet_texts(paths: List[str], fields: List[str]) -> Iterator[str]:
    """Yield text from parquet files via pyarrow.dataset with low memory usage."""
    try:
        import pyarrow.dataset as ds  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("pyarrow is required for parquet sources") from e
    if not paths:
        return iter(())  # type: ignore[return-value]
    try:
        dataset = ds.dataset(paths, format="parquet")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"failed to open parquet dataset: {e}") from e
    all_cols = list(dataset.schema.names or [])
    want = [c for c in (fields or ["text"]) if c in all_cols]
    if not want and all_cols:
        want = [all_cols[0]]
    if not want:
        return iter(())  # type: ignore[return-value]
    for batch in dataset.to_batches(columns=want, batch_size=1024):
        pdict = batch.to_pydict()
        # length by first chosen column
        length = len(pdict.get(want[0], []) or [])
        for i in range(length):
            parts: List[str] = []
            for c in want:
                col = pdict.get(c, [])
                v = col[i] if i < len(col) else None
                if v is None:
                    continue
                if isinstance(v, str):
                    parts.append(v)
                elif isinstance(v, list):
                    parts.extend(str(x) for x in v if isinstance(x, str))
                else:
                    try:
                        parts.append(str(v))
                    except Exception:
                        pass
            s = " ".join(p for p in parts if p).strip()
            if s:
                yield s


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_source(
    out_path: str,
    src: SourceCfg,
    min_chars: int,
    max_chars: int,
    rng: random.Random,
    *,
    seen_hashes: Optional[set[str]] = None,
    target_bytes_override: Optional[int] = None,
    near_dup_lsh: Optional[object] = None,
    seen_store: Optional[object] = None,
) -> Dict[str, int]:
    _ensure_dir(os.path.dirname(out_path))
    global_seen: set[str] = seen_hashes if seen_hashes is not None else set()
    written_bytes = 0
    kept = 0
    seen = 0
    if os.path.isfile(out_path):
            try:
                written_bytes = os.path.getsize(out_path)
            except Exception:
                written_bytes = 0
    eff_target_bytes = int(target_bytes_override) if target_bytes_override is not None else int(src.target_bytes)
    if src.kind == "hf":
        try:
            records = _iter_hf_records(src.dataset or "", src.subset, src.splits or ["train"])
            def gen_texts() -> Iterator[str]:
                for ex in records:
                    t = _extract_text(ex, src.text_fields or ["text"]) or ""
                    if t:
                        yield t
        except Exception as e:
            import warnings
            warnings.warn(f"[biomed_corpus] Skipping source {src.name}: {e}")
            return {"seen": 0, "kept": kept, "bytes": written_bytes}
    elif src.kind == "http_list":
        def gen_texts() -> Iterator[str]:
            for t in _iter_http_texts(src.urls or []):
                yield t
    elif src.kind == "parquet":
        def gen_texts() -> Iterator[str]:
            for t in _iter_parquet_texts(src.urls or [], src.text_fields or ["text"]):
                yield t
    else:
        def gen_texts() -> Iterator[str]:
            return iter(())
    use_append = os.path.isfile(out_path)
    if use_append:
        f = open(out_path, "a", encoding="utf-8")
    else:
        tmp_path = out_path + ".tmp"
        f = open(tmp_path, "w", encoding="utf-8")
    try:
        def _write_one(snippet: str) -> None:
            nonlocal kept, written_bytes
            if len(snippet) < min_chars:
                return
            h = _hash_3gram(snippet)
            try:
                if seen_store is not None and hasattr(seen_store, "exists") and seen_store.exists(h):  # type: ignore[attr-defined]
                    return
            except Exception:
                pass
            if h in global_seen:
                return
            if near_dup_lsh is not None:
                try:
                    mh, lsh = near_dup_lsh
                    m = mh(snippet)
                    if list(lsh.query(m)):
                        return
                    lsh.insert(h, m)
                except Exception:
                    pass
            remaining = max(1, eff_target_bytes - written_bytes)
            exp_b = min(max_chars, max(min_chars, len(snippet)))
            expected_remaining_samples = remaining / max(1, exp_b)
            keep_prob = 1.0 if written_bytes < eff_target_bytes else expected_remaining_samples / (expected_remaining_samples + 1000.0)
            if rng.random() > keep_prob:
                return
            rec = {"text": snippet, "source": src.name}
            line = json.dumps(rec, ensure_ascii=False)
            f.write(line + "\n")
            kept += 1
            written_bytes += len(line.encode("utf-8"))
            try:
                if seen_store is not None and hasattr(seen_store, "add"):
                    seen_store.add(h)  # type: ignore[attr-defined]
            except Exception:
                pass
            global_seen.add(h)
        for raw in gen_texts():
            if written_bytes >= eff_target_bytes:
                break
            seen += 1
            s = _norm_text(raw)
            if not s:
                continue
            if len(s) <= max_chars:
                _write_one(s)
            else:
                overlap = max(0, min(200, max_chars // 10))
                step = max_chars - overlap if max_chars > overlap else max_chars
                start = 0
                while start < len(s) and written_bytes < eff_target_bytes:
                    piece = s[start : start + max_chars]
                    _write_one(piece)
                    start += step
    finally:
        try:
        f.close()
        except Exception:
            pass
        if not use_append:
            try:
                if os.path.exists(tmp_path):
            os.replace(tmp_path, out_path)
            except Exception:
                pass
    return {"seen": seen, "kept": kept, "bytes": written_bytes}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build balanced biomedical corpus (~5 GB)")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--strict_sources", action="store_true", default=True)
    ap.add_argument("--preflight_only", action="store_true", default=False)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("random_seed", 17))
    rng = random.Random(seed)
    out_root = cfg.get("output_dir", "medical_tokalign/data/biomed_corpus")
    prevent_contam = bool(cfg.get("prevent_eval_contamination", False))
    near_dup_mode = str(cfg.get("near_dup", "none")).lower()
    defaults = cfg.get("defaults", {})
    min_chars = int(defaults.get("min_chars", 200))
    max_chars = int(defaults.get("max_chars", 20000))

    sources_cfg: List[SourceCfg] = []
    for name, sc in (cfg.get("sources") or {}).items():
        if not sc.get("enabled", True):
            continue
        sources_cfg.append(
            SourceCfg(
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

    # Preflight validation (fail fast on missing/inaccessible sources if strict)
    try:
        from . import dataset_preflight as preflight  # type: ignore[no-redef]
        specs = []
        for s in sources_cfg:
            specs.append((
                s.name,
                preflight.SourceSpec(
                    dataset=s.dataset or "",
                    subset=s.subset,
                    splits=s.splits or ["train"],
                    text_fields=s.text_fields or ["text"],
                ),
            ))
        ok, reports = preflight.preflight_sources(specs, strict=bool(getattr(args, "strict_sources", True)))
        print(preflight.format_reports(reports))
        if getattr(args, "preflight_only", False):
            return
        if getattr(args, "strict_sources", True) and not ok:
            raise SystemExit("Preflight failed for one or more sources (strict-sources). Aborting.")
    except Exception:
        # If preflight module missing for any reason, continue (CLI path also runs preflight)
        pass

    # Global target and global dedup set (in-memory only used when not using sqlite store)
    target_total = int(cfg.get("target_total_bytes", 0))
    global_seen: set[str] = set()
    # Optional blocklist from benchmark/eval JSONLs to prevent contamination
    if prevent_contam:
        try:
            bench_dir = os.path.join(os.path.dirname(out_root), "..", "medical", "benchmarks")
            if os.path.isdir(bench_dir):
                for name in sorted(os.listdir(bench_dir)):
                    if not name.endswith(".jsonl"):
                        continue
                    p = os.path.join(bench_dir, name)
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                rec = json.loads(line)
                                t = _norm_text(str(rec.get("text", "") or rec.get("context", "") or rec.get("question", "")))
                                if t:
                                    global_seen.add(_hash_3gram(t))
                            except Exception:
                                continue
        except Exception:
            pass
    summary: Dict[str, Dict[str, int]] = {}

    # Count bytes only (avoid rescanning large JSONLs)
    global_written = 0
    for s in sources_cfg:
        out_path = os.path.join(out_root, f"{s.name}.jsonl")
        try:
        if os.path.isfile(out_path):
                global_written += os.path.getsize(out_path)
            except Exception:
                pass

    # Optional MinHash near-dup (best effort)
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
        existing_bytes = os.path.getsize(out_path) if os.path.isfile(out_path) else 0
        if target_total > 0:
            remaining_global = max(0, target_total - global_written)
            if remaining_global <= 0:
                # No more global budget: record current stats and skip
                summary[s.name] = {
                    "seen": 0,
                    "kept": sum(1 for _ in open(out_path, "r", encoding="utf-8")) if os.path.isfile(out_path) else 0,
                    "bytes": existing_bytes,
                }
                continue
            # Allow writing up to min(per-source remaining, global remaining)
            per_source_remaining = max(0, s.target_bytes - existing_bytes)
            allowed_inc = min(per_source_remaining, remaining_global)
            eff_target = existing_bytes + allowed_inc
        else:
            eff_target = s.target_bytes

        stats = build_source(
            out_path,
            s,
            min_chars=min_chars,
            max_chars=max_chars,
            rng=rng,
            seen_hashes=global_seen,
            target_bytes_override=eff_target,
            near_dup_lsh=near_dup_lsh,
        )
        # Update global written by increment
        inc = max(0, int(stats.get("bytes", 0)) - int(existing_bytes))
        global_written += inc
        summary[s.name] = stats

    # Persist summary.json alongside printing
    try:
        with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as fsum:
            json.dump({
                "total_bytes": global_written,
                "target_total_bytes": target_total,
                "complete": bool(target_total > 0 and global_written >= target_total),
                "sources": summary
            }, fsum, indent=2)
    except Exception:
        pass

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)


