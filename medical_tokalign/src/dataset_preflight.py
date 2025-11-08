from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import hashlib
import os


@dataclass
class SourceSpec:
    dataset: str
    subset: Optional[str]
    splits: List[str]
    text_fields: List[str]
    kind: str = "hf"
    urls: Optional[List[str]] = None
    checksums: Optional[Dict[str, str]] = None


@dataclass
class SplitReport:
    split: str
    ok: bool
    reason: Optional[str]
    fields_present: List[str]


@dataclass
class SourceReport:
    dataset: str
    subset: Optional[str]
    ok: bool
    reason: Optional[str]
    split_reports: List[SplitReport]


def _env_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return token if token else None


def check_source(spec: SourceSpec) -> SourceReport:
    """Validate dataset availability and that at least one text field exists per split.

    Does not raise; returns a structured report. Uses a small streaming sample.
    """
    ds = spec.dataset
    sb = spec.subset
    token = _env_hf_token()
    split_reports: List[SplitReport] = []

    # Best-effort hub probe; do not fail if missing (script-based datasets may not have repos)
    hub_reason: Optional[str] = None
    try:
        try:
            from huggingface_hub import HfApi
            _ = HfApi().dataset_info(ds, token=token)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            # Acceptable for script-only datasets
            hub_reason = str(e)
    except Exception:
        hub_reason = None

    # Parquet preflight path (does not require datasets lib)
    if spec.kind.strip().lower() == "parquet":
        paths = list(spec.urls or [])
        if not paths:
            return SourceReport(dataset="parquet", subset=None, ok=False, reason="no parquet paths", split_reports=[])
        try:
            import pyarrow.dataset as pyds  # type: ignore
        except Exception as e:  # noqa: BLE001
            return SourceReport(dataset="parquet", subset=None, ok=False, reason=f"pyarrow missing: {e}", split_reports=[])
        try:
            dataset = pyds.dataset(paths, format="parquet")
            cols = set(dataset.schema.names or [])
            fields_present = [c for c in (spec.text_fields or ["text"]) if c in cols]
            # Attempt to read a tiny batch
            got_any = False
            for _ in dataset.to_batches(batch_size=1):
                got_any = True
                break
            ok = bool(fields_present) and got_any
            # Optional checksum verification
            if ok and spec.checksums:
                for p in paths:
                    want = spec.checksums.get(p)
                    if not want:
                        continue
                    h = hashlib.sha256()
                    try:
                        with open(p, "rb") as f:
                            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                                h.update(chunk)
                        got = h.hexdigest()
                        if got.lower() != want.lower():
                            ok = False
                            reason = f"checksum mismatch for {p}"
                            rep = SplitReport(split="parquet", ok=False, reason=reason, fields_present=fields_present)
                            return SourceReport(dataset="parquet", subset=None, ok=False, reason=reason, split_reports=[rep])
                    except Exception as e:  # noqa: BLE001
                        rep = SplitReport(split="parquet", ok=False, reason=f"checksum read failed: {e}", fields_present=fields_present)
                        return SourceReport(dataset="parquet", subset=None, ok=False, reason=str(e), split_reports=[rep])
            rep = SplitReport(split="parquet", ok=ok, reason=None if ok else "missing fields or empty", fields_present=fields_present)
            return SourceReport(dataset="parquet", subset=None, ok=ok, reason=None, split_reports=[rep])
        except Exception as e:  # noqa: BLE001
            rep = SplitReport(split="parquet", ok=False, reason=str(e), fields_present=[])
            return SourceReport(dataset="parquet", subset=None, ok=False, reason=str(e), split_reports=[rep])

    # HF datasets preflight (default)
    try:
        from datasets import load_dataset
    except Exception as e:  # noqa: BLE001
        return SourceReport(dataset=ds, subset=sb, ok=False, reason=f"datasets not available: {e}", split_reports=[])

    for sp in (spec.splits or ["train"]):
        ok = False
        reason = None
        fields_present: List[str] = []
        try:
            dargs: Dict[str, object] = {"split": sp}
            if sb is not None:
                try:
                    ds_obj = load_dataset(ds, sb, streaming=True, trust_remote_code=True, **dargs)  # type: ignore[call-arg]
                except TypeError:
                    ds_obj = load_dataset(ds, sb, streaming=True, **dargs)
            else:
                try:
                    ds_obj = load_dataset(ds, streaming=True, trust_remote_code=True, **dargs)  # type: ignore[call-arg]
                except TypeError:
                    ds_obj = load_dataset(ds, streaming=True, **dargs)
            # Pull one sample
            it = iter(ds_obj)
            ex = next(it, None)
            if ex is None:
                ok = False
                reason = "empty split"
            else:
                for f in (spec.text_fields or ["text"]):
                    if f in ex:
                        fields_present.append(f)
                # Treat nested dict/list as acceptable; extraction code will handle it
                ok = bool(ex) and (bool(fields_present) or any(isinstance(ex.get(f), (dict, list)) for f in (spec.text_fields or ["text"])) )
                if not ok:
                    reason = "no expected text fields"
        except Exception as e:  # noqa: BLE001
            ok = False
            reason = str(e)
        split_reports.append(SplitReport(split=sp, ok=ok, reason=reason, fields_present=fields_present))

    overall_ok = all(r.ok for r in split_reports) and True
    overall_reason = hub_reason
    return SourceReport(dataset=ds, subset=sb, ok=overall_ok, reason=overall_reason, split_reports=split_reports)


def preflight_sources(sources: List[Tuple[str, SourceSpec]], strict: bool = True) -> Tuple[bool, List[Tuple[str, SourceReport]]]:
    """Run preflight for a list of named sources.

    Returns (ok, reports). If strict and any fail, ok=False.
    """
    reports: List[Tuple[str, SourceReport]] = []
    all_ok = True
    for name, spec in sources:
        rep = check_source(spec)
        reports.append((name, rep))
        if strict and not rep.ok:
            all_ok = False
    return all_ok, reports


def format_reports(reports: List[Tuple[str, SourceReport]]) -> str:
    lines: List[str] = []
    for name, rep in reports:
        status = "OK" if rep.ok else "FAIL"
        subset = f"/{rep.subset}" if rep.subset else ""
        lines.append(f"[{status}] {name}: {rep.dataset}{subset} ({rep.reason or ''})")
        for sr in rep.split_reports:
            s = "ok" if sr.ok else f"fail: {sr.reason}"
            fields = ",".join(sr.fields_present) if sr.fields_present else "-"
            lines.append(f"  - split={sr.split}: {s}; fields={fields}")
    return "\n".join(lines)


