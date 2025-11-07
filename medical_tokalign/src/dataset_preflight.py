from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SourceSpec:
    dataset: str
    subset: Optional[str]
    splits: List[str]
    text_fields: List[str]


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

    try:
        from datasets import load_dataset
    except Exception as e:  # noqa: BLE001
        return SourceReport(dataset=ds, subset=sb, ok=False, reason=f"datasets not available: {e}", split_reports=[])

    for sp in (spec.splits or ["train"]):
        ok = False
        reason = None
        fields_present: List[str] = []
        try:
            dargs: Dict[str, object] = {"split": sp, "trust_remote_code": True}
            if sb is not None:
                ds_obj = load_dataset(ds, sb, streaming=True, **dargs)
            else:
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


