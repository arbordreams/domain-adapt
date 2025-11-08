#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from typing import List, Optional


def _load_dataset(ds_name: str, subset: Optional[str], split: str):
    from datasets import load_dataset
    dargs = {"split": split}
    if subset:
        try:
            return load_dataset(ds_name, subset, trust_remote_code=True, **dargs)  # type: ignore[call-arg]
        except TypeError:
            return load_dataset(ds_name, subset, **dargs)
    else:
        try:
            return load_dataset(ds_name, trust_remote_code=True, **dargs)  # type: ignore[call-arg]
        except TypeError:
            return load_dataset(ds_name, **dargs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Snapshot HF dataset splits to Parquet")
    ap.add_argument("--dataset", required=True, help="HF dataset id, e.g. zj88zj/PubMed_200k_RCT")
    ap.add_argument("--subset", default=None, help="Optional subset/config name")
    ap.add_argument("--splits", nargs="+", default=["train"], help="Splits to snapshot")
    ap.add_argument("--out_dir", required=True, help="Output directory for Parquet files")
    ap.add_argument("--columns", nargs="+", default=None, help="Optional columns to keep (default: all)")
    ap.add_argument("--max_rows", type=int, default=0, help="Optional cap per split (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for sp in args.splits:
        print(f"[snapshot] loading {args.dataset} {args.subset or ''} split={sp} ...")
        ds = _load_dataset(args.dataset, args.subset, sp)
        if args.columns:
            keep = [c for c in args.columns if c in ds.column_names]
            if keep:
                ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        if args.max_rows and args.max_rows > 0:
            ds = ds.select(range(min(args.max_rows, len(ds))))
        outp = os.path.join(args.out_dir, f"{sp}.parquet")
        print(f"[snapshot] writing {outp} ({len(ds)} rows) ...")
        ds.to_parquet(outp)
        print(f"[snapshot] done: {outp}")


if __name__ == "__main__":
    main()


