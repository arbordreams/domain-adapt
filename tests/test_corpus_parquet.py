import os
import tempfile

import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore

from medical_tokalign.src.dataset_preflight import SourceSpec, check_source


def test_parquet_preflight_ok():
    with tempfile.TemporaryDirectory() as tmpd:
        p = os.path.join(tmpd, "train.parquet")
        tbl = pa.table({"text": ["hello", "world"]})
        pq.write_table(tbl, p)
        spec = SourceSpec(
            dataset="",
            subset=None,
            splits=["parquet"],
            text_fields=["text"],
            kind="parquet",
            urls=[p],
            checksums=None,
        )
        rep = check_source(spec)
        assert rep.ok
        assert rep.split_reports[0].fields_present == ["text"]


def test_parquet_preflight_checksum_mismatch():
    with tempfile.TemporaryDirectory() as tmpd:
        p = os.path.join(tmpd, "train.parquet")
        tbl = pa.table({"text": ["x"]})
        pq.write_table(tbl, p)
        spec = SourceSpec(
            dataset="",
            subset=None,
            splits=["parquet"],
            text_fields=["text"],
            kind="parquet",
            urls=[p],
            checksums={p: "deadbeef"},
        )
        rep = check_source(spec)
        assert not rep.ok

