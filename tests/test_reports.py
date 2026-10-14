from __future__ import annotations

import numpy as np

from toylife2 import records
from toylife2.reports import render_report
from toylife2.specs import RunSpec


def test_report_reads_saved_array_without_changing_record(tmp_path, monkeypatch, canonical_tables):
    monkeypatch.setattr(
        records, "_source_archive", lambda path, source=None: records._write_bytes(path, b"source")
    )

    def table_archive(root, tables=None):
        path = root / "source" / "tables.zip"
        records._write_bytes(path, b"tables")
        return path

    monkeypatch.setattr(records, "_table_archive", table_archive)
    monkeypatch.setattr(records, "_archive_table_digest", lambda path: canonical_tables.digest)
    root = records.begin_record(RunSpec(seed=8), tmp_path / "record")
    records.finalize_record(
        root,
        arrays={"pattern": np.arange(12).reshape(3, 4)},
        summaries={"patterns": 1},
        metadata={"execution": {"actual_backend": "cpu"}},
    )
    report = render_report(root)
    assert report == tmp_path / "record-report.html"
    html = report.read_text(encoding="utf-8")
    assert "Saved array heatmap" in html
    assert "Scientific identity" in html
    assert records.verify_record(root)["ok"]
