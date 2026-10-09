from __future__ import annotations

from toylife2.cli import main
from toylife2.config import dump_spec
from toylife2.specs import RunSpec


def test_validate_reports_scientific_identity(tmp_path, capsys):
    path = tmp_path / "run.toml"
    path.write_text(dump_spec(RunSpec(seed=123)), encoding="utf-8")
    assert main(["validate", str(path)]) == 0
    output = capsys.readouterr().out
    assert '"valid": true' in output
    assert '"scientific_identity"' in output
    assert main(["validate", str(path), "--set", "model.secret=3"]) == 2
