"""Verify a wheel's tables, source capture, and replay outside the checkout.

Run with ``.venv/bin/python tests/check_wheel.py`` after ``uv build --wheel``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from toylife2.records import _hash, _source_archive
from toylife2.tables import Tables


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    wheel = root / "dist" / "toylife2-0.1.0-py3-none-any.whl"
    if not wheel.is_file():
        raise FileNotFoundError("build the wheel before checking it")
    with tempfile.TemporaryDirectory(prefix="toylife2-wheel-") as temporary:
        folder = Path(temporary)
        target = folder / "installed"
        environment = {**os.environ, "UV_CACHE_DIR": str(root / ".cache" / "uv")}
        subprocess.run(["uv", "pip", "install", "--python", sys.executable,
                        "--target", str(target), "--no-deps", "--offline", str(wheel)],
                       check=True, env=environment, capture_output=True, text=True)
        script = """
import json
from pathlib import Path
import toylife2
from toylife2 import RunSpec, Tables
from toylife2.runner import run, reproduce
from toylife2.records import inspect_record, verify_record
record = run(RunSpec(genotypes=('0'*40,)), destination=Path('run'))
copy = reproduce(record, Path('replayed'))
assert verify_record(record)['ok'] and verify_record(copy)['ok']
assert '_vendor' in Tables.load().source
print(json.dumps({'module': toylife2.__file__, 'digest': Tables.load().digest,
                  'source_sha256': inspect_record(record)['source_sha256'],
                  'replay': True}))
"""
        child = subprocess.run([sys.executable, "-c", script], cwd=folder,
                               env={**environment, "PYTHONPATH": str(target)},
                               check=True, capture_output=True, text=True)
        report = json.loads(child.stdout.strip().splitlines()[-1])
        assert Path(report["module"]).is_relative_to(target)
        assert report["digest"] == Tables.load().digest
        _source_archive(folder / "checkout-source.zip")
        assert report["source_sha256"] == _hash(folder / "checkout-source.zip")
        print(json.dumps({"installed_tables": "match", "source_archive": "match",
                          "installed_replay": report["replay"]}, sort_keys=True))


if __name__ == "__main__":
    main()
