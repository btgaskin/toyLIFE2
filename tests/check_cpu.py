"""Run tests on CPU in a fresh process; product defaults select an available GPU."""

import sys
from pathlib import Path

import pytest


class CPUBackend:
    def pytest_collection_modifyitems(self, items):
        from toylife2.specs import ExecutionSpec

        original = ExecutionSpec.__init__

        def execution(instance, *args, **kwargs):
            if not args and kwargs.get("backend", "auto") == "auto":
                kwargs["backend"] = "cpu"
            original(instance, *args, **kwargs)

        ExecutionSpec.__init__ = execution


if __name__ == "__main__":
    paths = sys.argv[1:] or [str(Path(__file__).resolve().parent)]
    raise SystemExit(pytest.main(paths, plugins=[CPUBackend()]))
