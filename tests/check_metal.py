"""Run numerical device contracts on physical Metal.

Run in a fresh process with ``.venv/bin/python tests/check_metal.py``.
The Engine disables backend fallback. This harness selects Metal for the
test modules' ExecutionSpec constructors and retains their existing assertions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


class MetalBackend:
    def __init__(self, backend="metal"):
        self.backend = backend

    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        from toylife2.specs import ExecutionSpec

        original = ExecutionSpec.__init__

        def execution(instance, *args, **kwargs):
            if args:
                args = (self.backend, *args[1:])
                kwargs.pop("backend", None)
            else:
                kwargs["backend"] = self.backend
            original(instance, *args, **kwargs)

        ExecutionSpec.__init__ = execution


def main(selected_backend="metal", extra_args=()) -> int:
    root = Path(__file__).resolve().parents[1]
    destination = root / "qualification"
    destination.mkdir(exist_ok=True)
    modules = (
        "runtime",
        "trajectory",
        "spatial",
        "maps",
        "chemistry",
        "rollout",
        "runtime_optimisation",
        "resident_world",
        "evaluation",
        "gene_copy",
        "molecular_interface",
    )
    code = pytest.main(
        [
            *(str(root / "tests" / f"test_{name}.py") for name in modules),
            "-q",
            "-m",
            "device and not integration and not qualification",
            f"--junitxml={destination / (selected_backend + '-tests.xml')}",
            *extra_args,
        ],
        plugins=[MetalBackend(selected_backend)],
    )
    from toylife2.runtime import Engine

    backend = Engine._runtime_backend
    report = {
        "pytest_exit_code": int(code),
        "backend": backend,
        "test_modules": list(modules),
        "pytest_args": list(extra_args),
    }
    (destination / (selected_backend + "-tests.json")).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if code == pytest.ExitCode.OK and backend != selected_backend:
        raise RuntimeError(f"tests did not initialize the {selected_backend} backend")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main(extra_args=sys.argv[1:]))
