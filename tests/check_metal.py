"""Run the existing numerical contract tests on physical Metal.

Run in a fresh process with ``.venv/bin/python tests/check_metal.py``.
The Engine disables backend fallback. This harness selects Metal for the
test modules' ExecutionSpec constructors and retains their existing assertions.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pytest


class MetalBackend:
    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        modules = {item.module for item in items if isinstance(item, pytest.Function)}
        for module in modules:
            assert module is not None
            module.ExecutionSpec = partial(module.ExecutionSpec, backend="metal")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    destination = root / "qualification"
    destination.mkdir(exist_ok=True)
    modules = ("runtime", "trajectory", "spatial", "maps")
    code = pytest.main(
        [*(str(root / "tests" / f"test_{name}.py") for name in modules),
         "-q", f"--junitxml={destination / 'metal-tests.xml'}"],
        plugins=[MetalBackend()],
    )
    from toylife2.runtime import Engine

    backend = Engine._runtime_backend
    report = {"pytest_exit_code": int(code), "backend": backend,
              "test_modules": list(modules)}
    (destination / "metal-tests.json").write_text(json.dumps(report, indent=2) + "\n")
    if code == pytest.ExitCode.OK and backend != "metal":
        raise RuntimeError("tests did not initialize the Metal backend")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
