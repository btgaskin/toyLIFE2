"""Select CI work from changed paths; unavailable history runs all routine checks."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


def checks(paths: list[str] | None, *, full: bool = False) -> dict[str, bool]:
    if full or paths is None:
        return {"tests": True, "wheel": True, "full": full}

    def editorial(path):
        return path in {"README.md", "CONTRIBUTING.md"} or (
            path.startswith("docs/") and path.endswith(".md") and path != "docs/decisions.md"
        )

    wheel = any(
        path in {"pyproject.toml", "uv.lock", "LICENSE", "docs/decisions.md"}
        or path.startswith(("src/", "upstream/", ".github/"))
        or path in {"tests/check_wheel.py", "tests/ci_changes.py"}
        for path in paths
    )
    return {"tests": any(not editorial(path) for path in paths), "wheel": wheel, "full": False}


def changed_paths(base: str | None) -> list[str] | None:
    if not base or not re.fullmatch(r"[0-9a-f]{40,64}", base) or set(base) == {"0"}:
        return None
    try:
        known = subprocess.run(
            ["git", "cat-file", "-e", f"{base}^{{commit}}"], capture_output=True
        )
        if known.returncode:
            subprocess.run(
                ["git", "fetch", "--no-tags", "--depth=1", "origin", base],
                check=True,
                capture_output=True,
            )
        output = subprocess.check_output(
            ["git", "diff", "--no-renames", "--name-only", "-z", base, "HEAD"]
        )
        return [os.fsdecode(path) for path in output.split(b"\0") if path]
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    kind = os.environ["GITHUB_EVENT_NAME"]
    full = kind == "release" or (
        kind == "workflow_dispatch" and str(event.get("inputs", {}).get("full", True)).lower()
        == "true"
    )
    base = event.get("pull_request", {}).get("base", {}).get("sha") or event.get("before")
    selected = checks(changed_paths(base), full=full)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        for name, enabled in selected.items():
            output.write(f"{name}={str(enabled).lower()}\n")
    print(json.dumps(selected, sort_keys=True))


if __name__ == "__main__":
    main()
