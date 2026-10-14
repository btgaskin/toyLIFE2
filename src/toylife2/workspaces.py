"""Named collections of portable run records."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .records import inspect_record, verify_record
from .specs import RunSpec, scientific_identity

WORKSPACE_SCHEMA = 1


def init_workspace(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "workspace.json"
    if marker.exists():
        data = json.loads(marker.read_text(encoding="utf-8"))
        if data.get("workspace_schema") != WORKSPACE_SCHEMA:
            raise ValueError("unsupported workspace schema")
    else:
        temporary = marker.with_name(marker.name + ".tmp")
        temporary.write_text(
            json.dumps(
                {"workspace_schema": WORKSPACE_SCHEMA, "created_at": datetime.now(UTC).isoformat()},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, marker)
    (root / "runs").mkdir(exist_ok=True)
    return root


def new_record_path(workspace: str | Path, spec: RunSpec, name: str | None = None) -> Path:
    root = init_workspace(workspace)
    if name is not None:
        if not name or not all(char.isalnum() or char in "_-" for char in name):
            raise ValueError("run name must contain only letters, numbers, '_' or '-'")
        stem = name
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        stem = f"{timestamp}-{scientific_identity(spec)[:12]}"
    destination = root / "runs" / stem
    if destination.exists():
        raise FileExistsError(destination)
    return destination


def list_records(workspace: str | Path) -> list[dict[str, Any]]:
    root = Path(workspace).expanduser().resolve()
    if not (root / "workspace.json").is_file():
        raise ValueError(f"not a toyLIFE workspace: {root}")
    records = []
    for directory in sorted((root / "runs").iterdir()):
        if directory.is_dir() and (directory / "manifest.json").is_file():
            manifest = inspect_record(directory)
            records.append(
                {
                    "name": directory.name,
                    "path": str(directory),
                    "status": manifest["status"],
                    "operation": manifest["operation"],
                    "scientific_identity": manifest["scientific_identity"],
                    "created_at": manifest["created_at"],
                }
            )
        elif (
            directory.is_dir()
            and (directory / "profile.toml").is_file()
            and (directory / "evolution" / "manifest.json").is_file()
        ):
            manifest = inspect_record(directory / "evolution")
            records.append(
                {
                    "name": directory.name,
                    "path": str(directory),
                    "status": manifest["status"],
                    "operation": "molecular_cells",
                    "scientific_identity": manifest["scientific_identity"],
                    "created_at": manifest["created_at"],
                }
            )
    return records


def check_workspace(workspace: str | Path) -> dict[str, Any]:
    records = list_records(workspace)
    checked = {}
    for entry in records:
        root = Path(entry["path"])
        if entry["operation"] == "molecular_cells":
            checks = {
                child.name: verify_record(child)
                for child in root.iterdir()
                if child.is_dir() and (child / "manifest.json").is_file()
            }
            errors = [
                f"{name}: {error}" for name, result in checks.items() for error in result["errors"]
            ]
            summary = root / "experiment.json"
            if summary.is_file():
                import hashlib

                for name, digest in json.loads(summary.read_text()).get("artifacts", {}).items():
                    artifact = root / name
                    if (
                        Path(name).name != name
                        or not artifact.is_file()
                        or hashlib.sha256(artifact.read_bytes()).hexdigest() != digest
                    ):
                        errors.append(f"invalid molecular artifact: {name}")
            checked[entry["name"]] = {"ok": not errors, "errors": errors, "records": checks}
        else:
            checked[entry["name"]] = verify_record(root)
    return {
        "path": str(Path(workspace).resolve()),
        "runs": len(records),
        "ok": all(item["ok"] for item in checked.values()),
        "records": checked,
    }
