"""Self-describing, integrity-checked run records.

The manifest is the commit point. A crash leaves a readable ``running`` record
and the last complete checkpoint; finalization changes its status atomically.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import uuid
import zipfile
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

from .config import dump_spec
from .results import ArrayResult
from .specs import RunSpec, digest, from_mapping, plain, scientific_identity

RECORD_SCHEMA = 1
STATUSES = frozenset({"running", "complete", "partial", "failed", "interrupted"})


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(
        path,
        (json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _source_archive(path: Path, source: Path | None = None) -> None:
    package = (source or Path(__file__).resolve().parent).resolve()
    checkout = next(
        (
            candidate
            for candidate in package.parents
            if (candidate / "pyproject.toml").is_file()
            and (candidate / "src" / "toylife2").resolve() == package
        ),
        None,
    )
    provenance = package / "_provenance"
    archive_files: dict[str, Path] = {}
    for item in package.rglob("*.py"):
        if (
            "__pycache__" not in item.parts
            and "_vendor" not in item.parts
            and "_provenance" not in item.parts
        ):
            archive_files["src/toylife2/" + item.relative_to(package).as_posix()] = item
    for relative in ("pyproject.toml", "uv.lock", "docs/decisions.md", "LICENSE"):
        candidates = []
        if checkout:
            candidates.append(checkout / relative)
        candidates.append(provenance / relative)
        if relative == "docs/decisions.md":
            candidates.append(provenance / "decisions.md")
        found = next((candidate for candidate in candidates if candidate.is_file()), None)
        if found:
            archive_files[relative] = found
    author_candidates = []
    if checkout:
        author_candidates.append(checkout / "upstream" / "toylife")
    author_candidates.append(provenance / "upstream" / "toylife")
    author_candidates.append(package / "_vendor" / "toylife")
    author_source = next((candidate for candidate in author_candidates if candidate.is_dir()), None)
    if author_source:
        for item in author_source.rglob("*"):
            if (
                item.is_file()
                and ".git" not in item.parts
                and "results" not in item.relative_to(author_source).parts
                and (item.suffix in {".cpp", ".h", ".md", ".pdf"} or item.name == "Makefile")
            ):
                name = "upstream/toylife/" + item.relative_to(author_source).as_posix()
                archive_files[name] = item
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, item in sorted(archive_files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, item.read_bytes())
    _write_bytes(path, buffer.getvalue())


def _table_archive(record: Path, tables=None) -> Path:
    """Deduplicate immutable table packs in a workspace, retaining portable links."""
    from .tables import DEFAULT_SOURCE, _source_digest, _source_files

    source = Path(tables.source if tables is not None else DEFAULT_SOURCE).resolve()
    files = _source_files(source)
    source_digest = _source_digest(source, files)
    workspace = record.parent.parent
    shared = (workspace / "workspace.json").is_file()
    target = (
        workspace / "packs" / f"tables-{source_digest}.zip"
        if shared
        else record / "source" / "tables.zip"
    )
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in files:
                info = zipfile.ZipInfo(
                    item.relative_to(source).as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, item.read_bytes())
        os.replace(temporary, target)
    if shared:
        link = record / "source" / "tables.zip"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(target, link.parent))
        return link
    return target


_ARCHIVE_DIGESTS: dict[str, str] = {}


def _archive_table_digest(path: Path) -> str:
    """Hash archived source in canonical order; memoize by compressed content."""
    from .tables import _CACHE_VERSION

    archive_hash = _hash(path)
    if archive_hash in _ARCHIVE_DIGESTS:
        return _ARCHIVE_DIGESTS[archive_hash]
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            Path(name).is_absolute() or ".." in Path(name).parts for name in names
        ):
            raise ValueError("invalid table archive paths")
        ordered = [
            "data/" + name
            for name in (
                "protein_gene.txt",
                "proteins_with_perimeter.txt",
                "proteins.txt",
                "perimeters.txt",
                "dimers.txt",
                "metabolites.txt",
            )
        ]
        ordered += sorted(name for name in names if Path(name).match("promoters/*_bind_prom*.txt"))
        ordered += sorted(name for name in names if Path(name).match("prot_metabolism/[4-8]_*.txt"))
        ordered += sorted(name for name in names if Path(name).match("dim_metabolism/[4-8]_*.txt"))
        ordered += ["prot_metabolism/prot_breaking.txt"]
        if len(ordered) < 300 or set(ordered) != set(names):
            raise ValueError("incomplete or unexpected table archive contents")
        sha = hashlib.sha256(f"canonical-tenths-{_CACHE_VERSION}\n".encode())
        for name in ordered:
            sha.update(name.encode() + b"\0")
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    sha.update(chunk)
    result = sha.hexdigest()
    if len(_ARCHIVE_DIGESTS) >= 8:
        _ARCHIVE_DIGESTS.pop(next(iter(_ARCHIVE_DIGESTS)))
    _ARCHIVE_DIGESTS[archive_hash] = result
    return result


def _record_identity(spec_id, input_hashes, components, table_digest, session_inputs):
    return digest(
        {
            "spec": spec_id,
            "inputs": input_hashes,
            "components": components,
            "tables": table_digest,
            "session_inputs": session_inputs,
        }
    )


def load_session_inputs(record: str | Path) -> dict:
    from .sessions import unpack

    root = Path(record)
    captured = _read_json(root / "manifest.json").get("session_inputs")
    if captured is None:
        return {}
    arrays = {
        name: np.load(root / item["path"], allow_pickle=False)
        for name, item in captured["arrays"].items()
    }
    return unpack(captured["state"], arrays)


@contextmanager
def captured_tables(record: str | Path):
    """Restore the executed molecular source, including alternate table packs."""
    from .tables import Tables

    root = Path(record)
    spec = load_record_spec(root)
    expected = _archive_table_digest(root / "source" / "tables.zip")
    current = None
    try:
        candidate = Tables.load(cache_dir=spec.execution.cache_dir)
        if candidate.digest == expected:
            candidate.verify_source()
            current = candidate
    except (ValueError, OSError):
        # The portable pack remains authoritative if local source/cache is unavailable.
        pass
    if current is not None:
        yield current
        return
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary)
        with zipfile.ZipFile(root / "source" / "tables.zip") as archive:
            for entry in archive.infolist():
                if not (source / entry.filename).resolve().is_relative_to(source.resolve()):
                    raise ValueError("unsafe table archive path")
            archive.extractall(source)
        tables = Tables.load(source=source, cache_dir=spec.execution.cache_dir)
        tables.verify_source()
        yield tables


def _record_files(record: Path):
    for file in sorted(record.rglob("*")):
        if file.is_file() and file.name != "manifest.json" and not file.name.endswith(".tmp"):
            yield file, file.relative_to(record).as_posix()


def _inventory(
    record: Path, manifest: Mapping[str, Any] | None = None, changed: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    previous = (manifest or {}).get("files", {})
    retained = set((manifest or {}).get("checkpoint_revisions", []))
    committed_traces = set((manifest or {}).get("trace", {}).get("chunks", []))
    inventory = {}
    for file, relative in _record_files(record):
        parts = Path(relative).parts
        if len(parts) > 1 and parts[0] == "checkpoints" and parts[1] not in retained:
            continue
        if len(parts) > 1 and parts[0] == "traces" and relative not in committed_traces:
            continue
        stat = file.stat()
        if relative in previous and relative not in (changed or set()):
            old = previous[relative]
            if stat.st_size != old["bytes"]:
                raise ValueError(f"committed record file changed: {relative}")
            if old.get("mtime_ns") == stat.st_mtime_ns:
                inventory[relative] = old
            elif _hash(file) == old["sha256"]:
                inventory[relative] = {**old, "mtime_ns": stat.st_mtime_ns}
            else:
                raise ValueError(f"committed record file changed: {relative}")
        else:
            inventory[relative] = {
                "sha256": _hash(file),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    for relative in set(previous) - set(inventory):
        parts = Path(relative).parts
        if len(parts) > 1 and parts[0] == "checkpoints" and parts[1] not in retained:
            continue
        raise ValueError(f"committed record file missing: {relative}")
    return inventory


def _commit(record: Path, manifest: dict[str, Any], changed: set[str] | None = None) -> None:
    manifest["updated_at"] = _now()
    manifest["files"] = _inventory(record, manifest, changed)
    _write_json(record / "manifest.json", manifest)


def _safe_name(name: str) -> str:
    if not name or not all(character.isalnum() or character in "_-" for character in name):
        raise ValueError(f"invalid record artifact name: {name!r}")
    return name


def _software_environment() -> dict[str, Any]:
    dependencies = {}
    distributions = {}
    for name in ("toylife2", "numpy", "quadrants", "tomli-w"):
        try:
            distribution = importlib_metadata.distribution(name)
        except importlib_metadata.PackageNotFoundError:
            dependencies[name] = None
            continue
        dependencies[name] = distribution.version
        record = distribution.read_text("RECORD")
        distributions[name] = hashlib.sha256(record.encode()).hexdigest() if record else None
    details = {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": dependencies,
        "distribution_records_sha256": distributions,
    }
    details["environment_sha256"] = digest(details)
    return details


def begin_record(
    spec: RunSpec,
    destination: str | Path,
    *,
    inputs: Mapping[str, str | Path] | None = None,
    source: Path | None = None,
    parent: Mapping[str, Any] | None = None,
    tables=None,
    session_inputs: Mapping[str, Any] | None = None,
) -> Path:
    """Create a replayable record before executing an experiment.

    ``inputs`` are copied and hashed, so replay never depends on mutable source
    paths. The source archive includes the executed package and lock file.
    """
    from .sessions import pack
    from .tables import Tables

    tables = tables if tables is not None else Tables.load(cache_dir=spec.execution.cache_dir)
    tables.verify_source()
    record = Path(destination).expanduser().resolve()
    if record.exists() and any(record.iterdir()):
        raise FileExistsError(f"record directory is not empty: {record}")
    record.mkdir(parents=True, exist_ok=True)
    _write_json(record / "spec.json", plain(spec))
    _write_bytes(record / "spec.toml", dump_spec(spec).encode("utf-8"))
    _source_archive(record / "source" / "source.zip", source)
    table_archive = _table_archive(record, tables)
    if _archive_table_digest(table_archive) != tables.digest:
        raise ValueError("captured table source differs from executing tables")
    input_metadata = {}
    for name, source_path in (inputs or {}).items():
        source_file = Path(source_path).expanduser().resolve()
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        target = record / "inputs" / _safe_name(name) / source_file.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
        input_metadata[name] = {
            "path": target.relative_to(record).as_posix(),
            "source_path": str(source_file),
            "sha256": _hash(target),
        }
    spec_scientific_id = scientific_identity(spec)
    input_hashes = {name: value["sha256"] for name, value in input_metadata.items()}
    from .components import capture

    captured = capture(spec) if spec.schema_version == 2 else None
    if captured is not None:
        _write_bytes(record / "source" / "components.zip", captured[0])
    session_arrays = {}
    session_state = pack(dict(session_inputs or {}), session_arrays)
    session_metadata = {"state": session_state, "arrays": {}}
    for name, values in session_arrays.items():
        path = record / "inputs" / "session" / f"{name}.npy"
        _array(path, values)
        session_metadata["arrays"][name] = {
            "path": path.relative_to(record).as_posix(),
            "sha256": _hash(path),
        }
    record_scientific_id = _record_identity(
        spec_scientific_id,
        input_hashes,
        captured[1]["identity"] if captured else None,
        tables.digest,
        session_metadata,
    )
    package = (source or Path(__file__).resolve().parent).resolve()
    lock_candidates = [
        candidate / "uv.lock"
        for candidate in package.parents
        if (candidate / "pyproject.toml").is_file()
        and (candidate / "src" / "toylife2").resolve() == package
    ]
    lock_candidates.append(package / "_provenance" / "uv.lock")
    lock = next((candidate for candidate in lock_candidates if candidate.is_file()), None)
    manifest = {
        "record_schema": 2 if spec.schema_version == 2 else RECORD_SCHEMA,
        "status": "running",
        "created_at": _now(),
        "scientific_identity": record_scientific_id,
        "scientific_identity_scheme": "spec-inputs-components-tables-session-v1",
        "table_source_digest": tables.digest,
        "session_inputs": session_metadata,
        "spec_scientific_identity": spec_scientific_id,
        "spec_sha256": digest(spec),
        "ruleset": spec.model.rules,
        "operation": spec.operation,
        "seed": spec.seed,
        "recording_mode": spec.recording.mode,
        "source_sha256": _hash(record / "source" / "source.zip"),
        "table_pack_sha256": _hash(table_archive),
        "dependency_lock_sha256": _hash(lock) if lock else None,
        "inputs": input_metadata,
        "execution": {
            "requested_backend": spec.execution.backend,
            "actual_backend": None,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "software_environment": _software_environment(),
        "status_units": {},
        "checkpoint_revisions": [],
        "artifacts": {"arrays": {}, "summary": None, "checkpoint": None},
    }
    if parent:
        manifest["parent"] = _jsonable(dict(parent))
    if captured is not None:
        manifest["components"] = captured[1]
    _commit(record, manifest)
    return record


def _array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npy")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_boundary(
    record: str | Path,
    state: Mapping[str, Any],
    *,
    arrays: Mapping[str, np.ndarray] | None = None,
    events: list[Mapping[str, Any]] | None = None,
) -> tuple[Path, bool]:
    """Commit a trace transition and its checkpoint with one manifest update.

    A rejected trace event is kept as ``pending_trace`` in the committed state.
    Files written before a failed manifest commit remain uncommitted orphans.
    """
    root = Path(record)
    manifest = _read_json(root / "manifest.json")
    if manifest["status"] != "running":
        raise ValueError("checkpoints can only be written to a running record")
    pending = list(events or [])
    accepted = True
    committed_state = dict(state)
    if pending:
        spec = load_record_spec(root)
        if spec.recording.mode != "full":
            raise ValueError("event traces require recording.mode='full'")
        trace = manifest.setdefault(
            "trace", {"bytes": 0, "rows": 0, "dropped_rows": 0, "truncated": False, "chunks": []}
        )
        lines = [
            (json.dumps(_jsonable(dict(event)), sort_keys=True, allow_nan=False) + "\n").encode()
            for event in pending
        ]
        payload = b"".join(lines)
        budget = spec.recording.trace_budget_mib * 1024 * 1024
        accepted = trace["bytes"] + len(payload) <= budget
        if accepted:
            path = root / "traces" / f"{uuid.uuid4().hex}.jsonl"
            _write_bytes(path, payload)
            trace.setdefault("chunks", []).append(path.relative_to(root).as_posix())
            trace["bytes"] += len(payload)
            trace["rows"] += len(lines)
            committed_state.pop("pending_trace", None)
            committed_state.pop("trace_event", None)
            committed_state.pop("recording_limit", None)
        else:
            committed_state["pending_trace"] = (
                _jsonable(dict(pending[0]))
                if len(pending) == 1
                else [_jsonable(dict(event)) for event in pending]
            )
            committed_state["recording_limit"] = "trace_budget"
            trace["truncated"] = True
            trace["pending_rows"] = len(pending)
    revision = uuid.uuid4().hex
    array_paths = {}
    for name, values in (arrays or {}).items():
        path = root / "checkpoints" / revision / f"{_safe_name(name)}.npy"
        _array(path, values)
        array_paths[name] = path.name
    checkpoint = root / "checkpoints" / revision / "checkpoint.json"
    _write_json(
        checkpoint,
        {
            "created_at": _now(),
            "revision": revision,
            "state": committed_state,
            "arrays": array_paths,
        },
    )
    manifest["artifacts"]["checkpoint"] = checkpoint.relative_to(root).as_posix()
    previous = manifest.get("checkpoint_revisions", [])
    manifest["checkpoint_revisions"] = [*previous, revision][-2:]
    _commit(root, manifest)
    for old in previous:
        if old not in manifest["checkpoint_revisions"]:
            shutil.rmtree(root / "checkpoints" / old, ignore_errors=True)
    return checkpoint, accepted


def write_checkpoint(
    record: str | Path,
    state: Mapping[str, Any],
    *,
    arrays: Mapping[str, np.ndarray] | None = None,
) -> Path:
    """Commit a checkpoint containing algorithm and RNG state."""
    checkpoint, _ = write_boundary(record, state, arrays=arrays)
    return checkpoint


def append_trace(record: str | Path, events: list[Mapping[str, Any]]) -> dict[str, int | bool]:
    """Commit one immutable JSONL trace chunk within the configured byte budget.

    Call before ``finalize_record``. Excess rows are counted as dropped in the
    manifest, so a bounded trace never looks complete when it was truncated.
    """
    root = Path(record)
    manifest = _read_json(root / "manifest.json")
    if manifest["status"] != "running":
        raise ValueError("trace rows can only be added to a running record")
    spec = load_record_spec(root)
    if spec.recording.mode != "full":
        raise ValueError("event traces require recording.mode='full'")
    trace = manifest.setdefault(
        "trace", {"bytes": 0, "rows": 0, "dropped_rows": 0, "truncated": False, "chunks": []}
    )
    budget = spec.recording.trace_budget_mib * 1024 * 1024
    remaining = max(0, budget - trace["bytes"])
    accepted = []
    for event in events:
        line = (json.dumps(_jsonable(dict(event)), sort_keys=True, allow_nan=False) + "\n").encode()
        if len(line) > remaining:
            break
        accepted.append(line)
        remaining -= len(line)
    if accepted:
        path = root / "traces" / f"{uuid.uuid4().hex}.jsonl"
        _write_bytes(path, b"".join(accepted))
        trace.setdefault("chunks", []).append(path.relative_to(root).as_posix())
        trace["bytes"] += sum(map(len, accepted))
        trace["rows"] += len(accepted)
    dropped = len(events) - len(accepted)
    trace["dropped_rows"] += dropped
    trace["truncated"] = trace["truncated"] or dropped > 0
    _commit(root, manifest)
    return {"written": len(accepted), "dropped": dropped, "truncated": trace["truncated"]}


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    if checkpoint.is_dir() and (checkpoint / "manifest.json").is_file():
        manifest = _read_json(checkpoint / "manifest.json")
        reference = manifest.get("artifacts", {}).get("checkpoint")
        if not reference:
            raise FileNotFoundError("record has no committed checkpoint")
        checkpoint /= reference
    elif checkpoint.is_dir():
        checkpoint = checkpoint / "checkpoint.json"
    for ancestor in checkpoint.parents:
        manifest_path = ancestor / "manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            relative = checkpoint.relative_to(ancestor).as_posix()
            if relative not in manifest.get("files", {}):
                raise ValueError(f"uncommitted checkpoint: {checkpoint}")
            break
    data = _read_json(checkpoint)
    data["loaded_arrays"] = {
        name: np.load(checkpoint.parent / relative, allow_pickle=False)
        for name, relative in data.get("arrays", {}).items()
    }
    return data


def finalize_record(
    record: str | Path,
    *,
    arrays: Mapping[str, np.ndarray | ArrayResult] | None = None,
    summaries: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    status: str = "complete",
) -> Path:
    """Commit arrays and summaries, then mark the record's final status."""
    if status not in STATUSES - {"running"}:
        raise ValueError(f"unsupported final record status: {status}")
    root = Path(record)
    manifest = _read_json(root / "manifest.json")
    if manifest["status"] != "running":
        raise ValueError("record has already been finalized")
    for name, result in (arrays or {}).items():
        _safe_name(name)
        if isinstance(result, ArrayResult) and status == "complete" and not result.complete:
            raise ValueError(f"{name} has non-complete units; finalize record as partial")
    for name, result in (arrays or {}).items():
        name = _safe_name(name)
        values = result.values if isinstance(result, ArrayResult) else result
        path = root / "arrays" / f"{name}.npy"
        _array(path, np.asarray(values))
        description = {
            "path": path.relative_to(root).as_posix(),
            "shape": list(np.shape(values)),
            "dtype": str(np.asarray(values).dtype),
        }
        if isinstance(result, ArrayResult):
            labels, counts = np.unique(result.status, return_counts=True)
            status_counts = {
                str(label): int(count) for label, count in zip(labels, counts, strict=True)
            }
            manifest["status_units"][name] = status_counts
            status_path = root / "arrays" / f"{name}_status.npy"
            _array(status_path, result.status)
            description["status_path"] = status_path.relative_to(root).as_posix()
            description["metadata"] = _jsonable(result.metadata)
        manifest["artifacts"]["arrays"][name] = description
    if summaries is not None:
        summary_path = root / "summary.json"
        _write_json(summary_path, summaries)
        manifest["artifacts"]["summary"] = "summary.json"
        scalar = {
            key: value
            for key, value in summaries.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(("metric", "value"))
        writer.writerows((key, value) for key, value in sorted(scalar.items()))
        _write_bytes(root / "summary.csv", output.getvalue().encode("utf-8"))
    if metadata:
        for key, value in metadata.items():
            if key == "table_digest" and manifest.get("table_source_digest") != value:
                raise ValueError("executing tables differ from the captured source")
            if key in {
                "scientific_identity_scheme",
                "table_source_digest",
                "session_inputs",
                "record_schema",
                "files",
                "spec_sha256",
                "scientific_identity",
                "spec_scientific_identity",
                "source_sha256",
                "table_pack_sha256",
                "dependency_lock_sha256",
                "inputs",
                "artifacts",
            }:
                raise ValueError(f"cannot overwrite record identity: {key}")
            manifest[key] = _jsonable(value)
    manifest["status"] = status
    _commit(root, manifest)
    return root


_IDENTITY_FIELDS = frozenset(
    {
        "scientific_identity_scheme",
        "table_source_digest",
        "session_inputs",
        "table_digest",
        "record_schema",
        "files",
        "spec_sha256",
        "scientific_identity",
        "spec_scientific_identity",
        "source_sha256",
        "table_pack_sha256",
        "dependency_lock_sha256",
        "inputs",
        "artifacts",
        "checkpoint_revisions",
        "status",
        "created_at",
        "updated_at",
        "status_units",
        "trace",
        "execution",
    }
)


def annotate_record(record: str | Path, metadata: Mapping[str, Any]) -> Path:
    """Add provenance such as a resume/reproduction parent without changing identity."""
    root = Path(record)
    manifest = _read_json(root / "manifest.json")
    for key, value in metadata.items():
        if key in _IDENTITY_FIELDS:
            raise ValueError(f"cannot overwrite record identity: {key}")
        manifest[key] = _jsonable(value)
    _commit(root, manifest)
    return root


def inspect_record(record: str | Path) -> dict[str, Any]:
    root = Path(record)
    manifest = _read_json(root / "manifest.json")
    manifest["path"] = str(root.resolve())
    return manifest


def load_record_spec(record: str | Path) -> RunSpec:
    return from_mapping(_read_json(Path(record) / "spec.json"))


def verify_record(record: str | Path) -> dict[str, Any]:
    """Check file integrity and identity before replay or inspection."""
    root = Path(record)
    manifest = _read_json(root / "manifest.json")
    if manifest.get("record_schema") not in {1, 2}:
        raise ValueError("unsupported record schema")
    errors = []
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
        elif path.stat().st_size != expected["bytes"] or _hash(path) != expected["sha256"]:
            errors.append(f"changed: {relative}")
    files = manifest.get("files", {})
    if files.get("source/tables.zip", {}).get("sha256") != manifest.get("table_pack_sha256"):
        errors.append("table archive identity changed")
    captured_inputs = [
        *manifest.get("inputs", {}).values(),
        *manifest.get("session_inputs", {}).get("arrays", {}).values(),
    ]
    for item in captured_inputs:
        if files.get(item["path"], {}).get("sha256") != item["sha256"]:
            errors.append(f"input identity changed: {item['path']}")
    retained = set(manifest.get("checkpoint_revisions", []))
    untracked = set()
    orphan_revisions = set()
    orphan_traces = set()
    committed_traces = set(manifest.get("trace", {}).get("chunks", []))
    for _, relative in _record_files(root):
        if relative in manifest.get("files", {}):
            continue
        parts = Path(relative).parts
        if len(parts) > 1 and parts[0] == "checkpoints" and parts[1] not in retained:
            orphan_revisions.add(parts[1])
        elif len(parts) > 1 and parts[0] == "traces" and relative not in committed_traces:
            orphan_traces.add(relative)
        else:
            untracked.add(relative)
    errors.extend(f"untracked: {path}" for path in sorted(untracked))
    if digest(load_record_spec(root)) != manifest.get("spec_sha256"):
        errors.append("spec identity changed")
    spec_scientific_id = scientific_identity(load_record_spec(root))
    if spec_scientific_id != manifest.get("spec_scientific_identity"):
        errors.append("spec scientific identity changed")
    input_hashes = {name: value["sha256"] for name, value in manifest.get("inputs", {}).items()}
    expected_scientific_id = (
        digest(
            {
                "spec": spec_scientific_id,
                "inputs": input_hashes,
                "components": manifest["components"]["identity"],
            }
        )
        if "components" in manifest
        else digest({"spec": spec_scientific_id, "inputs": input_hashes})
        if input_hashes
        else spec_scientific_id
    )
    if manifest.get("scientific_identity_scheme") == "spec-inputs-components-tables-session-v1":
        expected_scientific_id = _record_identity(
            spec_scientific_id,
            input_hashes,
            manifest.get("components", {}).get("identity"),
            manifest.get("table_source_digest"),
            manifest.get("session_inputs"),
        )
    elif manifest.get("scientific_identity_scheme") is not None:
        errors.append("unsupported scientific identity scheme")
    try:
        actual_tables = _archive_table_digest(root / "source" / "tables.zip")
        if manifest.get("table_source_digest", actual_tables) != actual_tables:
            errors.append("captured table source differs from scientific identity")
        if manifest.get("table_digest", actual_tables) != actual_tables:
            errors.append("executing tables differ from captured source")
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        errors.append(f"invalid table archive: {error}")
    if expected_scientific_id != manifest.get("scientific_identity"):
        errors.append("scientific identity changed")
    return {
        "ok": not errors,
        "errors": errors,
        "status": manifest.get("status"),
        "scientific_identity": manifest.get("scientific_identity"),
        "uncommitted_checkpoint_revisions": sorted(orphan_revisions),
        "uncommitted_trace_chunks": sorted(orphan_traces),
    }


def export_record(record: str | Path, destination: str | Path) -> Path:
    """Create a portable ZIP after verifying every committed file."""
    root = Path(record)
    checked = verify_record(root)
    if not checked["ok"]:
        raise ValueError("record integrity failed: " + "; ".join(checked["errors"]))
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    manifest = _read_json(root / "manifest.json")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in [*sorted(manifest["files"]), "manifest.json"]:
            archive.write(root / relative, arcname=f"{root.name}/{relative}")
    os.replace(temporary, target)
    return target
