"""Capture importable research components and their declared source/data inputs."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import io
import zipfile
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from .specs import RunSpec, digest


def resolve(entrypoint: str):
    module_name, separator, attribute = entrypoint.partition(":")
    if not separator or not module_name or not attribute or module_name == "__main__":
        raise ValueError("recorded components require an importable module:attribute entrypoint")
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise ValueError(f"component is not callable: {entrypoint}")
    return value


def declarations(spec: RunSpec) -> dict[str, dict[str, Any]]:
    result = {}
    for name in ("task", "objective"):
        component = getattr(spec, name)
        if component.entrypoint:
            result[name] = {"entrypoint": component.entrypoint, "config": component.config}
    for name in ("codec", "adapter"):
        entrypoint = spec.task.config.get(name + "_entrypoint")
        if entrypoint:
            result[name] = {"entrypoint": entrypoint, "config": spec.task.config}
    return result


def input_files(spec: RunSpec) -> dict[str, Path]:
    result = {"target": Path(spec.spatial.target)} if spec.spatial.target else {}
    for name in ("task", "objective"):
        declared = getattr(spec, name).config.get("inputs", {})
        if not isinstance(declared, Mapping):
            raise ValueError("component config.inputs must map names to file paths")
        for key, value in declared.items():
            artifact = f"{name}_{key}"
            if not key or not all(c.isalnum() or c in "_-" for c in key):
                raise ValueError("component input names use letters, digits, '_' or '-'")
            result[artifact] = Path(value)
    return result


def capture(spec: RunSpec) -> tuple[bytes, dict[str, Any]] | None:
    entries = declarations(spec)
    if not entries:
        return None
    files: dict[str, bytes] = {}
    descriptors = {}
    for name, declaration in entries.items():
        entrypoint = declaration["entrypoint"]
        function = resolve(entrypoint)
        module_name = entrypoint.partition(":")[0]
        modules = [module_name, *declaration["config"].get("source_modules", ())]
        for module_name in modules:
            module = importlib.import_module(module_name)
            origin = inspect.getsourcefile(module)
            if origin is None:
                raise ValueError(
                    f"component module requires a capturable source file: {module_name}"
                )
            source = Path(origin).resolve()
            top_name = module_name.split(".")[0]
            top = importlib.import_module(top_name)
            if hasattr(top, "__path__"):
                for root in top.__path__:
                    package = Path(root).resolve()
                    for item in sorted(package.rglob("*.py")):
                        if "__pycache__" not in item.parts:
                            files[top_name + "/" + item.relative_to(package).as_posix()] = (
                                item.read_bytes()
                            )
            else:
                files[module_name.replace(".", "/") + ".py"] = source.read_bytes()
        dependencies = {}
        for distribution_name in declaration["config"].get("dependencies", ()):
            distribution = metadata.distribution(distribution_name)
            record = distribution.read_text("RECORD")
            dependencies[distribution_name] = {
                "version": distribution.version,
                "record_sha256": hashlib.sha256(record.encode()).hexdigest() if record else None,
            }
        descriptors[name] = {
            "entrypoint": entrypoint,
            "dependencies": dependencies,
            "callable_module": function.__module__,
        }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, contents in sorted(files.items()):
            item = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o644 << 16
            archive.writestr(item, contents)
    payload = buffer.getvalue()
    descriptors["archive_sha256"] = hashlib.sha256(payload).hexdigest()
    descriptors["identity"] = digest(descriptors)
    return payload, descriptors


def verify_current(spec: RunSpec, expected: Mapping[str, Any]) -> None:
    actual = capture(spec)
    if actual is None or actual[1] != expected:
        raise ValueError("executed research components or declared dependencies differ from record")
