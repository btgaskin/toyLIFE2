"""Strict, portable TOML configuration for toyLIFE experiments."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tomli_w

from .specs import RunSpec, from_mapping, plain


def load_spec(path: str | Path) -> RunSpec:
    """Read and validate a complete experiment definition."""
    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    return from_mapping(data)


def dump_spec(spec: RunSpec) -> str:
    data = plain(spec)
    # TOML has no null. Missing optional fields reconstruct dataclass defaults.
    def omit_null(value):
        if isinstance(value, dict):
            return {key: omit_null(item) for key, item in value.items() if item is not None}
        if isinstance(value, list):
            return [omit_null(item) for item in value]
        return value

    return tomli_w.dumps(omit_null(data))


def write_spec(spec: RunSpec, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_spec(spec), encoding="utf-8")
    return target


def _parse_override(raw: str) -> tuple[list[str], Any]:
    key, separator, value = raw.partition("=")
    if not separator or not key:
        raise ValueError(f"override must be dotted.path=value: {raw!r}")
    parts = key.split(".")
    if any(not part.isidentifier() for part in parts):
        raise ValueError(f"invalid override path: {key!r}")
    try:
        parsed = tomllib.loads("value = " + value + "\n")["value"]
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"override value must be a TOML literal: {raw!r}") from error
    return parts, parsed


def apply_overrides(spec: RunSpec, overrides: list[str] | tuple[str, ...]) -> RunSpec:
    """Return a new validated specification; an override is part of the saved run."""
    data = plain(spec)
    for raw in overrides:
        parts, value = _parse_override(raw)
        parent: dict[str, Any] = data
        for part in parts[:-1]:
            child = parent.get(part)
            if not isinstance(child, dict):
                raise ValueError(f"unknown override path: {'.'.join(parts)}")
            parent = child
        if parts[-1] not in parent:
            raise ValueError(f"unknown override path: {'.'.join(parts)}")
        parent[parts[-1]] = value
    return from_mapping(data)


def mapping_spec(data: Mapping[str, Any]) -> RunSpec:
    return from_mapping(dict(data))
