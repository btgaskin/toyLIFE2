from __future__ import annotations

import tomllib

import pytest

from toylife2.config import apply_overrides, dump_spec, load_spec
from toylife2.specs import RunSpec, SpatialSpec, from_mapping, scientific_identity


def test_toml_round_trip_and_scientific_identity(tmp_path):
    spec = RunSpec(seed=42, genotypes=("0" * 40,))
    path = tmp_path / "run.toml"
    path.write_text(dump_spec(spec), encoding="utf-8")
    assert load_spec(path) == spec
    assert tomllib.loads(path.read_text(encoding="utf-8"))["seed"] == 42
    changed = apply_overrides(spec, ['execution.backend="metal"', "execution.batch_size=32"])
    assert changed.execution.backend == "metal"
    assert scientific_identity(changed) == scientific_identity(spec)


def test_unknown_and_invalid_fields_fail():
    with pytest.raises(ValueError, match="unknown RunSpec fields"):
        from_mapping({"seed": 1, "mystery": 2})
    with pytest.raises(ValueError, match="unknown ExecutionSpec fields"):
        from_mapping({"seed": 1, "execution": {"mystery": 2}})
    with pytest.raises(ValueError, match="unknown override path"):
        apply_overrides(RunSpec(seed=1), ["model.secret=3"])
    with pytest.raises(ValueError, match="TOML literal"):
        apply_overrides(RunSpec(seed=1), ["samples=undefined"])


def test_target_path_is_a_location_not_scientific_content():
    first = RunSpec(seed=3, operation="sswm", spatial=SpatialSpec(target="a.npy"))
    second = RunSpec(seed=3, operation="sswm", spatial=SpatialSpec(target="other/a.npy"))
    assert scientific_identity(first) == scientific_identity(second)
