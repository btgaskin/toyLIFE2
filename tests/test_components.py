import dataclasses
import importlib
import json
import sys
import zipfile

import numpy as np
import pytest

from toylife2.components import capture, verify_current
from toylife2.protocols import EpisodeSpec, TaskSpec, WorkLimits
from toylife2.records import inspect_record, verify_record
from toylife2.runner import resume, run
from toylife2.specs import ExecutionSpec, RunSpec


@pytest.fixture
def component(tmp_path, monkeypatch):
    package = tmp_path / "toy_custom"
    package.mkdir()
    (package / "__init__.py").write_text("from .world import factory\n")
    (package / "helper.py").write_text("def reward(actions):\n    return actions * 0 + 0.25\n")
    (package / "world.py").write_text("""import numpy as np
from .helper import reward
class World:
    def __init__(self, **kw): self.shape = kw["shape"]
    def reset(self): return {"tick": 0, "dose": np.zeros(self.shape, np.int32)}
    def observe(self, state): return {"dose": state["dose"]}
    def advance(self, state, actions):
        return {**state, "tick": state["tick"] + 1}, reward(actions), {}
def factory(**kwargs): return World(**kwargs)
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield package
    for name in list(sys.modules):
        if name == "toy_custom" or name.startswith("toy_custom."):
            sys.modules.pop(name)


def test_source_helpers_data_captured_and_relocated_resume(component, tmp_path):
    data = tmp_path / "stimuli.npy"
    np.save(data, [1, 2, 3])
    spec = RunSpec(
        schema_version=2,
        operation="rollout",
        seed=4,
        task=TaskSpec(
            kind="python",
            entrypoint="toy_custom:factory",
            config={"inputs": {"stimuli": str(data)}},
        ),
        episode=EpisodeSpec(horizon=2),
        limits=WorkLimits(max_world_ticks=1),
        execution=ExecutionSpec(cache_dir=str(tmp_path / "cache")),
    )
    record = run(spec, destination=tmp_path / "run")
    assert verify_record(record)["ok"]
    with zipfile.ZipFile(record / "source" / "components.zip") as archive:
        assert "toy_custom/helper.py" in archive.namelist()
    data.unlink()
    continued = resume(record, destination=tmp_path / "child", limits=WorkLimits())
    assert inspect_record(continued)["status"] == "complete"
    assert verify_record(continued)["ok"]
    manifest = inspect_record(record)
    (component / "helper.py").write_text("def reward(actions):\n    return actions * 0 + 0.5\n")
    with pytest.raises(ValueError, match="components"):
        verify_current(spec, manifest["components"])


def test_component_input_paths_do_not_change_scientific_spec_identity(component, tmp_path):
    from toylife2.specs import scientific_identity

    a = RunSpec(
        schema_version=2,
        seed=1,
        task=TaskSpec(
            kind="python", entrypoint="toy_custom:factory", config={"inputs": {"stimuli": "a.npy"}}
        ),
    )
    b = dataclasses.replace(
        a, task=dataclasses.replace(a.task, config={"inputs": {"stimuli": "b.npy"}})
    )
    assert scientific_identity(a) == scientific_identity(b)
    payload, metadata = capture(a)
    assert len(payload) and json.dumps(metadata)


def test_recorded_external_pending_optimizer_continuation(component, tmp_path):
    from toylife2.protocols import ObjectiveSpec
    from toylife2.specs import EvolutionSpec, RecordingSpec

    (component / "optimizer.py").write_text("""import numpy as np
from toylife2.adapters import CandidateBatch
class Adapter:
    def __init__(self, config, seed): self.asks = 0; self.tells = 0
    def ask(self):
        self.asks += 1
        return CandidateBatch(self.asks, np.zeros((2, 2), np.int32))
    def tell(self, batch, result): self.tells += 1
    def state_dict(self): return {"asks": self.asks, "tells": self.tells}
    def load_state_dict(self, state): self.asks = state["asks"]; self.tells = state["tells"]
def factory(**kwargs): return Adapter(**kwargs)
""")
    importlib.invalidate_caches()
    spec = RunSpec(
        schema_version=2,
        operation="external",
        seed=1,
        task=TaskSpec(
            kind="temporal_probe", config={"adapter_entrypoint": "toy_custom.optimizer:factory"}
        ),
        objective=ObjectiveSpec(kind="bounded_reward"),
        episode=EpisodeSpec(horizon=2),
        recording=RecordingSpec(mode="full"),
        evolution=EvolutionSpec(generations=2),
        limits=WorkLimits(max_evolution_updates=1),
        execution=ExecutionSpec(cache_dir=str(tmp_path / "cache")),
    )
    partial = run(spec, destination=tmp_path / "external")
    assert inspect_record(partial)["status"] == "partial"
    child = resume(partial, destination=tmp_path / "child", limits=WorkLimits())
    assert inspect_record(child)["status"] == "complete"
    from toylife2.records import read_checkpoint
    from toylife2.sessions import Checkpoint

    cp = Checkpoint.from_record(read_checkpoint(child))
    assert cp.payload["payload"]["optimizer"] == {"asks": 2, "tells": 2}
    assert verify_record(child)["ok"]
    rows = []
    for record in (partial, child):
        for relative in inspect_record(record).get("trace", {}).get("chunks", []):
            rows.extend(json.loads(line) for line in (record / relative).read_text().splitlines())
    nested = [row["event"] for row in rows if row["type"] == "rollout_transition"]
    assert sum(row["type"] == "world_tick" for row in nested) == 2
    assert sum(row["type"] == "chemical_update" for row in nested) == 2
    assert sum(row["type"] == "external_tell" for row in rows) == 2
