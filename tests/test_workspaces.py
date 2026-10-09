from __future__ import annotations

import pytest

from toylife2.specs import RunSpec
from toylife2.workspaces import init_workspace, list_records, new_record_path


def test_workspace_creation_and_record_naming(tmp_path):
    workspace = init_workspace(tmp_path / "research")
    assert (workspace / "workspace.json").is_file()
    assert list_records(workspace) == []
    destination = new_record_path(workspace, RunSpec(seed=1), "trial_1")
    assert destination == workspace / "runs" / "trial_1"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        new_record_path(workspace, RunSpec(seed=1), "trial_1")
    with pytest.raises(ValueError):
        new_record_path(workspace, RunSpec(seed=1), "../bad")
