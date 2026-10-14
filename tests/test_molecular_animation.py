import json
import re

import numpy as np
import pytest

from toylife2.molecular_animation import render_molecular_animation


def fixture(root):
    config = dict(field="fluctuating", domain=16, modes=[[1, 0]], mean=0.5, amplitude=0.4)
    (root / "experiment.json").write_text(
        json.dumps(
            dict(
                task=dict(kind="chemotaxis", config=config),
                episode=dict(dt=0.1),
                selected_genomes=[[1, 2]],
            )
        )
    )
    arrays = dict(
        position=np.zeros((11, 2, 3, 2)),
        initial_position=np.zeros((1, 2, 3, 2)),
        phases=np.zeros((11, 2, 3, 1)),
        initial_phases=np.zeros((1, 2, 3, 1)),
        theta=np.zeros((11, 2, 3)),
        sensed_exposure=np.ones((11, 2, 3)) * 0.5,
        tumbling=np.zeros((11, 2, 3)),
        **{"input_custom</script>": np.ones((11, 2, 3)), "readout_swim": np.zeros((11, 2, 3))},
    )
    np.savez(root / "behaviour.npz", **arrays)
    return arrays


def payload(path):
    html = path.read_text()
    return json.loads(
        re.search(
            r'<script id="animation-data" type="application/json">(.*?)</script>', html
        ).group(1)
    )


def test_sampling_labels_and_single_lane(tmp_path):
    fixture(tmp_path)
    path = render_molecular_animation(tmp_path, frame_stride=5)
    data = payload(path)
    assert data["lanes"][0]["steps"] == [0, 5, 10, 11]
    assert len(data["lanes"][0]["position"]) == 4
    assert len(data["lanes"]) == 1
    assert "custom</script>" in data["lanes"][0]["cues"]
    assert "swim" in data["lanes"][0]["readouts"]
    assert "\\u003c/script>" in path.read_text()


def test_comparison(tmp_path):
    arrays = fixture(tmp_path)
    np.savez(tmp_path / "initial_behaviour.npz", **arrays)
    assert len(payload(render_molecular_animation(tmp_path))["lanes"]) == 2
    arrays["initial_position"] = np.ones((1, 2, 3, 2))
    np.savez(tmp_path / "initial_behaviour.npz", **arrays)
    with pytest.raises(ValueError, match="scenarios must match"):
        render_molecular_animation(tmp_path)


@pytest.mark.parametrize("stride", [0, -1, 1.5, True])
def test_bad_stride(tmp_path, stride):
    with pytest.raises(ValueError, match="positive integer"):
        render_molecular_animation(tmp_path, frame_stride=stride)


def test_missing_positions(tmp_path):
    fixture(tmp_path)
    np.savez(tmp_path / "behaviour.npz", actions=np.zeros(1))
    with pytest.raises(ValueError, match="chemotaxis positions"):
        render_molecular_animation(tmp_path)
