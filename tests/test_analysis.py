import numpy as np
import pytest

from toylife2.analysis import bits, complexity, dictionary_count, pattern_summary, temporal_period
from toylife2.specs import SpatialSpec


def test_author_complexity_terminal_and_constants():
    assert complexity("0") == 0
    assert complexity("00") == 1
    assert dictionary_count("01") == dictionary_count("10") == 3
    assert complexity("01") == 3
    assert complexity("0101101001") == complexity("1001011010")
    with pytest.raises(ValueError):
        complexity("")
    assert bits(np.array([[0, 1], [2, 3]])) == "00011011"


def test_period_uses_complete_tissue_state():
    assert temporal_period(np.array([[0, 0], [1, 0], [1, 1], [1, 0]])) == {
        "transient": 1, "period": 2}
    assert temporal_period(np.array([[0, 0], [1, 0]]))["period"] is None


def test_future_injection_prevents_false_autonomous_cycle():
    spec = SpatialSpec(cells=1, steps=3, morphogen="explicit", injections=((2, 0, 1),))
    result = pattern_summary(np.array([[[0], [0], [1]]]), spec)
    assert result["period_autonomous_start"] == 2
    assert result["periods"] == [{"period": None, "transient": None}]
