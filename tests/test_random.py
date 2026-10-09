import numpy as np

from toylife2.random import Stream, philox


def test_random123_known_vectors():
    np.testing.assert_array_equal(philox(np.zeros(4, np.uint32), np.zeros(2, np.uint32)),
                                  [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8])
    np.testing.assert_array_equal(philox(np.full(4, 0xFFFFFFFF, np.uint32),
                                        np.full(2, 0xFFFFFFFF, np.uint32)),
                                  [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD])


def test_draws_are_identical_across_chunks_and_reordering():
    stream = Stream(913)
    ids = np.arange(117, dtype=np.uint64)
    options = dict(role="parents", trial=5, event=1001)
    upper = (1 << 63) + 123  # many rejection attempts
    full = stream.integers(upper, ids, **options)
    chunked = np.concatenate([stream.integers(upper, part, **options)
                              for part in np.array_split(ids, 9)])
    np.testing.assert_array_equal(full, chunked)
    np.testing.assert_array_equal(full[::-1], stream.integers(upper, ids[::-1], **options))
    assert np.all(full < upper)
    uniforms = stream.uniform(ids, **options)
    assert np.all((uniforms > 0) & (uniforms < 1))
    assert not np.array_equal(full, stream.integers(upper, ids, **{**options, "trial": 6}))


def test_uniform_maximum_counter_value_is_strictly_open(monkeypatch):
    stream = Stream(0)
    monkeypatch.setattr(stream, "words", lambda ids, **kwargs: np.full((4,), 0xFFFFFFFF,
                                                                    dtype=np.uint32))
    assert 0 < float(stream.uniform(0, role="test")) < 1
