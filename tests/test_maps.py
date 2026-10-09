from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

from toylife2.maps import _spatial_protocol, enumerate_map, merge_map_shards
from toylife2.reference import ReferenceModel
from toylife2.spatial import _morphogen, ca_from_descriptors
from toylife2.specs import ExecutionSpec, SpatialSpec
from toylife2.tables import Tables


def _key(descriptor: np.ndarray) -> str:
    return "".join(str(int(value)) for value in descriptor)


def test_weighted_pair_matches_all_256_reference_promoters(tmp_path) -> None:
    tables = Tables.load(cache_dir=tmp_path / "tables")
    execution = ExecutionSpec(batch_size=32, cache_dir=str(tmp_path / "qd"))
    spatial = SpatialSpec(cells=7, steps=6)
    assert _spatial_protocol(spatial) == _spatial_protocol(
        SpatialSpec(cells=7, steps=6, target="1111111")
    )
    left, right = 47, 49
    proteins = len(tables.protein_perimeter)
    pair = left * proteins + right
    result = enumerate_map(
        tables,
        execution,
        spatial=spatial,
        start_pair=pair,
        stop_pair=pair + 1,
        tile_pairs=1,
    )
    coding_to_protein = np.asarray(tables.gene_to_protein)
    count_left = int(np.count_nonzero(coding_to_protein == left))
    count_right = int(np.count_nonzero(coding_to_protein == right))
    representative_left = int(np.flatnonzero(coding_to_protein == left)[0])
    representative_right = int(np.flatnonzero(coding_to_protein == right)[0])
    weight = count_left * count_right
    reference = ReferenceModel(tables)
    logic: dict[str, int] = defaultdict(int)
    grn: dict[str, int] = defaultdict(int)
    ca: dict[str, int] = defaultdict(int)
    patterns: dict[str, int] = defaultdict(int)
    morphogen = _morphogen(spatial)
    for promoter_left in range(16):
        for promoter_right in range(16):
            genotype = np.asarray(
                [
                    (promoter_left << 16) | representative_left,
                    (promoter_right << 16) | representative_right,
                ],
                dtype=np.int32,
            )
            descriptor = reference.spatial_descriptor(genotype)
            logic[_key(descriptor[:6])] += weight
            grn[_key(descriptor)] += weight
            row = ca_from_descriptors(descriptor[None, :])[0]
            ca["".join(map(str, row))] += weight
            state = np.zeros(spatial.cells, dtype=np.int32)
            trajectory = np.zeros((spatial.steps, spatial.cells), dtype=np.int32)
            for tick in range(spatial.steps):
                incoming = state | morphogen[tick]
                next_state = np.zeros_like(state)
                for cell in range(spatial.cells):
                    neighbours = (
                        incoming[cell - 1] if cell else 0,
                        incoming[cell],
                        incoming[cell + 1] if cell + 1 < spatial.cells else 0,
                    )
                    next_state[cell] = row[neighbours[0] * 16 + neighbours[1] * 4 + neighbours[2]]
                trajectory[tick] = next_state
                state = next_state
            content = trajectory.astype("<i4").tobytes()
            pattern_id = hashlib.sha256(
                b"toylife2-pattern-v1\0" + _spatial_protocol(spatial).encode() + content
            ).hexdigest()
            patterns[pattern_id] += weight
    assert result["total_genotypes"] == 256 * weight
    assert result["logic_histogram"] == dict(logic)
    assert result["grn_histogram"] == dict(grn)
    assert result["ca_histogram"] == dict(ca)
    assert result["pattern_histogram"] == dict(patterns)


def test_weighted_shards_conserve_the_exact_pair_mass(tmp_path) -> None:
    tables = Tables.load(cache_dir=tmp_path / "tables")
    execution = ExecutionSpec(batch_size=32, cache_dir=str(tmp_path / "qd"))
    spatial = SpatialSpec(cells=7, steps=6)
    pair = 47 * len(tables.protein_perimeter) + 49
    together = enumerate_map(
        tables,
        execution,
        spatial=spatial,
        start_pair=pair,
        stop_pair=pair + 2,
        tile_pairs=2,
    )
    first = enumerate_map(
        tables,
        execution,
        spatial=spatial,
        start_pair=pair,
        stop_pair=pair + 1,
        tile_pairs=1,
    )
    second = enumerate_map(
        tables,
        execution,
        spatial=spatial,
        start_pair=pair + 1,
        stop_pair=pair + 2,
        tile_pairs=1,
    )
    assert together["total_genotypes"] == first["total_genotypes"] + second["total_genotypes"]
    for field in ("logic_histogram", "grn_histogram", "ca_histogram", "pattern_histogram"):
        merged: dict[str, int] = defaultdict(int)
        for shard in (first, second):
            for key, amount in shard[field].items():
                merged[key] += amount
        assert together[field] == dict(merged)
    assert merge_map_shards([second, first]) == together
    saved: list[dict[str, object]] = []
    partial = enumerate_map(
        tables,
        execution,
        spatial=spatial,
        start_pair=pair,
        stop_pair=pair + 2,
        tile_pairs=1,
        max_tiles=1,
        checkpoint_callback=saved.append,
    )
    assert partial["status"] == "partial"
    assert partial["next_pair"] == pair + 1
    resumed = enumerate_map(
        tables,
        execution,
        spatial=spatial,
        start_pair=pair,
        stop_pair=pair + 2,
        tile_pairs=2,
        resume_checkpoint=saved[-1],
    )
    assert resumed == together
    multiplicity = np.bincount(np.asarray(tables.gene_to_protein), minlength=2711)
    assert int(multiplicity.sum()) ** 2 * 256 == 1 << 40


def test_map_patterns_preserve_ab_under_later_explicit_injection(tmp_path) -> None:
    tables = Tables.load(cache_dir=tmp_path / "tables")
    execution = ExecutionSpec(cache_dir=str(tmp_path / "qd"))
    spatial = SpatialSpec(
        cells=1,
        steps=2,
        diffusion="none",
        morphogen="explicit",
        injections=((0, 0, 3), (1, 0, 2)),
    )
    pair = 47 * len(tables.protein_perimeter) + 49
    result = enumerate_map(
        tables,
        execution,
        spatial=spatial,
        start_pair=pair,
        stop_pair=pair + 1,
        tile_pairs=1,
    )
    coding_to_protein = np.asarray(tables.gene_to_protein)
    coding_a = int(np.flatnonzero(coding_to_protein == 47)[0])
    coding_b = int(np.flatnonzero(coding_to_protein == 49)[0])
    weight = int(np.count_nonzero(coding_to_protein == 47)) * int(
        np.count_nonzero(coding_to_protein == 49)
    )
    reference = ReferenceModel(tables)
    patterns: dict[str, int] = defaultdict(int)
    for promoter_a in range(16):
        for promoter_b in range(16):
            genotype = np.asarray(
                [
                    (promoter_a << 16) | coding_a,
                    (promoter_b << 16) | coding_b,
                ],
                dtype=np.int32,
            )
            descriptor = reference.spatial_descriptor(genotype)
            first = int(descriptor[3])
            second_input = 4 if first == 3 and descriptor[6] else first | 2
            trajectory = np.asarray([[first], [descriptor[second_input]]], dtype="<i4")
            key = hashlib.sha256(
                b"toylife2-pattern-v1\0"
                + _spatial_protocol(spatial).encode()
                + trajectory.tobytes()
            ).hexdigest()
            patterns[key] += weight
    assert result["pattern_histogram"] == dict(patterns)
