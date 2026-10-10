from __future__ import annotations

import numpy as np

from toylife2.reference import ReferenceModel
from toylife2.runtime import Engine
from toylife2.spatial import (
    _morphogen,
    _transported_state,
    ca_from_descriptors,
    compile_ca,
    pair_promoter_matrix,
    patterns,
    response_descriptors,
)
from toylife2.specs import ExecutionSpec, SpatialSpec
from toylife2.tables import Tables


def test_preformed_ab_survives_stronger_aa(tmp_path) -> None:
    tables = Tables.load(cache_dir=tmp_path / "tables")
    genotype = np.asarray([[33177, 950375]], dtype=np.int32)
    a = int(tables.gene_to_protein[genotype[0, 0] & 65535])
    b = int(tables.gene_to_protein[genotype[0, 1] & 65535])
    aa = tables.dimer_for(a, a)
    ab = tables.dimer_for(a, b)
    assert aa >= 0 and ab >= 0
    assert int(tables.dimer_energy[aa]) + 2 * int(tables.protein_energy[a]) < (
        int(tables.dimer_energy[ab]) + int(tables.protein_energy[a]) + int(tables.protein_energy[b])
    )
    descriptor = response_descriptors(
        tables, genotype, ExecutionSpec(cache_dir=str(tmp_path / "qd"))
    )[0]
    np.testing.assert_array_equal(
        descriptor, ReferenceModel(tables).spatial_descriptor(genotype[0])
    )
    assert descriptor[4] == 0
    assert descriptor[6] == 1


def test_ca_and_tissue_steps_are_synchronous(tmp_path) -> None:
    tables = Tables.load(cache_dir=tmp_path / "tables")
    genotype = np.asarray([[33177, 950375], [907050, 1048465]], dtype=np.int32)
    execution = ExecutionSpec(batch_size=1, cache_dir=str(tmp_path / "qd"))
    spec = SpatialSpec(cells=7, steps=6, boundary="open", diffusion="A")
    ca = compile_ca(tables, genotype, execution=execution)
    result = patterns(tables, genotype, spec, execution)
    assert ca.shape == (2, 64)
    assert result.values.shape == (2, 6, 7)
    morphogen = _morphogen(spec)
    expected = np.zeros_like(result.values)
    for row in range(2):
        state = np.zeros(spec.cells, dtype=np.int32)
        for t in range(spec.steps):
            next_state = np.zeros_like(state)
            for c in range(spec.cells):
                neighbours = []
                for position in (c - 1, c, c + 1):
                    if position < 0 or position >= spec.cells:
                        neighbours.append(0)
                        continue
                    neighbours.append(int(state[position]) | int(morphogen[t, position]))
                code = neighbours[0] * 16 + neighbours[1] * 4 + neighbours[2]
                next_state[c] = ca[row, code]
            expected[row, t] = next_state
            state = next_state
    np.testing.assert_array_equal(result.values, expected)


def test_all_diffusion_modes_preserve_molecular_identity() -> None:
    def molecules(value: int, dimer: bool) -> set[str]:
        if value == 0:
            return set()
        if value == 1:
            return {"B"}
        if value == 2:
            return {"A"}
        if value == 3:
            return {"AB"} if dimer else {"A", "B"}
        if value == 4:
            return {"AB", "A"}
        return {"AB", "B"}

    def independent(left: int, centre: int, right: int, dimer: bool, mode: str) -> int:
        local = molecules(centre, dimer)
        travelling = {
            "none": set(),
            "A": {"A"},
            "B": {"B"},
            "both": {"A", "B"},
            "all": {"A", "B", "AB"},
        }[mode]
        local.update((molecules(left, dimer) | molecules(right, dimer)) & travelling)
        if "AB" in local:
            if {"A", "B"} <= local:
                return 3
            if "A" in local:
                return 4
            if "B" in local:
                return 5
            return 3
        return (2 if "A" in local else 0) + (1 if "B" in local else 0)

    for mode in ("none", "A", "B", "both", "all"):
        for dimer in (False, True):
            descriptor = np.arange(7, dtype=np.int32)
            descriptor[6] = int(dimer)
            ca = ca_from_descriptors(descriptor[None], mode)[0]
            for left in range(4):
                for centre in range(4):
                    for right in range(4):
                        expected = independent(left, centre, right, dimer, mode)
                        assert _transported_state(left, centre, right, dimer, mode) == expected
                        assert ca[left * 16 + centre * 4 + right] == descriptor[expected]
            if dimer:
                extended = ca_from_descriptors(
                    descriptor[None],
                    mode,
                    input_states=6,
                )[0]
                for left in range(6):
                    for centre in range(6):
                        for right in range(6):
                            expected = independent(left, centre, right, dimer, mode)
                            assert (
                                _transported_state(
                                    left,
                                    centre,
                                    right,
                                    dimer,
                                    mode,
                                )
                                == expected
                            )
                            assert extended[left * 36 + centre * 6 + right] == descriptor[expected]
    assert _transported_state(3, 0, 0, True, "both") == 0
    assert _transported_state(3, 0, 3, True, "all") == 3


def test_explicit_initial_state_and_combined_injections(tmp_path) -> None:
    tables = Tables.load(cache_dir=tmp_path / "tables")
    genotype = np.asarray([[33177, 950375]], dtype=np.int32)
    spec = SpatialSpec(
        cells=3,
        steps=4,
        boundary="ring",
        diffusion="all",
        morphogen="explicit",
        initial_state=(1, 0, 2),
        injections=((0, 0, 2), (0, 0, 1), (1, 2, 1)),
    )
    morphogen = _morphogen(spec)
    assert morphogen.tolist() == [[3, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]
    execution = ExecutionSpec(cache_dir=str(tmp_path / "qd"))
    ca = compile_ca(tables, genotype, "all", execution)[0]
    result = patterns(tables, genotype, spec, execution).values[0]
    state = np.asarray(spec.initial_state, dtype=np.int32)
    expected = np.empty_like(result)
    for tick in range(spec.steps):
        incoming = state | morphogen[tick]
        next_state = np.empty_like(state)
        for cell in range(spec.cells):
            left = int(incoming[(cell - 1) % spec.cells])
            center = int(incoming[cell])
            right = int(incoming[(cell + 1) % spec.cells])
            next_state[cell] = ca[left * 16 + center * 4 + right]
        expected[tick] = next_state
        state = next_state
    np.testing.assert_array_equal(result, expected)


def test_pair_matrix_all_promoters_matches_fixed_inventory_reference(tmp_path) -> None:
    tables = Tables.load(cache_dir=tmp_path / "tables")
    pairs = np.asarray(
        [(0, 0), (47, 49), (47, 1), (1, 47), (47, 47), (100, 200)],
        dtype=np.int32,
    )
    engine = Engine(tables, ExecutionSpec(cache_dir=str(tmp_path / "qd")))
    responses, present = pair_promoter_matrix(tables, pairs, engine=engine)
    representatives = np.full(len(tables.protein_perimeter), -1, dtype=np.int32)
    for coding, protein in enumerate(tables.gene_to_protein):
        if representatives[protein] < 0:
            representatives[protein] = coding
    reference = ReferenceModel(tables)
    for index, (left, right) in enumerate(pairs):
        for promoter in range(16):
            genotype = np.asarray(
                [
                    (promoter << 16) | representatives[left],
                    (promoter << 16) | representatives[right],
                ],
                dtype=np.int32,
            )
            descriptor = reference.spatial_descriptor(genotype)
            assert present[index] == descriptor[6]
            if left > 0:
                np.testing.assert_array_equal(
                    responses[index, promoter] << 1,
                    descriptor[:6] & 2,
                )
            if right > 0:
                np.testing.assert_array_equal(
                    responses[index, promoter],
                    descriptor[:6] & 1,
                )


def test_lookup_backed_parity_rule_counterexamples(tmp_path) -> None:
    """Real source-table cases pin the rules isolated by the full-map experiment."""
    tables = Tables.load(cache_dir=tmp_path / "tables")
    cases = [
        # A valid repressor wins over a more stable protein with no promoter contact.
        (47, 1303, 14, [1, 1, 0, 0, 1, 0]),
        # A valid activator wins over a more stable nonbinding protein.
        (47, 180, 3, [0, 1, 0, 1, 1, 1]),
        # Identical, nondimerizing activators remain one competing identity.
        (174, 174, 3, [0, 1, 1, 1, 1, 1]),
        # Dimer/polymerase annihilation leaves A repressing in AB+A and B activating in AB+B.
        (90, 2350, 5, [1, 1, 1, 0, 0, 1]),
        # AB+A and AB+B have separate outputs (the author's missing-break example).
        (47, 264, 3, [0, 1, 0, 0, 0, 1]),
    ]
    pairs = np.asarray([(a, b) for a, b, _, _ in cases], dtype=np.int32)
    engine = Engine(tables, ExecutionSpec(cache_dir=str(tmp_path / "qd")))
    responses, _ = pair_promoter_matrix(tables, pairs, engine=engine)
    reference = ReferenceModel(tables)
    genotypes = []
    for index, (a, b, promoter, expected) in enumerate(cases):
        np.testing.assert_array_equal(responses[index, promoter], expected)
        coding = [int(np.flatnonzero(tables.gene_to_protein == x)[0]) for x in (a, b)]
        genotype = np.asarray([(promoter << 16) | c for c in coding], dtype=np.int32)
        genotypes.append(genotype)
        np.testing.assert_array_equal(reference.spatial_descriptor(genotype)[:6],
                                      np.asarray(expected) * 3)
    np.testing.assert_array_equal(
        response_descriptors(tables, np.stack(genotypes), engine=engine)[:, :6],
        np.asarray([expected for _, _, _, expected in cases]) * 3,
    )
    assert tables.dimer_for(174, 174) == -1
    tied_dimer = tables.dimer_for(90, 2350)
    assert (int(tables.dimer_energy[tied_dimer]) + int(tables.protein_energy[90])
            + int(tables.protein_energy[2350])
            + int(tables.dimer_promoter_energy[5, tied_dimer])) == -150
    assert int(tables.polymerase_energy[5]) == -150


def test_later_injection_enters_preformed_ab_state_four(tmp_path) -> None:
    tables = Tables.load(cache_dir=tmp_path / "tables")
    coding_a = 33177 & 65535
    coding_b = 950375 & 65535
    genotype = np.asarray([[(14 << 16) | coding_a, (14 << 16) | coding_b]], dtype=np.int32)
    descriptor = ReferenceModel(tables).spatial_descriptor(genotype[0])
    assert descriptor[3] == 3 and descriptor[4] == 0 and descriptor[6] == 1
    spec = SpatialSpec(
        cells=1,
        steps=2,
        diffusion="none",
        morphogen="explicit",
        injections=((0, 0, 3), (1, 0, 2)),
    )
    result = patterns(tables, genotype, spec, ExecutionSpec(cache_dir=str(tmp_path / "qd")))
    assert result.values[0, :, 0].tolist() == [3, 0]
    continuous = SpatialSpec(
        cells=1,
        steps=1,
        diffusion="none",
        morphogen="central_A_continuous",
        initial_state=(3,),
    )
    continuous_result = patterns(
        tables, genotype, continuous, ExecutionSpec(cache_dir=str(tmp_path / "qd"))
    )
    assert continuous_result.values[0, 0, 0] == descriptor[4]
