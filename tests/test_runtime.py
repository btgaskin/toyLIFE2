from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from toylife2.reference import ReferenceModel
from toylife2.runtime import Engine
from toylife2.specs import ExecutionSpec
from toylife2.tables import Tables


def test_preinitialized_cpu_accepts_native_arch_alias(tmp_path) -> None:
    # qd.cpu resolves to arm64 on Apple Silicon; compare enum identity, not text.
    script = """
import quadrants as qd
import sys
from toylife2._floating import preserve_host_floats
from toylife2.runtime import Engine
from toylife2.specs import ExecutionSpec
from toylife2.tables import Tables
with preserve_host_floats():
    qd.init(arch=qd.cpu, cpu_max_num_threads=1, enable_fallback=False,
            offline_cache_file_path=sys.argv[1])
tables = Tables.load()
try:
    Engine(tables, ExecutionSpec(backend='metal'))
except RuntimeError:
    pass
else:
    raise AssertionError('CPU program accepted as Metal')
engine = Engine(tables, ExecutionSpec(backend='cpu'))
assert engine.regulatory([0, 0]).complete
assert qd.lang.impl.current_cfg().cpu_max_num_threads == 1
"""
    cache = tmp_path / "qd"
    cache.mkdir()
    subprocess.run([sys.executable, "-c", script, str(cache)], check=True,
                   capture_output=True, text=True)


@pytest.fixture(scope="module")
def tables(tmp_path_factory: pytest.TempPathFactory) -> Tables:
    return Tables.load(cache_dir=tmp_path_factory.mktemp("tables"))


def test_regulatory_matches_scalar_and_preserves_batch_shape(tables: Tables, tmp_path) -> None:
    genes = np.random.default_rng(519).integers(0, 1 << 20, size=(2, 5, 3), dtype=np.int32)
    engine = Engine(
        tables, ExecutionSpec(batch_size=3, max_steps=32, cache_dir=str(tmp_path / "qd"))
    )
    result = engine.regulatory(genes)
    assert result.values.shape == (2, 5, 8)
    assert result.status.shape == (2, 5)
    assert result.complete
    oracle = ReferenceModel(tables)
    for index in np.ndindex(2, 5):
        np.testing.assert_array_equal(result.values[index], oracle.regulatory(genes[index]))


def test_metabolic_positive_and_cycle_match_scalar(tables: Tables, tmp_path) -> None:
    # Includes a real cleavage and a two-step intact metabolic cycle.
    genes = np.asarray(
        [
            [907050, 1048465, 769001],
            [1005689, 644498, 941775],
        ],
        dtype=np.int32,
    )
    environment = ["00010001", "00000011"]
    engine = Engine(
        tables, ExecutionSpec(batch_size=1, max_steps=16, cache_dir=str(tmp_path / "qd"))
    )
    result = engine.metabolic(genes, environment)
    oracle = ReferenceModel(tables)
    for index, genotype in enumerate(genes):
        values, status = oracle.metabolic(genotype, environment, max_steps=16)
        np.testing.assert_array_equal(result.values[index], values)
        np.testing.assert_array_equal(result.status[index], status)
    assert result.values[0, 0] == 1
    assert result.status[1, 1] == "complete"


def test_metabolic_marks_unfinished_attractor(tables: Tables, tmp_path) -> None:
    genes = np.asarray([[907050, 1048465, 769001]], dtype=np.int32)
    engine = Engine(tables, ExecutionSpec(max_steps=1, cache_dir=str(tmp_path / "qd")))
    result = engine.metabolic(genes, ["00010001"])
    values, status = ReferenceModel(tables).metabolic(genes[0], ["00010001"], max_steps=1)
    np.testing.assert_array_equal(result.values[0], values)
    np.testing.assert_array_equal(result.status[0], status)


def test_default_step_cap_uses_finite_state_history(tables: Tables, tmp_path) -> None:
    genotype = np.repeat(np.asarray([[907050, 1048465, 769001]], dtype=np.int32), 16, axis=0)
    engine = Engine(
        tables,
        ExecutionSpec(
            batch_size=16,
            memory_budget_mib=128,
            max_steps=100_000,
            cache_dir=str(tmp_path / "qd"),
        ),
    )
    result = engine.metabolic(genotype, ["00010001", "00000011", "01101010", "10100101"])
    assert result.values.shape == (16, 4)
    assert result.complete
    np.testing.assert_array_equal(result.values[:, 0], np.ones(16, dtype=np.uint8))


def test_four_gene_metabolic_batch_matches_reference(tables: Tables, tmp_path) -> None:
    genes = np.random.default_rng(1972).integers(0, 1 << 20, size=(8, 4), dtype=np.int32)
    metabolites = tuple(
        met
        for met, excluded in zip(tables.met_strings, tables.met_unbreakable, strict=True)
        if not excluded
    )[:5]
    engine = Engine(
        tables, ExecutionSpec(batch_size=3, max_steps=16, cache_dir=str(tmp_path / "qd"))
    )
    result = engine.metabolic(genes, metabolites)
    oracle = ReferenceModel(tables)
    for index, genotype in enumerate(genes):
        values, status = oracle.metabolic(genotype, metabolites, max_steps=16)
        np.testing.assert_array_equal(result.values[index], values)
        np.testing.assert_array_equal(result.status[index], status)


def test_table_pack_is_admitted_before_device_upload(tables: Tables, tmp_path) -> None:
    engine = Engine(tables, ExecutionSpec(memory_budget_mib=64, cache_dir=str(tmp_path / "qd")))
    genotype = np.asarray([[907050, 1048465, 769001]], dtype=np.int32)
    with pytest.raises(MemoryError, match="table pack"):
        engine.metabolic(genotype, ["00010001"])
    assert engine._table_bytes <= 64 * 1024 * 1024


def test_32_gene_perturbed_expression_preserves_distant_outputs(
    tables: Tables,
    tmp_path,
) -> None:
    base = [907050, 1048465, 769001]
    genotypes = np.asarray(
        [base + [0] * 29, base[:2] + [0] * 29 + [base[2]]],
        dtype=np.int32,
    )
    engine = Engine(
        tables,
        ExecutionSpec(
            max_steps=3,
            memory_budget_mib=512,
            cache_dir=str(tmp_path / "qd"),
        ),
    )
    result = engine.metabolic(genotypes, ["00010001"])
    reference = ReferenceModel(tables)
    for index, genotype in enumerate(genotypes):
        values, status = reference.metabolic(genotype, ["00010001"], max_steps=3)
        np.testing.assert_array_equal(result.values[index], values)
        np.testing.assert_array_equal(result.status[index], status)
    assert result.values[:, 0].tolist() == [1, 1]


def test_weaker_dimer_recomputes_capacity_after_stronger_consumption(
    tables: Tables,
    tmp_path,
) -> None:
    representatives = np.full(len(tables.protein_perimeter), -1, dtype=np.int32)
    for coding, protein in enumerate(tables.gene_to_protein):
        if representatives[protein] < 0:
            representatives[protein] = coding
    genotype = np.asarray(
        [
            (7 << 16) | representatives[53],
            representatives[53],
            representatives[913],
            representatives[390],
            representatives[390],
        ],
        dtype=np.int32,
    )
    reference = ReferenceModel(tables)
    state = reference._regulatory_state(genotype, 31)
    assert state.dimers[(53, 913)] == 1
    assert state.dimers[(53, 390)] == 1
    stale = state.copy()
    stale.dimers[(53, 390)] = 0
    stale.proteins[53] += 1
    stale.proteins[390] += 1
    assert reference.promoter_active(7, stale)
    assert not reference.promoter_active(7, state)
    engine = Engine(tables, ExecutionSpec(cache_dir=str(tmp_path / "qd")))
    np.testing.assert_array_equal(
        engine.regulatory(genotype).values, reference.regulatory(genotype)
    )
