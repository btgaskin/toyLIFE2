from __future__ import annotations

import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from toylife2.tables import (
    DEFAULT_SOURCE,
    Tables,
    _array_checksum,
    _parsed_source_checksum,
    canonical_metabolite,
)


@pytest.fixture(scope="module")
def tables() -> Tables:
    return Tables.load()


def test_published_universe_and_perimeter_factorization(tables: Tables) -> None:
    assert tables.gene_to_protein.shape == (65536,)
    assert tables.protein_energy.shape == (2711,)
    assert tables.perimeter_sides.shape == (379, 8)
    assert tables.dimer_energy.shape == (20207,)
    assert len(tables.met_strings) == 274
    assert np.count_nonzero(tables.met_unbreakable) == 60
    assert tables.protein_promoter_energy.shape == (16, 379)
    assert tables.protein_met_energy.shape == (274, 379)
    assert canonical_metabolite("10000000") == "00000001"
    assert not tables.gene_to_protein.flags.writeable
    with pytest.raises(ValueError):
        tables.protein_energy[1] = 0


def test_geometry_retains_both_breaking_constituents(tables: Tables) -> None:
    assert tables.breaking_energy.shape[1:] == (379, 2)
    met = tables.met_index["00000001"]
    dimer = 1904
    assert tables.dimer_met_breakable[met, dimer]
    sequence = int(tables.dimer_met_seq[met, dimer])
    assert any(energy < 0 for _, energy in tables.breaking_options(sequence, 40))


def test_cache_roundtrip(tmp_path) -> None:
    first = Tables.load(cache_dir=tmp_path)
    second = Tables.load(cache_dir=tmp_path)
    assert first.digest == second.digest
    np.testing.assert_array_equal(first.breaking_energy, second.breaking_energy)
    assert second.met_index["0000"] == 0


@pytest.fixture(scope="module")
def cache_payload(tmp_path_factory):
    directory = tmp_path_factory.mktemp("table-cache")
    tables = Tables.load(cache_dir=directory)
    with np.load(directory / f"tables-{tables.digest}.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _copy_source(directory: Path) -> Path:
    for name in ("data", "promoters", "prot_metabolism", "dim_metabolism"):
        shutil.copytree(DEFAULT_SOURCE / name, directory / name, copy_function=os.link)
    return directory


@pytest.mark.parametrize("protein", [-1, 2711, 4294967297])
def test_source_rejects_invalid_coding_protein_ids(tmp_path, protein) -> None:
    source = _copy_source(tmp_path / "source")
    path = source / "data" / "protein_gene.txt"
    genes = path.read_text().splitlines()
    genes[0] = str(protein)
    path.unlink()  # The other source files share read-only hardlinks with the fixture.
    path.write_text("\n".join(genes) + "\n")
    with pytest.raises(ValueError, match="gene_to_protein.*invalid protein ID"):
        Tables.load(source)


def test_valid_custom_source_is_verified(tmp_path, tables) -> None:
    source = _copy_source(tmp_path / "source")
    path = source / "data" / "protein_gene.txt"
    genes = path.read_text().splitlines()
    genes[0] = "1"
    path.unlink()
    path.write_text("\n".join(genes) + "\n")
    custom = Tables.load(source, cache_dir=tmp_path / "cache")
    assert custom.gene_to_protein[0] == 1
    assert custom.digest != tables.digest
    custom.verify_source()
    with pytest.raises(ValueError, match="digest does not match"):
        tables.verify_source(source)


@pytest.mark.parametrize("name,change", [
    ("gene_to_protein", lambda a: a[:-1]),
    ("gene_to_protein", lambda a: a.astype(np.int64)),
    ("gene_to_protein", lambda a: np.full_like(a, 2711)),
    ("protein_perimeter", lambda a: np.full_like(a, -1)),
    ("perimeter_sides", lambda a: np.full_like(a, 16)),
    ("dimer_id", lambda a: np.full_like(a, -1)),
    ("breaking_seq_to_index", lambda a: np.full_like(a, -1)),
])
def test_tables_validate_synthetic_array_structure(tables, name, change) -> None:
    with pytest.raises(ValueError, match=name):
        replace(tables, **{name: change(getattr(tables, name))})


@pytest.mark.parametrize("name,change,rechecksum", [
    ("gene_to_protein", lambda a: a[:-1], True),
    ("gene_to_protein", lambda a: a.astype(np.float64), True),
    ("gene_to_protein", lambda a: np.full_like(a, 2711), True),
    ("dimer_id", lambda a: np.full_like(a, -1), True),
    ("breaking_seq_to_index", lambda a: np.full_like(a, -1), True),
    ("gene_to_protein", lambda a: np.where(np.arange(a.size) == 0, 1, a), False),
])
def test_corrupt_cache_rebuilds_from_source(
        tmp_path, tables, cache_payload, name, change, rechecksum) -> None:
    payload = dict(cache_payload)
    payload[name] = change(payload[name])
    if rechecksum:
        arrays = {key: value for key, value in payload.items() if key not in {
            "cache_format_version", "source_digest", "content_checksum",
        }}
        payload["content_checksum"] = np.asarray(_array_checksum(tables.digest, arrays))
    path = tmp_path / f"tables-{tables.digest}.npz"
    np.savez(path, **payload)
    loaded = Tables.load(cache_dir=tmp_path)
    np.testing.assert_array_equal(getattr(loaded, name), getattr(tables, name))
    loaded.verify_source()
    with np.load(path, allow_pickle=False) as archive:
        assert archive["content_checksum"].item() == loaded.content_checksum()


def test_valid_cache_load_uses_checksum_without_reparsing(
        tmp_path, tables, cache_payload, monkeypatch) -> None:
    np.savez(tmp_path / f"tables-{tables.digest}.npz", **cache_payload)

    def unexpected_parse(cls, source, digest):
        raise AssertionError("valid cached tables should load without parsing source text")

    monkeypatch.setattr(Tables, "_parse", classmethod(unexpected_parse))
    loaded = Tables.load(cache_dir=tmp_path)
    assert loaded.content_checksum() == tables.content_checksum()


def test_source_verification_rejects_cache_with_rewritten_checksum(
        tmp_path, tables, cache_payload) -> None:
    payload = dict(cache_payload)
    genes = payload["gene_to_protein"].copy()
    genes[0] = 1
    payload["gene_to_protein"] = genes
    arrays = {key: value for key, value in payload.items() if key not in {
        "cache_format_version", "source_digest", "content_checksum",
    }}
    payload["content_checksum"] = np.asarray(_array_checksum(tables.digest, arrays))
    np.savez(tmp_path / f"tables-{tables.digest}.npz", **payload)
    loaded = Tables.load(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="executing table arrays do not match"):
        loaded.verify_source()


def test_source_verification_checks_in_memory_values_and_reuses_parse(tables, monkeypatch) -> None:
    _parsed_source_checksum.cache_clear()
    parse = Tables._parse
    calls = []

    def counted_parse(cls, source, digest):
        calls.append(source)
        return parse(source, digest)

    monkeypatch.setattr(Tables, "_parse", classmethod(counted_parse))
    tables.verify_source()
    tables.verify_source()
    assert len(calls) == 1
    changed = tables.gene_to_protein.copy()
    changed[0] = 1
    synthetic = replace(tables, gene_to_protein=changed)
    with pytest.raises(ValueError, match="executing table arrays do not match"):
        synthetic.verify_source()
    # Frozen dataclasses and array flags do not replace value verification.
    changed.flags.writeable = True
    changed[0] = 2
    with pytest.raises(ValueError, match="executing table arrays do not match"):
        synthetic.verify_source()
