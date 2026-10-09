"""Immutable, integer-energy representation of the published toyLIFE lookup universe.

The source text files are part of the model. Energies are stored in tenths, so
all equality and ordering decisions are exact. Zero denotes a missing bond.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType

import numpy as np

_PACKAGE_SOURCE = Path(__file__).resolve().parent / "_vendor" / "toylife"
_CHECKOUT_SOURCE = Path(__file__).resolve().parents[2] / "upstream" / "toylife"
DEFAULT_SOURCE = _PACKAGE_SOURCE if _PACKAGE_SOURCE.is_dir() else _CHECKOUT_SOURCE
_CACHE_VERSION = 2


def _tenths(value: str) -> int:
    number = float(value)
    scaled = round(number * 10)
    if abs(number * 10 - scaled) > 1e-6:
        raise ValueError(f"energy is not an exact tenth: {value}")
    return scaled


def canonical_metabolite(sequence: str) -> str:
    """Canonicalize reversal-equivalent toyMetabolites, including fragments."""
    if not sequence or set(sequence) - {"0", "1"}:
        raise ValueError("a metabolite is a nonempty binary string")
    return min(sequence, sequence[::-1])


def _source_files(source: Path) -> list[Path]:
    names = [source / "data" / name for name in (
        "protein_gene.txt", "proteins_with_perimeter.txt", "proteins.txt",
        "perimeters.txt", "dimers.txt", "metabolites.txt",
    )]
    names.extend(sorted((source / "promoters").glob("*_bind_prom*.txt")))
    names.extend(sorted((source / "prot_metabolism").glob("[4-8]_*.txt")))
    names.extend(sorted((source / "dim_metabolism").glob("[4-8]_*.txt")))
    names.append(source / "prot_metabolism" / "prot_breaking.txt")
    if any(not path.is_file() for path in names) or len(names) < 300:
        raise FileNotFoundError(f"incomplete toyLIFE source tables: {source}")
    return names


def _source_digest(source: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(f"canonical-tenths-{_CACHE_VERSION}\n".encode())
    for path in paths:
        digest.update(str(path.relative_to(source)).encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _contact_energy() -> np.ndarray:
    """Energy of two facing four-site sides (the second side is reversed)."""
    result = np.zeros((16, 16), dtype=np.int16)
    for first in range(16):
        for second in range(16):
            energy = 0
            for position in range(4):
                a = (first >> position) & 1
                b = (second >> (3 - position)) & 1
                energy -= 20 if a and b else 3 if a or b else 0
            result[first, second] = energy
    return result


def _breaking_geometry(sequences: np.ndarray, sides: np.ndarray) -> np.ndarray:
    """Find unique protein-side attachments to each constituent of a bound dimer.

    The upstream ``prot_breaking`` table retains only the strongest interaction
    over both constituents. A weaker interaction with the other constituent can
    give a more stable total complex after its folding energy is included. We
    therefore retain one unique best contact per constituent.
    """
    contact = _contact_energy()
    unique = np.zeros(sides.shape, dtype=bool)
    for perimeter, row in enumerate(sides):
        seen: set[int] = set()
        for index, side in enumerate(row):
            if int(side) not in seen:
                unique[perimeter, index] = True
                seen.add(int(side))
    output = np.zeros((len(sequences), len(sides), 2), dtype=np.int16)
    for index, sequence in enumerate(sequences):
        value = int(sequence)
        target = np.array([(value >> shift) & 15 for shift in (12, 8, 4, 0)],
                          dtype=np.uint8)
        scores = contact[sides[:, :, None], target[None, None, :]]
        scores = np.where(unique[:, :, None], scores, 100)
        for constituent in range(2):
            half = scores[:, :, constituent * 2:constituent * 2 + 2]
            minimum = np.min(half, axis=(1, 2))
            singular = np.sum(half == minimum[:, None, None], axis=(1, 2)) == 1
            output[index, :, constituent] = np.where(
                singular & (minimum < -26), minimum, 0,
            )
    return output


@dataclass(frozen=True)
class Tables:
    gene_to_protein: np.ndarray
    protein_perimeter: np.ndarray
    protein_energy: np.ndarray
    perimeter_sides: np.ndarray
    dimer_id: np.ndarray
    dimer_energy: np.ndarray
    dimer_perimeter_a: np.ndarray
    dimer_perimeter_b: np.ndarray
    protein_promoter_energy: np.ndarray
    protein_promoter_pol: np.ndarray
    dimer_promoter_energy: np.ndarray
    dimer_promoter_pol: np.ndarray
    polymerase_energy: np.ndarray
    met_strings: tuple[str, ...]
    met_unbreakable: np.ndarray
    protein_met_energy: np.ndarray
    dimer_met_energy: np.ndarray
    dimer_met_seq: np.ndarray
    dimer_met_pos: np.ndarray
    dimer_met_orientation: np.ndarray
    dimer_met_breakable: np.ndarray
    breaking_sequences: np.ndarray
    breaking_seq_to_index: np.ndarray
    breaking_energy: np.ndarray
    digest: str
    source: str
    met_index: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for data_field in fields(self):
            if data_field.name == "met_index":
                continue
            value = getattr(self, data_field.name)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        object.__setattr__(self, "met_index", MappingProxyType(
            {sequence: index for index, sequence in enumerate(self.met_strings)}))

    @classmethod
    def load(cls, source: str | Path | None = None,
             cache_dir: str | Path | None = None) -> Tables:
        directory = Path(source) if source is not None else DEFAULT_SOURCE
        directory = directory.expanduser().resolve()
        source_files = _source_files(directory)
        digest = _source_digest(directory, source_files)
        cache_path = None
        if cache_dir is not None:
            cache_root = Path(cache_dir).expanduser().resolve()
            cache_path = cache_root / f"tables-{digest}.npz"
            if cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as archive:
                    values = {name: archive[name] for name in archive.files}
                met_strings = tuple(str(s) for s in values.pop("met_strings"))
                return cls(**values, met_strings=met_strings, digest=digest,
                           source=str(directory))
        tables = cls._parse(directory, digest)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
            arrays = {field.name: getattr(tables, field.name) for field in fields(tables)
                      if isinstance(getattr(tables, field.name), np.ndarray)}
            arrays["met_strings"] = np.asarray(tables.met_strings)
            try:
                with temp_path.open("wb") as stream:
                    np.savez(stream, **arrays)
                os.replace(temp_path, cache_path)
            finally:
                temp_path.unlink(missing_ok=True)
        return tables

    @classmethod
    def _parse(cls, source: Path, digest: str) -> Tables:
        data = source / "data"
        genes = np.fromfile(data / "protein_gene.txt", dtype=np.int32, sep=" ")
        if genes.shape != (65536,):
            raise ValueError("protein_gene.txt must contain 65,536 rows")
        rows = [tuple(map(int, line.split())) for line in
                (data / "proteins_with_perimeter.txt").open()]
        if [row[0] for row in rows] != list(range(len(rows))):
            raise ValueError("protein IDs must be contiguous")
        protein_perimeter = np.asarray([row[1] for row in rows], dtype=np.int16)
        protein_energy = np.zeros(len(rows), dtype=np.int16)
        protein_sides = np.zeros((len(rows), 8), dtype=np.uint8)
        for line in (data / "proteins.txt").open():
            tokens = line.split()
            index = int(tokens[0])
            protein_energy[index] = _tenths(tokens[1])
            if index and len(tokens) == 10:
                protein_sides[index] = [int(s) for s in tokens[2:]]
        perimeters = [tuple(map(int, line.split())) for line in
                      (data / "perimeters.txt").open()]
        if [row[0] for row in perimeters] != list(range(len(perimeters))):
            raise ValueError("perimeter IDs must be contiguous")
        sides = np.asarray([row[1:] for row in perimeters], dtype=np.uint8)
        for protein_id in range(1, len(rows)):
            if not np.array_equal(protein_sides[protein_id],
                                  sides[protein_perimeter[protein_id]]):
                raise ValueError(f"protein {protein_id} has inconsistent perimeter geometry")
        nper = len(sides)
        dimer_id = np.full((nper, nper), -1, dtype=np.int32)
        dimer_energy_list: list[int] = []
        per_a: list[int] = []
        per_b: list[int] = []
        for line in (data / "dimers.txt").open():
            tokens = line.split()
            a, b = int(tokens[0]), int(tokens[1])
            if dimer_id[a, b] != -1:
                raise ValueError(f"duplicate dimer perimeter pair {(a, b)}")
            dimer_id[a, b] = len(dimer_energy_list)
            dimer_energy_list.append(_tenths(tokens[3]))
            per_a.append(a)
            per_b.append(b)
        dimer_energy = np.asarray(dimer_energy_list, dtype=np.int16)
        prot_prom_energy = np.zeros((16, nper), dtype=np.int16)
        prot_prom_pol = np.zeros((16, nper), dtype=bool)
        dim_prom_energy = np.zeros((16, len(dimer_energy)), dtype=np.int16)
        dim_prom_pol = np.zeros((16, len(dimer_energy)), dtype=bool)
        for promoter in range(16):
            seen: dict[int, tuple[int, bool]] = {}
            path = source / "promoters" / f"proteins_bind_prom{promoter}.txt"
            for line in path.open():
                protein, energy, pol = line.split()
                perimeter = int(protein_perimeter[int(protein)])
                value = (_tenths(energy), bool(int(pol)))
                if perimeter in seen and seen[perimeter] != value:
                    raise ValueError("promoter binding differs within a perimeter")
                seen[perimeter] = value
                prot_prom_energy[promoter, perimeter] = value[0]
                prot_prom_pol[promoter, perimeter] = value[1]
            path = source / "promoters" / f"dimers_bind_prom{promoter}.txt"
            for line in path.open():
                dimer, energy, pol = line.split()
                dim_prom_energy[promoter, int(dimer)] = _tenths(energy)
                dim_prom_pol[promoter, int(dimer)] = bool(int(pol))
        polymerase = np.zeros(16, dtype=np.int16)
        for promoter, energy in {5: -150, 7: -153, 11: -139, 13: -153,
                                 14: -139, 15: -156}.items():
            polymerase[promoter] = energy
        met_rows = [line.split() for line in (data / "metabolites.txt").open()]
        met_strings = tuple(row[0] for row in met_rows)
        if len(met_strings) != len(set(met_strings)) or any(
                canonical_metabolite(s) != s for s in met_strings):
            raise ValueError("metabolites must be unique reversal representatives")
        met_index = {s: i for i, s in enumerate(met_strings)}
        met_unbreakable = np.asarray([bool(int(row[1])) for row in met_rows])
        protein_met = np.zeros((len(met_strings), nper), dtype=np.int16)
        dim_met_energy = np.zeros((len(met_strings), len(dimer_energy)), dtype=np.int16)
        dim_met_seq = np.zeros_like(dim_met_energy, dtype=np.uint16)
        dim_met_pos = np.zeros_like(dim_met_energy, dtype=np.uint8)
        dim_met_orientation = np.zeros_like(dim_met_energy, dtype=np.uint8)
        dim_met_breakable = np.zeros_like(dim_met_energy, dtype=bool)
        sequence_set: set[int] = set()
        for met, index in met_index.items():
            suffix = f"{len(met)}_{int(met, 2)}.txt"
            seen_met: dict[int, int] = {}
            for line in (source / "prot_metabolism" / suffix).open():
                tokens = line.split()
                perimeter = int(protein_perimeter[int(tokens[0])])
                energy = _tenths(tokens[1])
                if perimeter in seen_met and seen_met[perimeter] != energy:
                    raise ValueError("metabolite binding differs within a perimeter")
                seen_met[perimeter] = energy
                protein_met[index, perimeter] = energy
            for line in (source / "dim_metabolism" / suffix).open():
                tokens = line.split()
                dimer = int(tokens[0])
                dim_met_energy[index, dimer] = _tenths(tokens[1])
                sequence = tokens[2]
                dim_met_seq[index, dimer] = int(sequence, 2)
                dim_met_pos[index, dimer] = int(tokens[3])
                dim_met_orientation[index, dimer] = int(tokens[4])
                if len(sequence) == 16:
                    dim_met_breakable[index, dimer] = True
                    sequence_set.add(int(sequence, 2))
        sequences = np.asarray(sorted(sequence_set), dtype=np.uint16)
        seq_to_index = np.full(65536, -1, dtype=np.int32)
        seq_to_index[sequences] = np.arange(len(sequences), dtype=np.int32)
        breaking_energy = _breaking_geometry(sequences, sides)
        return cls(genes, protein_perimeter, protein_energy, sides, dimer_id,
                   dimer_energy, np.asarray(per_a, dtype=np.int16),
                   np.asarray(per_b, dtype=np.int16), prot_prom_energy, prot_prom_pol,
                   dim_prom_energy, dim_prom_pol, polymerase, met_strings,
                   met_unbreakable, protein_met, dim_met_energy, dim_met_seq,
                   dim_met_pos, dim_met_orientation, dim_met_breakable,
                   sequences, seq_to_index, breaking_energy, digest, str(source))

    def dimer_for(self, protein_a: int, protein_b: int) -> int:
        perimeter_a = int(self.protein_perimeter[protein_a])
        perimeter_b = int(self.protein_perimeter[protein_b])
        return int(self.dimer_id[min(perimeter_a, perimeter_b),
                                 max(perimeter_a, perimeter_b)])

    def breaking_options(self, sequence: int,
                         breaker_perimeter: int) -> tuple[tuple[int, int], ...]:
        index = int(self.breaking_seq_to_index[sequence])
        if index < 0:
            return ()
        return tuple((subunit, int(energy)) for subunit, energy in
                     enumerate(self.breaking_energy[index, breaker_perimeter]) if energy < 0)
