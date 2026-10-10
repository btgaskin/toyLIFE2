"""Immutable, integer-energy representation of the published toyLIFE lookup universe.

The source text files are part of the model. Energies are stored in tenths, so
all equality and ordering decisions are exact. Zero denotes a missing bond.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from zipfile import BadZipFile

import numpy as np

_PACKAGE_SOURCE = Path(__file__).resolve().parent / "_vendor" / "toylife"
_CHECKOUT_SOURCE = Path(__file__).resolve().parents[2] / "upstream" / "toylife"
DEFAULT_SOURCE = _PACKAGE_SOURCE if _PACKAGE_SOURCE.is_dir() else _CHECKOUT_SOURCE
_CACHE_VERSION = 2
# The scientific source digest above is stable across cache format revisions.
_CACHE_FORMAT_VERSION = 1


def _array_checksum(digest: str, arrays: Mapping[str, np.ndarray]) -> str:
    """Bind the complete typed array payload to its scientific source digest."""
    checksum = hashlib.sha256(f"tables-cache-{_CACHE_FORMAT_VERSION}\n{digest}\n".encode())
    for name, value in sorted(arrays.items()):
        checksum.update(f"{name}\0{value.dtype.str}\0{value.shape}\0".encode())
        checksum.update(np.ascontiguousarray(value).tobytes())
    return checksum.hexdigest()


@lru_cache(maxsize=8)
def _parsed_source_checksum(source: str, digest: str) -> str:
    return Tables._parse(Path(source), digest).content_checksum()


def _tenths(value: str) -> int:
    number = float(value)
    scaled = round(number * 10)
    if abs(number * 10 - scaled) > 1e-6:
        raise ValueError(f"energy is not an exact tenth: {value}")
    if not -32768 <= scaled <= 32767:
        raise ValueError(f"energy is outside the int16 range: {value}")
    return scaled


def _index(value: str | int, size: int, name: str) -> int:
    index = int(value)
    if not 0 <= index < size:
        raise ValueError(f"{name} index {index} is outside [0, {size})")
    return index


def _flag(value: str) -> bool:
    return bool(_index(value, 2, "boolean"))


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
        self.validate()
        for data_field in fields(self):
            if data_field.name == "met_index":
                continue
            value = getattr(self, data_field.name)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        object.__setattr__(self, "met_index", MappingProxyType(
            {sequence: index for index, sequence in enumerate(self.met_strings)}))

    def validate(self) -> None:
        """Check the typed lookup universe before any host or device indexing."""
        vectors = {
            "protein_energy": np.int16, "perimeter_sides": np.uint8,
            "dimer_energy": np.int16, "breaking_sequences": np.uint16,
        }
        for name, dtype in vectors.items():
            value = getattr(self, name)
            ndim = 2 if name == "perimeter_sides" else 1
            if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or (
                    value.ndim != ndim):
                raise ValueError(f"{name} must be a {ndim}-dimensional {np.dtype(dtype)} array")
        nprot, nper, ndim, nseq, nmet = (
            len(self.protein_energy), len(self.perimeter_sides), len(self.dimer_energy),
            len(self.breaking_sequences), len(self.met_strings),
        )
        if not nprot or not nper or not nmet:
            raise ValueError("protein, perimeter and metabolite tables must be nonempty")
        shapes = {
            "gene_to_protein": ((65536,), np.int32),
            "protein_perimeter": ((nprot,), np.int16),
            "perimeter_sides": ((nper, 8), np.uint8),
            "dimer_id": ((nper, nper), np.int32),
            "dimer_perimeter_a": ((ndim,), np.int16),
            "dimer_perimeter_b": ((ndim,), np.int16),
            "protein_promoter_energy": ((16, nper), np.int16),
            "protein_promoter_pol": ((16, nper), np.bool_),
            "dimer_promoter_energy": ((16, ndim), np.int16),
            "dimer_promoter_pol": ((16, ndim), np.bool_),
            "polymerase_energy": ((16,), np.int16),
            "met_unbreakable": ((nmet,), np.bool_),
            "protein_met_energy": ((nmet, nper), np.int16),
            "dimer_met_energy": ((nmet, ndim), np.int16),
            "dimer_met_seq": ((nmet, ndim), np.uint16),
            "dimer_met_pos": ((nmet, ndim), np.uint8),
            "dimer_met_orientation": ((nmet, ndim), np.uint8),
            "dimer_met_breakable": ((nmet, ndim), np.bool_),
            "breaking_seq_to_index": ((65536,), np.int32),
            "breaking_energy": ((nseq, nper, 2), np.int16),
        }
        for name, (shape, dtype) in shapes.items():
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != shape or (
                    value.dtype != np.dtype(dtype)):
                raise ValueError(f"{name} must have shape {shape} and dtype {np.dtype(dtype)}")
        for name, lower, upper in (
                ("gene_to_protein", 0, nprot), ("protein_perimeter", 0, nper),
                ("perimeter_sides", 0, 16), ("dimer_id", -1, ndim),
                ("dimer_perimeter_a", 0, nper), ("dimer_perimeter_b", 0, nper),
                ("dimer_met_pos", 0, 8), ("dimer_met_orientation", 0, 3),
                ("breaking_seq_to_index", -1, nseq)):
            value = getattr(self, name)
            if np.any(value < lower) or np.any(value >= upper):
                raise ValueError(f"{name} values must be in [{lower}, {upper})")
        expected_dimers = np.full((nper, nper), -1, dtype=np.int32)
        a, b = self.dimer_perimeter_a, self.dimer_perimeter_b
        if np.any(a > b) or len(set(zip(a.tolist(), b.tolist(), strict=True))) != ndim:
            raise ValueError("dimer perimeter pairs must be unique and ordered")
        expected_dimers[a, b] = np.arange(ndim, dtype=np.int32)
        if not np.array_equal(self.dimer_id, expected_dimers):
            raise ValueError("dimer_id must agree with the dimer perimeter pairs")
        if nseq and np.any(self.breaking_sequences[1:] <= self.breaking_sequences[:-1]):
            raise ValueError("breaking_sequences must be unique and sorted")
        expected_sequences = np.full(65536, -1, dtype=np.int32)
        expected_sequences[self.breaking_sequences] = np.arange(nseq, dtype=np.int32)
        if not np.array_equal(self.breaking_seq_to_index, expected_sequences):
            raise ValueError("breaking_seq_to_index must agree with breaking_sequences")
        bound_sequences = self.dimer_met_seq[self.dimer_met_breakable]
        if np.any(self.breaking_seq_to_index[bound_sequences] < 0):
            raise ValueError("breakable dimer metabolites must have breaking sequence entries")
        if not isinstance(self.met_strings, tuple) or any(
                not isinstance(s, str) or not 4 <= len(s) <= 8 or canonical_metabolite(s) != s
                for s in self.met_strings) or len(set(self.met_strings)) != nmet:
            raise ValueError("met_strings must contain unique canonical binary strings, length 4-8")
        lengths = np.asarray([len(s) for s in self.met_strings])[:, None]
        if np.any(self.dimer_met_pos >= lengths) or np.any(
                self.dimer_met_breakable & (self.dimer_met_pos == 0)):
            raise ValueError("dimer_met_pos must identify a valid metabolite breaking position")

    def _cache_arrays(self) -> dict[str, np.ndarray]:
        arrays = {f.name: getattr(self, f.name) for f in fields(self)
                  if isinstance(getattr(self, f.name), np.ndarray)}
        arrays["met_strings"] = np.asarray(self.met_strings)
        return arrays

    def content_checksum(self) -> str:
        """Checksum executing values, shapes and dtypes, bound to the source digest."""
        return _array_checksum(self.digest, self._cache_arrays())

    def verify_source(self, source: str | Path | None = None) -> None:
        """Verify these executing arrays against the actual source pack.

        Source bytes and in-memory arrays are checked on each call. The parsed
        reference checksum is reused for a source path/digest within this process.
        Synthetic tables remain usable for rule tests; recording them requires a
        corresponding source pack.
        """
        directory = Path(self.source if source is None else source).expanduser().resolve()
        digest = _source_digest(directory, _source_files(directory))
        if digest != self.digest:
            raise ValueError("table digest does not match the source pack")
        self.validate()
        if self.content_checksum() != _parsed_source_checksum(str(directory), digest):
            raise ValueError("executing table arrays do not match the source pack")

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
                try:
                    with np.load(cache_path, allow_pickle=False) as archive:
                        values = {name: archive[name] for name in archive.files}
                    if values.pop("cache_format_version").item() != _CACHE_FORMAT_VERSION:
                        raise ValueError("unsupported table cache format")
                    if values.pop("source_digest").item() != digest:
                        raise ValueError("table cache has a different source digest")
                    checksum = values.pop("content_checksum").item()
                    if checksum != _array_checksum(digest, values):
                        raise ValueError("table cache content checksum mismatch")
                    strings = values.pop("met_strings")
                    if strings.ndim != 1 or strings.dtype.kind != "U":
                        raise ValueError("cached metabolites must be a string vector")
                    met_strings = tuple(str(s) for s in strings)
                    return cls(**values, met_strings=met_strings, digest=digest,
                               source=str(directory))
                except (ValueError, TypeError, KeyError, OSError, EOFError, BadZipFile):
                    # A cache is disposable; authoritative source tables rebuild it.
                    pass
        tables = cls._parse(directory, digest)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
            arrays = tables._cache_arrays()
            arrays["cache_format_version"] = np.asarray(_CACHE_FORMAT_VERSION)
            arrays["source_digest"] = np.asarray(digest)
            arrays["content_checksum"] = np.asarray(tables.content_checksum())
            try:
                with temp_path.open("wb") as stream:
                    np.savez(stream, allow_pickle=False, **arrays)
                os.replace(temp_path, cache_path)
            finally:
                temp_path.unlink(missing_ok=True)
        return tables

    @classmethod
    def _parse(cls, source: Path, digest: str) -> Tables:
        data = source / "data"
        genes = np.loadtxt(data / "protein_gene.txt", dtype=np.int64)
        if genes.shape != (65536,):
            raise ValueError("protein_gene.txt must contain 65,536 rows")
        rows = [tuple(map(int, line.split())) for line in
                (data / "proteins_with_perimeter.txt").open()]
        if [row[0] for row in rows] != list(range(len(rows))):
            raise ValueError("protein IDs must be contiguous")
        if np.any(genes < 0) or np.any(genes >= len(rows)):
            raise ValueError("gene_to_protein contains an invalid protein ID")
        genes = genes.astype(np.int32)
        protein_perimeter = np.asarray([row[1] for row in rows], dtype=np.int16)
        protein_energy = np.zeros(len(rows), dtype=np.int16)
        protein_sides = np.zeros((len(rows), 8), dtype=np.uint8)
        for line in (data / "proteins.txt").open():
            tokens = line.split()
            index = _index(tokens[0], len(rows), "protein")
            protein_energy[index] = _tenths(tokens[1])
            if index and len(tokens) == 10:
                protein_sides[index] = [int(s) for s in tokens[2:]]
        perimeters = [tuple(map(int, line.split())) for line in
                      (data / "perimeters.txt").open()]
        if [row[0] for row in perimeters] != list(range(len(perimeters))):
            raise ValueError("perimeter IDs must be contiguous")
        sides = np.asarray([row[1:] for row in perimeters], dtype=np.uint8)
        if sides.shape != (len(perimeters), 8) or np.any(sides >= 16):
            raise ValueError("perimeter_sides must contain eight sides in [0, 16)")
        if np.any(protein_perimeter < 0) or np.any(protein_perimeter >= len(sides)):
            raise ValueError("protein_perimeter contains an invalid perimeter ID")
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
            a, b = (_index(tokens[0], nper, "dimer perimeter"),
                    _index(tokens[1], nper, "dimer perimeter"))
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
                perimeter = int(protein_perimeter[_index(protein, len(rows), "protein")])
                value = (_tenths(energy), _flag(pol))
                if perimeter in seen and seen[perimeter] != value:
                    raise ValueError("promoter binding differs within a perimeter")
                seen[perimeter] = value
                prot_prom_energy[promoter, perimeter] = value[0]
                prot_prom_pol[promoter, perimeter] = value[1]
            path = source / "promoters" / f"dimers_bind_prom{promoter}.txt"
            for line in path.open():
                dimer, energy, pol = line.split()
                dimer_index = _index(dimer, len(dimer_energy), "dimer")
                dim_prom_energy[promoter, dimer_index] = _tenths(energy)
                dim_prom_pol[promoter, dimer_index] = _flag(pol)
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
        met_unbreakable = np.asarray([_flag(row[1]) for row in met_rows])
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
                perimeter = int(protein_perimeter[_index(tokens[0], len(rows), "protein")])
                energy = _tenths(tokens[1])
                if perimeter in seen_met and seen_met[perimeter] != energy:
                    raise ValueError("metabolite binding differs within a perimeter")
                seen_met[perimeter] = energy
                protein_met[index, perimeter] = energy
            for line in (source / "dim_metabolism" / suffix).open():
                tokens = line.split()
                dimer = _index(tokens[0], len(dimer_energy), "dimer")
                dim_met_energy[index, dimer] = _tenths(tokens[1])
                sequence = tokens[2]
                if not sequence or len(sequence) > 16 or set(sequence) - {"0", "1"}:
                    raise ValueError("dimer metabolite sequences must be binary and fit uint16")
                dim_met_seq[index, dimer] = int(sequence, 2)
                dim_met_pos[index, dimer] = _index(tokens[3], len(met), "breaking position")
                dim_met_orientation[index, dimer] = _index(tokens[4], 3, "dimer orientation")
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
