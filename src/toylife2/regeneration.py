"""Independent offline checks of fold and contact geometry.

The compact 4×4 walks, HP contact energies, and reversal/symmetry-normalized
perimeters are generated here rather than read from the protein lookup table.
This is a qualification tool; runtime simulation loads the supplied universe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tables import Tables, _contact_energy


def _adjacency() -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(4 * (y + dy) + x + dx for dx, dy in
                       ((1, 0), (-1, 0), (0, 1), (0, -1))
                       if 0 <= x + dx < 4 and 0 <= y + dy < 4)
                 for y in range(4) for x in range(4))


def _transform(cell: int, symmetry: int) -> int:
    x, y = cell % 4, cell // 4
    if symmetry & 4:
        x = 3 - x
    for _ in range(symmetry & 3):
        x, y = 3 - y, x
    return 4 * y + x


def enumerate_compact_folds() -> tuple[tuple[int, ...], ...]:
    """Enumerate the 38 Hamiltonian 4×4 walks modulo D4 and chain reversal."""
    adjacency = _adjacency()
    paths: list[tuple[int, ...]] = []

    def visit(path: tuple[int, ...], used: int) -> None:
        if len(path) == 16:
            paths.append(path)
            return
        for next_cell in adjacency[path[-1]]:
            if not (used >> next_cell) & 1:
                visit(path + (next_cell,), used | (1 << next_cell))

    for cell in range(16):
        visit((cell,), 1 << cell)
    folds = {min(tuple(_transform(cell, symmetry) for cell in direction)
                 for symmetry in range(8) for direction in (path, path[::-1]))
             for path in paths}
    if len(paths) != 552 or len(folds) != 38:
        raise AssertionError("unexpected compact-walk count")
    return tuple(sorted(folds))


def _canonical_boundary() -> np.ndarray:
    boundary = tuple(cell for cell in range(16)
                     if cell // 4 in (0, 3) or cell % 4 in (0, 3))
    position = {cell: index for index, cell in enumerate(boundary)}
    result = np.zeros(1 << 12, dtype=np.uint16)
    for code in range(1 << 12):
        result[code] = min(sum(((code >> index) & 1) << position[
            _transform(cell, symmetry)] for index, cell in enumerate(boundary))
                           for symmetry in range(8))
    return result


@dataclass(frozen=True)
class FoldUniverse:
    functional: np.ndarray
    energy: np.ndarray
    boundary_signature: np.ndarray
    folds: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        for array in (self.functional, self.energy, self.boundary_signature):
            array.flags.writeable = False


def regenerate_folding() -> FoldUniverse:
    """Fold every binary 16-mer using integer HP energies and revised tie rule.

    A sequence with minimum-energy alternatives remains functional when those
    alternatives have the same perimeter after D4 normalization. The all-polar
    chain has no stabilizing contact and is nonfunctional in the source data.
    """
    folds = enumerate_compact_folds()
    oriented = folds + tuple(path[::-1] for path in folds)
    adjacency = _adjacency()
    lattice_edges = tuple((a, b) for a in range(16) for b in adjacency[a] if a < b)
    boundary = tuple(cell for cell in range(16)
                     if cell // 4 in (0, 3) or cell % 4 in (0, 3))
    canonical = _canonical_boundary()
    sequence = np.arange(65536, dtype=np.uint16)
    bits = ((sequence[:, None] >> np.arange(15, -1, -1)) & 1).astype(np.uint8)
    minimum = np.full(65536, 999, dtype=np.int16)
    perimeter_h = np.full(65536, 99, dtype=np.uint8)
    signature = np.full(65536, -1, dtype=np.int32)
    ambiguous = np.zeros(65536, dtype=bool)
    for walk in oriented:
        sequence_at = {cell: index for index, cell in enumerate(walk)}
        energy = np.zeros(65536, dtype=np.int16)
        for cell_a, cell_b in lattice_edges:
            a, b = sequence_at[cell_a], sequence_at[cell_b]
            if abs(a - b) == 1:
                continue
            first, second = bits[:, a], bits[:, b]
            energy -= np.where(first & second, 20, np.where(first | second, 3, 0))
        edge_bits = bits[:, [sequence_at[cell] for cell in boundary]]
        hydrophobic = edge_bits.sum(axis=1).astype(np.uint8)
        code = np.sum(edge_bits.astype(np.uint16) *
                      (1 << np.arange(12, dtype=np.uint16)), axis=1, dtype=np.uint16)
        current_signature = canonical[code]
        better = (energy < minimum) | ((energy == minimum) &
                                       (hydrophobic < perimeter_h))
        tied = (energy == minimum) & (hydrophobic == perimeter_h) & ~better
        ambiguous = np.where(better, False,
                             ambiguous | (tied & (signature != current_signature)))
        signature = np.where(better, current_signature, signature)
        minimum = np.where(better, energy, minimum)
        perimeter_h = np.where(better, hydrophobic, perimeter_h)
    functional = ~ambiguous
    functional[0] = False
    return FoldUniverse(functional, minimum, signature, folds)


@dataclass(frozen=True)
class RegenerationReport:
    folds: int
    functional_sequences: int
    distinct_perimeters: int
    energy_mismatches: int
    functional_mismatches: int
    perimeter_identity_mismatches: int
    attachment_rows_checked: int
    attachment_mismatches: int


def validate_regeneration(tables: Tables, check_attachments: bool = True) -> RegenerationReport:
    """Compare independently regenerated folds and contact choices to source data."""
    folded = regenerate_folding()
    encoded = tables.gene_to_protein
    functional = encoded > 0
    functional_mismatches = int(np.count_nonzero(folded.functional != functional))
    source_energies = tables.protein_energy[encoded[functional]]
    energy_mismatches = int(np.count_nonzero(folded.energy[functional] != source_energies))
    assigned_perimeters = tables.protein_perimeter[encoded[functional]]
    identity_pairs = np.stack((assigned_perimeters,
                               folded.boundary_signature[functional]), axis=1)
    perimeter_identity_mismatches = int(
        len(np.unique(identity_pairs, axis=0)) - len(np.unique(assigned_perimeters))
    )
    perimeter_identity_mismatches += int(
        len(np.unique(identity_pairs, axis=0)) -
        len(np.unique(folded.boundary_signature[functional]))
    )
    rows_checked = 0
    attachment_mismatches = 0
    if check_attachments:
        source = Path(tables.source)
        path = source / "prot_metabolism" / "prot_breaking.txt"
        for line in path.open():
            protein, sequence_text, energy_text, side_text = line.split()
            sequence = int(sequence_text, 2)
            seq_index = int(tables.breaking_seq_to_index[sequence])
            if seq_index < 0:
                continue  # this perceived sequence is absent from the supplied D-M universe
            perimeter = int(tables.protein_perimeter[int(protein)])
            side = int(side_text) // 2
            observed = int(tables.breaking_energy[seq_index, perimeter, side])
            expected = round(float(energy_text) * 10)
            rows_checked += 1
            attachment_mismatches += observed != expected
    return RegenerationReport(len(folded.folds), int(folded.functional.sum()),
        len(np.unique(folded.boundary_signature[folded.functional])),
        energy_mismatches, functional_mismatches, perimeter_identity_mismatches, rows_checked,
        attachment_mismatches)


def contact_energy(first_side: int, second_side: int) -> int:
    """Independent public geometry primitive, in tenths of an energy unit."""
    if not 0 <= first_side < 16 or not 0 <= second_side < 16:
        raise ValueError("a toyProtein side has four binary positions")
    return int(_contact_energy()[first_side, second_side])


@dataclass(frozen=True)
class BindingRegenerationReport:
    """Mismatch counts for contacts reconstructed without ligand lookup rows."""

    dimer_bonds_checked: int
    dimer_bond_mismatches: int
    protein_promoter_checked: int
    protein_promoter_mismatches: int
    protein_promoter_pol_mismatches: tuple[tuple[int, int], ...]
    protein_metabolite_checked: int
    protein_metabolite_mismatches: int


def regenerate_protein_ligands(tables: Tables) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild protein ligand tables from the two perimeter orientations.

    A distinct perimeter orientation is counted once. A binding site is a
    position on that orientation, so two positions with the same four-bit
    chemistry still cause a tie. A metabolite may have several equally good
    windows at one protein site without making the protein attachment
    ambiguous. The adjacent side before the winning promoter site determines
    polymerase recruitment.
    """
    contact = _contact_energy()
    perimeters = tables.perimeter_sides.reshape((-1, 2, 4))
    promoter_energy = np.zeros_like(tables.protein_promoter_energy)
    promoter_pol = np.zeros_like(tables.protein_promoter_pol)
    metabolite_energy = np.zeros_like(tables.protein_met_energy)
    metabolite_windows = tuple(tuple(int(met[index:index + 4], 2)
                                     for index in range(len(met) - 3))
                               for met in tables.met_strings)
    for perimeter, orientations in enumerate(perimeters):
        distinct = orientations[:1] if np.array_equal(*orientations) else orientations
        for promoter in range(16):
            candidates = [(int(contact[side, promoter]), orientation, position)
                          for orientation, sides in enumerate(distinct)
                          for position, side in enumerate(sides)]
            minimum = min(candidate[0] for candidate in candidates)
            winners = [candidate for candidate in candidates
                       if candidate[0] == minimum]
            if minimum < -26 and len(winners) == 1:
                promoter_energy[promoter, perimeter] = minimum
                _, orientation, position = winners[0]
                adjacent = int(distinct[orientation, (position - 1) % 4])
                promoter_pol[promoter, perimeter] = contact[adjacent, 10] < -26
        for metabolite, windows in enumerate(metabolite_windows):
            candidates = [min(int(contact[side, window]) for window in windows)
                          for sides in distinct for side in sides]
            minimum = min(candidates)
            if minimum < -26 and candidates.count(minimum) == 1:
                metabolite_energy[metabolite, perimeter] = minimum
    for array in (promoter_energy, promoter_pol, metabolite_energy):
        array.flags.writeable = False
    return promoter_energy, promoter_pol, metabolite_energy


def validate_binding_regeneration(tables: Tables) -> BindingRegenerationReport:
    """Check listed dimer bonds and all protein ligand cells against geometry."""
    contact = _contact_energy()
    bond_mismatches = 0
    for dimer, (first, second) in enumerate(zip(tables.dimer_perimeter_a,
                                                  tables.dimer_perimeter_b,
                                                  strict=True)):
        candidates = contact[np.ix_(tables.perimeter_sides[first],
                                    tables.perimeter_sides[second])]
        bond_mismatches += int(int(candidates.min()) != int(tables.dimer_energy[dimer]))
    promoter, pol, metabolite = regenerate_protein_ligands(tables)
    pol_mismatch = np.argwhere(pol != tables.protein_promoter_pol)
    return BindingRegenerationReport(
        len(tables.dimer_energy), bond_mismatches,
        promoter.size, int(np.count_nonzero(promoter != tables.protein_promoter_energy)),
        tuple((int(p), int(perimeter)) for p, perimeter in pol_mismatch),
        metabolite.size, int(np.count_nonzero(metabolite != tables.protein_met_energy)),
    )
