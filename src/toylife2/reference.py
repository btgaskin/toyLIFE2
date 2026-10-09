"""Deterministic scalar oracle for the canonical toyLIFE chemistry.

The reference engine favors explicit molecule accounting over speed. Its state
transitions define the contract checked by the Quadrants batched implementation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from .genotypes import validate
from .tables import Tables, canonical_metabolite

Dimers = Counter[tuple[int, int]]
Complexes = Counter[tuple[str, int, int, str]]


@dataclass
class CellState:
    """Molecule counts at a cell-step boundary or within an explicit step.

    Dimer protein IDs are sorted. A bound complex key is ``(kind, p1, p2,
    metabolite)`` with kind ``P`` or ``D`` and p2=-1 for a protein complex.
    """

    proteins: Counter[int] = field(default_factory=Counter)
    dimers: Dimers = field(default_factory=Counter)
    metabolites: Counter[str] = field(default_factory=Counter)
    bound: Complexes = field(default_factory=Counter)

    def __post_init__(self) -> None:
        for population in (self.proteins, self.dimers, self.metabolites, self.bound):
            if any(type(count) is not int or count < 0 for count in population.values()):
                raise ValueError("molecule counts must be nonnegative integers")
        if any(canonical_metabolite(met) != met for met in self.metabolites):
            raise ValueError("CellState metabolites must use reversal-canonical sequences")
        if any(canonical_metabolite(key[3]) != key[3] for key in self.bound):
            raise ValueError("bound metabolites must use reversal-canonical sequences")

    def copy(self) -> CellState:
        return CellState(Counter(self.proteins), Counter(self.dimers),
                         Counter(self.metabolites), Counter(self.bound))

    def key(self) -> tuple:
        return tuple(tuple(sorted((k, v) for k, v in population.items() if v > 0))
                     for population in (self.proteins, self.dimers,
                                        self.metabolites, self.bound))


@dataclass(frozen=True)
class StepResult:
    state: CellState
    expression: int
    fissions: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class _Candidate:
    energy: int
    reactants: tuple[tuple[str, object, int], ...]
    product: tuple[str, object] | None
    fission: tuple[str, str, str] | None = None


def _genotype(genes: np.ndarray) -> np.ndarray:
    parsed = validate(genes)
    if parsed.ndim != 1:
        raise ValueError("the scalar reference accepts one genotype")
    return parsed


class ReferenceModel:
    def __init__(self, tables: Tables):
        self.tables = tables

    def translated(self, genotype: np.ndarray, expression: int) -> Counter[int]:
        genes = _genotype(genotype)
        if expression < 0 or expression >= 1 << len(genes):
            raise ValueError("expression state is outside the genotype")
        result: Counter[int] = Counter()
        for index, gene in enumerate(genes):
            if expression & (1 << (len(genes) - index - 1)):
                protein = int(self.tables.gene_to_protein[int(gene) & 65535])
                if protein > 0:  # active nonfolding genes still have an expression bit
                    result[protein] += 1
        return result

    def _dimer_id(self, pair: tuple[int, int]) -> int:
        return self.tables.dimer_for(*pair)

    def _binding_candidates(self, state: CellState) -> list[_Candidate]:
        t = self.tables
        result: list[_Candidate] = []
        proteins = sorted(p for p, count in state.proteins.items() if count > 0 and p > 0)
        dimers = sorted(pair for pair, count in state.dimers.items() if count > 0)
        mets = sorted(s for s, count in state.metabolites.items() if count > 0)
        for index, p1 in enumerate(proteins):
            for p2 in proteins[index:]:
                if p1 == p2 and state.proteins[p1] < 2:
                    continue
                pair = (p1, p2)
                dimer = self._dimer_id(pair)
                if dimer < 0:
                    continue
                energy = int(t.protein_energy[p1]) + int(t.protein_energy[p2]) + int(
                    t.dimer_energy[dimer])
                reactants = (("P", p1, 2),) if p1 == p2 else (
                    ("P", p1, 1), ("P", p2, 1))
                result.append(_Candidate(energy, reactants, ("D", pair)))
        for protein in proteins:
            perimeter = int(t.protein_perimeter[protein])
            for met in mets:
                met_index = t.met_index.get(canonical_metabolite(met))
                if met_index is None:
                    continue
                contact = int(t.protein_met_energy[met_index, perimeter])
                if contact < 0:
                    result.append(_Candidate(int(t.protein_energy[protein]) + contact,
                        (("P", protein, 1), ("M", met, 1)),
                        ("B", ("P", protein, -1, met))))
        for pair in dimers:
            dimer = self._dimer_id(pair)
            for met in mets:
                met_index = t.met_index.get(canonical_metabolite(met))
                if met_index is None:
                    continue
                contact = int(t.dimer_met_energy[met_index, dimer])
                if contact < 0:
                    energy = (int(t.protein_energy[pair[0]]) +
                              int(t.protein_energy[pair[1]]) +
                              int(t.dimer_energy[dimer]) + contact)
                    result.append(_Candidate(energy,
                        (("D", pair, 1), ("M", met, 1)),
                        ("B", ("D", pair[0], pair[1], met))))
        for bound, count in sorted(state.bound.items()):
            if count <= 0 or bound[0] != "D":
                continue
            _, p1, p2, met = bound
            pair = (p1, p2)
            dimer = self._dimer_id(pair)
            met_index = t.met_index.get(canonical_metabolite(met))
            if met_index is None or not t.dimer_met_breakable[met_index, dimer]:
                continue
            sequence = int(t.dimer_met_seq[met_index, dimer])
            split = int(t.dimer_met_pos[met_index, dimer])
            if not 0 < split < len(met):
                continue
            old_energy = (int(t.protein_energy[p1]) + int(t.protein_energy[p2]) +
                          int(t.dimer_energy[dimer]))
            for breaker in proteins:
                perimeter = int(t.protein_perimeter[breaker])
                for subunit, contact in t.breaking_options(sequence, perimeter):
                    new_energy = int(t.protein_energy[breaker]) + int(
                        t.protein_energy[pair[subunit]]) + contact
                    if new_energy < old_energy:
                        result.append(_Candidate(new_energy,
                            (("P", breaker, 1), ("B", bound, 1)), None,
                            (met, met[:split], met[split:])))
        return result

    @staticmethod
    def _pool(state: CellState, kind: str) -> Counter[Any]:
        return cast(Counter[Any], {"P": state.proteins, "D": state.dimers,
                                   "M": state.metabolites, "B": state.bound}[kind])

    def react(self, state: CellState) -> tuple[CellState, tuple[tuple[str, str, str], ...]]:
        """Resolve all current candidates by total energy and simultaneous demand."""
        current = state.copy()
        groups: dict[int, list[_Candidate]] = defaultdict(list)
        for candidate in self._binding_candidates(current):
            groups[candidate.energy].append(candidate)
        fissions: list[tuple[str, str, str]] = []
        for energy in sorted(groups):
            proposed: list[tuple[_Candidate, int]] = []
            demand: Counter[tuple[str, object]] = Counter()
            for candidate in groups[energy]:
                capacity = min(self._pool(current, kind)[key] // units
                               for kind, key, units in candidate.reactants)
                if capacity <= 0:
                    continue
                proposed.append((candidate, capacity))
                for kind, key, units in candidate.reactants:
                    demand[(kind, key)] += capacity * units
            accepted = [(candidate, capacity) for candidate, capacity in proposed
                        if all(demand[(kind, key)] <= self._pool(current, kind)[key]
                               for kind, key, _ in candidate.reactants)]
            for candidate, capacity in accepted:
                for kind, key, units in candidate.reactants:
                    self._pool(current, kind)[key] -= capacity * units
                if candidate.product is not None:
                    product_kind, product_key = candidate.product
                    self._pool(current, product_kind)[product_key] += capacity
                if candidate.fission is not None:
                    original, left, right = candidate.fission
                    for _ in range(capacity):
                        fissions.append((original, left, right))
                    current.metabolites[canonical_metabolite(left)] += capacity
                    current.metabolites[canonical_metabolite(right)] += capacity
        for kind in ("P", "D", "M", "B"):
            population = self._pool(current, kind)
            for key in list(population):
                if population[key] <= 0:
                    del population[key]
        return current, tuple(fissions)

    def promoter_active(self, promoter: int, state: CellState) -> bool:
        t = self.tables
        candidates: dict[int, list[tuple[str, object, bool]]] = defaultdict(list)
        poly = int(t.polymerase_energy[promoter])
        if poly < 0:
            candidates[poly].append(("polymerase", 0, True))
        for protein, copies in state.proteins.items():
            if protein <= 0 or copies <= 0:
                continue
            perimeter = int(t.protein_perimeter[protein])
            contact = int(t.protein_promoter_energy[promoter, perimeter])
            if contact < 0:
                total = int(t.protein_energy[protein]) + contact
                candidates[total].append(("protein", protein,
                                          bool(t.protein_promoter_pol[promoter, perimeter])))
        for pair, copies in state.dimers.items():
            if copies <= 0:
                continue
            dimer = self._dimer_id(pair)
            contact = int(t.dimer_promoter_energy[promoter, dimer])
            if contact < 0:
                total = int(t.protein_energy[pair[0]]) + int(t.protein_energy[pair[1]])
                total += int(t.dimer_energy[dimer]) + contact
                candidates[total].append(("dimer", pair,
                                          bool(t.dimer_promoter_pol[promoter, dimer])))
        for energy in sorted(candidates):
            if len(candidates[energy]) == 1:
                return candidates[energy][0][2]
        return False

    def expression(self, genotype: np.ndarray, state: CellState) -> int:
        genes = _genotype(genotype)
        bits = 0
        for gene in genes:
            bits = (bits << 1) | int(self.promoter_active(int(gene) >> 16, state))
        return bits

    def promoter_response(self, genotype: np.ndarray,
                          proteins: Counter[int] | dict[int, int] | None = None,
                          dimers: Dimers | dict[tuple[int, int], int] | None = None) -> int:
        """Evaluate an already assembled inventory without altering its species."""
        return self.expression(genotype, CellState(Counter(proteins or {}),
                                                   Counter(dimers or {})))

    def functional_output(self, genotype: np.ndarray, expression: int) -> int:
        """Two-gene tissue output, masking expressed genes with unfolded products."""
        genes = _genotype(genotype)
        if len(genes) != 2:
            raise ValueError("functional tissue output requires two genes")
        products = [int(self.tables.gene_to_protein[int(gene) & 65535]) for gene in genes]
        return expression & ((2 if products[0] else 0) | (1 if products[1] else 0))

    def spatial_descriptor(self, genotype: np.ndarray) -> np.ndarray:
        """Seven-symbol GRN descriptor for the paper's six chemical inputs.

        Inputs 3/4/5 contain a preformed AB dimer when AB can bind. An added A
        or B in inputs 4/5 remains free; an AA or BB bond cannot rearrange the
        existing AB dimer during promoter evaluation.
        """
        genes = _genotype(genotype)
        if len(genes) != 2:
            raise ValueError("spatial descriptor requires two genes")
        a, b = [int(self.tables.gene_to_protein[int(gene) & 65535]) for gene in genes]
        ab = self.tables.dimer_for(a, b) if a and b else -1
        inventories = [CellState() for _ in range(6)]
        if b:
            inventories[1].proteins[b] += 1
        if a:
            inventories[2].proteins[a] += 1
        if ab >= 0:
            pair = (min(a, b), max(a, b))
            for index in (3, 4, 5):
                inventories[index].dimers[pair] += 1
            inventories[4].proteins[a] += 1
            inventories[5].proteins[b] += 1
        else:
            if a:
                inventories[3].proteins[a] += 1
                inventories[4].proteins[a] += 2
                inventories[5].proteins[a] += 1
            if b:
                inventories[3].proteins[b] += 1
                inventories[4].proteins[b] += 1
                inventories[5].proteins[b] += 2
            for index in (3, 4, 5):
                inventories[index], _ = self.react(inventories[index])
        result = [self.functional_output(genes, self.expression(genes, inventory))
                  for inventory in inventories]
        result.append(int(ab >= 0))
        return np.asarray(result, dtype=np.uint8)

    def _regulatory_state(self, genotype: np.ndarray, expression: int) -> CellState:
        state = CellState(proteins=self.translated(genotype, expression))
        state, _ = self.react(state)
        return state

    def regulatory(self, genotype: np.ndarray, max_truth_states: int = 1_048_576) -> np.ndarray:
        genes = _genotype(genotype)
        if len(genes) > 63:
            raise ValueError("encoded regulatory states require at most 63 genes")
        count = 1 << len(genes)
        if count > max_truth_states:
            raise ValueError("truth table exceeds max_truth_states")
        return np.asarray([self.expression(genes, self._regulatory_state(genes, state))
                           for state in range(count)], dtype=np.uint64)

    def attractor(self, genotype: np.ndarray, max_steps: int = 100_000) -> tuple[int, ...]:
        genes = _genotype(genotype)
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        visited: dict[int, int] = {}
        sequence: list[int] = []
        expression = 0
        for _ in range(max_steps):
            if expression in visited:
                return tuple(sequence[visited[expression]:])
            visited[expression] = len(sequence)
            sequence.append(expression)
            expression = self.expression(genes,
                                         self._regulatory_state(genes, expression))
        raise RuntimeError("regulatory attractor incomplete: step limit reached")

    def step(self, genotype: np.ndarray, state: CellState) -> StepResult:
        """Perform the two chemistry phases and a promoter snapshot."""
        genes = _genotype(genotype)
        after_input, first_fissions = self.react(state)
        expression = self.expression(genes, after_input)
        after_input.proteins = self.translated(genes, expression)
        after_input.dimers.clear()
        after_output, second_fissions = self.react(after_input)
        all_fissions = first_fissions + second_fissions
        retained = Counter()
        for (_, _, _, met), count in after_output.bound.items():
            retained[met] += count
        generated = Counter()
        for _, left, right in all_fissions:
            generated[canonical_metabolite(left)] += 1
            generated[canonical_metabolite(right)] += 1
        for met, count in generated.items():
            retained[met] += min(count, after_output.metabolites[met])
        return StepResult(CellState(after_output.proteins, after_output.dimers,
                                    retained, Counter()), expression, all_fissions)

    def _assay_phase(self, genotype: np.ndarray, state: CellState,
                     max_steps: int) -> str:
        seen: set[tuple] = set()
        for _ in range(max_steps):
            key = state.key()
            if key in seen:
                return "intact"
            seen.add(key)
            result = self.step(genotype, state)
            if result.fissions:
                return "cleaved"
            state = result.state
            if not state.metabolites and not state.bound:
                return "intact"
        return "incomplete"

    def metabolic(self, genotype: np.ndarray, environment: Iterable[str] | None = None,
                  max_steps: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
        genes = _genotype(genotype)
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        mets = tuple(environment) if environment is not None else tuple(
            met for met, unbreakable in zip(self.tables.met_strings,
                                            self.tables.met_unbreakable, strict=True)
            if not unbreakable)
        for met in mets:
            if not 4 <= len(met) <= 8 or set(met) - {"0", "1"}:
                raise ValueError("assay metabolites must be binary strings of length 4 through 8")
        try:
            phases = self.attractor(genes, max_steps)
        except RuntimeError:
            return (np.zeros(len(mets), dtype=np.uint8),
                    np.full(len(mets), "incomplete", dtype="<U10"))
        values = np.zeros(len(mets), dtype=np.uint8)
        statuses = np.full(len(mets), "complete", dtype="<U10")
        for index, met in enumerate(mets):
            results = []
            for phase in phases:
                initial = self._regulatory_state(genes, phase)
                initial.metabolites[canonical_metabolite(met)] += 1
                outcome = self._assay_phase(genes, initial, max_steps)
                results.append(outcome)
                if outcome == "cleaved":
                    values[index] = 1
                    break
            if not values[index] and "incomplete" in results:
                statuses[index] = "incomplete"
        return values, statuses
