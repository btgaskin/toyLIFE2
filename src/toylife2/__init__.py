"""Python/Quadrants toyLIFE: immutable tables, batched phenotypes, reproducible runs."""

from __future__ import annotations

import numpy as np

from .genotypes import GenomeBatch, decode, encode, neighbours, validate
from .reference import CellState, ReferenceModel, StepResult
from .results import ArrayResult
from .runtime import Engine
from .specs import ExecutionSpec, GeneCopySpec, RunSpec, SpatialSpec
from .tables import Tables

__version__ = "0.1.0"
__all__ = [
    "ArrayResult",
    "CellState",
    "ExecutionSpec",
    "GeneCopySpec",
    "GenomeBatch",
    "Model",
    "ReferenceModel",
    "RunSpec",
    "SpatialSpec",
    "StepResult",
    "Tables",
    "decode",
    "encode",
    "neighbours",
]


class Model:
    """All leading array axes are preserved; the final axis holds ordered genes."""

    def __init__(self, tables: Tables | None = None, execution: ExecutionSpec | None = None):
        self.execution = execution or ExecutionSpec()
        self.tables = tables or Tables.load(cache_dir=self.execution.cache_dir)
        self.engine = Engine(self.tables, self.execution)

    def regulatory(self, genotypes: np.ndarray) -> ArrayResult:
        return self.engine.regulatory(genotypes)

    def metabolic(self, genotypes: np.ndarray, environment=None) -> ArrayResult:
        return self.engine.metabolic(genotypes, environment)

    def neighbourhood(
        self, genotypes: np.ndarray, *, phenotype: str = "metabolic", environment=None
    ) -> ArrayResult:
        if phenotype not in ("regulatory", "metabolic"):
            raise ValueError("phenotype must be regulatory or metabolic")
        genes = validate(genotypes)
        candidate_bytes = genes.nbytes * 20 * genes.shape[-1]
        self.engine._check_memory(
            2 * candidate_bytes + 2 * genes.nbytes,
            message="neighbour genotypes exceed the execution memory budget",
        )
        candidates = neighbours(genes)
        with self.engine._temporary_bytes(
            genes.nbytes + candidates.nbytes,
            message="neighbour genotypes exceed the execution memory budget",
        ):
            if phenotype == "regulatory":
                return self.regulatory(candidates)
            return self.metabolic(candidates, environment)

    def robustness(
        self, genotypes: np.ndarray, *, phenotype: str = "metabolic", environment=None
    ) -> dict:
        """Exact neutral fraction and distinct nonneutral phenotype counts per genotype."""
        genes = validate(genotypes)
        from .analysis import neighbourhood_summary

        with self.engine._temporary_bytes(
            genes.nbytes, message="robustness inputs exceed the execution memory budget"
        ):
            base = (
                self.regulatory(genes)
                if phenotype == "regulatory"
                else self.metabolic(genes, environment)
            )
            with self.engine._temporary_bytes(
                base.values.nbytes + base.status.nbytes,
                message="base phenotype exceeds the execution memory budget",
            ):
                adjacent = self.neighbourhood(genes, phenotype=phenotype, environment=environment)
                self.engine._check_memory(
                    adjacent.status.nbytes
                    + 5 * adjacent.values.nbytes
                    + adjacent.values.size // adjacent.values.shape[-1] * 16,
                    message="robustness summary exceeds the execution memory budget",
                )
                return neighbourhood_summary(base, adjacent)

    def patterns(self, genotypes: np.ndarray, spec: SpatialSpec | None = None) -> ArrayResult:
        from .spatial import patterns

        return patterns(self.tables, genotypes, spec, self.execution, engine=self.engine)

    def cell_step(self, genotype: np.ndarray, state: CellState) -> StepResult:
        """Explicit inventories use the scalar oracle with full product provenance."""
        genes = validate(genotype)
        if genes.ndim != 1:
            raise ValueError("cell_step requires one genotype")
        return ReferenceModel(self.tables).step(genes, state)


# Import after Model so orchestration can share the public model without a cycle.
from .adapters import CandidateBatch, ExternalSession  # noqa: E402
from .batch import BatchRunner  # noqa: E402
from .evaluation import EvaluationResult, Evaluator, ObjectiveOutput  # noqa: E402
from .population import PopulationSession  # noqa: E402
from .protocols import (  # noqa: E402
    BinaryOutputSpec,
    ControllerSpec,
    EpisodeSpec,
    ObjectiveSpec,
    PortSpec,
    ReadoutSpec,
    StructuralInterface,
    TaskSpec,
    WorkLimits,
)
from .sessions import Checkpoint, StepReport  # noqa: E402

__all__ += [
    "BinaryOutputSpec",
    "BatchRunner",
    "Checkpoint",
    "ControllerSpec",
    "StructuralInterface",
    "EpisodeSpec",
    "ObjectiveSpec",
    "PortSpec",
    "ReadoutSpec",
    "StepReport",
    "TaskSpec",
    "WorkLimits",
    "CandidateBatch",
    "ExternalSession",
    "Evaluator",
    "EvaluationResult",
    "ObjectiveOutput",
    "PopulationSession",
]
