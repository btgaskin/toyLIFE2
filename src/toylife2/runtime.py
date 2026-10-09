"""Device-resident batched evaluation of toyLIFE phenotypes."""

# Quadrants' public type hints describe Python calls but cannot type-check
# kernel specializations or the dynamic qd.lang runtime surface.
# pyright: reportCallIssue=false, reportOptionalMemberAccess=false

from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np

from .genotypes import validate
from .results import ArrayResult
from .specs import RULESET, ExecutionSpec
from .tables import Tables


class Engine:
    """One Quadrants backend and one immutable table pack per process."""

    _runtime_backend: str | None = None
    _runtime_cpu_threads: int | None = None

    def __init__(self, tables: Tables, execution: ExecutionSpec | None = None):
        import quadrants as qd

        self.tables = tables
        self.execution = execution or ExecutionSpec()
        self._reserved_session_bytes = 0
        if (
            Engine._runtime_backend is not None
            and Engine._runtime_backend != self.execution.backend
        ):
            raise RuntimeError("a process owns one Quadrants backend; use a new process")
        try:
            program = qd.lang.impl.get_runtime().prog
        except qd.lang.exception.QuadrantsRuntimeError:
            program = None
        if program is None:
            from ._floating import preserve_host_floats

            cache = Path(self.execution.cache_dir)
            cache.mkdir(parents=True, exist_ok=True)
            with preserve_host_floats():
                options: dict[str, object] = {}
                if self.execution.cpu_threads is not None:
                    options["cpu_max_num_threads"] = self.execution.cpu_threads
                qd.init(
                    arch=qd.cpu if self.execution.backend == "cpu" else qd.metal,
                    default_ip=qd.i32,
                    fast_math=False,
                    enable_fallback=False,
                    raise_on_templated_floats=True,
                    offline_cache=True,
                    src_ll_cache=True,
                    offline_cache_file_path=str(cache),
                    **options,  # type: ignore[arg-type]  # Quadrants runtime options are dynamic.
                )
            Engine._runtime_cpu_threads = self.execution.cpu_threads
        elif Engine._runtime_backend is None:
            actual = qd.lang.impl.current_cfg().arch
            expected = qd.cpu if self.execution.backend == "cpu" else qd.metal
            if actual != expected:
                raise RuntimeError("Quadrants already uses a different backend")
        Engine._runtime_backend = self.execution.backend
        if (
            self.execution.backend == "cpu"
            and self.execution.cpu_threads is not None
            and self.execution.cpu_threads != Engine._runtime_cpu_threads
        ):
            raise RuntimeError("a process owns one CPU thread configuration; use a new process")
        self.qd = qd
        self._buffers: dict[str, object] = {}
        self._table_bytes = 0

    def _table(self, name: str):
        if name not in self._buffers:
            array = np.asarray(getattr(self.tables, name), dtype=np.int32, order="C")
            budget = self.execution.memory_budget_mib * 1024 * 1024
            if self._table_bytes + array.nbytes > budget:
                raise MemoryError("immutable table pack exceeds the execution memory budget")
            buffer = self.qd.ndarray(self.qd.i32, shape=array.shape)
            buffer.from_numpy(array)
            self._buffers[name] = buffer
            self._table_bytes += array.nbytes
        return self._buffers[name]

    def _regulatory_tables(self) -> tuple[object, ...]:
        return tuple(
            self._table(name)
            for name in (
                "gene_to_protein",
                "protein_perimeter",
                "protein_energy",
                "dimer_id",
                "dimer_energy",
                "protein_promoter_energy",
                "protein_promoter_pol",
                "dimer_promoter_energy",
                "dimer_promoter_pol",
                "polymerase_energy",
            )
        )

    def _i32(self, shape: tuple[int, ...], initial: np.ndarray | None = None):
        buffer = self.qd.ndarray(self.qd.i32, shape=shape)
        if initial is not None:
            buffer.from_numpy(np.asarray(initial, dtype=np.int32, order="C"))
        return buffer

    def regulatory(self, genotypes: np.ndarray) -> ArrayResult:
        """Return the complete one-step Boolean function for each genotype."""
        from ._kernels import regulatory_step

        genes = validate(genotypes)
        leading = genes.shape[:-1]
        n_genes = genes.shape[-1]
        states = 1 << n_genes
        if states > self.execution.max_truth_states:
            raise ValueError("truth table exceeds max_truth_states")
        if n_genes >= 31:
            raise ValueError("truth table exceeds signed state indexing")
        flat = genes.reshape(-1, n_genes)
        tables = self._regulatory_tables()
        capacity = n_genes * (n_genes + 1) // 2
        per_lane = 4 * (2 * n_genes + 6 * capacity + 1)
        budget = self.execution.memory_budget_mib * 1024 * 1024
        output_bytes = len(flat) * states * 4
        available = budget - self._table_bytes - output_bytes
        if available <= 0:
            raise MemoryError("regulatory output exceeds the execution memory budget")
        if available < states * per_lane:
            raise MemoryError("one genotype truth table exceeds the execution memory budget")
        chunk = min(self.execution.batch_size, max(1, available // (states * per_lane)))
        output = np.empty((len(flat), states), dtype=np.int32)
        for start in range(0, len(flat), chunk):
            batch = flat[start : start + chunk]
            lanes = len(batch) * states
            device_genes = self._i32(batch.shape, batch)
            pids = self._i32((lanes, n_genes))
            counts = self._i32((lanes, n_genes))
            candidate = [self._i32((lanes, capacity)) for _ in range(6)]
            truth = self._i32((len(batch), states))
            regulatory_step(device_genes, *tables, pids, counts, *candidate, truth)
            output[start : start + len(batch)] = truth.to_numpy()
        return ArrayResult(
            output.reshape((*leading, states)),
            np.full(leading or (), "complete", dtype="<U20"),
            {
                "ruleset": RULESET,
                "backend": self.execution.backend,
                "table_digest": str(self.tables.digest),
            },
        )

    def metabolic(
        self,
        genotypes: np.ndarray,
        environment: tuple[str, ...] | list[str] | None = None,
    ) -> ArrayResult:
        """Assay cleavage across regulatory attractor phases on Quadrants."""
        from ._kernels import metabolic_assay, regulatory_step
        from .tables import canonical_metabolite

        genes = validate(genotypes)
        leading = genes.shape[:-1]
        n_genes = genes.shape[-1]
        if n_genes >= 5:
            from ._trajectory import metabolic_reachable

            return metabolic_reachable(self, genes, environment)
        states = 1 << n_genes
        if states > self.execution.max_truth_states:
            raise ValueError("truth table exceeds max_truth_states")
        env = (
            tuple(environment)
            if environment is not None
            else tuple(
                met
                for met, unbreakable in zip(
                    self.tables.met_strings, self.tables.met_unbreakable, strict=True
                )
                if not unbreakable
            )
        )
        if not env:
            raise ValueError("metabolic environment must contain at least one metabolite")
        for met in env:
            if not isinstance(met, str) or not 4 <= len(met) <= 8 or set(met) - {"0", "1"}:
                raise ValueError("metabolites must be binary strings of length four through eight")
        met_indices = np.asarray(
            [self.tables.met_index[canonical_metabolite(met)] for met in env], dtype=np.int32
        )
        met_lengths = np.asarray([len(met) for met in env], dtype=np.int32)
        flat = genes.reshape(-1, n_genes)
        reg_tables = self._regulatory_tables()
        met_tables = tuple(
            self._table(name)
            for name in (
                "protein_met_energy",
                "dimer_met_energy",
                "dimer_met_seq",
                "dimer_met_pos",
                "dimer_met_breakable",
                "breaking_seq_to_index",
                "breaking_energy",
            )
        )
        dim_capacity = n_genes * (n_genes + 1) // 2
        candidate_capacity = 2 * dim_capacity + 3 * n_genes
        # A full boundary state has one free metabolite and at most n_genes
        # protein-derived molecules. This stars-and-bars bound caps the exact
        # seen-state history independently of a large max_steps setting.
        species_capacity = n_genes + dim_capacity
        history_capacity = min(
            self.execution.max_steps,
            comb(n_genes + species_capacity, n_genes),
        )
        source_lane_bytes = 4 * (2 * n_genes + 6 * dim_capacity)
        assay_lane_bytes = 4 * (
            4 * n_genes
            + 7 * dim_capacity
            + 6 * candidate_capacity
            + history_capacity * (n_genes + dim_capacity)
        )
        output_bytes = len(flat) * len(env) * (1 + np.dtype("<U20").itemsize)
        fixed = self._table_bytes + 4 * len(env) * 2 + output_bytes
        source_sample_bytes = states * source_lane_bytes + 4 * states
        per_sample = source_sample_bytes + len(env) * (assay_lane_bytes + 8)
        budget = self.execution.memory_budget_mib * 1024 * 1024
        if fixed + source_sample_bytes + assay_lane_bytes + 8 > budget:
            raise MemoryError("one metabolic assay exceeds the execution memory budget")
        if fixed + per_sample <= budget:
            chunk = min(self.execution.batch_size, max(1, (budget - fixed) // per_sample))
            met_chunk = len(env)
        else:
            chunk = 1
            met_chunk = max(
                1, min(len(env), (budget - fixed - source_sample_bytes) // (assay_lane_bytes + 8))
            )
        output = np.empty((len(flat), len(env)), dtype=np.uint8)
        statuses = np.empty((len(flat), len(env)), dtype="<U20")
        for start in range(0, len(flat), chunk):
            batch = flat[start : start + chunk]
            source_lanes = len(batch) * states
            device_genes = self._i32(batch.shape, batch)
            source_pids = self._i32((source_lanes, n_genes))
            source_pcounts = self._i32((source_lanes, n_genes))
            source_candidates = [self._i32((source_lanes, dim_capacity)) for _ in range(6)]
            truth = self._i32((len(batch), states))
            regulatory_step(
                device_genes, *reg_tables, source_pids, source_pcounts, *source_candidates, truth
            )
            transitions = truth.to_numpy()
            cycles = np.zeros((len(batch), states), dtype=np.int32)
            lengths = np.zeros(len(batch), dtype=np.int32)
            unfinished = np.zeros(len(batch), dtype=bool)
            for row in range(len(batch)):
                seen: dict[int, int] = {}
                sequence: list[int] = []
                state = 0
                for _ in range(self.execution.max_steps):
                    if state in seen:
                        cycle = sequence[seen[state] :]
                        cycles[row, : len(cycle)] = cycle
                        lengths[row] = len(cycle)
                        break
                    seen[state] = len(sequence)
                    sequence.append(state)
                    state = int(transitions[row, state])
                else:
                    unfinished[row] = True
            device_cycles = self._i32(cycles.shape, cycles)
            device_lengths = self._i32(lengths.shape, lengths)
            for met_start in range(0, len(env), met_chunk):
                met_end = min(met_start + met_chunk, len(env))
                met_part = met_indices[met_start:met_end]
                length_part = met_lengths[met_start:met_end]
                device_met_indices = self._i32(met_part.shape, met_part)
                device_met_lengths = self._i32(length_part.shape, length_part)
                assay_lanes = len(batch) * len(met_part)
                pids = self._i32((assay_lanes, n_genes))
                pcounts = self._i32((assay_lanes, n_genes))
                prev_pcounts = self._i32((assay_lanes, n_genes))
                dims = [self._i32((assay_lanes, dim_capacity)) for _ in range(6)]
                candidates = [self._i32((assay_lanes, candidate_capacity)) for _ in range(6)]
                p_demand = self._i32((assay_lanes, n_genes))
                d_demand = self._i32((assay_lanes, dim_capacity))
                history_p = self._i32((assay_lanes, history_capacity, n_genes))
                history_d = self._i32((assay_lanes, history_capacity, dim_capacity))
                values = self._i32((len(batch), len(met_part)))
                status = self._i32((len(batch), len(met_part)))
                metabolic_assay(
                    device_genes,
                    *reg_tables[:3],
                    *reg_tables[4:5],
                    *reg_tables[5:],
                    *met_tables,
                    device_met_indices,
                    device_met_lengths,
                    device_cycles,
                    device_lengths,
                    source_pids,
                    source_pcounts,
                    source_candidates[0],
                    source_candidates[1],
                    source_candidates[2],
                    source_candidates[5],
                    pids,
                    pcounts,
                    prev_pcounts,
                    dims[0],
                    dims[1],
                    dims[2],
                    dims[3],
                    dims[4],
                    dims[5],
                    *candidates,
                    p_demand,
                    d_demand,
                    history_p,
                    history_d,
                    values,
                    status,
                    self.execution.max_steps,
                )
                output[start : start + len(batch), met_start:met_end] = values.to_numpy().astype(
                    np.uint8
                )
                numeric = status.to_numpy()
                row_status = np.where(numeric != 0, "incomplete", "complete")
                row_status[unfinished] = "incomplete"
                statuses[start : start + len(batch), met_start:met_end] = row_status
        return ArrayResult(
            output.reshape((*leading, len(env))),
            statuses.reshape((*leading, len(env))),
            {
                "ruleset": RULESET,
                "backend": self.execution.backend,
                "table_digest": str(self.tables.digest),
                "environment": env,
            },
        )
