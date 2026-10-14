"""Device-resident batched evaluation of toyLIFE phenotypes."""

# Quadrants' public type hints describe Python calls but cannot type-check
# kernel specializations or the dynamic qd.lang runtime surface.
# pyright: reportCallIssue=false, reportOptionalMemberAccess=false

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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
        self._reserved_assay_bytes = 0
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
                    arch=getattr(qd, self.execution.backend),
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
            expected = getattr(qd, self.execution.backend)
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
        from ._chemistry_cache import GraphCache

        self._chemistry_cache = GraphCache(self.execution.memory_budget_mib * 1024 * 1024)
        self._chemistry_block_size = 8
        from .resident_world import RandomBlockCache

        self._world_block_size = 8
        self._world_random_cache = RandomBlockCache(
            min(16 * 1024 * 1024, self.execution.memory_budget_mib * 1024 * 1024 // 32),
            engine=self,
        )

    _REGULATORY_TABLES = (
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
    _METABOLIC_TABLES = (
        "protein_met_energy",
        "dimer_met_energy",
        "dimer_met_seq",
        "dimer_met_pos",
        "dimer_met_breakable",
        "breaking_seq_to_index",
        "breaking_energy",
    )

    def _missing_table_bytes(self, names: tuple[str, ...], *, peak: bool = False) -> int:
        retained = maximum = 0
        for name in dict.fromkeys(names):
            if name not in self._buffers:
                array = np.asarray(getattr(self.tables, name))
                size = array.size * 4
                retained += size
                conversion = size if array.dtype != np.int32 or not array.flags.c_contiguous else 0
                maximum = max(maximum, retained + conversion)
        return maximum if peak else retained

    def _available_bytes(
        self,
        *,
        table_names: tuple[str, ...] = (),
        fixed_bytes: int = 0,
        minimum_bytes: int = 0,
        message: str = "numerical buffers exceed the execution memory budget",
    ) -> int:
        """Plan against live sessions, retained tables and enclosing assay scratch."""
        available = (
            self.execution.memory_budget_mib * 1024 * 1024
            - self._table_bytes
            - self._reserved_session_bytes
            - self._reserved_assay_bytes
        )
        table_peak = self._missing_table_bytes(table_names, peak=True)
        self._chemistry_cache.reclaim(available - fixed_bytes - minimum_bytes - table_peak)
        available -= self._chemistry_cache.bytes
        self._world_random_cache.reclaim(available - fixed_bytes - minimum_bytes - table_peak)
        available -= self._world_random_cache.bytes
        if table_peak > available:
            raise MemoryError("immutable table pack exceeds the execution memory budget")
        available -= table_peak + fixed_bytes
        if available < minimum_bytes:
            raise MemoryError(message)
        return available

    def _check_memory(self, required_bytes: int, *, message: str) -> None:
        self._available_bytes(minimum_bytes=required_bytes, message=message)

    @staticmethod
    def _snapshot_bytes(value: object) -> int:
        """Count every requested array occurrence, matching owned snapshot copies."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        if isinstance(value, Mapping):
            return sum(Engine._snapshot_bytes(child) for child in value.values())
        if isinstance(value, (tuple, list)):
            return sum(Engine._snapshot_bytes(child) for child in value)
        return 0

    def _check_snapshot_memory(self, session: object, required_bytes: int) -> None:
        unregistered = max(
            0, getattr(session, "buffer_bytes", 0) - getattr(session, "_memory_reserved_bytes", 0)
        )
        self._check_memory(
            required_bytes + unregistered,
            message="requested snapshot exceeds execution memory budget; select fields",
        )

    @contextmanager
    def _temporary_bytes(self, required_bytes: int, *, message: str) -> Iterator[None]:
        """Reserve host working arrays that remain live across nested assays."""
        self._check_memory(required_bytes, message=message)
        self._reserved_assay_bytes += required_bytes
        try:
            yield
        finally:
            self._reserved_assay_bytes -= required_bytes

    def _table_pack(self, names: tuple[str, ...]) -> tuple[object, ...]:
        # Admit the complete missing set before converting or uploading any table.
        self._available_bytes(
            table_names=names, message="immutable table pack exceeds the execution memory budget"
        )
        staged = {}
        for name in dict.fromkeys(names):
            if name not in self._buffers:
                array = np.asarray(getattr(self.tables, name), dtype=np.int32, order="C")
                buffer = self.qd.ndarray(self.qd.i32, shape=array.shape)
                buffer.from_numpy(array)
                staged[name] = buffer
                del array
        required = self._missing_table_bytes(names)
        self._buffers.update(staged)
        self._table_bytes += required
        return tuple(self._buffers[name] for name in names)

    def _table(self, name: str):
        return self._table_pack((name,))[0]

    def _regulatory_tables(self) -> tuple[object, ...]:
        return self._table_pack(self._REGULATORY_TABLES)

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
        capacity = n_genes * (n_genes + 1) // 2
        # Device genes/workspace/truth, host upload and truth transfer per chunk.
        per_sample = 8 * n_genes + states * 4 * (2 * n_genes + 6 * capacity + 2)
        output_bytes = 2 * len(flat) * (states * 4 + np.dtype("<U20").itemsize)
        # ArrayResult takes owned immutable copies of values and status.
        fixed_bytes = genes.nbytes + output_bytes
        available = self._available_bytes(
            table_names=self._REGULATORY_TABLES,
            fixed_bytes=fixed_bytes,
            minimum_bytes=per_sample,
            message="one genotype truth table exceeds the execution memory budget",
        )
        tables = self._regulatory_tables()
        chunk = min(self.execution.batch_size, available // per_sample)
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
            del device_genes, pids, counts, candidate, truth
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

            return metabolic_reachable(self, genes, environment, _validated=True)
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
        output_bytes = 2 * len(flat) * len(env) * (1 + np.dtype("<U20").itemsize)
        # Source arrays remain live during each metabolite chunk. Include both
        # cycle/transition host arrays and their device counterparts and transfers.
        fixed = genes.nbytes + 8 * len(env) + output_bytes
        source_sample_bytes = states * source_lane_bytes + 16 * states + 8 * n_genes + 9
        # Metabolite IDs/lengths on device, values/status device and host readout.
        met_lane_bytes = assay_lane_bytes + 16 + 4 + 4 + 80
        per_sample = source_sample_bytes + len(env) * met_lane_bytes
        names = self._REGULATORY_TABLES + self._METABOLIC_TABLES
        available = self._available_bytes(
            table_names=names,
            fixed_bytes=fixed,
            minimum_bytes=source_sample_bytes + met_lane_bytes,
            message="one metabolic assay exceeds the execution memory budget",
        )
        buffers = self._table_pack(names)
        reg_tables = buffers[: len(self._REGULATORY_TABLES)]
        met_tables = buffers[len(self._REGULATORY_TABLES) :]
        if available >= per_sample:
            chunk = min(self.execution.batch_size, available // per_sample)
            met_chunk = len(env)
        else:
            chunk = 1
            met_chunk = min(len(env), (available - source_sample_bytes) // met_lane_bytes)
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
                del device_met_indices, device_met_lengths, pids, pcounts, prev_pcounts
                del dims, candidates, p_demand, d_demand, history_p, history_d, values, status
                del numeric, row_status
            del device_genes, source_pids, source_pcounts, source_candidates, truth
            del transitions, cycles, lengths, unfinished, device_cycles, device_lengths
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
