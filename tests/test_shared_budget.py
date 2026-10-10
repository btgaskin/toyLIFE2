"""Admission of mixed live sessions and fixed assays under one Engine budget."""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from toylife2 import BatchRunner, ControllerSpec, Model, PortSpec
from toylife2.maps import enumerate_map
from toylife2.runtime import Engine
from toylife2.specs import ExecutionSpec, SpatialSpec
from toylife2.tables import Tables


class SharedBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="toylife2-shared-budget-")
        cls.tables = Tables.load(cache_dir=cls.directory.name)
        cls.controller = ControllerSpec(ports=PortSpec(metabolites=()))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def runner(self, mib):
        return BatchRunner(
            ExecutionSpec(memory_budget_mib=mib, cache_dir=self.directory.name), tables=self.tables
        )

    def test_live_session_rejects_lazy_tables_then_release_admits_assay(self):
        with self.runner(5) as runner:
            engine = runner.model.engine
            session = runner.chemistry(np.zeros((18000, 2), np.uint32), self.controller)
            reserved = engine._reserved_session_bytes
            self.assertEqual(reserved, session.buffer_bytes)
            with self.assertRaisesRegex(MemoryError, "table pack"):
                runner.model.regulatory(np.zeros((100, 2), np.uint32))
            self.assertEqual(engine._table_bytes, 0)
            self.assertEqual(engine._buffers, {})
            self.assertEqual(engine._reserved_session_bytes, reserved)
            runner.release(session)
            result = runner.model.regulatory(np.zeros((100, 2), np.uint32))
            np.testing.assert_array_equal(result.values, 0)
            table_bytes = engine._table_bytes
            with self.assertRaises(MemoryError):
                runner.chemistry(np.zeros((18000, 2), np.uint32), self.controller)
            self.assertEqual(engine._reserved_session_bytes, 0)
            self.assertEqual(engine._table_bytes, table_bytes)

    def test_multiple_sessions_release_and_close_preserve_other_runner(self):
        model = Model(
            self.tables, ExecutionSpec(memory_budget_mib=4, cache_dir=self.directory.name)
        )
        first = BatchRunner(model=model)
        second = BatchRunner(model=model)
        a = first.chemistry(np.zeros((9000, 2), np.uint32), self.controller)
        b = second.chemistry(np.zeros((10784, 2), np.uint32), self.controller)
        before = model.engine._reserved_session_bytes
        self.assertLess(4 * 1024 * 1024 - before, 212)
        with self.assertRaises(MemoryError):
            first.chemistry(np.zeros((1, 2), np.uint32), self.controller)
        self.assertEqual(model.engine._reserved_session_bytes, before)
        first.release(a)
        first.release(a)
        self.assertEqual(model.engine._reserved_session_bytes, b.buffer_bytes)
        c = first.chemistry(np.zeros((9000, 2), np.uint32), self.controller)
        first.close()
        self.assertEqual(model.engine._reserved_session_bytes, b.buffer_bytes)
        self.assertGreater(c.buffer_bytes, 0)
        second.close()
        self.assertEqual(model.engine._reserved_session_bytes, 0)

    def test_tissue_output_competes_with_warm_tables_and_live_session(self):
        with self.runner(5) as runner:
            runner.model.regulatory(np.zeros((1, 2), np.uint32))
            session = runner.chemistry(np.zeros((5000, 2), np.uint32), self.controller)
            genes = np.zeros((100, 2), np.uint32)
            spec = SpatialSpec(cells=31, steps=50)
            reserved = runner.model.engine._reserved_session_bytes
            with self.assertRaises(MemoryError):
                runner.model.patterns(genes, spec)
            self.assertEqual(runner.model.engine._reserved_session_bytes, reserved)
            self.assertEqual(runner.model.engine._reserved_assay_bytes, 0)
            runner.release(session)
            result = runner.model.patterns(genes, spec)
            self.assertEqual(result.values.shape, (100, 50, 31))
            self.assertEqual(runner.model.engine._reserved_assay_bytes, 0)

    def test_metabolic_outputs_compete_with_sessions_for_both_gene_paths(self):
        with self.runner(110) as runner:
            # Table conversions have their own numerical peak during the first upload.
            runner.model.metabolic(np.zeros((1, 2), np.uint32), ["0000"])
            session = runner.chemistry(np.zeros((5000, 2), np.uint32), self.controller)
            engine = runner.model.engine
            before = (engine._reserved_session_bytes, engine._table_bytes)
            for genes, samples in ((2, 75000), (5, 68000)):
                with self.subTest(genes=genes):
                    with self.assertRaises(MemoryError):
                        runner.model.metabolic(np.zeros((samples, genes), np.uint32), ["0000"])
                    self.assertEqual((engine._reserved_session_bytes, engine._table_bytes), before)
                    self.assertEqual(engine._reserved_assay_bytes, 0)
            runner.release(session)
            for genes in (2, 5):
                result = runner.model.metabolic(np.zeros((1, genes), np.uint32), ["0000"])
                np.testing.assert_array_equal(result.status, "complete")

    def test_owned_result_copies_are_admitted_at_the_return_peak(self):
        with self.runner(6) as runner:
            engine = runner.model.engine
            runner.model.regulatory(np.zeros((1, 2), np.uint32))
            before = engine._table_bytes
            with self.assertRaises(MemoryError):
                runner.model.regulatory(np.zeros((20000, 2), np.uint32))
            self.assertEqual(engine._table_bytes, before)
            self.assertEqual(engine._reserved_assay_bytes, 0)
            result = runner.model.regulatory(np.zeros((12000, 2), np.uint32))
            self.assertEqual(result.values.shape, (12000, 4))
            self.assertFalse(result.values.flags.writeable)
            self.assertFalse(result.status.flags.writeable)

    def test_snapshot_fields_are_admitted_before_copying(self):
        with self.runner(4) as runner:
            session = runner.chemistry(np.zeros((19784, 2), np.uint32), self.controller)
            with patch(
                "toylife2.chemistry._owned", side_effect=AssertionError("copy before admit")
            ):
                with self.assertRaises(MemoryError):
                    session.snapshot(fields=("genotypes",))
            self.assertEqual(dict(session.snapshot(fields=())), {})
            self.assertEqual(runner.model.engine._reserved_session_bytes, session.buffer_bytes)

    def test_tissue_snapshot_accounts_for_other_live_sessions(self):
        with self.runner(5) as runner:
            tissue = runner.tissue(np.zeros((100, 2), np.uint32), SpatialSpec(cells=31, steps=1))
            engine = runner.model.engine
            available = engine._available_bytes()
            lanes = (available - 13000) // 212
            chemistry = runner.chemistry(np.zeros((lanes, 2), np.uint32), self.controller)
            with self.assertRaises(MemoryError):
                tissue.snapshot()
            selected = tissue.snapshot(fields=("tissue",))
            self.assertEqual(selected["tissue"].shape, (100, 31))
            self.assertFalse(selected["tissue"].flags.writeable)
            runner.release(chemistry)
            self.assertEqual(tissue.snapshot()["genotypes"].shape, (100, 2))

    def test_rollout_snapshot_selects_fields_before_derived_arrays(self):
        from toylife2.protocols import EpisodeSpec

        with self.runner(1) as runner:
            rollout = runner.rollout(
                np.zeros((4, 2), np.uint32), protocol=EpisodeSpec(horizon=1),
                controller=self.controller,
            )
            engine = runner.model.engine
            lanes = engine._available_bytes() // 212
            runner.chemistry(np.zeros((lanes, 2), np.uint32), self.controller)
            with self.assertRaises(MemoryError):
                rollout.snapshot(include_pending=True)
            self.assertEqual(rollout.snapshot(fields=("tick",)), {"tick": 0})
            self.assertEqual(engine._reserved_assay_bytes, 0)

    def test_robustness_retains_base_and_neighbour_outputs_in_admission(self):
        with self.runner(6) as runner:
            runner.model.regulatory(np.zeros((1, 2), np.uint32))
            with self.assertRaises(MemoryError):
                runner.model.robustness(np.zeros((500, 2), np.uint32), phenotype="regulatory")
            self.assertEqual(runner.model.engine._reserved_assay_bytes, 0)
            result = runner.model.robustness(np.zeros((1, 2), np.uint32), phenotype="regulatory")
            self.assertEqual(result["robustness"].shape, (1,))

    def test_upload_failure_keeps_table_counters_and_pack_coherent(self):
        engine = Engine(self.tables, ExecutionSpec(cache_dir=self.directory.name))
        allocate = engine.qd.ndarray
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected upload failure")
            return allocate(*args, **kwargs)

        fake = SimpleNamespace(ndarray=fail_second, i32=engine.qd.i32)
        with patch.object(engine, "qd", fake):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                engine._table_pack(("gene_to_protein", "protein_energy"))
        self.assertEqual(engine._table_bytes, 0)
        self.assertEqual(engine._buffers, {})
        self.assertEqual(engine._reserved_assay_bytes, 0)
        engine._table_pack(("gene_to_protein", "protein_energy"))
        self.assertEqual(engine._table_bytes, (65536 + 2711) * 4)

    def test_weighted_map_uses_shared_engine_admission(self):
        with self.runner(5) as runner:
            session = runner.chemistry(np.zeros((18000, 2), np.uint32), self.controller)
            engine = runner.model.engine
            with self.assertRaises(MemoryError):
                enumerate_map(self.tables, engine=engine, stop_pair=1, tile_pairs=1)
            self.assertEqual(engine._table_bytes, 0)
            self.assertEqual(engine._reserved_assay_bytes, 0)
            runner.release(session)
            result = enumerate_map(self.tables, engine=engine, stop_pair=1, tile_pairs=1)
            self.assertEqual(result["status"], "complete")
            self.assertGreater(engine._table_bytes, 0)
            self.assertEqual(engine._reserved_assay_bytes, 0)


if __name__ == "__main__":
    unittest.main()
