"""Statistics comparator contracts using real audits and fabricated SDK memory.

These tests reuse the native adapter's fake runtime. No compiler, simulator or
hardware is accessed, and fixture values are not a model-accuracy result.
"""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
if (ROOT / "models/meshgraphnet/native_appliance").is_dir() and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from models.meshgraphnet.native_appliance import test_appliance as fixtures
    from models.meshgraphnet.native_appliance import saved_audit as common
    from models.meshgraphnet.native_appliance import compare_rollouts as resident
except ModuleNotFoundError:
    # Installed as validations/native-gnn/tests/test_compare_statistics.py.
    for directory in (HERE, HERE.parent):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import test_appliance as fixtures
    import saved_audit as common
    import compare_rollouts as resident

# Locally the comparator is adjacent; exported tests live one level below it.
comparator_path = HERE / "compare_statistics.py"
if not comparator_path.is_file():
    comparator_path = HERE.parent / "compare_statistics.py"
spec = importlib.util.spec_from_file_location("e209_compare_statistics", comparator_path)
comparison = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {"saved_audit": common, "compare_rollouts": resident}):
    spec.loader.exec_module(comparison)


class StatisticsComparisonTests(unittest.TestCase):
    resident = fixtures.ResidentWorkerTests.resident
    make_capture = fixtures.RolloutComparisonTests.make_capture

    def setUp(self):
        fixtures.WorkerTests.setUp(self)
        self.root_statistics = 0
        self.extra_params = {}

    def arguments(self, name="capture", backend="hardware"):
        # Freeze the selected mode before the real worker binds its case/build.
        self.manifest["params"].update(ROOT_STATS=self.root_statistics, **self.extra_params)
        common.write_json(self.case / "case.json", self.manifest)
        common.write_json(self.build / "out.json", dict(params=self.manifest["params"]))
        self.archive.write_bytes(fixtures.tar_bytes([
            ("case/" + filename, (self.case / filename).read_bytes())
            for filename in sorted(set(self.manifest["files"]) | {"case.json"})]))
        return fixtures.WorkerTests.arguments(self, name, backend)

    def pair(self, variation=None, backend="hardware"):
        self.root_statistics = 0
        first = self.make_capture("replicated", [2, 2])
        self.root_statistics = 1
        second = self.make_capture("root", [2, 2], variation, backend)
        return first, second

    def replace_raw(self, folder, change):
        path = folder / "runtime/raw.npz"
        arrays = common.load_npz(path, common.CAPTURE_LIMIT)
        change(arrays)
        np.savez_compressed(path, **arrays)
        execution_path = folder / "runtime/execution.json"
        execution = common.read_json(execution_path)
        execution["raw_sha256"] = common.sha(path, common.CAPTURE_LIMIT)
        common.write_json(execution_path, execution)

    def test_matching_modes_accept_different_clocks_and_compare_every_checkpoint(self):
        first, second = self.pair()
        def change_clocks(raw):
            for index in range(2):
                raw[f"launch{index}_clocks"][:, 3] = 5+index
        self.replace_raw(second, change_clocks)
        report = comparison.compare(first, second)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["physical_steps"], 4)
        fields = common.checkpoint_fields(common.load(first / "case"))
        self.assertEqual(set(report["bit_equal_checkpoint_arrays"]),
                         {name for name in fields if not name.endswith("_clocks")})
        for index in range(2):
            self.assertIn(f"launch{index}_lifecycle", report["bit_equal_checkpoint_arrays"])
            self.assertIn(f"launch{index}_stats", report["bit_equal_checkpoint_arrays"])
        self.assertEqual(report["intervals"][0]["per_pe_sum_cycles"], [2]*4)
        self.assertEqual(report["intervals"][1]["per_pe_sum_cycles"], [9]*4)

    def test_changed_uploaded_inputs_or_original_graph_rejected_after_successful_audits(self):
        self.root_statistics = 0
        first = self.make_capture("replicated", [2, 2])
        self.root_statistics = 1
        for variation, message in (("graph", "frozen inputs.npz"), ("original", "original model and graph")):
            with self.subTest(variation=variation):
                second = self.make_capture(variation, [2, 2], variation)
                self.assertEqual(common.audit_report(second)["numerical_status"], "PASS")
                with self.assertRaisesRegex(ValueError, message):
                    comparison.compare(first, second)

    def test_one_ulp_early_checkpoint_difference_rejected_even_when_each_audit_passes(self):
        first, second = self.pair()
        def change(raw):
            values = raw["launch0_delta_h"]
            values[0, 0] = np.nextafter(values[0, 0], np.float32(1))
        self.replace_raw(second, change)
        self.assertEqual(common.audit_report(first)["numerical_status"], "PASS")
        self.assertEqual(common.audit_report(second)["numerical_status"], "PASS")
        with self.assertRaisesRegex(ValueError, "all checkpoint outputs array bits: launch0_delta_h"):
            comparison.compare(first, second)

    def test_statistics_order_and_other_parameter_mismatch_rejected(self):
        first, second = self.pair()
        with self.assertRaisesRegex(ValueError, "baseline then root statistics"):
            comparison.compare(second, first)
        with self.assertRaisesRegex(ValueError, "baseline then root statistics"):
            comparison.compare(first, first)
        self.extra_params = {"S": 2}
        changed = self.make_capture("changed-blocks", [2, 2])
        self.assertEqual(common.audit_report(changed)["numerical_status"], "PASS")
        with self.assertRaisesRegex(ValueError, "matched parameters"):
            comparison.compare(first, changed)

    def test_numerical_failure_and_incomplete_execution_stop_comparison(self):
        first, second = self.pair()
        def change(raw):
            raw["launch0_delta_h"][0, 0] += np.float32(1)
        self.replace_raw(second, change)
        audit = common.audit_report(second)
        self.assertEqual(audit["execution_status"], "PASS")
        self.assertEqual(audit["numerical_status"], "FAIL")
        with self.assertRaisesRegex(ValueError, "both complete numerical audits must pass"):
            comparison.compare(first, second)
        record = common.read_json(first / "capture.json")
        record["status"] = "incomplete"
        common.write_json(first / "capture.json", record)
        with self.assertRaisesRegex(ValueError, "completed capture"):
            comparison.compare(first, second)

    def test_source_provenance_and_backend_mismatch_rejected(self):
        self.root_statistics = 0
        first = self.make_capture("replicated", [2, 2])
        self.root_statistics = 1
        for variation, message in (("source", "same frozen device source"),
                                   ("provenance", "same originating provenance")):
            with self.subTest(variation=variation):
                second = self.make_capture(variation, [2, 2], variation)
                self.assertEqual(common.audit_report(second)["numerical_status"], "PASS")
                with self.assertRaisesRegex(ValueError, message):
                    comparison.compare(first, second)
        second = self.make_capture("simulator", [2, 2], backend="simulator")
        with self.assertRaisesRegex(ValueError, "same execution backend"):
            comparison.compare(first, second)

    def test_cli_failure_is_nonzero_and_does_not_overwrite_existing_report(self):
        first, second = self.pair()
        destination = self.base / "comparison.json"
        arguments = [str(Path(comparison.__file__)), str(second), str(first), "--out", str(destination)]
        with mock.patch.object(sys, "argv", arguments):
            self.assertEqual(comparison.main(), 1)
            self.assertEqual(common.read_json(destination)["status"], "FAIL")
            before = destination.read_bytes()
            with self.assertRaisesRegex(ValueError, "fresh comparison report"):
                comparison.main()
            self.assertEqual(destination.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
