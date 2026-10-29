"""Cross-arm bit equality, provenance joins and wrapper-cwd regression tests."""
import importlib
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np

HERE = Path(__file__).resolve().parent
if not (HERE/"saved_audit.py").exists():
    HERE = HERE.parent
PACKAGE = "_e215_control_appliance_compare_test"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(HERE)]
    sys.modules[PACKAGE] = package
common = importlib.import_module(PACKAGE+".saved_audit")
runner = importlib.import_module(PACKAGE+".sdk_runner")
comparison = importlib.import_module(PACKAGE+".compare_arms")
helper_path = HERE/"test_appliance.py"
if not helper_path.exists():
    helper_path = HERE/"tests/test_appliance.py"
spec = importlib.util.spec_from_file_location(PACKAGE+".helpers", helper_path)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def inventory():
    return dict(files={"kernel.elf": dict(sha256="0"*64, bytes=64, static_sram_end=1024,
                                         sram_ranges=[[0, 1024]], outside_sram_ranges=[])},
                artifact_bytes=64, max_static_sram_end=1024, scope="synthetic test inventory")


class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="e215-compare-tests-")
        cls.root = Path(cls.temporary.name)
        if (HERE/"cases").exists():
            cls.case = next((HERE/"cases").iterdir())
        else:
            driver = importlib.import_module(PACKAGE+".run")
            old = driver.RUNS
            driver.RUNS = cls.root
            try:
                cls.case = cls.root/"mean"
                driver.prepare(cls.case, "mean")
            finally:
                driver.RUNS = old
        cls.manifest = common.load(cls.case)
        build = cls.root/"fake-build"
        fixtures.fake_build(build, cls.manifest)
        cls.runtimes = []
        for selected in common.ARMS:
            output = cls.root/selected/"runtime"
            sdk, _ = fixtures.fake_sdk(cls.case)
            runner.main(cls.case, build, output, "simulator", selected=selected, sdk=sdk)
            cls.runtimes.append(output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def scratch(self):
        return Path(tempfile.mkdtemp(prefix="test-", dir=self.root))

    def copies(self, destination):
        result = []
        for original in self.runtimes:
            output = destination/original.parent.name/"runtime"
            shutil.copytree(original, output)
            result.append(output)
        return result

    def rewrite_raw(self, output, raw):
        np.savez_compressed(output/"raw.npz", **raw)
        record = common.read_json(output/"execution.json")
        record["raw_sha256"] = common.sha(output/"raw.npz")
        common.write_json(output/"execution.json", record)

    def test_exact_final_bits_and_explicit_absent_compile_provenance(self):
        result = comparison.compare(case=self.case, runtimes=list(reversed(self.runtimes)))
        self.assertEqual(result["cross_arm_final_bits"], "PASS")
        self.assertEqual(result["build_identity"]["status"], "UNAVAILABLE")
        self.assertEqual(set(result["compared_final_outputs"]), {"history", "w0", "w1", "stats", "release", "checkpoint_step", "run_summary"})
        self.assertIn("no speedup", result["timing_scope"])

    def test_individually_tolerated_bit_difference_is_rejected(self):
        runtimes = self.copies(self.scratch())
        arrays = common.load_npz(self.case/"inputs.npz")
        pe = int(np.flatnonzero(arrays["slots"][:, 0] != 65535)[0])
        raw = common.load_npz(runtimes[1]/"raw.npz")
        raw["history"][pe, 0] = np.nextafter(raw["history"][pe, 0], np.float32(np.inf))
        self.rewrite_raw(runtimes[1], raw)
        self.assertEqual(common.audit(self.case, runtimes[1])["status"], "PASS")
        with self.assertRaisesRegex(ValueError, "cross-arm final bits"):
            comparison.compare(case=self.case, runtimes=runtimes)

    def test_individual_failure_and_duplicate_arm_are_rejected(self):
        runtimes = self.copies(self.scratch())
        raw = common.load_npz(runtimes[2]/"raw.npz")
        raw["marker1_summary"][0, 0] += 1
        self.rewrite_raw(runtimes[2], raw)
        with self.assertRaisesRegex(ValueError, "generation marker"):
            comparison.compare(case=self.case, runtimes=runtimes)
        duplicate = self.scratch()/"runtime"
        shutil.copytree(self.runtimes[0], duplicate)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            comparison.compare(case=self.case, runtimes=[self.runtimes[0], duplicate, self.runtimes[2]])

    def test_local_compile_receipts_must_match(self):
        root = self.scratch()
        case = root/"case"
        shutil.copytree(self.case, case)
        runtimes = self.copies(root)
        record = dict(status="PASS", returncode=0, backend="simulator", case_sha256=common.sha(case/"case.json"),
                      host_sha256={}, inventory=inventory())
        common.write_json(case/"compile.json", record)
        for selected, runtime in zip(common.ARMS, runtimes):
            common.write_json(runtime.parent/"request.json", dict(backend="simulator", arm=selected,
                              case_sha256=record["case_sha256"], compile_sha256=common.sha(case/"compile.json")))
        self.assertEqual(comparison.compare(case=case, runtimes=runtimes)["build_identity"]["status"], "BOUND")
        request = common.read_json(runtimes[2].parent/"request.json")
        request["compile_sha256"] = "f"*64
        common.write_json(runtimes[2].parent/"request.json", request)
        with self.assertRaisesRegex(ValueError, "same local compile"):
            comparison.compare(case=case, runtimes=runtimes)
        (runtimes[2].parent/"request.json").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete local compile"):
            comparison.compare(case=case, runtimes=runtimes)

    def appliance_captures(self, root, mismatch=False):
        results = []
        for index, (selected, runtime) in enumerate(zip(common.ARMS, self.runtimes)):
            folder = root/selected
            shutil.copytree(self.case, folder/"case")
            shutil.copytree(runtime, folder/"runtime")
            execution = common.read_json(folder/"runtime/execution.json")
            execution["backend"] = "hardware"
            common.write_json(folder/"runtime/execution.json", execution)
            fixtures.fake_build(folder/"build", self.manifest)
            common.write_json(folder/"build/inventory.json", inventory())
            (folder/"host").mkdir()
            for name in common.HOST_FILES:
                shutil.copyfile(HERE/name, folder/"host"/name)
            build = dict(schema=1, action="compile", backend="hardware", compile_workers=1,
                         compile_resource_cpu=1000, compiler_args=common.compiler_arguments(self.manifest["params"], "hardware", 1),
                         params=self.manifest["params"], case_name="test-case", case_sha256=common.sha(self.case/"case.json"),
                         case_files=self.manifest["files"], artifact_sha256=("2" if mismatch and index == 2 else "1")*64,
                         host_sha256={name: common.sha(HERE/name) for name in common.HOST_FILES})
            common.write_json(folder/"manifest.json", build)
            request = dict(schema=1, arm=selected, run_id="test-"+selected, case_name="test-case", manifest_sha256=common.sha(folder/"manifest.json"))
            common.write_json(folder/"request.json", request)
            common.write_json(folder/"capture.json", dict(status="COMPLETED", backend="hardware", run_id=request["run_id"],
                              manifest_sha256=request["manifest_sha256"], request_sha256=common.sha(folder/"request.json"),
                              compiled_metadata_sha256={name: common.sha(folder/"build"/name)
                                                        for name in ("out.json", "bin/out_rpc.json", "inventory.json")}))
            results.append(folder)
        return results

    def test_appliance_artifact_and_inventory_join(self):
        captures = self.appliance_captures(self.scratch())
        result = comparison.compare(captures=captures)
        self.assertEqual(result["backend"], "hardware")
        self.assertEqual(result["build_identity"]["artifact_sha256"], "1"*64)
        changed = self.appliance_captures(self.scratch(), mismatch=True)
        for folder in changed:
            self.assertEqual(common.admit_capture(folder)["status"], "PASS")
        with self.assertRaisesRegex(ValueError, "same appliance artifact"):
            comparison.compare(captures=changed)

    def test_runtime_generated_files_cannot_mutate_shared_artifact(self):
        if not (HERE/"run.py").exists():
            self.skipTest("repository local driver only")
        driver = importlib.import_module(PACKAGE+".run")
        root = self.scratch()
        case, out = root/"case", root/"capture/runtime"
        shutil.copytree(self.case, case)
        fixtures.fake_build(case/"build", self.manifest)
        (case/"build/kernel.elf").write_bytes(b"synthetic fixture bytes")
        canonical = common.sha(case/"build/kernel.elf")
        (out.parent/"sdk-work").mkdir(parents=True)
        with mock.patch.object(driver, "RUNS", root), mock.patch.object(driver.common, "compiled_inventory", return_value=inventory()) as scan:
            work, build = driver.stage_runtime_build(case, out, dict(inventory=inventory()))
            scan.assert_called_once_with(build)
        self.assertEqual(build, out.parent/"sdk-work/build")
        self.assertEqual(common.sha(build/"kernel.elf"), canonical)
        (build/"generated").mkdir()
        (build/"generated/default.elf").write_bytes(b"SDK-generated artifact")
        (build/"kernel.elf").write_bytes(b"mutation in private runtime copy")
        self.assertFalse((case/"build/generated").exists())
        self.assertEqual(common.sha(case/"build/kernel.elf"), canonical)
        # A failed copied-inventory check cannot admit a runtime copy.
        other = root/"second/runtime"
        (other.parent/"sdk-work").mkdir(parents=True)
        with mock.patch.object(driver, "RUNS", root), mock.patch.object(driver.common, "compiled_inventory", return_value={}):
            with self.assertRaisesRegex(ValueError, "copied runtime ELF inventory"):
                driver.stage_runtime_build(case, other, dict(inventory=inventory()))

    def test_wrapper_source_visibility_precedes_runtime_chdir(self):
        if not (HERE/"run.py").exists():
            self.skipTest("repository SDK wrapper regression; portable package has no local driver")
        driver = importlib.import_module(PACKAGE+".run")
        case, out = driver.RUNS/"test-case", driver.RUNS/"test-output"
        request = driver.simulation_request(case, out, "resident")
        self.assertEqual(request["wrapper_cwd"], str(driver.ROOT))
        self.assertEqual(request["cwd"], str(driver.ROOT))
        self.assertEqual(request["runtime_cwd"], str(out/"sdk-work"))
        self.assertTrue(Path(request["command"][2]).is_relative_to(Path(request["wrapper_cwd"])))
        self.assertTrue(Path(request["command"][0]).is_relative_to(Path(request["wrapper_cwd"])))
        self.assertIn(str(out/"runtime"), request["command"])


if __name__ == "__main__":
    unittest.main()
