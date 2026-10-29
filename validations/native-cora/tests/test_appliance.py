"""SDK-free native-feature, independent audit, ABI and cleanup gates."""
import argparse
import importlib
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
PACKAGE = "_e217_native_cora_test"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(HERE)]
    sys.modules[PACKAGE] = package
common = importlib.import_module(PACKAGE+".saved_audit")
runner = importlib.import_module(PACKAGE+".sdk_runner")


class Types:
    MEMCPY_16BIT, MEMCPY_32BIT, ROW_MAJOR = "u16", "u32", "row"


def fake_build(build, manifest):
    (build/"bin").mkdir(parents=True)
    common.write_json(build/"out.json", dict(params=manifest["params"]))
    symbols = []
    for name, (_, dtype) in dict(manifest["inputs"], **manifest["outputs"]).items():
        symbols.append(dict(name=name, id=len(symbols), kind="Var", type={"float32": "[*]f32", "uint16": "[*]u16", "uint32": "[*]u32"}[dtype]))
    symbols.append(dict(name="f_run", id=len(symbols), kind="Func", type="void", inputs=[]))
    common.write_json(build/"bin/out_rpc.json", dict(rpc_symbols=symbols))


def fake_sdk(case):
    manifest, instances = common.load(case), []
    expected = common.load_npz(case/"expected.npz")
    class SdkRuntime:
        def __init__(self, build, **options):
            self.uploads, self.launched, self.stops = {}, False, 0
            self.build, self.options = build, options
            instances.append(self)
        def get_id(self, name):
            return name
        def load(self):
            pass
        def run(self):
            pass
        def memcpy_h2d(self, name, value, x, y, c, r, count, **kwargs):
            assert kwargs["nonblock"] is False and kwargs["order"] == Types.ROW_MAJOR
            if manifest["inputs"][name][1] == "uint16":
                assert value.dtype == np.uint32 and np.all(value < 65536)
                assert kwargs["data_type"] == Types.MEMCPY_16BIT
            self.uploads[name] = value.reshape(c*r, count).copy()
        def launch(self, symbol, *args, **kwargs):
            assert symbol == "f_run" and not args and kwargs == dict(nonblock=False)
            assert not self.launched
            self.launched = True
        def memcpy_d2h(self, value, symbol, x, y, c, r, count, **kwargs):
            assert kwargs["nonblock"] is False
            if symbol in self.uploads:
                content = self.uploads[symbol]
            elif symbol == "stats" and not self.launched:
                content = np.zeros((c*r, 8), np.uint32)
            elif symbol == "clocks":
                content = np.tile(np.array([10, 0, 0, 1010, 0, 0], np.uint32), (c*r, 1))
            else:
                content = expected[symbol].astype(manifest["outputs"][symbol][1])
            value[:] = content.ravel()
        def stop(self):
            self.stops += 1
            assert self.stops == 1
    return (SdkRuntime, Types, Types), instances


class NativeCoraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="e217-host-tests-")
        cls.root = Path(cls.temporary.name)
        if (HERE/"cases").exists():
            cases = sorted((HERE/"cases").iterdir())
            cls.case = next((case for case in cases if common.read_json(case/"case.json")["kind"] == "tiny"), cases[0])
            cls.native = None
        else:
            driver = importlib.import_module(PACKAGE+".run")
            with mock.patch.object(driver, "RUNS", cls.root):
                cls.case, cls.native = cls.root/"tiny", cls.root/"native64"
                driver.prepare(cls.case)
                driver.prepare(cls.native, "cora", 64, 8, 2, 8)
        cls.manifest = common.load(cls.case)
        cls.build = cls.root/"build"
        fake_build(cls.build, cls.manifest)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def capture(self, case=None):
        case = case or self.case
        build = self.build
        if case != self.case:
            build = self.root/"native-build"
            if not build.exists():
                fake_build(build, common.load(case))
        output = Path(tempfile.mkdtemp(prefix="capture-", dir=self.root))/"runtime"
        sdk, instances = fake_sdk(case)
        runner.main(case, build, output, "simulator", sdk=sdk)
        return output, instances[0]

    def test_complete_inference_capture_and_independent_audit(self):
        output, runtime = self.capture()
        result = common.audit(self.case, output)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["predictions"]["all_nodes_disagree"], 0)
        self.assertEqual(runtime.stops, 1)
        self.assertEqual(runtime.options, dict(cmaddr=None, simfab_numthreads=1, suppress_simfab_trace=True))
        self.assertEqual(result["local_interval_min_max"], [1000, 1000])

    def test_beyond_first_sixteen_feature_and_shard_guard(self):
        manifest = self.manifest
        if manifest["kind"] != "tiny":
            self.skipTest("local adversarial fixture only")
        provenance = common.read_json(self.case/"provenance.json")
        self.assertGreater(provenance["beyond16_feature_logit_difference"], 1e-3)
        arrays = common.load_npz(self.case/"inputs.npz")
        nodes = common.load_npz(self.case/"node_ids.npz")["nodes"]
        original = common.native_inputs(self.case, manifest)
        params = manifest["params"]
        for pe in range(params["C"]*params["R"]):
            live = int(arrays["x_ptr"][pe, -1])
            features = arrays["x_col"][pe, :live].astype(np.int64)*params["C"]+pe % params["C"]
            arrays["x_val"][pe, :live][features >= 16] = 0
        with self.assertRaises(ValueError):
            common.decode_uploads(params, arrays, nodes, original)

    def test_native_calibration_keeps_all1433_and_has_no_accuracy(self):
        if self.native is None:
            self.skipTest("repository native calibration preparation only")
        manifest = common.load(self.native)
        self.assertEqual(manifest["features"], 1433)
        self.assertEqual(manifest["params"]["FW"], 180)
        raw = common.native_inputs(self.native, manifest)
        self.assertEqual(raw["x"].shape, (64, 1433))
        output, _ = self.capture(self.native)
        result = common.audit(self.native, output)
        self.assertTrue(result["calibration_only"])
        self.assertNotIn("device_validation_accuracy", result["predictions"])

    def test_saved_capture_rejects_padding_inputs_counters_and_carriers(self):
        output, _ = self.capture()
        original = common.load_npz(output/"raw.npz")
        changes = [("pre", (0, 0), 1), ("logits", (0, 0), 1), ("stats", (0, 4), 1),
                   ("before_w0", (0, 0), 1e-7), ("after_w1", (0, 0), 1e-7),
                   ("before_x_ptr", (0, 0), 65536), ("clocks", (0, 0), 65536), ("initial_stats", (0, 0), 1)]
        params = self.manifest["params"]
        first_padding = 7-((7-1)//params["T"])*params["T"]
        if first_padding < params["T"]:
            changes.append(("panel_sum", (0, first_padding), 1e-8))
        for name, index, delta in changes:
            with self.subTest(field=name):
                changed = {key: value.copy() for key, value in original.items()}
                if name in ("before_w0", "after_w1"):
                    changed[name].view(np.uint32)[index] ^= np.uint32(1)
                else:
                    changed[name][index] += delta
                self.assertFalse(np.array_equal(changed[name].view(np.uint32), original[name].view(np.uint32)))
                np.savez_compressed(output/"raw.npz", **changed)
                record = common.read_json(output/"execution.json")
                record["raw_sha256"] = common.sha(output/"raw.npz")
                common.write_json(output/"execution.json", record)
                self.assertEqual(common.admit_execution(self.case, output)["numerical_status"], "NOT_RUN")
                with self.assertRaises(ValueError):
                    common.audit(self.case, output)

    def test_constructor_journal_failure_retains_cleanup_handle(self):
        sdk, instances = fake_sdk(self.case)
        def fail(record):
            if record["api"] == ["construct"]:
                raise OSError("injected constructor-exit journal failure")
        output = self.root/"constructor-failure"
        with self.assertRaises(RuntimeError):
            runner.main(self.case, self.build, output, "simulator", sdk=sdk, persist_hook=fail)
        self.assertEqual(instances[0].stops, 1)
        self.assertEqual(common.read_json(output/"execution.json")["failed_api"], "construct")

    def test_source_pin_and_compiler_launch_abi(self):
        build = self.root/"bad-build"
        fake_build(build, self.manifest)
        rpc = common.read_json(build/"bin/out_rpc.json")
        rpc["rpc_symbols"][-1]["name"] = "run"
        common.write_json(build/"bin/out_rpc.json", rpc)
        with self.assertRaises(ValueError):
            common.check_abi(build, self.manifest)
        case = self.root/"changed-source"
        shutil.copytree(self.case, case)
        with (case/"source/pe.csl").open("a") as stream:
            stream.write("\n// changed\n")
        with self.assertRaises(ValueError):
            common.load(case)

    def test_rejected_forecast_cannot_construct_appliance_compiler(self):
        appliance = importlib.import_module(PACKAGE+".appliance")
        root = self.root/"rejected-package"
        case = root/"cases/rejected"
        shutil.copytree(self.case, case)
        manifest = common.read_json(case/"case.json")
        manifest["admission"].update(code_reserve_bytes=44*1024,
                                     forecast_bytes=common.named_bytes(manifest["params"])+44*1024,
                                     admitted=False)
        common.write_json(case/"case.json", manifest)
        for name in common.HOST_FILES:
            shutil.copyfile(HERE/name, root/name)
        common.write_json(root/"package.json", dict(schema=1, family="native-cora",
                          cases={"rejected": common.sha(case/"case.json")},
                          host_sha256={name: common.sha(root/name) for name in common.HOST_FILES}))
        compiler = mock.Mock()
        args = argparse.Namespace(case="rejected", backend="hardware", compile_workers=1,
                                  namespace="job-operator", out=root/"compile")
        with self.assertRaisesRegex(ValueError, "static forecast rejected"):
            appliance.compile_program(args, root=root, compiler_type=compiler)
        compiler.assert_not_called()
        self.assertFalse(args.out.exists())

    def test_wrapper_cwd_and_private_runtime_artifact(self):
        if not (HERE/"run.py").exists():
            self.skipTest("local driver only")
        driver = importlib.import_module(PACKAGE+".run")
        request = driver.simulation_request(driver.RUNS/"case", driver.RUNS/"out")
        self.assertEqual(request["wrapper_cwd"], str(driver.ROOT))
        self.assertTrue(Path(request["command"][2]).is_relative_to(driver.ROOT))
        case, output = self.root/"copy-case", self.root/"copy-out/runtime"
        shutil.copytree(self.case, case)
        fake_build(case/"build", self.manifest)
        (case/"build/kernel.elf").write_bytes(b"synthetic source bytes")
        digest = common.sha(case/"build/kernel.elf")
        (output.parent/"sdk-work").mkdir(parents=True)
        with mock.patch.object(driver, "RUNS", self.root), mock.patch.object(driver.common, "compiled_inventory", return_value={"synthetic": True}):
            work, build = driver.stage_runtime_build(case, output, dict(inventory={"synthetic": True}))
        (build/"generated").mkdir()
        (build/"generated/default.elf").write_bytes(b"runtime addition")
        self.assertFalse((case/"build/generated").exists())
        self.assertEqual(common.sha(case/"build/kernel.elf"), digest)


if __name__ == "__main__":
    unittest.main()
