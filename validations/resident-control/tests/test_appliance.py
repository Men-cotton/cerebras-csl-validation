"""SDK-free finite schedule, native carrier, saved-audit and cleanup tests."""
import importlib
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
if not (HERE/"saved_audit.py").exists():
    HERE = HERE.parent
PACKAGE = "_e215_control_appliance_test"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(HERE)]
    sys.modules[PACKAGE] = package
common = importlib.import_module(PACKAGE+".saved_audit")
runner = importlib.import_module(PACKAGE+".sdk_runner")


class Types:
    MEMCPY_16BIT = "u16"
    MEMCPY_32BIT = "u32"
    ROW_MAJOR = "row"


def fake_build(build, manifest):
    (build/"bin").mkdir(parents=True)
    common.write_json(build/"out.json", dict(params=manifest["params"]))
    symbols = []
    for name, (_, dtype) in dict(manifest["inputs"], **manifest["outputs"]).items():
        symbols.append(dict(name=name, id=len(symbols), kind="Var",
                            type={"float32": "[*]f32", "uint32": "[*]u32", "uint16": "[*]u16"}[dtype]))
    for name, args in (("f_run", []), ("f_resume", [dict(name="expected", type="u32")])):
        symbols.append(dict(name=name, id=len(symbols), kind="Func", type="void", inputs=args))
    common.write_json(build/"bin/out_rpc.json", dict(rpc_symbols=symbols))


def fake_sdk(case):
    manifest = common.load(case)
    expected = common.load_npz(case/"expected.npz")
    params = manifest["params"]
    instances = []

    class SdkRuntime:
        def __init__(self, build, **options):
            self.uploads, self.launches, self.stops, self.generation = {}, [], 0, 0
            self.options = options
            instances.append(self)

        def get_id(self, name):
            return name

        def load(self):
            pass

        def run(self):
            pass

        def memcpy_h2d(self, name, values, x, y, width, height, count, **options):
            assert options["nonblock"] is False and options["streaming"] is False
            assert options["order"] == Types.ROW_MAJOR
            if manifest["inputs"][name][1] == "uint16":
                assert values.dtype == np.uint32 and np.all(values < 65536)
                assert options["data_type"] == Types.MEMCPY_16BIT
            self.uploads[name] = values.reshape(width*height, count).copy()

        def launch(self, name, *args, **options):
            assert options == dict(nonblock=False)
            self.launches.append((name, args))
            if name == "f_run":
                assert not args and self.generation == 0
                self.generation = 8 if self.uploads["control"][0, 0] == 0 else 1
            else:
                assert name == "f_resume" and len(args) == 1 and type(args[0]) is np.uint32
                assert int(args[0]) == self.generation
                self.generation += 1

        def memcpy_d2h(self, values, name, x, y, width, height, count, **options):
            assert options["nonblock"] is False
            selected = "resident" if self.uploads["control"][0, 0] == 0 else "staged"
            if name == "history":
                value = expected["history"+str(self.generation)].astype(np.float32)
            elif name in ("run_summary", "checkpoint_step"):
                value = common.markers(params, self.generation, selected)["summary" if name == "run_summary" else "tags"]
            elif name == "control_status":
                value = np.tile(np.array([3, 0 if selected == "resident" else 7, 0, 1], np.uint32), (width*height, 1))
            elif name == "step_times":
                value = np.zeros((width*height, 96), np.uint32)
                for pe in range(width*height):
                    for step in range(8):
                        value[pe, step*6] = 1000*pe+100*step+10
                        value[pe, step*6+3] = 1000*pe+100*step+60
            elif name in manifest["outputs"]:
                value = expected[name].astype(np.dtype(manifest["outputs"][name][1]))
            else:
                value = self.uploads[name]
            values[:] = value.ravel()

        def stop(self):
            self.stops += 1
            assert self.stops == 1

    return (SdkRuntime, Types, Types), instances


class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="e215-host-tests-")
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
                # Signed duplicate-edge, empty-owner and tail feature fixture.
                adverse = cls.root/"adversarial"
                driver.prepare(adverse, "adversarial")
                cls.adverse = adverse
            finally:
                driver.RUNS = old
        cls.manifest = common.load(cls.case)
        cls.build = cls.root/"build"
        fake_build(cls.build, cls.manifest)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def capture(self, selected, case=None):
        case = case or self.case
        build = self.build
        if case != self.case:
            build = self.root/"adverse-build"
            if not build.exists():
                fake_build(build, common.load(case))
        out = Path(tempfile.mkdtemp(prefix="capture-", dir=self.root))/"runtime"
        sdk, instances = fake_sdk(case)
        runner.main(case, build, out, "simulator", selected=selected, sdk=sdk)
        return out, instances[0]

    def test_three_finite_schedules_and_carriers(self):
        for selected in common.ARMS:
            with self.subTest(arm=selected):
                out, runtime = self.capture(selected)
                result = common.audit(self.case, out)
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(runtime.stops, 1)
                self.assertNotIn("release", runtime.uploads)
                self.assertEqual(len(runtime.launches), 1 if selected == "resident" else 8)
                self.assertEqual(runtime.options, dict(cmaddr=None, simfab_numthreads=1, suppress_simfab_trace=True))
                self.assertGreater(result["host_timing"]["launch_to_last_marker_ns"], 0)
                for name, args in runtime.launches[1:]:
                    self.assertIs(type(args[0]), np.uint32)
                self.assertEqual(result["active_cycles"][0], [50]*8)

    def test_adversarial_download(self):
        if not hasattr(self, "adverse"):
            self.skipTest("repository fixture preparation only; exported case is already frozen")
        out, _ = self.capture("download", self.adverse)
        self.assertEqual(common.audit(self.adverse, out)["status"], "PASS")

    def test_numerical_and_native_carrier_rejections(self):
        out, _ = self.capture("download")
        original = common.load_npz(out/"raw.npz")
        changes = (
            ("release", (0, 0), 1),
            ("marker1_summary", (0, 0), 1),
            ("control_status", (0, 1), 1),
            ("after_x_val", (0, 0), .5),
            ("after_x_ptr", (0, 0), 65536),
            ("step_times", (0, 0), 65536),
            ("step_times", (0, 95), 1),
            ("history1", (0, 0), .5),
            ("history", (0, 0), .5),
            ("w1", (0, 0), 1e-7),
        )
        for name, index, delta in changes:
            with self.subTest(field=name, index=index):
                raw = {key: value.copy() for key, value in original.items()}
                raw[name][index] += delta
                np.savez_compressed(out/"raw.npz", **raw)
                report = common.read_json(out/"execution.json")
                report["raw_sha256"] = common.sha(out/"raw.npz")
                common.write_json(out/"execution.json", report)
                self.assertEqual(common.admit_execution(self.case, out)["numerical_status"], "NOT_RUN")
                with self.assertRaises(ValueError):
                    common.audit(self.case, out)

    def test_typed_receipt_and_operation_rejection(self):
        out, _ = self.capture("staged")
        original = common.read_json(out/"execution.json")
        report = __import__("copy").deepcopy(original)
        report["launch_typed_arguments"][1][0]["dtype"] = "int64"
        common.write_json(out/"execution.json", report)
        with self.assertRaisesRegex(ValueError, "typed"):
            common.admit_execution(self.case, out)
        report = __import__("copy").deepcopy(original)
        report["api"][0] = "start"
        common.write_json(out/"execution.json", report)
        with self.assertRaisesRegex(ValueError, "sequence"):
            common.admit_execution(self.case, out)

    def test_constructor_journal_failure_stops_retained_handle(self):
        sdk, instances = fake_sdk(self.case)
        def fail(record):
            if record["api"] == ["construct"]:
                raise OSError("injected constructor-exit journal failure")
        out = self.root/"failed-construct"
        with self.assertRaises(RuntimeError):
            runner.main(self.case, self.build, out, "simulator", sdk=sdk, persist_hook=fail)
        self.assertEqual(instances[0].stops, 1)
        report = common.read_json(out/"execution.json")
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["api"], ["construct", "stop"])

    def test_stop_entry_keyboard_interrupt_still_calls_stop_once(self):
        sdk, instances = fake_sdk(self.case)
        def fail(record):
            if record.get("pending_api") == "stop":
                raise KeyboardInterrupt("injected stop-entry journal interrupt")
        out = self.root/"failed-stop"
        with self.assertRaises(RuntimeError):
            runner.main(self.case, self.build, out, "simulator", sdk=sdk, persist_hook=fail)
        self.assertEqual(instances[0].stops, 1)
        self.assertEqual(common.read_json(out/"execution.json")["status"], "FAILED")

    def test_failed_api_survives_cleanup(self):
        sdk, instances = fake_sdk(self.case)
        original = sdk[0].load
        def load(self):
            raise RuntimeError("injected native load failure")
        sdk[0].load = load
        out = self.root/"failed-load"
        with self.assertRaises(RuntimeError):
            runner.main(self.case, self.build, out, "simulator", sdk=sdk)
        self.assertEqual(instances[0].stops, 1)
        self.assertEqual(common.read_json(out/"execution.json")["failed_api"], "load")
        sdk[0].load = original

    def test_frozen_payload_and_typed_compiler_abi(self):
        build = self.root/"wrong-abi"
        fake_build(build, self.manifest)
        rpc = common.read_json(build/"bin/out_rpc.json")
        rpc["rpc_symbols"][-1]["inputs"][0]["type"] = "i32"
        common.write_json(build/"bin/out_rpc.json", rpc)
        with self.assertRaisesRegex(ValueError, "typed ABI"):
            common.check_abi(build, self.manifest)
        frozen = self.root/"tampered-case"
        shutil.copytree(self.case, frozen)
        with (frozen/"source/pe.csl").open("a") as stream:
            stream.write("\n// changed\n")
        with self.assertRaisesRegex(ValueError, "identity"):
            common.load(frozen)


if __name__ == "__main__":
    unittest.main()
