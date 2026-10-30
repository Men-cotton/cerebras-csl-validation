"""SDK-free GraphSAGE export, independent audit and transport failure gates.

Local fixtures use the device packer; their audit uses the separate original-
occurrence implementation. Exported tests use the bundled semantic fixture.
"""
import argparse
import importlib
import importlib.util
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import types
import unittest
from unittest import mock
import numpy as np

HERE = Path(__file__).resolve().parent
if not (HERE/"saved_audit.py").exists():
    HERE = HERE.parent
PACKAGE = "_graphsage_cora_appliance_test"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(HERE)]
    sys.modules[PACKAGE] = package
common = importlib.import_module(PACKAGE+".saved_audit")
runner = importlib.import_module(PACKAGE+".sdk_runner")
exporter = importlib.import_module(PACKAGE+".export_appliance")
appliance = importlib.import_module(PACKAGE+".appliance")
worker = importlib.import_module(PACKAGE+".appliance_worker")


class Types:
    MEMCPY_16BIT, MEMCPY_32BIT, ROW_MAJOR = "u16", "u32", "row"


def fake_build(build, manifest):
    (build/"bin").mkdir(parents=True)
    common.write_json(build/"out.json", dict(params=manifest["params"]))
    symbols = []
    for name, (_, dtype) in dict(manifest["inputs"], **manifest["outputs"]).items():
        symbols.append(dict(name=name, id=len(symbols), kind="Var",
            type={"float32": "[*]f32", "uint16": "[*]u16", "uint32": "[*]u32"}[dtype]))
    symbols.append(dict(name="f_run", id=len(symbols), kind="Func", type="void", inputs=[]))
    common.write_json(build/"bin/out_rpc.json", dict(rpc_symbols=symbols))


def fake_sdk(case, fail_upload=None):
    manifest, instances = common.load(case), []
    expected = common.load_npz(case/"expected.npz")
    class SdkRuntime:
        def __init__(self, build, **options):
            self.uploads, self.reads, self.launched, self.stops = {}, [], False, 0
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
            if name == fail_upload:
                raise RuntimeError("injected upload failure")
            if manifest["inputs"][name][1] == "uint16":
                assert value.dtype == np.uint32 and np.all(value < 65536)
                assert kwargs["data_type"] == Types.MEMCPY_16BIT
            self.uploads[name] = value.reshape(c*r, count).copy()
        def launch(self, symbol, *args, **kwargs):
            assert symbol == "f_run" and not args and kwargs == dict(nonblock=False)
            assert not self.launched and list(self.uploads)[-1] == "slots"
            self.launched = True
        def memcpy_d2h(self, value, symbol, x, y, c, r, count, **kwargs):
            assert kwargs["nonblock"] is False
            self.reads.append(symbol)
            if symbol == "slots":
                content = self.uploads[symbol]
            elif symbol == "stats" and not self.launched:
                content = np.zeros((c*r, 10), np.uint32)
            elif symbol == "clocks":
                assert value.dtype == np.uint32 and kwargs["data_type"] == Types.MEMCPY_16BIT
                content = np.tile(np.array([10, 0, 0, 1010, 0, 0], np.uint32), (c*r, 1))
            else:
                content = expected[symbol].astype(manifest["outputs"][symbol][1])
            value[:] = content.ravel()
        def stop(self):
            self.stops += 1
            assert self.stops == 1
    return (SdkRuntime, Types, Types), instances


def fixture():
    rng = np.random.RandomState(219)
    x = rng.uniform(-.5, 1, (7, 5)).astype(np.float32)
    x[np.abs(x) < .3] = 0
    return dict(x=x, src=np.array([0, 0, 1, 2, 1, 2], np.int64),
        dst=np.array([2, 2, 2, 0, 0, 1], np.int64), labels=np.arange(7, dtype=np.int64) % 3,
        train_idx=np.array([0], np.int64), val_idx=np.array([1], np.int64), test_idx=np.array([], np.int64),
        ws0=rng.uniform(-.5, .5, (5, 4)).astype(np.float32), wn0=rng.uniform(-.5, .5, (5, 4)).astype(np.float32),
        b0=np.array([.25, -.125, 0, .125], np.float32),
        ws1=rng.uniform(-1, 1, (4, 3)).astype(np.float32), wn1=rng.uniform(-1, 1, (4, 3)).astype(np.float32),
        b1=np.array([.5, -.5, 0], np.float32))


def local_prepared(path):
    device = HERE.parent/"cora_device"
    spec = importlib.util.spec_from_file_location("_graphsage_cora_test_packer", device/"model.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    data = fixture()
    packed = model.pack(data, 2, 4)
    path.mkdir()
    np.savez(path/"input_model.npz", **data)
    np.savez(path/"arrays.npz", **packed["arrays"])
    np.save(path/"node_ids.npy", packed["node_ids"])
    common.write_json(path/"case.json", dict(params=packed["params"], metadata=packed["metadata"],
        source_sha256={"models/graphsage/cora_device/"+name: common.sha(device/name) for name in ("layout.csl", "pe.csl")},
        files={name: common.sha(path/name) for name in ("input_model.npz", "arrays.npz", "node_ids.npy")},
        original_checkpoint_sha256=common.sha(path/"input_model.npz"), original_nodes=7, selected_nodes=7,
        input_kind="synthetic", dataset_scope="synthetic-semantic", model_contract=common.MODEL_CONTRACT,
        capture_contract=common.CAPTURE_CONTRACT))
    return device


class GraphSageApplianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="graphsage-appliance-host-tests-")
        cls.root = Path(cls.temporary.name)
        if (HERE/"cases").exists():
            candidates = sorted((HERE/"cases").iterdir())
            cls.case = next((case for case in candidates if common.read_json(case/"case.json")["kind"] == "tiny"), candidates[0])
            cls.prepared = None
        else:
            cls.prepared = cls.root/"prepared"
            cls.device = local_prepared(cls.prepared)
            cls.case = cls.root/"portable"
            exporter.prepare_case(cls.prepared, cls.case, cls.device)
        cls.manifest = common.load(cls.case)
        cls.build = cls.root/"build"
        fake_build(cls.build, cls.manifest)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def capture(self, backend="simulator"):
        output = Path(tempfile.mkdtemp(prefix="capture-", dir=self.root))/"runtime"
        sdk, instances = fake_sdk(self.case)
        runner.main(self.case, self.build, output, backend, "mock-cmaddr" if backend == "hardware" else None, sdk=sdk)
        return output, instances[0]

    def rebind_raw(self, output, raw):
        np.savez_compressed(output/"raw.npz", **raw)
        record = common.read_json(output/"execution.json")
        record["raw_sha256"] = common.sha(output/"raw.npz")
        common.write_json(output/"execution.json", record)

    def test_complete_capture_has_one_launch_and_bounded_reads(self):
        output, runtime = self.capture()
        result = common.audit(self.case, output)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["predictions"]["all_nodes_disagree"], 0)
        self.assertEqual(runtime.reads, ["slots", "stats", "stats", "clocks", "pre", "hidden", "logits"])
        self.assertEqual(list(runtime.uploads), list(common.schema(self.manifest["params"])))
        self.assertEqual(runtime.stops, 1)
        self.assertEqual(runtime.options, dict(cmaddr=None, simfab_numthreads=1, suppress_simfab_trace=True))
        self.assertEqual(result["local_interval_min_max"], [1000, 1000])
        if self.manifest["dataset_scope"] != "full-checkpoint":
            self.assertNotIn("declared_splits", result["predictions"])
        hardware, runtime = self.capture("hardware")
        self.assertEqual(runtime.options, dict(cmaddr="mock-cmaddr"))
        self.assertEqual(common.audit(self.case, hardware)["backend"], "hardware")

    def test_every_output_witness_counter_and_carrier_is_audited(self):
        output, _ = self.capture()
        original = common.load_npz(output/"raw.npz")
        mutations = [("pre", (0, 0), 1), ("hidden", (0, 0), 1), ("logits", (0, 0), 1),
            ("stats", (0, 4), 1), ("upload_tail", (0, 0), 1), ("initial_stats", (0, 0), 1),
            ("clocks", (0, 0), 65536)]
        for name, index, delta in mutations:
            with self.subTest(field=name):
                changed = {key: value.copy() for key, value in original.items()}
                changed[name][index] += delta
                self.rebind_raw(output, changed)
                self.assertEqual(common.admit_execution(self.case, output)["numerical_status"], "NOT_RUN")
                with self.assertRaises(ValueError):
                    common.audit(self.case, output)
        ids = common.load_npz(self.case/"node_ids.npz")["nodes"]
        if np.any(ids < 0):
            pe, local = np.argwhere(ids < 0)[0]
            changed = {key: value.copy() for key, value in original.items()}
            changed["hidden"][pe, local*self.manifest["params"]["H"]] = np.float32(-0.0)
            self.rebind_raw(output, changed)
            with self.assertRaisesRegex(ValueError, "exact zero owner padding"):
                common.audit(self.case, output)

    def test_original_occurrence_and_parameter_decoding_rejects_mutations(self):
        original = common.checkpoint(self.case, self.manifest)
        pristine = common.load_npz(self.case/"inputs.npz")
        nodes = common.load_npz(self.case/"node_ids.npz")["nodes"]
        for name in ("ws0", "wn0", "ws1", "wn1", "b0", "b1", "x_val", "g_val"):
            with self.subTest(field=name):
                arrays = {key: value.copy() for key, value in pristine.items()}
                arrays[name].view(np.uint32)[0, 0] ^= np.uint32(1)
                with self.assertRaises(ValueError):
                    common.decode_uploads(self.manifest["params"], arrays, nodes, original)
        # Removing a duplicate from the original occurrence stream cannot be
        # masked by retaining the former degree coefficient in the packed CSR.
        pairs = list(zip(original["src"].tolist(), original["dst"].tolist()))
        if len(set(pairs)) < len(pairs):
            duplicate = next(i for i, pair in enumerate(pairs) if pair in pairs[:i])
            altered = {name: value.copy() for name, value in original.items()}
            altered["src"], altered["dst"] = np.delete(altered["src"], duplicate), np.delete(altered["dst"], duplicate)
            with self.assertRaises(ValueError):
                common.decode_uploads(self.manifest["params"], pristine, nodes, altered)

    def test_zero_neighborhood_and_zero_norm_keep_raw_logits(self):
        data = fixture()
        data["x"][:] = 0
        data["b0"][:] = 0
        data["src"], data["dst"] = data["src"][:0], data["dst"][:0]
        ref = common.reference(data)
        self.assertTrue(np.all(ref["hidden"] == 0))
        np.testing.assert_array_equal(ref["logits"], np.tile(data["b1"], (7, 1)))
        self.assertLess(ref["logits"][0, 1], 0)

    def test_failed_constructor_journal_or_upload_stops_once(self):
        sdk, instances = fake_sdk(self.case)
        def fail(record):
            if record["api"] == ["construct"]:
                raise OSError("injected constructor-exit journal failure")
        with self.assertRaises(RuntimeError):
            runner.main(self.case, self.build, self.root/"constructor-failure", "simulator", sdk=sdk, persist_hook=fail)
        self.assertEqual(instances[0].stops, 1)
        sdk, instances = fake_sdk(self.case, "ws0")
        output = self.root/"upload-failure"
        with self.assertRaises(RuntimeError):
            runner.main(self.case, self.build, output, "simulator", sdk=sdk)
        self.assertEqual(instances[0].stops, 1)
        self.assertEqual(common.read_json(output/"execution.json")["failed_api"], "h2d:ws0")
        self.assertFalse(instances[0].launched)

    def test_source_abi_and_generated_elf_admission(self):
        build = self.root/"bad-build"
        fake_build(build, self.manifest)
        rpc = common.read_json(build/"bin/out_rpc.json")
        rpc["rpc_symbols"][-1]["inputs"] = [dict(type="u16")]
        common.write_json(build/"bin/out_rpc.json", rpc)
        with self.assertRaises(ValueError):
            common.check_abi(build, self.manifest)
        changed = self.root/"changed-source"
        shutil.copytree(self.case, changed)
        with (changed/"source/pe.csl").open("a") as stream:
            stream.write("\n// changed\n")
        with self.assertRaises(ValueError):
            common.load(changed)
        inventory = dict(files={"kernel.elf": dict(bytes=64, sha256="0"*64, static_sram_end=44*1024)},
                         artifact_bytes=64, max_static_sram_end=44*1024)
        with self.assertRaisesRegex(ValueError, "static SRAM"):
            common.admit_compiled_inventory(inventory)

    def test_export_registered_steps_and_prepared_byte_identity(self):
        if self.prepared is None:
            root = HERE
        else:
            root = self.root/"exported"
            exporter.export({"tiny": self.prepared}, root, "graphsage-test-request", device_source=self.device)
        package = common.read_json(root/"package.json")
        self.assertEqual(package["family"], "graphsage-cora")
        request = common.read_json(root/"request-template.json")
        for offset in range(0, len(request["steps"]), 3):
            compile_step, run_step, audit_step = request["steps"][offset:offset+3]
            self.assertEqual([s["kind"] for s in (compile_step, run_step, audit_step)], ["appliance", "appliance", "host"])
            self.assertEqual(compile_step["args"][0], "compile")
            self.assertEqual(run_step["args"][0], "run")
            self.assertEqual(audit_step["script"], "validations/graphsage-cora/saved_audit.py")
        if self.prepared is not None:
            copied = root/"cases/tiny"
            for source, destination in (("input_model.npz", "input_model.npz"), ("arrays.npz", "inputs.npz"), ("node_ids.npy", "node_ids.npy")):
                self.assertEqual(common.sha(self.prepared/source), common.sha(copied/destination))

    def test_one_compile_and_worker_capture_binding(self):
        root = self.root/"transport-package"
        case = root/"cases/tiny"
        shutil.copytree(self.case, case)
        for name in common.HOST_FILES:
            shutil.copyfile(HERE/name, root/name)
        common.write_json(root/"package.json", dict(schema=1, family="graphsage-cora",
            cases={"tiny": common.sha(case/"case.json")}, host_sha256={name: common.sha(root/name) for name in common.HOST_FILES}))
        calls = []
        class Compiler:
            def __init__(self, **options):
                self.options = options
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def compile(self, source, layout, flags, destination):
                calls.append((source, layout, flags, destination, self.options))
                artifact = Path(destination)/"artifact.zip"
                artifact.write_bytes(b"mock compiler artifact, not a generated ELF")
                return str(artifact)
        args = argparse.Namespace(case="tiny", backend="simulator", compile_workers=1,
            namespace="job-operator", out=self.root/"compile")
        appliance.compile_program(args, root=root, compiler_type=Compiler)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][4]["resource_cpu"], 1000)
        manifest_path = args.out/"native-build.json"
        record, _, _ = appliance.load_build(manifest_path, "simulator", "tiny", root=root)
        request = self.root/"worker-request.json"
        common.write_json(request, dict(schema=1, run_id="mock-run", case_name="tiny", manifest_sha256=common.sha(manifest_path)))
        inventory = dict(files={"kernel.elf": dict(bytes=64, sha256="0"*64, static_sram_end=1024)},
            artifact_bytes=64, max_static_sram_end=1024)
        def execute(case_path, build, output, backend, cmaddr):
            sdk, _ = fake_sdk(case_path)
            runner.main(case_path, build, output, backend, cmaddr, sdk=sdk)
        capture = self.root/"worker-capture"
        args = argparse.Namespace(manifest=manifest_path, request=request, backend="simulator",
            case_archive=manifest_path.parent/"case.tar.gz", out=capture, build=self.build, cmaddr=None)
        with mock.patch.object(worker.common, "compiled_inventory", return_value=inventory):
            worker.capture(args, root=root, runner=execute)
        self.assertEqual(common.audit_report(capture)["status"], "PASS")
        self.assertEqual(common.admit_capture(capture)["artifact_sha256"], record["artifact_sha256"])
        with (capture/"build/out.json").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "compiled metadata bytes"):
            common.admit_capture(capture)


if __name__ == "__main__":
    unittest.main()
