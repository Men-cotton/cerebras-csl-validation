"""SDK-free checks for the dedicated resident launch/audit contract.

The signed asymmetric graph has duplicates, a zero edge, empty owner lanes,
external IDs above u16, source-slot holes and nonzero reductions on both axes.
Mocks establish host sequencing and fail-closed audits, not CSL execution.
"""
import argparse
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
PACKAGE = "_bounded_adjoint_appliance_test"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(HERE)]
    sys.modules[PACKAGE] = package
common = importlib.import_module(PACKAGE+".saved_audit")
runner = importlib.import_module(PACKAGE+".sdk_runner")
appliance = importlib.import_module(PACKAGE+".appliance")


class Types:
    MEMCPY_16BIT, MEMCPY_32BIT, ROW_MAJOR = "u16", "u32", "row"


def fixture(case, cursor=1):
    case.mkdir()
    (case/"source").mkdir()
    q = dict(C=2, R=2, B=2, T=4, F=3, L=3, G=2, U=5, E=7, CURSOR=cursor)
    values = (np.arange(24).reshape(8, 3) % 11-5).astype(np.float32)/8
    d = ((np.arange(24).reshape(8, 3)*3 % 17)-8).astype(np.float32)/16
    raw = dict(ids=np.array([7, 4000000000, 11, 4294967290, 90, 3, 65537, 19], np.uint32),
               owner=np.array([0, 0, 1, 1, 1, 2, 2, 3], np.int64), values=values,
               src=np.array([4, 0, 4, 5, 1, 6, 7, 2, 0, 5, 0, 0], np.int64),
               dst=np.array([0, 0, 0, 0, 1, 1, 5, 5, 5, 0, 2, 7], np.int64),
               coeff=np.array([.5, -.25, .75, 1, 0, -1, .25, .5, 1, -.125, -.5, .25], np.float32),
               derivative0=d, derivative1=-d+np.float32(.125),
               labels=np.arange(8, dtype=np.int32) % 2,
               train=np.arange(8) < 2, valid=np.arange(8) >= 2)
    arrays, _, _, _ = common.independent_inputs(raw, q)
    np.savez(case/"inputs.npz", **arrays)
    np.savez(case/"original.npz", **raw)
    expected, _ = common.independent_expected(case, dict(params=q))
    np.savez(case/"reference.npz", **expected)
    for name in ("layout.csl", "pe.csl"):
        (case/"source"/name).write_text("// Mock fixture; never compile this source.\n")
    named = common.named_bytes(q)
    metadata = dict(forecast_static_bytes=named+35*1024, code_sdk_reserve_bytes=35*1024,
                    declared_admission="ADMITTED")
    parent = dict(experiment="E214", params=q, metadata=metadata,
                  files={n: common.sha(case/n) for n in common.CASE_REQUIRED-{"source-manifest.json"}},
                  sources={"models/hierarchical_gnn/grouped_csr/bounded_panels/adjoint_capacity/"+n:
                           common.sha(case/"source"/n) for n in ("layout.csl", "pe.csl")})
    common.write_json(case/"source-manifest.json", parent)
    manifest = dict(schema=1, family="bounded-adjoint", params=q, inputs=common.schema(q), outputs=common.schema(q, True),
                    launches=[dict(entrypoint="run", arguments=[], update=None), dict(entrypoint="run", arguments=[], update="derivative")],
                    source_case_sha256=common.sha(case/"source-manifest.json"),
                    files={n: common.sha(case/n) for n in common.CASE_REQUIRED}, metadata=metadata,
                    admission=dict(named_bytes=named, code_reserve_bytes=35*1024, forecast_bytes=named+35*1024, admitted=True),
                    atol=common.ATOL, rtol=common.RTOL, clock_scope=common.CLOCK_SCOPE, scope="mock signed asymmetric fixture")
    common.write_json(case/"case.json", manifest)
    return common.load(case)


def fake_build(build, manifest):
    (build/"bin").mkdir(parents=True)
    common.write_json(build/"out.json", dict(params=manifest["params"]))
    symbols = [dict(name=name, id=i, kind="Var", type={"float32": "[*]f32", "uint16": "[*]u16", "uint32": "[*]u32"}[dtype])
               for i, (name, (_, dtype)) in enumerate(dict(manifest["inputs"], **manifest["outputs"]).items())]
    symbols.append(dict(name="run", id=len(symbols), kind="Func", type="void", inputs=[]))
    common.write_json(build/"bin/out_rpc.json", dict(rpc_symbols=symbols))


def fake_sdk(case, *, partial=False, incomplete_clock=False):
    manifest, instances = common.load(case), []
    expected, panels = common.independent_expected(case, manifest)
    q = manifest["params"]
    class SdkRuntime:
        def __init__(self, build, **options):
            self.uploads, self.epoch, self.stops, self.updates = {}, -1, 0, []
            self.options = options
            instances.append(self)
        def get_id(self, name):
            return name
        def load(self):
            pass
        def run(self):
            pass
        def memcpy_h2d(self, name, value, x, y, c, r, count, **kwargs):
            assert (x, y, c, r) == (0, 0, q["C"], q["R"])
            assert kwargs["nonblock"] is False and kwargs["order"] == Types.ROW_MAJOR
            if manifest["inputs"][name][1] == "uint16":
                assert value.dtype == np.uint32 and np.all(value < 65536)
                assert kwargs["data_type"] == Types.MEMCPY_16BIT
            if self.epoch >= 0:
                assert self.epoch == 0 and name == "derivative"
                self.updates.append(name)
            self.uploads[name] = value.reshape(c*r, count).copy()
        def launch(self, symbol, *args, **kwargs):
            assert symbol == "run" and not args and kwargs == dict(nonblock=False)
            self.epoch += 1
            assert self.epoch < 2
            np.testing.assert_array_equal(self.uploads["derivative"], expected[f"epoch{self.epoch}_derivative"])
        def memcpy_d2h(self, value, symbol, x, y, c, r, count, **kwargs):
            assert kwargs["nonblock"] is False
            if partial and symbol == "group_begin":
                value[0] = np.uint32(0x10007)
                raise RuntimeError("partial native read")
            if symbol in self.uploads:
                content = self.uploads[symbol]
            elif symbol == "stats" and self.epoch == -1:
                content = np.zeros((c*r, 12), np.uint32)
            elif symbol == "clocks":
                content = np.tile(np.array([10, 0, 0, 1010, 0, 0], np.uint32), (c*r, 1))
            elif symbol in ("panel_local", "panel_home", "panel_receive"):
                panel = panels[self.epoch]
                content = np.zeros((c*r, count), np.float32)
                if symbol == "panel_local":
                    content[:] = panel["panel_local"]
                else:
                    row = slice(panel["last_group"]*c, (panel["last_group"]+1)*c)
                    content[row] = panel["source_row_home" if symbol == "panel_home" else "source_row_receive"]
            else:
                content = expected[f"epoch{self.epoch}_"+symbol]
            if incomplete_clock and symbol == "clocks":
                # Leave the high start word untouched. The old legal-word
                # sentinel0xDEAD would form a positive wrapped delta <2^47.
                saved = value[2]
                value[:] = np.asarray(content, dtype=value.dtype).ravel()
                value[2] = saved
            else:
                value[:] = np.asarray(content, dtype=value.dtype).ravel()
        def stop(self):
            self.stops += 1
    return (SdkRuntime, Types, Types), instances


class ApplianceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.case, self.build = self.root/"case", self.root/"build"
        self.manifest = fixture(self.case)
        fake_build(self.build, self.manifest)

    def capture(self, name="runtime"):
        sdk, instances = fake_sdk(self.case)
        output = self.root/name
        runner.main(self.case, self.build, output, "simulator", sdk=sdk)
        return output, instances[0]

    def mutate_raw(self, output, mutate):
        raw = common.load_npz(output/"raw.npz")
        mutate(raw)
        np.savez(output/"raw.npz", **raw)
        record = common.read_json(output/"execution.json")
        record["raw_sha256"] = common.sha(output/"raw.npz")
        common.write_json(output/"execution.json", record)

    def test_two_launches_d_only_update_complete_audit(self):
        output, sdk = self.capture()
        result = common.audit(self.case, output)
        self.assertEqual(result["arrays"], 81)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(sdk.epoch, 1)
        self.assertEqual(sdk.updates, ["derivative"])
        self.assertEqual(sdk.stops, 1)
        self.assertEqual(sdk.options, dict(cmaddr=None, simfab_numthreads=1, suppress_simfab_trace=True))
        raw = common.original_arrays(self.case, self.manifest["params"])
        # Explicit asymmetric witness: AZ and A-transpose-Z must differ.
        self.assertFalse(np.array_equal(common.scatter(raw, raw["values"]), common.scatter(raw, raw["values"], True)))

    def test_missing_update_and_changed_api_order_rejected(self):
        output, _ = self.capture()
        record = common.read_json(output/"execution.json")
        record["api"].remove("h2d:epoch1_derivative")
        common.write_json(output/"execution.json", record)
        with self.assertRaisesRegex(ValueError, "SDK sequence"):
            common.admit_execution(self.case, output)

    def test_stale_gradient_counters_padding_and_native_high_bits_rejected(self):
        mutations = [lambda r: r["epoch1_gradient"].__setitem__(slice(None), r["epoch0_gradient"]),
                     lambda r: r["epoch1_stats"].__setitem__((0, 9), 1),
                     lambda r: r["epoch1_traversal"].__setitem__((0, 5), 0),
                     lambda r: r["epoch0_output"].__setitem__((0, 3), np.float32(1e-30)),
                     lambda r: r["epoch0_after_group_begin_native_u32"].__setitem__((0, 0), 65536),
                     lambda r: r["epoch1_clocks"].__setitem__((0, 0), 65536)]
        for i, mutate in enumerate(mutations):
            with self.subTest(i=i):
                output, _ = self.capture(f"mutation{i}")
                self.mutate_raw(output, mutate)
                with self.assertRaises(ValueError):
                    common.audit(self.case, output)

    def test_corrupt_original_directory_reference_rejected_independently(self):
        output, _ = self.capture()
        arrays = common.load_npz(self.case/"inputs.npz")
        arrays["edge_value"][0, 0] += np.float32(.125)
        # Rebind transport hashes: the independent original-occurrence decoder
        # must still reject an internally self-consistent wrong packed input.
        np.savez(self.case/"inputs.npz", **arrays)
        parent = common.read_json(self.case/"source-manifest.json")
        parent["files"]["inputs.npz"] = common.sha(self.case/"inputs.npz")
        common.write_json(self.case/"source-manifest.json", parent)
        manifest = common.read_json(self.case/"case.json")
        for name in ("inputs.npz", "source-manifest.json"):
            manifest["files"][name] = common.sha(self.case/name)
        manifest["source_case_sha256"] = manifest["files"]["source-manifest.json"]
        common.write_json(self.case/"case.json", manifest)
        with self.assertRaisesRegex(ValueError, "original ordered upload"):
            common.independent_expected(self.case, common.load(self.case))

    def test_partial_native_read_retained_and_runtime_stopped(self):
        sdk, instances = fake_sdk(self.case, partial=True)
        output = self.root/"partial"
        with self.assertRaisesRegex(RuntimeError, "incomplete SDK capture"):
            runner.main(self.case, self.build, output, "simulator", sdk=sdk)
        self.assertEqual(instances[0].stops, 1)
        raw = common.load_npz(output/"raw.npz")
        self.assertEqual(int(raw["epoch0_before_group_begin_native_u32"][0, 0]), 0x10007)
        self.assertEqual(int(raw["epoch0_before_group_begin"][0, 0]), 7)
        self.assertEqual(common.read_json(output/"execution.json")["status"], "FAILED")

    def test_successful_partial_clock_read_is_not_numerical_pass(self):
        sdk, instances = fake_sdk(self.case, incomplete_clock=True)
        output = self.root/"incomplete-clock"
        runner.main(self.case, self.build, output, "simulator", sdk=sdk)
        self.assertEqual(instances[0].stops, 1)
        self.assertEqual(common.admit_execution(self.case, output)["numerical_status"], "NOT_RUN")
        with self.assertRaisesRegex(ValueError, "clock words have zero high halves"):
            common.audit(self.case, output)

    def test_constructor_exit_journal_failure_still_stops_once(self):
        sdk, instances = fake_sdk(self.case)
        fired = []
        def persist(record):
            if "construct" in record["api"] and not fired:
                fired.append(True)
                raise OSError("journal failure")
        with self.assertRaisesRegex(RuntimeError, "incomplete SDK capture"):
            runner.main(self.case, self.build, self.root/"journal-fail", "simulator", sdk=sdk, persist_hook=persist)
        self.assertEqual(instances[0].stops, 1)

    def test_native_u16_clock_u32_and_zero_argument_run_abi(self):
        common.check_abi(self.build, self.manifest)
        rpc = common.read_json(self.build/"bin/out_rpc.json")
        next(x for x in rpc["rpc_symbols"] if x["name"] == "clocks")["type"] = "[*]u16"
        common.write_json(self.build/"bin/out_rpc.json", rpc)
        with self.assertRaisesRegex(ValueError, "run/pointer ABI"):
            common.check_abi(self.build, self.manifest)

    def test_source_binding_before_runtime(self):
        root = self.root/"package"
        (root/"cases").mkdir(parents=True)
        shutil.copytree(self.case, root/"cases/tiny")
        for name in common.HOST_FILES:
            shutil.copyfile(HERE/name, root/name)
        common.write_json(root/"package.json", dict(schema=1, family="bounded-adjoint",
                          cases={"tiny": common.sha(root/"cases/tiny/case.json")},
                          host_sha256={n: common.sha(root/n) for n in common.HOST_FILES}))
        appliance.identity(root, "tiny")
        (root/"cases/tiny/source/pe.csl").write_text("changed")
        with self.assertRaisesRegex(ValueError, "frozen file identity"):
            appliance.identity(root, "tiny")

    def test_compile_artifact_hash_and_one_worker_binding(self):
        root = self.root/"package"
        (root/"cases").mkdir(parents=True)
        shutil.copytree(self.case, root/"cases/tiny")
        for name in common.HOST_FILES:
            shutil.copyfile(HERE/name, root/name)
        common.write_json(root/"package.json", dict(schema=1, family="bounded-adjoint",
                          cases={"tiny": common.sha(root/"cases/tiny/case.json")},
                          host_sha256={n: common.sha(root/n) for n in common.HOST_FILES}))
        class Compiler:
            def __init__(self, **options):
                assert options["resource_cpu"] == 1000
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def compile(self, source, layout, flags, out):
                assert "--max-parallelism=1" in flags and layout == "layout.csl"
                artifact = Path(out)/"artifact.bin"
                artifact.write_bytes(b"mock compiler artifact")
                return artifact
        args = argparse.Namespace(case="tiny", backend="hardware", compile_workers=1,
                                  namespace="job-operator", out=self.root/"compiled")
        appliance.compile_program(args, root, Compiler)
        record, artifact, _ = appliance.load_build(args.out/"native-build.json", "hardware", "tiny", root)
        self.assertEqual(record["compile_resource_cpu"], 1000)
        artifact.write_bytes(b"changed artifact")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            appliance.load_build(args.out/"native-build.json", "hardware", "tiny", root)

    def test_scan_and_cursor_expect_same_math_distinct_inspections(self):
        scan = self.root/"scan"
        fixture(scan, cursor=0)
        a, _ = common.independent_expected(self.case, self.manifest)
        b, _ = common.independent_expected(scan, common.load(scan))
        for epoch in range(2):
            for field in ("output", "gradient", "stats", "derivative"):
                np.testing.assert_array_equal(a[f"epoch{epoch}_{field}"], b[f"epoch{epoch}_{field}"])
        self.assertFalse(np.array_equal(a["epoch0_traversal"], b["epoch0_traversal"]))


if __name__ == "__main__":
    unittest.main()
