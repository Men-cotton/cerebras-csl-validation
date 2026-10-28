"""Installed flat adapter, real frozen package and three-step request checks.

Install beside stream_appliance.py in validations/primitives. For a local
preinstallation check, E210_PRIMITIVES, E210_STREAM_PACKAGE and
E210_STREAM_REQUEST may point to the corresponding staged files. No gnn-csl
source tree, SDK import or device is required. Only the compiler/runtime are
synthetic; all package, request, ABI, capture and numerical checks are real.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
import weakref

import numpy as np

FAMILY = Path(os.environ.get("E210_PRIMITIVES", Path(__file__).resolve().parent)).resolve()
PACKAGE = Path(os.environ.get("E210_STREAM_PACKAGE", FAMILY/"halo_stream_cases")).resolve()
REQUEST_ID = "tank-halo-streamed-capture-20261028-r001"
REQUEST_PATH = Path(os.environ.get("E210_STREAM_REQUEST", FAMILY.parents[1]/"requests/primitives"/(REQUEST_ID+".json"))).resolve()
sys.path.insert(0, str(FAMILY))
import stream_appliance as app
import stream_worker as worker

contract, chunks, halo = app.contract, app.chunks, app.halo
CASE_NAME = "halo-stream-3x3-c128"


def option(step, name):
    args = step["args"]
    contract.require(args.count(name) == 1 and args.index(name)+1 < len(args), "unique option: "+name)
    return args[args.index(name)+1]


def inspect_request(request):
    contract.require(request["schema"] == 1 and request["id"] == REQUEST_ID and len(request["steps"]) == 3,
                     "pilot request inventory")
    steps = request["steps"]
    contract.require([s["args"][0] for s in steps] == ["compile", "run", "audit"], "compile/run/immediate audit order")
    contract.require(len({s["name"] for s in steps}) == 3, "unique step names")
    outputs = []
    for step, kind, timeout in zip(steps, ("appliance", "appliance", "host"), (1200, 1200, 120)):
        contract.require(step["script"] == "validations/primitives/stream_appliance.py" and step["kind"] == kind
                         and step["timeout_seconds"] == timeout, "registered bounded entrypoint")
        contract.require(option(step, "--case") == CASE_NAME and
                         option(step, "--package") == "validations/primitives/halo_stream_cases", "registered case/package")
        output = option(step, "--out")
        contract.require(output.startswith("{output}/") and output not in outputs, "fresh unique output")
        outputs.append(output)
    contract.require(option(steps[0], "--backend") == option(steps[1], "--backend") == "hardware", "hardware backend")
    contract.require(option(steps[1], "--artifact-record") == outputs[0]+"/build.json", "run owns prior compile")
    contract.require(option(steps[1], "--host-timeout") == "240", "worker timeout")
    contract.require(option(steps[2], "--capture") == outputs[1], "audit owns immediately preceding capture")


class InstalledStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package, cls.case_dir, cls.case, cls.pin = contract.package_identity(PACKAGE, CASE_NAME)
        spec = importlib.util.spec_from_file_location("installed_stream_reference", cls.case_dir/"oracle.py")
        cls.oracle = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.oracle
        spec.loader.exec_module(cls.oracle)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source_pins = contract.source_identity(contract.APP_FILES)
        self.ids = {name: index+10 for index, name in enumerate(self.case["symbols"])}
        self.build = self.root/"synthetic-build"
        (self.build/"bin").mkdir(parents=True)
        contract.write_json(self.build/"out.json", dict(params=self.case["params"]))
        types = {"float32": "[*]f32", "uint32": "[*]u32", "uint16": "[*]u16"}
        rpc = [dict(name=name, id=self.ids[name], kind="Var", type=types[value[1]]) for name, value in self.case["symbols"].items()]
        rpc += [dict(name=name, kind="Func", type="void", inputs=value) for name, value in self.case["functions"].items()]
        contract.write_json(self.build/"bin/out_rpc.json", dict(rpc_symbols=rpc))
        (self.build/"bin/synthetic.elf").write_bytes(b"synthetic fixture; never loaded by an SDK")
        for side in ("east", "west"):
            (self.build/side).mkdir()
            contract.write_json(self.build/side/"out.json", dict(memcpy_side=side))

    def tearDown(self):
        self.assertEqual(contract.source_identity(contract.APP_FILES), self.source_pins)

    def compile(self):
        parent = self
        class Compiler:
            def __init__(self, **options):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *unused):
                pass
            def compile(self, source, layout, flags, destination):
                parent.assertEqual(Path(source), parent.case_dir/"source")
                parent.assertEqual(layout, "layout.csl")
                artifact = Path(destination)/"synthetic.tar.gz"
                with tarfile.open(artifact, "w:gz") as archive:
                    archive.add(parent.build, arcname="out")
                return str(artifact)
        args = argparse.Namespace(package=PACKAGE, case=CASE_NAME, backend="hardware", namespace=None,
                                  out=self.root/"compile", compiler=None)
        app.compile_program(args, Compiler)
        return args.out/"build.json"

    def capture(self, mutation=None):
        record = self.compile()
        parent = self
        data = argparse.Namespace(MEMCPY_16BIT="u16", MEMCPY_32BIT="u32")
        order = argparse.Namespace(ROW_MAJOR="row-major")
        class Runtime:
            def __init__(self, build, **options):
                parent.assertEqual(Path(build), parent.build)
                parent.assertEqual(options, dict(cmaddr="synthetic-hardware"))
                self.previous = None
            def get_id(self, name):
                return parent.ids[name]
            def load(self):
                pass
            def run(self):
                pass
            def launch(self, symbol, length, phase, **options):
                parent.assertEqual(symbol, "apply_epochs")
                parent.assertIsInstance(length, np.int16)
                parent.assertIsInstance(phase, np.int16)
                parent.assertEqual(options, dict(nonblock=False))
                self.values = parent.oracle.vectorized_control(3, 3, int(length), int(phase))
                clocks = np.zeros((3, 3, 6), dtype=np.uint32)
                clocks[:, :, 3] = np.arange(9, dtype=np.uint32).reshape(3, 3)+100
                self.values["clocks"] = clocks
                if mutation == "tail" and int(length) == 17:
                    self.values["y"][2, 2, 31] = 1
            def memcpy_d2h(self, values, sid, x, y, width, height, count, **options):
                if self.previous is not None:
                    parent.assertIsNone(self.previous(), "previous D2H destination must be released")
                self.previous = weakref.ref(values)
                name = next(n for n, i in parent.ids.items() if i == sid)
                parent.assertEqual(options, dict(streaming=False, nonblock=False, order=order.ROW_MAJOR,
                                                data_type=data.MEMCPY_16BIT if name == "clocks" else data.MEMCPY_32BIT))
                parent.assertEqual(count, parent.case["symbols"][name][0])
                values[:] = self.values[name][y:y+height, x:x+width].reshape(-1)
            def stop(self):
                pass
        class Launcher:
            def __init__(self, artifact, **options):
                self.staged = {}
            def __enter__(self):
                return self
            def __exit__(self, *unused):
                pass
            def stage(self, path):
                self.staged[Path(path).name] = Path(path)
            def run(self, command):
                parent.assertEqual(set(self.staged), set(contract.WORKER_FILES) | {"case.json", "request.json", "build.json"})
                parent.assertIn("240s", command)
                self.capture = parent.root/"remote-capture"
                args = argparse.Namespace(name=parent.build, case=self.staged["case.json"], request=self.staged["request.json"],
                                          build_record=self.staged["build.json"], out=self.capture, backend="hardware",
                                          cmaddr="synthetic-hardware")
                execution = worker.capture(args, lambda: (Runtime, data, order))
                parent.assertEqual(execution["execution_status"], "PASS")
                return "synthetic SDK fixture"
            def download_artifact(self, remote, local):
                path = Path(local)
                if remote == "capture":
                    path = path.with_name(path.name+".tar.gz")
                    with tarfile.open(path, "w:gz") as archive:
                        archive.add(self.capture, arcname="capture")
                else:
                    path.write_bytes(b"0\n" if remote == "host-exit.txt" else b"synthetic SDK fixture\n")
                return str(path)
        args = argparse.Namespace(package=PACKAGE, case=CASE_NAME, backend="hardware", artifact_record=record,
                                  out=self.root/"run", host_timeout=240)
        status = app.run_program(args, Launcher)
        self.assertEqual((status["execution_status"], status["numerical_status"]), ("PASS", "NOT_RUN"))
        return args.out

    def audit(self, output):
        return app.audit_program(argparse.Namespace(package=PACKAGE, case=CASE_NAME, capture=output, out=self.root/"audit.json"))

    def test_actual_package_request_and_entrypoint_registration(self):
        self.assertEqual(app.identity(PACKAGE, CASE_NAME), self.pin)
        self.assertEqual(self.case["grid"], [3, 3])
        self.assertEqual(self.case["capture"], dict(chunk_bytes=128, total_bytes=7668, max_chunks=66))
        self.assertFalse(any(name.endswith(".npz") for name in contract.inventory(PACKAGE)))
        inspect_request(json.loads(REQUEST_PATH.read_text()))
        registration = json.loads((FAMILY/"validation.json").read_text())
        self.assertIn("stream_appliance.py", registration["entrypoints"])

    def test_request_mutations_reject_wrong_compile_capture_order_and_output(self):
        original = json.loads(REQUEST_PATH.read_text())
        mutations = ((1, "--artifact-record", "{output}/wrong/build.json"),
                     (2, "--capture", "{output}/wrong-run"),
                     (1, "--out", option(original["steps"][0], "--out")))
        for index, name, value in mutations:
            request = copy.deepcopy(original)
            args = request["steps"][index]["args"]
            args[args.index(name)+1] = value
            with self.assertRaises(ValueError):
                inspect_request(request)
        request = copy.deepcopy(original)
        request["steps"][1], request["steps"][2] = request["steps"][2], request["steps"][1]
        with self.assertRaisesRegex(ValueError, "immediate audit"):
            inspect_request(request)

    def test_compiler_artifact_and_abi_metadata_mutation_rejected(self):
        record = self.compile()
        bound, artifact = app.load_build(record, PACKAGE, CASE_NAME, "hardware")
        self.assertEqual(bound["compiled_metadata_sha256"], app.artifact_metadata(artifact, self.case))
        rpc = contract.read_json(self.build/"bin/out_rpc.json")
        rpc["rpc_symbols"][0]["id"] += 100
        contract.write_json(self.build/"bin/out_rpc.json", rpc)
        changed = self.root/"changed.tar.gz"
        with tarfile.open(changed, "w:gz") as archive:
            archive.add(self.build, arcname="out")
        self.assertNotEqual(bound["compiled_metadata_sha256"], app.artifact_metadata(changed, self.case))
        artifact.write_bytes(changed.read_bytes())
        with self.assertRaisesRegex(ValueError, "compiled artifact"):
            app.load_build(record, PACKAGE, CASE_NAME, "hardware")

    def test_installed_launcher_capture_and_independent_oracle_audit(self):
        output = self.capture()
        report = self.audit(output)
        self.assertEqual(report["numerical_status"], "PASS")
        self.assertEqual((report["payload_bytes"], report["chunks"], report["peak_chunk_bytes"]), (7668, 66, 128))
        self.assertEqual(report["numerical_words"], 1755)
        self.assertTrue(all(v["pe_count"] == 9 for v in report["clocks"].values()))

    def test_payload_and_captured_abi_mutations_rejected(self):
        output = self.capture()
        payload = next((output/"capture/chunks").glob("*.bin"))
        original = payload.read_bytes()
        changed = bytearray(original)
        changed[0] ^= 1
        payload.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.audit(output)
        payload.write_bytes(original)
        metadata = output/"capture/out_rpc.json"
        metadata.write_bytes(metadata.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError, "compiled artifact"):
            self.audit(output)

    def test_numerical_mismatch_preserves_complete_execution(self):
        output = self.capture(mutation="tail")
        with self.assertRaisesRegex(ValueError, "numerical mismatch"):
            self.audit(output)
        self.assertEqual(contract.read_json(output/"capture/execution.json")["execution_status"], "PASS")
        self.assertEqual(contract.read_json(output/"collection.json")["status"], "complete")


if __name__ == "__main__":
    unittest.main()
