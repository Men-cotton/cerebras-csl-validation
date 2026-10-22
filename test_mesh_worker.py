"""Source-only SDK mocks; these tests submit no jobs or numerical SIM claims."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

import numpy as np

import mesh_worker as worker


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.case = self.root / "case"
        self.build = self.root / "out"
        self.case.mkdir()
        (self.build / "bin").mkdir(parents=True)
        self.inputs = {"velocity": np.arange(8, dtype=np.float32).reshape(4, 2),
                       "counts": np.ones((4, 2), dtype=np.uint32)}
        np.savez(self.case / "inputs.npz", **self.inputs)
        self.manifest = dict(params=dict(C=2, R=2, MODE=0, STEPS=3),
                             inputs=dict(velocity=[2, "float32"], counts=[2, "uint32"]),
                             outputs=dict(stats=[16, "uint32"], lifecycle=[4, "uint32"],
                                          clocks=[6, "uint16"], h=[2, "float32"]),
                             residency=dict(schema=1, chunks=[1, 2]),
                             files={"inputs.npz": worker.sha((self.case / "inputs.npz").read_bytes())})
        self.symbols = []
        for index, (name, (_, dtype)) in enumerate(dict(self.manifest["inputs"], **self.manifest["outputs"]).items()):
            self.symbols.append(dict(id=index, name=name, kind="Var",
                                     type={"float32": "[*]f32", "uint32": "[*]u32", "uint16": "[*]u16"}[dtype]))
        self.symbols.append(dict(id=len(self.symbols), name="run", kind="Func", type="void",
                                 inputs=[dict(name="steps", type="u16")]))
        self.compiled = {"params": {k: str(v) for k, v in self.manifest["params"].items()}}
        self.rpc = {"rpc_symbols": self.symbols}
        self.request = dict(schema=1, run_id=str(uuid.uuid4()), backend="hardware", case_name="fixture",
                            case_sha256="", package_manifest_sha256="1"*64,
                            artifact_sha256="2"*64,
                            worker_sha256=worker.sha(Path(worker.__file__).read_bytes()))
        self.persist()

    def persist(self):
        worker.write_json(self.case / "case.json", self.manifest)
        self.request["case_sha256"] = worker.sha((self.case / "case.json").read_bytes())
        worker.write_json(self.root / "request.json", self.request)
        worker.write_json(self.build / "out.json", self.compiled)
        worker.write_json(self.build / "bin/out_rpc.json", self.rpc)

    def fake(self, fail=None, bad_ids=False):
        seen = {"calls": [], "launches": [], "transfers": []}
        ids = {s["name"]: s["id"] for s in self.symbols if s["kind"] == "Var"}
        dtype = SimpleNamespace(MEMCPY_16BIT="16", MEMCPY_32BIT="32")
        order = SimpleNamespace(ROW_MAJOR="ROW_MAJOR")

        class Runtime:
            def __init__(self, name, **options):
                seen.update(name=name, options=options)
                self.memory = {}
                self.hit("construct")

            def hit(self, label):
                seen["calls"].append(label)
                if label == fail:
                    raise RuntimeError("injected " + label + " at secret-address")

            def get_id(self, name):
                self.hit("get_id:"+name)
                return ids[name] + int(bad_ids)

            def load(self):
                self.hit("load")

            def run(self):
                self.hit("run")

            def launch(self, name, steps, **options):
                self.hit("launch:"+name)
                seen["launches"].append((name, steps, options))

            def memcpy_h2d(self, symbol, value, x, y, columns, rows, count, **options):
                self.hit("h2d")
                self.memory[symbol] = value.copy()
                seen["transfers"].append(("h2d", value.dtype, (x, y, columns, rows, count), options))

            def memcpy_d2h(self, value, symbol, x, y, columns, rows, count, **options):
                self.hit("d2h")
                value[:] = self.memory.get(symbol, np.zeros(value.shape, value.dtype))
                seen["transfers"].append(("d2h", value.dtype, (x, y, columns, rows, count), options))

            def stop(self):
                self.hit("stop")

        return lambda: (Runtime, dtype, order), seen

    def run_capture(self, backend="hardware", fail=None, bad_ids=False, cmaddr="default"):
        self.request["backend"] = backend
        self.persist()
        if cmaddr == "default":
            cmaddr = "secret-address" if backend == "hardware" else None
        args = SimpleNamespace(case=self.case, name=self.build, out=self.root / "capture",
                               request=self.root / "request.json", backend=backend, cmaddr=cmaddr)
        factory, seen = self.fake(fail, bad_ids)
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(worker, "sdk_identity", return_value={"mock": True}):
            result = worker.capture(args, factory)
        return result, seen, args

    def test_hardware_preserves_api_sequence_and_carriers(self):
        result, seen, args = self.run_capture()
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(seen["options"], {"cmaddr": "secret-address"})
        schema = dict(self.manifest["inputs"], **self.manifest["outputs"])
        expected = ["construct"] + ["get_id:"+name for name in schema] + ["load", "run"]
        expected += ["h2d:"+name for name in self.inputs] + ["d2h:initial_stats", "d2h:initial_lifecycle"]
        expected += ["d2h:before_"+name for name in self.inputs]
        for i in range(2):
            expected += ["launch:run"] + [f"d2h:chunk{i}_"+name for name in self.manifest["outputs"]]
            expected += [f"d2h:chunk{i}_velocity"]
        expected += ["d2h:after_counts", "stop"]
        self.assertEqual(result["api"], expected)
        self.assertEqual([v[1] for v in seen["launches"]], [1, 2])
        self.assertTrue(all(isinstance(v[1], np.uint16) and v[2] == {"nonblock": False} for v in seen["launches"]))
        for direction, dtype, geometry, options in seen["transfers"]:
            self.assertEqual(geometry[:4], (0, 0, 2, 2))
            self.assertEqual(options["order"], "ROW_MAJOR")
            self.assertIs(options["streaming"], False)
            self.assertIs(options["nonblock"], False)
            if options["data_type"] == "16":
                self.assertEqual(dtype, np.dtype("uint32"))
        self.assertEqual(result["request_sha256"], worker.sha(args.request.read_bytes()))
        self.assertEqual(result["compiled_sha256"], {
            "out.json": worker.sha((args.out / "out.json").read_bytes()),
            "out_rpc.json": worker.sha((args.out / "out_rpc.json").read_bytes())})
        self.assertEqual(result["raw_sha256"], worker.sha((args.out / "raw.npz").read_bytes()))
        for name in ("backend", "run_id", "case_sha256", "artifact_sha256", "worker_sha256", "package_manifest_sha256"):
            self.assertEqual(result[name], self.request[name])
        self.assertNotIn("secret-address", (args.out / "execution.json").read_text())
        with np.load(args.out / "raw.npz", allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw["before_counts"], self.inputs["counts"])
            self.assertEqual(raw["chunk0_clocks"].dtype, np.dtype("uint32"))

    def test_appliance_sim_has_simulator_options(self):
        result, seen, _ = self.run_capture("appliance-sim")
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(seen["options"], dict(cmaddr=None, simfab_numthreads=1, suppress_simfab_trace=True))

    def test_local_sim_has_simulator_options(self):
        result, seen, _ = self.run_capture("local-sim")
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(seen["options"], dict(cmaddr=None, simfab_numthreads=1, suppress_simfab_trace=True))

    def test_mode2_additional_inputs_outputs_keep_the_same_sequence(self):
        self.manifest["params"].update(MODE=2, Q=2)
        self.manifest["inputs"]["round_sources"] = [2, "uint32"]
        self.manifest["outputs"]["comm_stats"] = [12, "uint32"]
        self.inputs["round_sources"] = np.full((4, 2), 4, dtype=np.uint32)
        np.savez(self.case / "inputs.npz", **self.inputs)
        self.manifest["files"]["inputs.npz"] = worker.sha((self.case / "inputs.npz").read_bytes())
        self.compiled["params"] = {k: str(v) for k, v in self.manifest["params"].items()}
        self.symbols = []
        for index, (name, (_, dtype)) in enumerate(dict(self.manifest["inputs"], **self.manifest["outputs"]).items()):
            self.symbols.append(dict(id=index, name=name, kind="Var",
                                     type={"float32": "[*]f32", "uint32": "[*]u32", "uint16": "[*]u16"}[dtype]))
        self.symbols.append(dict(id=len(self.symbols), name="run", kind="Func", type="void",
                                 inputs=[dict(name="steps", type="u16")]))
        self.rpc["rpc_symbols"] = self.symbols
        result, _, args = self.run_capture()
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["api"].count("h2d:round_sources"), 1)
        self.assertIn("d2h:chunk1_comm_stats", result["api"])
        self.assertIn("d2h:after_round_sources", result["api"])
        with np.load(args.out / "raw.npz", allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw["before_round_sources"], self.inputs["round_sources"])
            np.testing.assert_array_equal(raw["after_round_sources"], self.inputs["round_sources"])

    def test_hardware_requires_address_before_sdk(self):
        result, seen, args = self.run_capture(cmaddr=None)
        self.assertEqual(result["status"], "FAILED")
        self.assertFalse(seen["calls"])
        self.assertTrue((args.out / "raw.npz").is_file())

    def test_simulator_rejects_address_before_sdk(self):
        result, seen, _ = self.run_capture("appliance-sim", cmaddr="secret-address")
        self.assertEqual(result["status"], "FAILED")
        self.assertFalse(seen["calls"])

    def test_failed_read_retains_carriers_and_stops(self):
        result, seen, args = self.run_capture(fail="d2h")
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(seen["calls"][-1], "stop")
        self.assertNotIn("secret-address", json.dumps(result))
        self.assertEqual(result["api"][-1], "stop")
        with np.load(args.out / "raw.npz", allow_pickle=False) as raw:
            self.assertEqual(raw.files, ["initial_stats"])
            np.testing.assert_array_equal(raw["initial_stats"], np.full((4, 16), 0xdead, np.uint32))

    def test_failed_stop_is_not_completed(self):
        result, _, _ = self.run_capture(fail="stop")
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("stop_error", result)
        self.assertNotIn("stop", result["api"])

    def test_failed_constructor_leaves_empty_capture(self):
        result, _, args = self.run_capture(fail="construct")
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["api"], [])
        with np.load(args.out / "raw.npz", allow_pickle=False) as raw:
            self.assertEqual(raw.files, [])

    def test_runtime_symbol_mismatch_stops_before_load(self):
        result, seen, _ = self.run_capture(bad_ids=True)
        self.assertEqual(result["status"], "FAILED")
        self.assertNotIn("load", seen["calls"])
        self.assertEqual(seen["calls"][-1], "stop")

    def test_input_hash_mismatch_rejected_before_sdk(self):
        self.manifest["files"]["inputs.npz"] = "0"*64
        result, seen, _ = self.run_capture()
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("input hash mismatch", result["error"])
        self.assertFalse(seen["calls"])

    def test_worker_hash_mismatch_rejected_before_sdk(self):
        self.request["worker_sha256"] = "0"*64
        result, seen, _ = self.run_capture()
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("worker hash mismatch", result["error"])
        self.assertFalse(seen["calls"])

    def test_abi_rejects_wrong_argument_missing_symbol_and_parameter(self):
        wrong = copy.deepcopy(self.rpc)
        wrong["rpc_symbols"][-1]["inputs"][0]["type"] = "u32"
        with self.assertRaisesRegex(ValueError, "single u16"):
            worker.check_abi(self.manifest, self.compiled, wrong)
        wrong = copy.deepcopy(self.rpc)
        wrong["rpc_symbols"].pop(0)
        with self.assertRaisesRegex(ValueError, "symbol count"):
            worker.check_abi(self.manifest, self.compiled, wrong)
        wrong = copy.deepcopy(self.compiled)
        wrong["params"]["MODE"] = "1"
        with self.assertRaisesRegex(ValueError, "parameters differ"):
            worker.check_abi(self.manifest, wrong, self.rpc)

    def test_abi_rejects_duplicate_ids(self):
        wrong = copy.deepcopy(self.rpc)
        wrong["rpc_symbols"][1]["id"] = wrong["rpc_symbols"][0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate compiled symbol ID"):
            worker.check_abi(self.manifest, self.compiled, wrong)


if __name__ == "__main__":
    unittest.main()
