"""Mocked single-action appliance and frozen-case admission tests; no SDK jobs."""
import io
import importlib.util
import json
import logging
from pathlib import Path
import sys
import tarfile
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock
import zipfile

import numpy as np

try:
    from models.meshgraphnet.csl_training import appliance as app, saved_audit as common
    from models.meshgraphnet.csl_training import export_appliance as exporter
    from models.meshgraphnet.csl_training import appliance_worker as worker
except ModuleNotFoundError:
    import appliance as app
    import saved_audit as common
    import export_appliance as exporter
    import appliance_worker as worker


def tar_bytes(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, payload in files:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return output.getvalue()


def make_case(folder):
    (folder/"source").mkdir(parents=True)
    for name in common.CASE_FILES:
        if name.endswith(".npz"):
            np.savez(folder/name, fixture=np.zeros(1, np.float32))
        else:
            (folder/name).write_text("fixture "+name)
    p = dict(C=2, R=2, H=4, S=2, L=2, E=3, W=578)
    ins, outs = common.schemas(p)
    manifest = dict(schema=1, experiment="E198", params=p, steps=2,
                    inputs={k: list(v) for k, v in ins.items()}, outputs={k: list(v) for k, v in outs.items()},
                    files={name: common.sha(folder/name) for name in common.CASE_FILES}, scope="mock only")
    common.write_json(folder/"case.json", manifest)
    return manifest


class ApplianceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base/"family"
        make_case(self.root/"cases/minimum")
        for name in (*common.HOST_FILES, "export_appliance.py", "test_appliance.py"):
            (self.root/name).write_text("fixture "+name)
        common.write_json(self.root/"package.json", dict(schema=1, family="csl-training",
            cases={"minimum": common.sha(self.root/"cases/minimum/case.json")},
            host_sha256={name: common.sha(self.root/name) for name in common.HOST_FILES}))
        self.compiles, self.runs, self.options, self.staged = [], [], [], {}
        self.launch_failure = False
        self.download_failure = None
        self.download_collision = None
        self.worker_exit = "0"
        self.wrong_request = False
        self.audit_changes = {}
        parent = self

        class Compiler:
            def __init__(self, **kwargs):
                parent.options.append(kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def compile(self, source, layout, flags, output):
                parent.compiles.append((source, layout, flags, output))
                logging.warning("Job id: wsjob-compile-training-mock")
                path = Path(output)/"artifact.bin"
                path.write_bytes(b"compiled mock artifact")
                return str(path)

        class Launcher:
            def __init__(self, artifact, **kwargs):
                parent.options.append(kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def stage(self, filename):
                path = Path(filename)
                parent.staged[path.name] = path.read_bytes()

            def run(self, command):
                parent.runs.append(command)
                logging.warning("Job id: wsjob-run-training-mock")
                if parent.launch_failure:
                    raise RuntimeError("mock launcher failure")
                return "complete"

            def download_artifact(self, remote, requested):
                if remote == parent.download_failure:
                    raise RuntimeError("mock missing download")
                path = Path(requested)
                if remote == parent.download_collision:
                    return str(path.parent/"request.json")
                if remote == "capture":
                    request = parent.staged["run-request.json"]
                    if parent.wrong_request:
                        request += b" "
                    files = [("capture/capture.json", b"{}"), ("capture/runtime/raw.npz", b"mock raw"),
                             ("capture/runtime/execution.json", b"{}"),
                             ("capture/manifest.json", parent.staged["training-build.json"]),
                             ("capture/request.json", request)]
                    path = path.with_name("capture.tar.gz")
                    path.write_bytes(tar_bytes(files))
                elif remote == "host-exit.txt":
                    path.write_text(parent.worker_exit+"\n")
                else:
                    path.write_text("mock log\n")
                return path.name

        self.Compiler, self.Launcher = Compiler, Launcher
        self.audit_patch = mock.patch.object(common, "admit_capture", side_effect=self.mock_audit)
        self.audit = self.audit_patch.start()
        self.addCleanup(self.audit_patch.stop)

    def mock_audit(self, folder):
        request = common.read_json(folder/"request.json")
        build = common.read_json(folder/"manifest.json")
        result = dict(status="PASS", run_id=request["run_id"], backend=build["backend"],
                      manifest_sha256=request["manifest_sha256"])
        result.update(self.audit_changes)
        return result

    def compile(self, backend="hardware"):
        args = app.parser().parse_args(["compile", "--case", "minimum", "--backend", backend,
            "--namespace", "job-operator", "--out", str(self.base/"build")])
        self.assertEqual(app.compile_program(args, self.root, self.Compiler), 0)
        return self.base/"build/training-build.json"

    def run_args(self, name="run", backend="hardware"):
        return app.parser().parse_args(["run", "--case", "minimum", "--backend", backend,
            "--artifact-record", str(self.base/"build/training-build.json"),
            "--host-timeout", "300", "--out", str(self.base/name)])

    def test_one_compile_one_run_and_pinned_job_records(self):
        path = self.compile()
        build = common.read_json(path)
        self.assertEqual(len(self.compiles), 1)
        self.assertIn("--fabric-dims=762,1172", build["compiler_args"])
        self.assertEqual(self.options[0]["mgmt_namespace"], "job-operator")
        self.assertFalse(self.options[0]["disable_version_check"])
        self.assertEqual(build["case_sha256"], common.sha(self.root/"cases/minimum/case.json"))
        self.assertIn("wsjob-compile-training-mock", (path.parent/"client.log").read_text())
        self.assertEqual(app.run_program(self.run_args(), self.root, self.Launcher), 0)
        self.assertEqual(len(self.runs), 1)
        self.assertEqual(set(self.staged), {*common.HOST_FILES, "case.tar.gz", "training-build.json", "run-request.json"})
        self.assertIn("--cmaddr %CMADDR%", self.runs[0])
        collection = common.read_json(self.base/"run/collection.json")
        self.assertEqual(collection["status"], "complete")
        self.assertEqual(collection["execution_status"], "PASS")
        self.assertEqual(collection["numerical_status"], "NOT_RUN")
        self.assertNotIn("audit", collection)

    def test_simulator_uses_case_fabric_without_hardware_address(self):
        path = self.compile("simulator")
        self.assertIn("--fabric-dims=9,4", common.read_json(path)["compiler_args"])
        self.assertEqual(app.run_program(self.run_args(backend="simulator"), self.root, self.Launcher), 0)
        self.assertTrue(self.options[-1]["simulator"])
        self.assertNotIn("--cmaddr", self.runs[0])

    def test_artifact_and_frozen_case_mutation_reject_before_launch(self):
        path = self.compile()
        artifact = path.parent/"artifact.bin"
        original = artifact.read_bytes()
        artifact.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            app.run_program(self.run_args(), self.root, self.Launcher)
        artifact.write_bytes(original)
        (self.root/"cases/minimum/source/pe.csl").write_text("changed")
        with self.assertRaisesRegex(ValueError, "frozen file changed"):
            app.run_program(self.run_args(), self.root, self.Launcher)
        self.assertFalse(self.runs)
        self.assertFalse((self.base/"run").exists())

    def test_host_and_case_archive_mutation_reject_before_launch(self):
        path = self.compile()
        archive = path.parent/"case.tar.gz"
        original = archive.read_bytes()
        archive.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "case archive hash"):
            app.run_program(self.run_args(), self.root, self.Launcher)
        archive.write_bytes(original)
        (self.root/"appliance_worker.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "host source identity"):
            app.run_program(self.run_args(), self.root, self.Launcher)
        self.assertFalse(self.runs)

    def test_launch_failure_retains_partial_downloads(self):
        self.compile()
        self.launch_failure = True
        with self.assertRaisesRegex(RuntimeError, "partial downloads"):
            app.run_program(self.run_args(), self.root, self.Launcher)
        summary = common.read_json(self.base/"run/collection.json")
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(set(summary["downloaded"]), {"host.log", "host-exit.txt", "capture"})
        self.audit.assert_not_called()

    def test_timeout_wrong_request_and_missing_or_colliding_download_fail(self):
        self.compile()
        conditions = [("worker_exit", "124", "worker failed"),
                      ("wrong_request", True, "captured request differs"),
                      ("download_failure", "host-exit.txt", "collection failed"),
                      ("download_collision", "host-exit.txt", "collection failed")]
        for index, (field, value, message) in enumerate(conditions):
            old = getattr(self, field)
            setattr(self, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                app.run_program(self.run_args("failure"+str(index)), self.root, self.Launcher)
            self.assertEqual(common.read_json(self.base/f"failure{index}/collection.json")["status"], "incomplete")
            setattr(self, field, old)

    def test_capture_admission_is_bound_to_request(self):
        self.compile()
        for index, values in enumerate(({"run_id": "wrong"}, {"backend": "simulator"}, {"status": "FAIL"})):
            self.audit_changes = values
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "capture admission"):
                app.run_program(self.run_args("different"+str(index)), self.root, self.Launcher)

    def test_compile_failure_and_existing_output_preserved(self):
        class Failing(self.Compiler):
            def compile(self, *args):
                raise RuntimeError("mock compile failed")
        args = app.parser().parse_args(["compile", "--case", "minimum", "--backend", "hardware", "--out", str(self.base/"bad")])
        with self.assertRaisesRegex(RuntimeError, "mock compile failed"):
            app.compile_program(args, self.root, Failing)
        self.assertEqual(common.read_json(self.base/"bad/compile-status.json")["status"], "incomplete")
        with self.assertRaises(FileExistsError):
            app.compile_program(args, self.root, self.Compiler)

    def test_export_is_byte_preserving_and_generates_single_action_steps(self):
        output = self.base/"exported"
        package = exporter.export({"minimum": self.root/"cases/minimum"}, output, root=self.root)
        self.assertEqual(common.load(output/"cases/minimum"), common.load(self.root/"cases/minimum"))
        self.assertEqual(app.identity(output, "minimum")["case_sha256"], package["cases"]["minimum"])
        steps = common.read_json(output/"request-template.json")["steps"]
        self.assertEqual([row["args"][0] for row in steps], ["compile", "run"])
        self.assertTrue(all(row["timeout_seconds"] == 1200 for row in steps))
        with self.assertRaises(FileExistsError):
            exporter.export({"minimum": self.root/"cases/minimum"}, output, root=self.root)


class WorkerTests(unittest.TestCase):
    """Real worker/runner/auditor with fabricated device memory, never a device.

    The hardware/simulator labels exercise backend plumbing only. Every byte
    stays in TemporaryDirectory; these hand-written expectations are neither
    a gradient oracle nor evidence of CSL execution.
    """

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="csl-training-worker-mock-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = Path(app.__file__).resolve().parent
        self.case = self.base/"case"
        manifest = make_case(self.case)
        p = manifest["params"]
        self.owners, self.steps = p["C"]*p["R"], manifest["steps"]
        self.sdk_calls, self.gradient_error, self.launch_failure = [], False, False
        # In the exported family the runner belongs to each frozen case.
        runner_source = self.root/"sdk_runner.py"
        if not runner_source.is_file():
            names = common.read_json(self.root/"package.json")["cases"]
            runner_source = self.root/"cases"/next(iter(names))/"sdk_runner.py"
        (self.case/"sdk_runner.py").write_bytes(runner_source.read_bytes())
        ins, outs = common.schemas(p)
        self.fixed = {name: np.zeros((self.owners, ins[name][0]), dtype=ins[name][1])
                      for name in ("counts", "src_owner", "src_slot", "dst_slot", "src_begin", "edge_input", "weights")}
        self.fixed["counts"][0] = [1, 1]
        self.fixed["src_begin"][0, 1:] = 1
        self.fixed["weights"][:] = np.float32(.5)
        samples = {name: np.zeros((self.steps, self.owners, ins[name][0]), dtype=ins[name][1])
                   for name in ("node_input", "target", "loss_mask", "scalars")}
        samples["loss_mask"][:, 0, 0] = 1
        samples["scalars"][:] = np.array([.01, 1.], np.float32)
        self.expected = {}
        weights = self.fixed["weights"][0].copy()
        for step in range(self.steps):
            gradient = np.full(p["W"], .125*(step+1), np.float32)
            weights = weights-np.float32(.01)*gradient
            self.expected.update({f"step{step}_prediction": np.array([[.1*(step+1), -.2]], np.float32),
                                  f"step{step}_loss": np.array([.25], np.float32),
                                  f"step{step}_gradients": gradient,
                                  f"step{step}_weights": weights.copy()})
        node_ids = np.full((self.owners, p["L"]), -1, np.int32)
        node_ids[0, 0] = 0
        for filename, values in (("inputs.npz", self.fixed), ("samples.npz", samples),
                                 ("expected.npz", self.expected), ("mapping.npz", {"node_ids": node_ids})):
            np.savez(self.case/filename, **values)
        manifest["scope"] = "fabricated worker test; no SDK or numerical gradient evidence"
        manifest["files"] = {name: common.sha(self.case/name) for name in common.CASE_FILES}
        common.write_json(self.case/"case.json", manifest)
        self.archive = self.base/"case.tar.gz"
        self.archive.write_bytes(tar_bytes([("case/"+name, (self.case/name).read_bytes())
                                           for name in sorted(common.CASE_FILES | {"case.json"})]))
        self.build = self.base/"build"
        (self.build/"bin").mkdir(parents=True)
        common.write_json(self.build/"out.json", {"params": p})
        symbols = []
        for index, (name, (_, dtype)) in enumerate(dict(ins, **outs).items()):
            symbols.append(dict(id=index, name=name, kind="Var",
                                type={"float32": "[*]f32", "uint32": "[*]u32", "uint16": "[*]u16"}[dtype]))
        symbols.append(dict(id=len(symbols), name="run", kind="Func", type="void", inputs=[]))
        common.write_json(self.build/"bin/out_rpc.json", {"rpc_symbols": symbols})
        parent = self

        class Runtime:
            def __init__(self, build, **options):
                parent.sdk_calls.append(("construct", build, options))
                self.values = {"lifecycle": np.zeros((parent.owners, 2), np.uint32)}
                self.step = 0

            def get_id(self, name):
                return name

            def load(self):
                pass

            def run(self):
                pass

            def memcpy_h2d(self, name, value, *args, **kwargs):
                self.values[name] = value.reshape(parent.owners, -1).copy()

            def memcpy_d2h(self, value, name, *args, **kwargs):
                value[:] = self.values[name].ravel()

            def launch(self, name, **kwargs):
                parent.sdk_calls.append(("launch", name))
                if parent.launch_failure:
                    raise RuntimeError("fabricated SDK launch failure")
                for field in ("loss", "gradients", "weights"):
                    self.values[field] = np.tile(parent.expected[f"step{self.step}_{field}"].reshape(1, -1),
                                                 (parent.owners, 1))
                if parent.gradient_error:
                    self.values["gradients"][:, 0] += np.float32(1.)
                self.values["prediction"] = np.zeros((parent.owners, 2*p["L"]), np.float32)
                self.values["prediction"][0, :2] = parent.expected[f"step{self.step}_prediction"][0]
                self.values["lifecycle"][:] = [1, self.step+1]
                self.values["clocks"] = np.tile(np.array([1, 0, 0, 2, 0, 0], np.uint32), (parent.owners, 1))
                # Literal H4/S2, four-owner protocol counts keep this fixture
                # independent of the production counter expression.
                self.values["stats"] = np.tile(np.array(
                    [0x600D, 1, 2, 0, 0, 0, 0, 0, 16, 31, 0, 8, 1, 0, 0, 0], np.uint32),
                    (parent.owners, 1))
                self.values["stats"][0] = np.array(
                    [0x600D, 1, 2, 1, 1, 2, 2, 1, 16, 31, 7, 8, 1, 1, 1, 0], np.uint32)
                self.step += 1

            def stop(self):
                parent.sdk_calls.append(("stop",))

        modules = {name: ModuleType(name) for name in
                   ("cerebras", "cerebras.sdk", "cerebras.sdk.runtime", "cerebras.sdk.runtime.sdkruntimepybind")}
        for name, module in modules.items():
            if name != "cerebras.sdk.runtime.sdkruntimepybind":
                module.__path__ = []
        leaf = modules["cerebras.sdk.runtime.sdkruntimepybind"]
        leaf.SdkRuntime = Runtime
        leaf.MemcpyDataType = SimpleNamespace(MEMCPY_16BIT=16, MEMCPY_32BIT=32)
        leaf.MemcpyOrder = SimpleNamespace(ROW_MAJOR=0)
        patch = mock.patch.dict(sys.modules, modules)
        patch.start()
        self.addCleanup(patch.stop)

    def arguments(self, name, backend):
        manifest = dict(schema=1, action="compile", backend=backend, case_name="worker-fixture",
                        host_sha256={name: common.sha(self.root/name) for name in common.HOST_FILES},
                        case_archive_sha256=common.sha(self.archive), case_sha256=common.sha(self.case/"case.json"),
                        params=common.load(self.case)["params"])
        build_record, request = self.base/(name+"-build.json"), self.base/(name+"-request.json")
        common.write_json(build_record, manifest)
        common.write_json(request, dict(schema=1, run_id="fabricated-"+name, case_name="worker-fixture",
                                        manifest_sha256=common.sha(build_record)))
        return SimpleNamespace(manifest=build_record, request=request, case_archive=self.archive,
                               build=self.build, out=self.base/name, backend=backend,
                               cmaddr="mock-address" if backend == "hardware" else None)

    def test_real_worker_runner_and_audit_accept_both_backends(self):
        for backend in ("simulator", "hardware"):
            with self.subTest(backend=backend):
                args = self.arguments(backend, backend)
                self.assertEqual(worker.capture(args, root=self.root), 0)
                result = common.audit_capture(args.out)
                self.assertEqual(result["backend"], backend)
                self.assertEqual(len(result["reports"]), self.steps)
                self.assertTrue(all(set(row["errors"].values()) == {0.} for row in result["reports"]))
                record = common.read_json(args.out/"capture.json")
                self.assertEqual(set(record["compiled_metadata_sha256"]), {"out.json", "bin/out_rpc.json"})
                options = [call[2] for call in self.sdk_calls if call[0] == "construct"][-1]
                self.assertEqual(options["cmaddr"], args.cmaddr)
                if backend == "simulator":
                    self.assertEqual(options["simfab_numthreads"], 1)
                    self.assertTrue(options["suppress_simfab_trace"])
                else:
                    self.assertNotIn("simfab_numthreads", options)
        self.assertEqual(sum(call[0] == "launch" for call in self.sdk_calls), 2*self.steps)

    def test_worker_preserves_sdk_failure_capture(self):
        self.launch_failure = True
        args = self.arguments("launch-failure", "simulator")
        with self.assertRaisesRegex(RuntimeError, "fabricated SDK launch failure"):
            worker.capture(args, root=self.root)
        self.assertEqual(common.read_json(args.out/"capture.json")["status"], "incomplete")
        self.assertEqual(common.read_json(args.out/"runtime/execution.json")["status"], "FAILED")
        self.assertTrue((args.out/"runtime/raw.npz").is_file())
        self.assertIn(("stop",), self.sdk_calls)

    def test_gradient_mismatch_passes_execution_but_fails_local_audit(self):
        self.gradient_error = True
        args = self.arguments("bad-gradient", "hardware")
        with mock.patch.object(common, "audit", side_effect=AssertionError("remote numerical audit forbidden")):
            self.assertEqual(worker.capture(args, root=self.root), 0)
            self.assertEqual(common.admit_capture(args.out)["execution_status"], "PASS")
        record = common.read_json(args.out/"capture.json")
        self.assertEqual(record["status"], "COMPLETED")
        self.assertEqual(record["numerical_status"], "NOT_RUN")
        self.assertFalse((args.out/"audit.json").exists())
        with self.assertRaisesRegex(AssertionError, "step0_gradients"):
            common.audit_capture(args.out)
        # Exercise the actual archive admission used by collection.
        archive = self.base/"capture.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(args.out, arcname="capture")
        restored = common.unpack_archive(archive, self.base/"restored", "capture", common.CAPTURE_LIMIT)
        self.assertEqual(common.admit_capture(restored)["numerical_status"], "NOT_RUN")
        report_path = self.base/"numerical-failure.json"
        self.assertEqual(common.main([str(archive), "--out", str(report_path)]), 1)
        report = common.read_json(report_path)
        self.assertEqual((report["execution_status"], report["numerical_status"]), ("PASS", "FAIL"))
        self.assertIn("step0_gradients", report["numerical_error"])
        self.assertEqual(common.read_json(args.out/"capture.json"), record)

    def test_local_audit_recovers_legacy_worker_failure_without_rewriting(self):
        self.gradient_error = True
        args = self.arguments("legacy", "hardware")
        worker.capture(args, root=self.root)
        record = common.read_json(args.out/"capture.json")
        record.update(status="incomplete", error="historical numerical failure")
        common.write_json(args.out/"capture.json", record)
        original = (args.out/"capture.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "complete capture backend"):
            common.admit_capture(args.out)
        result = common.audit_report(args.out)
        self.assertEqual(result["execution_status"], "PASS")
        self.assertEqual(result["numerical_status"], "FAIL")
        self.assertEqual(result["worker_error"], record["error"])
        self.assertEqual((args.out/"capture.json").read_bytes(), original)

    def test_execution_rejects_missing_stop_errors_and_incomplete_status(self):
        args = self.arguments("bad-execution", "hardware")
        worker.capture(args, root=self.root)
        path = args.out/"runtime/execution.json"
        original = common.read_json(path)
        changes = ({"calls": original["calls"][:-1]}, {"status": "FAILED"},
                   {"stop_error": "stop failed"}, {"error": "launch failed"})
        for change in changes:
            with self.subTest(change=change):
                common.write_json(path, dict(original, **change))
                with self.assertRaises(ValueError):
                    common.admit_capture(args.out)
        report_path = self.base/"execution-failure.json"
        self.assertEqual(common.main([str(args.out), "--out", str(report_path)]), 2)
        result = common.read_json(report_path)
        self.assertEqual((result["execution_status"], result["numerical_status"]), ("FAIL", "NOT_RUN"))

    def test_execution_rejects_missing_wrong_shape_and_unbound_arrays(self):
        args = self.arguments("bad-arrays", "hardware")
        worker.capture(args, root=self.root)
        path = args.out/"runtime/raw.npz"
        execution_path = args.out/"runtime/execution.json"
        execution = common.read_json(execution_path)
        raw = common.load_npz(path)
        for mode in ("missing", "shape", "hash"):
            with self.subTest(mode=mode):
                values = dict(raw)
                if mode == "missing":
                    del values["step0_gradients"]
                elif mode == "shape":
                    values["step0_gradients"] = values["step0_gradients"][:, :-1]
                else:
                    values["step0_gradients"] = values["step0_gradients"] + 1
                np.savez(path, **values)
                common.write_json(execution_path, dict(execution, raw_sha256=(
                    execution["raw_sha256"] if mode == "hash" else common.sha(path))))
                with self.assertRaises(ValueError):
                    common.admit_capture(args.out)

    def test_nonfinite_values_are_local_numerical_failure(self):
        args = self.arguments("nonfinite", "hardware")
        worker.capture(args, root=self.root)
        path = args.out/"runtime/raw.npz"
        raw = common.load_npz(path)
        raw["step0_gradients"][:, 0] = np.nan
        np.savez(path, **raw)
        execution_path = args.out/"runtime/execution.json"
        execution = common.read_json(execution_path)
        common.write_json(execution_path, dict(execution, raw_sha256=common.sha(path)))
        self.assertEqual(common.admit_capture(args.out)["execution_status"], "PASS")
        result = common.audit_report(args.out)
        self.assertEqual(result["numerical_status"], "FAIL")
        self.assertIn("finite gradients", result["numerical_error"])

    def test_local_cli_success_and_report_collision(self):
        args = self.arguments("cli-pass", "hardware")
        worker.capture(args, root=self.root)
        report_path = self.base/"numerical-pass.json"
        self.assertEqual(common.main([str(args.out), "--out", str(report_path)]), 0)
        result = common.read_json(report_path)
        self.assertEqual(result["numerical_status"], "PASS")
        with self.assertRaisesRegex(ValueError, "fresh"):
            common.main([str(args.out), "--out", str(report_path)])

    def test_worker_rejects_missing_capture(self):
        args = self.arguments("missing-capture", "simulator")
        with self.assertRaisesRegex(ValueError, "regular file required"):
            worker.capture(args, root=self.root, runner=lambda *args: None)
        self.assertEqual(common.read_json(args.out/"capture.json")["status"], "incomplete")

    def test_worker_rejects_runtime_backend_substitution(self):
        args = self.arguments("wrong-backend", "hardware")

        def wrong_backend(case, build, output, backend, cmaddr):
            spec = importlib.util.spec_from_file_location("worker_backend_fixture", case/"sdk_runner.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.main(case, build, output, backend, cmaddr)
            execution = common.read_json(output/"execution.json")
            execution["backend"] = "host-test"
            common.write_json(output/"execution.json", execution)

        with self.assertRaisesRegex(ValueError, "runtime backend"):
            worker.capture(args, root=self.root, runner=wrong_backend)
        self.assertEqual(common.read_json(args.out/"capture.json")["status"], "incomplete")

    def test_saved_capture_rejects_host_test_backend(self):
        args = self.arguments("host-backend", "simulator")
        worker.capture(args, root=self.root)
        for name in ("capture.json", "manifest.json"):
            value = common.read_json(args.out/name)
            value["backend"] = "host-test"
            common.write_json(args.out/name, value)
        with self.assertRaisesRegex(ValueError, "complete capture backend"):
            common.audit_capture(args.out)


class BoundsTests(unittest.TestCase):
    def test_timeout_bounds_and_shell_arguments(self):
        for value in (0, -1, True, 1801):
            with self.subTest(value=value), self.assertRaises(ValueError):
                app.host_command("hardware", value)
        self.assertIn("timeout --signal=INT --kill-after=15s 300s", app.host_command("hardware", 300))
        with self.assertRaises(ValueError):
            app.sdk_options("bad; namespace")

    def test_tar_links_paths_duplicates_and_bounds_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            contents = [tar_bytes([("capture/../escape", b"x")]),
                        tar_bytes([("capture/raw", b"a"), ("capture/raw", b"b")]),
                        tar_bytes([("capture/large", b"x"*1025)])]
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w:gz") as archive:
                member = tarfile.TarInfo("capture/link")
                member.type, member.linkname = tarfile.SYMTYPE, "/tmp/escape"
                archive.addfile(member)
            contents.append(stream.getvalue())
            for index, value in enumerate(contents):
                path = root/f"{index}.tar.gz"
                path.write_bytes(value)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    common.unpack_archive(path, root/f"out{index}", "capture", 1024)
            self.assertFalse((root/"escape").exists())

    def test_npz_duplicate_and_object_members_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez(root/"object.npz", bad=np.array([object()], dtype=object))
            with self.assertRaises(ValueError):
                common.load_npz(root/"object.npz")
            stream = io.BytesIO()
            np.save(stream, np.zeros(2, np.float32))
            with zipfile.ZipFile(root/"duplicate.npz", "w") as archive:
                archive.writestr("a.npy", stream.getvalue())
                archive.writestr("a.npy", stream.getvalue())
            with self.assertRaises(ValueError):
                common.load_npz(root/"duplicate.npz")


if __name__ == "__main__":
    unittest.main()
