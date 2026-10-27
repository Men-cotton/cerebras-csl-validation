"""Mocked native GNN adapter checks: no SDK jobs and no physical measurements."""
import io
import json
import logging
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock
import zipfile
import numpy as np

try:
    from models.meshgraphnet.native_appliance import appliance as app, saved_audit as common
    from models.meshgraphnet.native_appliance import appliance_worker as worker
    from models.meshgraphnet.native_appliance import export_appliance as exporter
    from models.meshgraphnet.native_appliance import compare_rollouts as comparison
except ModuleNotFoundError:
    import appliance as app
    import saved_audit as common
    import appliance_worker as worker
    import export_appliance as exporter
    import compare_rollouts as comparison


def tar_bytes(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, payload in files:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return output.getvalue()


def make_case(folder):
    (folder / "source").mkdir(parents=True)
    (folder / "source/pe.csl").write_text("fixture pe only, never compile")
    (folder / "source/layout.csl").write_text("fixture layout only, never compile")
    inputs = dict(h_in=np.arange(8, dtype=np.float32).reshape(4, 2) / 8,
                  bank=np.full((4, 3), .25, np.float32),
                  counts=np.ones((4, 2), np.uint32))
    expected = dict(h_out=np.tile(np.array([.5, 0], np.float32), (4, 1)),
                    stats=np.tile(np.array([0x600D, 1, 2], np.uint32), (4, 1)))
    np.savez(folder / "inputs.npz", **inputs)
    np.savez(folder / "expected.npz", **expected)
    np.savez(folder / "zero_masks.npz", h_out=np.tile(np.array([0, 1], np.uint32), (4, 1)))
    common.write_json(folder / "origin-case.json", dict(schema=1, files={}))
    ins = {name: [value.shape[1], str(value.dtype)] for name, value in inputs.items()}
    outs = {name: [value.shape[1], str(value.dtype)] for name, value in expected.items()}
    outs["clocks"] = [6, "uint16"]
    manifest = dict(schema=1, family="native-gnn", experiment="fabricated-test-only",
        params=dict(C=2, R=2, H=128, T=64, L=1, E=1), inputs=ins, outputs=outs,
        launch=dict(entrypoint="run", arguments=[]), immutable_inputs=list(ins),
        checks=dict(h_out=dict(kind="approx", rtol=.0003, atol=.0003),
                    stats=dict(kind="exact"), clocks=dict(kind="clock")),
        scope="fabricated host tests only; no device or model correctness evidence",
        origin_case_sha256=common.sha(folder / "origin-case.json"), origin_files={},
        files={name: common.sha(folder / name) for name in common.CASE_REQUIRED})
    common.write_json(folder / "case.json", manifest)
    return manifest


def minimal_elf(static_end=4352):
    ident = b"\x7fELF" + bytes([1, 1, 1]) + bytes(9)
    header = struct.pack("<HHIIIIIHHHHHH", 2, 0, 1, 0, 0, 52, 0, 52, 0, 0, 40, 2, 0)
    section = struct.pack("<IIIIIIIIII", 0, 8, 2, 4096, 0, static_end-4096, 0, 0, 4, 0)
    return ident + header + bytes(40) + section


class ApplianceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base/"family"
        make_case(self.root/"cases/minimum")
        for name in (*common.HOST_FILES, "export_appliance.py", "compare_rollouts.py", "test_appliance.py", "README.md"):
            (self.root/name).write_text("fixture "+name)
        common.write_json(self.root/"package.json", dict(schema=1, family="native-gnn",
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
                logging.warning("Job id: wsjob-compile-native-mock")
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
                logging.warning("Job id: wsjob-run-native-mock")
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
                             ("capture/manifest.json", parent.staged["native-build.json"]),
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

    def compile(self, backend="hardware", workers=1, name="build"):
        args = app.parser().parse_args(["compile", "--case", "minimum", "--backend", backend,
            "--namespace", "job-operator", "--compile-workers", str(workers), "--out", str(self.base/name)])
        self.assertEqual(app.compile_program(args, self.root, self.Compiler), 0)
        return self.base/name/"native-build.json"

    def run_args(self, name="run", backend="hardware"):
        return app.parser().parse_args(["run", "--case", "minimum", "--backend", backend,
            "--artifact-record", str(self.base/"build/native-build.json"),
            "--host-timeout", "300", "--out", str(self.base/name)])

    def test_one_compile_one_run_and_pinned_job_records(self):
        path = self.compile()
        build = common.read_json(path)
        self.assertEqual(len(self.compiles), 1)
        self.assertIn("--fabric-dims=762,1172", build["compiler_args"])
        self.assertEqual(self.options[0]["mgmt_namespace"], "job-operator")
        self.assertEqual(self.options[0]["resource_cpu"], 1000)
        self.assertEqual(build["compile_workers"], 1)
        self.assertEqual(build["compile_resource_cpu"], 1000)
        self.assertFalse(self.options[0]["disable_version_check"])
        self.assertEqual(build["case_sha256"], common.sha(self.root/"cases/minimum/case.json"))
        self.assertIn("wsjob-compile-native-mock", (path.parent/"client.log").read_text())
        self.assertEqual(app.run_program(self.run_args(), self.root, self.Launcher), 0)
        self.assertEqual(len(self.runs), 1)
        self.assertEqual(set(self.staged), {*common.HOST_FILES, "case.tar.gz", "native-build.json", "run-request.json"})
        self.assertIn("--cmaddr %CMADDR%", self.runs[0])
        collection = common.read_json(self.base/"run/collection.json")
        self.assertEqual(collection["status"], "complete")
        self.assertEqual(collection["execution_status"], "PASS")
        self.assertEqual(collection["numerical_status"], "NOT_RUN")
        self.assertNotIn("audit", collection)
        transport = collection["downloaded"]["capture"]
        self.assertFalse(transport["retained"])
        self.assertGreater(transport["bytes"], 0)
        self.assertFalse((self.base/"run"/transport["path"]).exists())
        self.assertTrue((self.base/"run/capture-data/runtime/raw.npz").is_file())

    def test_transport_cleanup_failure_does_not_fail_completed_execution(self):
        self.compile()
        original = Path.unlink
        def refuse_transport(path, *args, **kwargs):
            if path.name == "capture.tar.gz":
                raise PermissionError("cleanup denied")
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, "unlink", refuse_transport):
            self.assertEqual(app.run_program(self.run_args(), self.root, self.Launcher), 0)
        summary = common.read_json(self.base/"run/collection.json")
        self.assertEqual(summary["execution_status"], "PASS")
        transport = summary["downloaded"]["capture"]
        self.assertTrue(transport["retained"])
        self.assertEqual(transport["cleanup_error"], "cleanup denied")
        self.assertTrue((self.base/"run"/transport["path"]).is_file())

    def test_compile_workers_scale_compiler_cpu_only(self):
        for workers in (4, 8):
            with self.subTest(workers=workers):
                name = "build" + str(workers)
                path = self.compile(workers=workers, name=name)
                build = common.read_json(path)
                self.assertEqual(build["compile_workers"], workers)
                self.assertEqual(build["compile_resource_cpu"], workers*1000)
                self.assertEqual(self.options[-1]["resource_cpu"], workers*1000)
                self.assertIn(f"--max-parallelism={workers}", build["compiler_args"])
                args = self.run_args(name="run" + str(workers))
                args.artifact_record = path
                app.run_program(args, self.root, self.Launcher)
                self.assertEqual(self.options[-1]["resource_cpu"], 1000)

    def test_invalid_compile_workers_and_consistent_record_tampering_rejected(self):
        for workers in (0, 2, 16, True, 4.0):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                common.compile_workers(workers)
        with self.assertRaises(SystemExit):
            app.parser().parse_args(["compile", "--case", "minimum", "--backend", "hardware",
                                    "--out", str(self.base/"bad"), "--compile-workers", "2"])
        path = self.compile(workers=4)
        good = common.read_json(path)
        for field, value in (("compile_workers", 8), ("compile_resource_cpu", 1000),
                             ("compiler_args", good["compiler_args"].replace("--max-parallelism=4", "--max-parallelism=1"))):
            with self.subTest(field=field):
                changed = dict(good, **{field: value})
                common.write_json(path, changed)
                common.write_json(path.parent/"request.json", {k: v for k, v in changed.items()
                                                               if k not in ("artifact_path", "artifact_sha256")})
                common.write_json(path.parent/"compile-status.json", dict(status="complete", compile_id=changed["compile_id"],
                                                                          manifest_sha256=common.sha(path)))
                with self.assertRaisesRegex(ValueError, "binding"):
                    app.load_build(path, "hardware", "minimum", self.root)
        self.assertFalse(self.runs)

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
        transport = summary["downloaded"]["capture"]
        self.assertTrue(transport["retained"])
        self.assertTrue((self.base/"run"/transport["path"]).is_file())
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
        self.assertEqual([row["args"][0] for row in steps[:2]], ["compile", "run"])
        self.assertEqual([row["kind"] for row in steps], ["appliance", "appliance", "host"])
        at = steps[0]["args"].index("--compile-workers")
        self.assertEqual(steps[0]["args"][at+1], "1")
        self.assertEqual([row["timeout_seconds"] for row in steps], [1200, 1200, 120])
        self.assertEqual(steps[2], dict(name="audit-minimum", kind="host",
            script="validations/native-gnn/saved_audit.py", timeout_seconds=120,
            args=["{output}/run-minimum/capture-data", "--out", "{output}/numerical-minimum.json"]))
        descriptor = common.read_json(output/"validation.json")
        self.assertEqual(descriptor["entrypoints"], ["appliance.py", "saved_audit.py", "compare_rollouts.py"])
        self.assertEqual(package["comparator_sha256"], common.sha(output / "compare_rollouts.py"))
        with self.assertRaises(FileExistsError):
            exporter.export({"minimum": self.root/"cases/minimum"}, output, root=self.root)

    def test_exporter_orders_each_audit_before_the_next_case(self):
        output = self.base/"exported-sequence"
        exporter.export({"minimum": self.root/"cases/minimum", "larger": self.root/"cases/minimum"},
                        output, root=self.root)
        steps = common.read_json(output/"request-template.json")["steps"]
        self.assertEqual([row["name"] for row in steps],
                         ["compile-minimum", "run-minimum", "audit-minimum",
                          "compile-larger", "run-larger", "audit-larger"])
        allowed = {"validations/native-gnn/" + name
                   for name in common.read_json(output/"validation.json")["entrypoints"]}
        self.assertTrue(all(row["script"] in allowed for row in steps))
        for index, name in ((2, "minimum"), (5, "larger")):
            self.assertEqual(steps[index]["args"],
                             ["{output}/run-" + name + "/capture-data", "--out",
                              "{output}/numerical-" + name + ".json"])

    def test_exporter_records_requested_compile_parallelism(self):
        output = self.base/"exported-four"
        package = exporter.export({"minimum": self.root/"cases/minimum"}, output, root=self.root, compile_workers=4)
        self.assertEqual(package["compile_workers"], 4)
        steps = common.read_json(output/"request-template.json")["steps"]
        at = steps[0]["args"].index("--compile-workers")
        self.assertEqual(steps[0]["args"][at+1], "4")
        self.assertNotIn("--compile-workers", steps[1]["args"])
        with self.assertRaises(ValueError):
            exporter.export({"minimum": self.root/"cases/minimum"}, self.base/"invalid-export", root=self.root, compile_workers=2)


class WorkerTests(unittest.TestCase):
    """Use the production worker/runner with hand-written fake SDK memory."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="native-worker-mock-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = Path(app.__file__).resolve().parent
        self.case = self.base / "case"
        self.manifest = make_case(self.case)
        self.inputs = common.load_npz(self.case / "inputs.npz")
        self.expected = common.load_npz(self.case / "expected.npz")
        self.mode = "normal"
        self.sdk_calls = []
        self.archive = self.base / "case.tar.gz"
        self.archive.write_bytes(tar_bytes([("case/" + name, (self.case / name).read_bytes())
            for name in sorted(set(self.manifest["files"]) | {"case.json"})]))
        self.build = self.base / "build"
        (self.build / "bin").mkdir(parents=True)
        common.write_json(self.build / "out.json", dict(params=self.manifest["params"]))
        symbols = []
        for index, (name, (_, dtype)) in enumerate(dict(self.manifest["inputs"], **self.manifest["outputs"]).items()):
            symbols.append(dict(id=index, name=name, kind="Var", type={"float32": "[*]f32", "uint32": "[*]u32", "uint16": "[*]u16"}[dtype]))
        symbols.append(dict(id=len(symbols), name="run", kind="Func", type="void", inputs=[]))
        common.write_json(self.build / "bin/out_rpc.json", dict(rpc_symbols=symbols))
        (self.build / "bin/out_0_0.elf").write_bytes(minimal_elf())
        parent = self

        class Runtime:
            def __init__(self, build, **options):
                parent.sdk_calls.append(("construct", options))
                self.values = {"stats": np.zeros((4, 3), np.uint32)}
                self.values["lifecycle"] = np.zeros((4, 4), np.uint32)
                self.launch_count = 0

            def get_id(self, name):
                return name

            def load(self):
                pass

            def run(self):
                pass

            def memcpy_h2d(self, name, values, *args, **kwargs):
                self.values[name] = values.reshape(4, -1).copy()

            def memcpy_d2h(self, values, name, *args, **kwargs):
                values[:] = self.values[name].ravel()

            def launch(self, name, *arguments, **kwargs):
                parent.sdk_calls.append(("launch", name, arguments) if arguments else ("launch", name))
                if parent.mode == "launch_failure" or parent.mode == "launch_failure_second" and self.launch_count == 1:
                    raise RuntimeError("fabricated SDK launch failure")
                if parent.manifest["schema"] == 2:
                    index = 0 if parent.mode == "replay" else self.launch_count
                    prefix = f"launch{index}_"
                    self.values.update({key[len(prefix):]: value.copy() for key, value in parent.expected.items()
                                        if key.startswith(prefix)})
                else:
                    self.values.update({name: value.copy() for name, value in parent.expected.items()})
                self.launch_count += 1
                self.values["clocks"] = np.tile(np.array([1, 0, 0, 2, 0, 0], np.uint32), (4, 1))
                if parent.mode == "nan":
                    self.values["h_out"][0, 0] = np.nan
                elif parent.mode == "numerical":
                    self.values["h_out"][0, 0] += 1
                elif parent.mode == "padding":
                    self.values["h_out"][0, 1] = np.float32(1e-5)
                elif parent.mode == "counter":
                    self.values["stats"][0, 0] += 1
                elif parent.mode == "immutable":
                    self.values["bank"][0, 0] += 1
                elif parent.mode == "clock":
                    self.values["clocks"][0, 3] = 1
                elif parent.mode == "boundary_ulp":
                    self.values["h_out"][0, 0] = np.nextafter(self.values["h_out"][0, 0], np.float32(1))
                elif parent.mode == "physical_tolerance":
                    self.values["h_out"][0, 0] += np.float32(1e-4)

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

    def arguments(self, name="capture", backend="hardware"):
        manifest = dict(schema=1, action="compile", backend=backend, case_name="mock-native",
            compile_workers=1, compile_resource_cpu=1000,
            compiler_args=common.compiler_arguments(self.manifest["params"], backend, 1),
            host_sha256={n: common.sha(self.root / n) for n in common.HOST_FILES},
            artifact_sha256="f" * 64, case_archive_sha256=common.sha(self.archive),
            case_sha256=common.sha(self.case / "case.json"), case_files=self.manifest["files"], params=self.manifest["params"])
        build_record, request = self.base / (name + "-build.json"), self.base / (name + "-request.json")
        common.write_json(build_record, manifest)
        common.write_json(request, dict(schema=1, run_id="fabricated-" + name, case_name="mock-native", manifest_sha256=common.sha(build_record)))
        return SimpleNamespace(manifest=build_record, request=request, case_archive=self.archive,
                               build=self.build, out=self.base / name, backend=backend,
                               cmaddr="mock.invalid:9000" if backend == "hardware" else None)

    def test_capture_and_numerical_audit_have_separate_status(self):
        args = self.arguments()
        self.assertEqual(worker.capture(args), 0)
        admission = common.admit_capture(args.out)
        import zipfile
        with zipfile.ZipFile(args.out/"runtime/raw.npz") as archive:
            self.assertTrue(all(item.compress_type == zipfile.ZIP_DEFLATED for item in archive.infolist()))
        self.assertEqual(admission["execution_status"], "PASS")
        self.assertEqual(admission["numerical_status"], "NOT_RUN")
        self.assertEqual(admission["compiled_inventory"]["max_static_sram_end"], 4352)
        self.assertNotIn("files", admission["compiled_inventory"])
        self.assertEqual(admission["compiled_elf_files"], 1)
        self.assertEqual(admission["compiled_inventory_sha256"], common.sha(args.out/"build/inventory.json"))
        self.assertEqual(common.audit_report(args.out)["numerical_status"], "PASS")
        self.assertEqual(self.sdk_calls[0][1], {"cmaddr": "mock.invalid:9000"})
        self.assertEqual(sum(row[0] == "launch" for row in self.sdk_calls), 1)
        raw = common.load_npz(args.out / "runtime/raw.npz")
        for name in self.inputs:
            np.testing.assert_array_equal(raw["before_" + name], self.inputs[name])
            np.testing.assert_array_equal(raw["after_" + name], self.inputs[name])

    def test_numerical_failures_preserve_complete_capture(self):
        for mode in ("nan", "numerical", "padding", "counter", "immutable", "clock"):
            with self.subTest(mode=mode):
                self.mode = mode
                args = self.arguments(mode)
                self.assertEqual(worker.capture(args), 0)
                self.assertEqual(common.read_json(args.out / "capture.json")["status"], "COMPLETED")
                before_hash = common.sha(args.out / "runtime/raw.npz")
                report = common.audit_report(args.out)
                self.assertEqual(report["execution_status"], "PASS")
                self.assertEqual(report["numerical_status"], "FAIL")
                self.assertEqual(common.sha(args.out / "runtime/raw.npz"), before_hash)

    def test_nonzero_boundary_mask_checks_bits_after_complete_capture(self):
        masks = dict(h_out=np.zeros_like(self.expected["h_out"], dtype=np.uint32))
        masks["h_out"][0, 0] = 1
        np.savez(self.case / "exact_masks.npz", **masks)
        self.manifest["files"]["exact_masks.npz"] = common.sha(self.case / "exact_masks.npz")
        self.manifest["exact_mask_counts"] = dict(h_out=1)
        common.write_json(self.case / "case.json", self.manifest)
        self.archive.write_bytes(tar_bytes([("case/" + name, (self.case / name).read_bytes())
            for name in sorted(set(self.manifest["files"]) | {"case.json"})]))
        args = self.arguments("boundary-good")
        self.assertEqual(worker.capture(args), 0)
        self.assertEqual(common.audit_report(args.out)["numerical_status"], "PASS")
        self.mode = "boundary_ulp"
        args = self.arguments("boundary-bad")
        self.assertEqual(worker.capture(args), 0)
        before_hash = common.sha(args.out / "runtime/raw.npz")
        report = common.audit_report(args.out)
        self.assertEqual(report["execution_status"], "PASS")
        self.assertEqual(report["numerical_status"], "FAIL")
        self.assertIn("exact masked values h_out", report["numerical_error"])
        self.assertEqual(common.sha(args.out / "runtime/raw.npz"), before_hash)

    def test_per_output_physical_tolerance_is_enforced_locally(self):
        self.manifest["checks"]["h_out"] = dict(kind="approx", rtol=3e-5, atol=1e-6)
        common.write_json(self.case / "case.json", self.manifest)
        self.archive.write_bytes(tar_bytes([("case/" + name, (self.case / name).read_bytes())
            for name in sorted(set(self.manifest["files"]) | {"case.json"})]))
        self.mode = "physical_tolerance"
        args = self.arguments()
        self.assertEqual(worker.capture(args), 0)
        report = common.audit_report(args.out)
        self.assertEqual(report["execution_status"], "PASS")
        self.assertEqual(report["numerical_status"], "FAIL")

    def test_failed_sdk_launch_retains_partial_arrays(self):
        self.mode = "launch_failure"
        args = self.arguments()
        with self.assertRaisesRegex(RuntimeError, "partial arrays retained"):
            worker.capture(args)
        self.assertEqual(common.read_json(args.out / "capture.json")["status"], "incomplete")
        self.assertTrue((args.out / "runtime/raw.npz").is_file())
        self.assertEqual(self.sdk_calls[-1], ("stop",))
        with self.assertRaisesRegex(ValueError, "completed capture"):
            common.admit_capture(args.out)

    def test_simulator_backend_uses_single_thread_and_no_address(self):
        args = self.arguments(backend="simulator")
        worker.capture(args)
        self.assertEqual(self.sdk_calls[0][1], dict(cmaddr=None, simfab_numthreads=1, suppress_simfab_trace=True))

    def test_abi_mutation_rejects_before_sdk_construction(self):
        common.write_json(self.build / "out.json", dict(params={"C": 2}))
        with self.assertRaisesRegex(ValueError, "compiled parameter ABI"):
            worker.capture(self.arguments())
        self.assertFalse(self.sdk_calls)

    def test_worker_compile_parallelism_binding_rejects_before_sdk(self):
        args = self.arguments()
        record = common.read_json(args.manifest)
        record["compile_workers"] = 4
        common.write_json(args.manifest, record)
        request = common.read_json(args.request)
        request["manifest_sha256"] = common.sha(args.manifest)
        common.write_json(args.request, request)
        with self.assertRaisesRegex(ValueError, "compiler CPU resource binding"):
            worker.capture(args)
        self.assertFalse(self.sdk_calls)

    def test_static_sram_limit_rejects_before_sdk_and_keeps_inventory(self):
        (self.build / "bin/out_0_0.elf").write_bytes(minimal_elf(44*1024+1))
        args = self.arguments()
        with self.assertRaisesRegex(ValueError, "static SRAM exceeds"):
            worker.capture(args)
        self.assertFalse(self.sdk_calls)
        self.assertEqual(common.read_json(args.out / "build/inventory.json")["max_static_sram_end"], 44*1024+1)
        self.assertEqual(common.read_json(args.out / "capture.json")["status"], "incomplete")

    def test_changed_compiled_or_host_capture_bytes_rejected(self):
        args = self.arguments()
        worker.capture(args)
        path = args.out / "build/inventory.json"
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "compiled metadata bytes"):
            common.admit_capture(args.out)
        path.write_bytes(original)
        (args.out / "host/sdk_runner.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "captured host source"):
            common.admit_capture(args.out)

    def test_execution_inventory_corruption_is_not_numerical_failure(self):
        args = self.arguments()
        worker.capture(args)
        execution = common.read_json(args.out / "runtime/execution.json")
        execution["api"].remove("launch:run")
        common.write_json(args.out / "runtime/execution.json", execution)
        with self.assertRaisesRegex(ValueError, "exact SDK API sequence"):
            common.audit_report(args.out)

    def test_cli_emits_numerical_failure_without_overwriting_capture(self):
        self.mode = "nan"
        args = self.arguments()
        worker.capture(args)
        record_before = (args.out / "capture.json").read_bytes()
        report = self.base / "report.json"
        self.assertEqual(common.main([str(args.out), "--out", str(report)]), 1)
        self.assertEqual(common.read_json(report)["numerical_status"], "FAIL")
        self.assertEqual((args.out / "capture.json").read_bytes(), record_before)
        with self.assertRaisesRegex(ValueError, "fresh report"):
            common.main([str(args.out), "--out", str(report)])

    def test_generated_host_audit_exit_contract_and_capture_retention(self):
        exported = self.base / "exported"
        export_source = self.base / "export-source"
        export_source.mkdir()
        for name in (*common.HOST_FILES, "export_appliance.py", "compare_rollouts.py", "test_appliance.py", "README.md"):
            (export_source / name).write_text("fixture " + name)
        exporter.export({"minimum": self.case}, exported, root=export_source)
        step = common.read_json(exported / "request-template.json")["steps"][2]
        for mode, exit_code, execution, numerical in (
                ("normal", 0, "PASS", "PASS"),
                ("nan", 1, "PASS", "FAIL"),
                ("corrupt", 2, "FAIL", "NOT_RUN")):
            with self.subTest(mode=mode):
                self.mode = mode
                output = self.base / mode
                (output / "run-minimum").mkdir(parents=True)
                args = self.arguments("capture-" + mode)
                args.out = output / "run-minimum/capture-data"
                self.assertEqual(worker.capture(args), 0)
                if mode == "corrupt":
                    path = args.out / "runtime/execution.json"
                    record = common.read_json(path)
                    record["api"].remove("launch:run")
                    common.write_json(path, record)
                retained = {str(path.relative_to(args.out)): common.sha(path)
                            for path in args.out.rglob("*") if path.is_file()}
                calls = list(self.sdk_calls)
                argv = [arg.replace("{output}", str(output)) for arg in step["args"]]
                self.assertEqual(common.main(argv), exit_code)
                report = common.read_json(output / "numerical-minimum.json")
                self.assertEqual(report["execution_status"], execution)
                self.assertEqual(report["numerical_status"], numerical)
                self.assertEqual(self.sdk_calls, calls)
                self.assertEqual({str(path.relative_to(args.out)): common.sha(path)
                                  for path in args.out.rglob("*") if path.is_file()}, retained)


class IntegrityTests(unittest.TestCase):
    def test_inventory_json_has_explicit_larger_bound_only(self):
        self.assertEqual(common.JSON_LIMIT, 4 << 20)
        self.assertEqual(common.INVENTORY_JSON_LIMIT, 16 << 20)
        self.assertEqual(common.CASE_LIMIT, 128 << 20)
        self.assertEqual(common.CAPTURE_LIMIT, 256 << 20)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"inventory.json"
            value = dict(padding="x"*(common.JSON_LIMIT+1))
            with self.assertRaisesRegex(ValueError, "JSON output bound"):
                common.write_json(path, value)
            self.assertFalse(path.exists())
            common.write_json(path, value, limit=common.INVENTORY_JSON_LIMIT)
            with self.assertRaisesRegex(ValueError, "file exceeds bound"):
                common.read_json(path)
            self.assertEqual(common.read_json(path, limit=common.INVENTORY_JSON_LIMIT), value)
            oversized = dict(padding="x"*common.INVENTORY_JSON_LIMIT)
            with self.assertRaisesRegex(ValueError, "JSON output bound"):
                common.write_json(path, oversized, limit=common.INVENTORY_JSON_LIMIT)
            self.assertEqual(common.read_json(path, limit=common.INVENTORY_JSON_LIMIT), value)

    def test_compiled_elf_total_bound_remains_enforced(self):
        self.assertEqual(common.ELF_TOTAL_LIMIT, 1 << 30)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            payload = minimal_elf()
            for name in ("a.elf", "b.elf"):
                (path/name).write_bytes(payload)
            with mock.patch.object(common, "ELF_TOTAL_LIMIT", 2*len(payload)):
                inventory = common.compiled_inventory(path)
                common.admit_compiled_inventory(inventory)
                self.assertEqual(inventory["artifact_bytes"], 2*len(payload))
            with mock.patch.object(common, "ELF_TOTAL_LIMIT", 2*len(payload)-1):
                with self.assertRaisesRegex(ValueError, "compiled ELF total byte bound"):
                    common.compiled_inventory(path)
                with self.assertRaisesRegex(ValueError, "compiled ELF total byte bound"):
                    common.admit_compiled_inventory(inventory)
            inventory["files"]["a.elf"]["bytes"] = (32 << 20)+1
            with self.assertRaisesRegex(ValueError, "compiled ELF entry bounds"):
                common.admit_compiled_inventory(inventory)

    def test_compiled_inventory_fails_closed_on_unreadable_subtree(self):
        def unreadable_walk(path, *, followlinks, onerror):
            onerror(PermissionError("unreadable generated artifact subtree"))
            yield path, [], []
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(common.os, "walk", side_effect=unreadable_walk):
                with self.assertRaisesRegex(PermissionError, "unreadable"):
                    common.compiled_inventory(temporary)

    def test_freeze_preserves_origin_files_when_expectation_names_collide(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            case = base / "source"
            original = make_case(case)
            del original["family"]
            common.write_json(case / "case.json", original)
            original_bytes = (case / "case.json").read_bytes()

            def oracle(path):
                expected = common.load_npz(path / "expected.npz")
                expected["h_out"][0, 0] += np.float32(.25)
                return dict(expectedarrays=expected, exactfields=["stats"],
                            zeromasks=common.load_npz(path / "zero_masks.npz"), rtol=.0003, atol=.0003,
                            scope="fabricated oracle fixture; no physical prediction")

            frozen = exporter.freeze(case, base / "portable", oracle=oracle)
            self.assertEqual((base / "portable/origin-case.json").read_bytes(), original_bytes)
            for name, digest in original["files"].items():
                self.assertEqual(common.sha(base / "portable" / frozen["origin_files"][name]), digest)
            self.assertEqual(frozen["origin_files"]["expected.npz"], "origin-files/expected.npz")
            self.assertEqual(common.load(base / "portable"), frozen)

    def test_full_module_hook_reuses_equal_archives_and_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            case = base / "source"
            original = make_case(case)
            del original["family"]
            original["portable_oracle_module"] = "models.meshgraphnet.native_full.driver"
            original["experiment"] = "full-model-fixture"
            values = common.load_npz(case / "expected.npz")
            # Retain a compressed ZIP, although exporter normally writes an
            # uncompressed NPZ. Equality is about arrays, not archive encoding.
            np.savez_compressed(case / "expected.npz", **values)
            original["files"]["expected.npz"] = common.sha(case / "expected.npz")
            bit_masks = dict(h_out=np.tile(np.array([1, 0], np.uint32), (4, 1)))
            np.savez(case / "exact_masks.npz", **bit_masks)
            original["files"]["exact_masks.npz"] = common.sha(case / "exact_masks.npz")
            common.write_json(case / "case.json", original)

            def oracle(path):
                return dict(expectedarrays=common.load_npz(path / "expected.npz"), exactfields=["stats"],
                            zeromasks=common.load_npz(path / "zero_masks.npz"), exactmasks=bit_masks,
                            rtol=3e-4, atol=3e-4, tolerances=dict(h_out=dict(rtol=3e-5, atol=1e-6)),
                            scope="fabricated full-model contract")

            with mock.patch.object(exporter.importlib, "import_module", return_value=SimpleNamespace(portable_expectations=oracle)) as load_module:
                frozen = exporter.freeze(case, base / "portable")
            load_module.assert_called_once_with("models.meshgraphnet.native_full.driver")
            for name in ("expected.npz", "zero_masks.npz", "exact_masks.npz"):
                self.assertEqual(frozen["origin_files"][name], name)
                self.assertEqual((base / "portable" / name).read_bytes(), (case / name).read_bytes())
                self.assertFalse((base / "portable/origin-files" / name).exists())
            self.assertEqual(frozen["exact_mask_counts"], dict(h_out=4))
            self.assertEqual(frozen["checks"]["h_out"], dict(kind="approx", rtol=3e-5, atol=1e-6))
            self.assertEqual(common.load(base / "portable"), frozen)

    def test_full_exact_masks_preserve_different_origin_and_nested_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            case = base / "source"
            original = make_case(case)
            del original["family"]
            old = dict(h_out=np.tile(np.array([1, 0], np.uint32), (4, 1)))
            new = dict(h_out=old["h_out"].copy())
            new["h_out"][0, 0] = 0
            np.savez(case / "exact_masks.npz", **old)
            (case / "origin-files").mkdir()
            (case / "origin-files/exact_masks.npz").write_bytes(b"opaque original collision")
            for name in ("exact_masks.npz", "origin-files/exact_masks.npz"):
                original["files"][name] = common.sha(case / name)
            common.write_json(case / "case.json", original)

            def oracle(path):
                return dict(expectedarrays=common.load_npz(path / "expected.npz"), exactfields=["stats"],
                            zeromasks=common.load_npz(path / "zero_masks.npz"), exactmasks=new,
                            rtol=3e-4, atol=3e-4, scope="fabricated collision fixture")

            frozen = exporter.freeze(case, base / "portable", oracle=oracle)
            self.assertEqual(frozen["origin_files"]["exact_masks.npz"], "origin-files/origin-files/exact_masks.npz")
            for name, digest in original["files"].items():
                self.assertEqual(common.sha(base / "portable" / frozen["origin_files"][name]), digest)
            self.assertEqual(frozen["exact_mask_counts"], dict(h_out=3))
            self.assertEqual(common.load(base / "portable"), frozen)

    def test_array_reuse_distinguishes_signed_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "values.npz"
            np.savez(path, value=np.array([0.], np.float32))
            self.assertTrue(exporter.same_arrays(path, dict(value=np.array([0.], np.float32))))
            self.assertFalse(exporter.same_arrays(path, dict(value=np.array([-0.], np.float32))))

    def test_exact_mask_abi_coverage_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for failure in ("dtype", "shape", "values", "empty", "field", "count", "hash", "unlisted", "nonfinite"):
                with self.subTest(failure=failure):
                    case = base / failure
                    manifest = make_case(case)
                    masks = dict(h_out=np.tile(np.array([1, 0], np.uint32), (4, 1)))
                    if failure == "dtype":
                        masks["h_out"] = masks["h_out"].astype(np.float32)
                    elif failure == "shape":
                        masks["h_out"] = masks["h_out"][:1]
                    elif failure == "values":
                        masks["h_out"][0, 0] = 2
                    elif failure == "empty":
                        masks["h_out"][:] = 0
                    elif failure == "field":
                        masks["unknown"] = masks.pop("h_out")
                    np.savez(case / "exact_masks.npz", **masks)
                    manifest["files"]["exact_masks.npz"] = common.sha(case / "exact_masks.npz")
                    manifest["exact_mask_counts"] = dict(h_out=5 if failure == "count" else 4)
                    if failure == "hash":
                        manifest["files"]["exact_masks.npz"] = "0" * 64
                    elif failure == "unlisted":
                        del manifest["files"]["exact_masks.npz"]
                    elif failure == "nonfinite":
                        expected = common.load_npz(case / "expected.npz")
                        expected["h_out"][0, 0] = np.inf
                        np.savez(case / "expected.npz", **expected)
                        manifest["files"]["expected.npz"] = common.sha(case / "expected.npz")
                    common.write_json(case / "case.json", manifest)
                    with self.assertRaises(ValueError):
                        common.load(case)
            case = base / "no-boundary"
            manifest = make_case(case)
            np.savez(case / "exact_masks.npz", h_out=np.zeros((4, 2), np.uint32))
            manifest["files"]["exact_masks.npz"] = common.sha(case / "exact_masks.npz")
            manifest["exact_mask_counts"] = dict(h_out=0)
            common.write_json(case / "case.json", manifest)
            self.assertEqual(common.load(case), manifest)

    def test_exact_mask_bytes_are_included_in_capture_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "case"
            manifest = make_case(case)
            np.savez(case / "exact_masks.npz", h_out=np.tile(np.array([1, 0], np.uint32), (4, 1)))
            manifest["files"]["exact_masks.npz"] = common.sha(case / "exact_masks.npz")
            manifest["exact_mask_counts"] = dict(h_out=4)
            common.write_json(case / "case.json", manifest)
            # Four PEs: 2*(2+3+2) input words + (2+3+6) outputs
            # + three initial counter words =112 total 32-bit carriers.
            case_bytes = sum((case / name).stat().st_size for name in set(manifest["files"]) | {"case.json"})
            exact_bound = case_bytes + 112 * 4 + (32 << 20)
            with mock.patch.object(common, "CAPTURE_LIMIT", exact_bound):
                self.assertEqual(common.load(case), manifest)
            with mock.patch.object(common, "CAPTURE_LIMIT", exact_bound-1):
                with self.assertRaisesRegex(ValueError, "projected complete capture"):
                    common.load(case)

    def test_unsupported_oracle_and_invalid_output_tolerances_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            case = base / "source"
            original = make_case(case)
            del original["family"]
            original["portable_oracle_module"] = "unsupported.module"
            common.write_json(case / "case.json", original)
            with mock.patch.object(exporter.importlib, "import_module") as load_module:
                with self.assertRaisesRegex(ValueError, "supported native oracle"):
                    exporter.freeze(case, base / "unsupported")
                load_module.assert_not_called()
            invalid = [dict(stats=dict(rtol=0., atol=0.)), dict(unknown=dict(rtol=0., atol=0.)),
                       dict(h_out=dict(rtol=True, atol=0.)), dict(h_out=dict(rtol=float("nan"), atol=0.)),
                       dict(h_out=dict(rtol=.01, atol=0.)), dict(h_out=dict(rtol=0.))]
            for index, overrides in enumerate(invalid):
                def oracle(path):
                    return dict(expectedarrays=common.load_npz(path / "expected.npz"), exactfields=["stats"],
                                rtol=3e-4, atol=3e-4, tolerances=overrides, scope="invalid tolerance fixture")
                with self.assertRaises(ValueError):
                    exporter.freeze(case, base / str(index), oracle=oracle)

    def test_archive_traversal_and_duplicate_members_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for index, members in enumerate([[("case/../escape", b"x")],
                                             [("case/a", b"x"), ("case/a", b"y")]]):
                archive = base / f"bad{index}.tar.gz"
                archive.write_bytes(tar_bytes(members))
                with self.assertRaisesRegex(ValueError, "archive path or duplicate"):
                    common.unpack_archive(archive, base / f"out{index}", "case", 1024)

    def test_npz_shape_extent_checked_before_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.npz"
            payload = io.BytesIO()
            np.lib.format.write_array_header_1_0(payload, dict(descr="<f4", fortran_order=False, shape=(100000000,)))
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("bad.npy", payload.getvalue())
            with self.assertRaisesRegex(ValueError, "exact NPY extent"):
                common.load_npz(path)

    def test_origin_pin_and_missing_numerical_contract_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "case"
            manifest = make_case(case)
            manifest["origin_files"] = {"missing": "source/pe.csl"}
            common.write_json(case / "case.json", manifest)
            with self.assertRaisesRegex(ValueError, "origin file mapping"):
                common.load(case)
            manifest["origin_files"] = {}
            del manifest["checks"]["h_out"]
            common.write_json(case / "case.json", manifest)
            with self.assertRaisesRegex(ValueError, "complete numerical contract"):
                common.load(case)


class ResidentWorkerTests(unittest.TestCase):
    """Exercise advancing checkpoints through the real worker and fake SDK."""

    setUp = WorkerTests.setUp
    arguments = WorkerTests.arguments

    def resident(self):
        self.manifest["schema"] = 2
        self.manifest.pop("launch")
        self.manifest["launches"] = [dict(entrypoint="run", steps=2), dict(entrypoint="run", steps=2)]
        self.manifest["params"]["MAX_STEPS"] = 4
        self.manifest["outputs"]["lifecycle"] = [4, "uint32"]
        self.manifest["checks"]["lifecycle"] = dict(kind="exact")
        base = self.expected
        self.expected, masks, exact = {}, {}, {}
        for index in range(2):
            prefix = f"launch{index}_"
            values = {name: value.copy() for name, value in base.items()}
            values["h_out"][:, 0] += np.float32(index / 4)
            values["stats"][:, 1] = 2
            values["lifecycle"] = np.tile(np.array([1, index+1, 2*(index+1), 1], np.uint32), (4, 1))
            self.expected.update({prefix + name: value for name, value in values.items()})
            masks[prefix + "h_out"] = np.tile(np.array([0, 1], np.uint32), (4, 1))
            exact[prefix + "h_out"] = np.tile(np.array([1, 0], np.uint32), (4, 1))
        for name, values in (("expected.npz", self.expected), ("zero_masks.npz", masks), ("exact_masks.npz", exact)):
            np.savez_compressed(self.case / name, **values)
            self.manifest["files"][name] = common.sha(self.case / name)
        self.manifest["exact_mask_counts"] = {name: 4 for name in exact}
        common.write_json(self.case / "case.json", self.manifest)
        self.archive.write_bytes(tar_bytes([("case/" + name, (self.case / name).read_bytes())
            for name in sorted(set(self.manifest["files"]) | {"case.json"})]))
        common.write_json(self.build / "out.json", dict(params=self.manifest["params"]))
        symbols = common.read_json(self.build / "bin/out_rpc.json")["rpc_symbols"]
        symbols[-1]["inputs"] = [dict(type="u32")]
        symbols.append(dict(id=len(symbols), name="lifecycle", kind="Var", type="[*]u32"))
        common.write_json(self.build / "bin/out_rpc.json", dict(rpc_symbols=symbols))

    def test_two_advancing_checkpoints_upload_once_and_check_every_output(self):
        self.resident()
        args = self.arguments()
        self.assertEqual(worker.capture(args), 0)
        report = common.audit_report(args.out)
        self.assertEqual(report["numerical_status"], "PASS")
        self.assertEqual([row["cumulative_steps"] for row in report["checkpoints"]], [2, 4])
        self.assertEqual(report["local_cycles_max"], 2)
        self.assertEqual(set(report["errors"]), set(self.expected))
        raw = common.load_npz(args.out / "runtime/raw.npz", common.CAPTURE_LIMIT)
        self.assertEqual(set(raw), set(common.raw_schema(self.manifest)))
        execution = common.read_json(args.out / "runtime/execution.json")
        self.assertEqual(execution["launches"], self.manifest["launches"])
        self.assertEqual(execution["api"], common.api_sequence(common.load(self.case)))
        last_upload = max(i for i, label in enumerate(execution["api"]) if label.startswith("h2d:"))
        first_launch = execution["api"].index("launch:run")
        self.assertLess(last_upload, first_launch)
        calls = [row for row in self.sdk_calls if row[0] == "launch"]
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(type(row[2][0]) is np.uint32 and row[2][0] == 2 for row in calls))
        with zipfile.ZipFile(args.out / "runtime/raw.npz") as archive:
            self.assertTrue(all(row.compress_type == zipfile.ZIP_DEFLATED for row in archive.infolist()))

    def test_replayed_state_and_bad_values_remain_completed_numerical_failures(self):
        self.resident()
        for mode in ("replay", "nan", "padding", "counter", "immutable", "clock", "boundary_ulp"):
            with self.subTest(mode=mode):
                self.mode = mode
                args = self.arguments(mode)
                self.assertEqual(worker.capture(args), 0)
                digest = common.sha(args.out / "runtime/raw.npz")
                report = common.audit_report(args.out)
                self.assertEqual(report["execution_status"], "PASS")
                self.assertEqual(report["numerical_status"], "FAIL")
                self.assertEqual(digest, common.sha(args.out / "runtime/raw.npz"))

    def test_interrupted_second_launch_preserves_first_checkpoint(self):
        self.resident()
        self.mode = "launch_failure_second"
        args = self.arguments()
        with self.assertRaisesRegex(RuntimeError, "partial arrays retained"):
            worker.capture(args)
        raw = common.load_npz(args.out / "runtime/raw.npz")
        self.assertIn("launch0_h_out", raw)
        self.assertNotIn("launch1_h_out", raw)
        self.assertEqual(common.read_json(args.out / "runtime/execution.json")["launches"], self.manifest["launches"][:1])
        with self.assertRaisesRegex(ValueError, "completed capture"):
            common.admit_capture(args.out)

    def test_u32_entrypoint_abi_required_before_runtime(self):
        self.resident()
        symbols = common.read_json(self.build / "bin/out_rpc.json")
        next(row for row in symbols["rpc_symbols"] if row["name"] == "run")["inputs"] = []
        common.write_json(self.build / "bin/out_rpc.json", symbols)
        with self.assertRaisesRegex(ValueError, "compiled symbol ABI"):
            worker.capture(self.arguments())
        self.assertFalse(self.sdk_calls)

    def test_launch_argument_and_checkpoint_inventory_corruption_rejected(self):
        self.resident()
        args = self.arguments()
        worker.capture(args)
        path = args.out / "runtime/execution.json"
        original = common.read_json(path)
        for argument in (1, True):
            with self.subTest(argument=argument):
                changed = dict(original, launches=[dict(entrypoint="run", steps=argument), self.manifest["launches"][1]])
                common.write_json(path, changed)
                with self.assertRaisesRegex(ValueError, "launch argument"):
                    common.admit_capture(args.out)
        common.write_json(path, original)
        raw_path = args.out / "runtime/raw.npz"
        raw = common.load_npz(raw_path)
        del raw["launch1_h_out"]
        np.savez_compressed(raw_path, **raw)
        original["raw_sha256"] = common.sha(raw_path)
        common.write_json(path, original)
        with self.assertRaisesRegex(ValueError, "complete raw capture inventory"):
            common.admit_capture(args.out)

    def test_schedule_bounds_and_expanded_capture_bound_remain_strict(self):
        self.resident()
        good = self.manifest
        for launches in ([], [dict(entrypoint="run", steps=0)], [dict(entrypoint="run", steps=True)],
                         [dict(entrypoint="other", steps=2)], [dict(entrypoint="run", steps=2)] * 3,
                         [dict(entrypoint="run", steps=1)] * 17):
            with self.subTest(launches=launches), self.assertRaises(ValueError):
                common.launch_contract(dict(good, launches=launches))
        with self.assertRaisesRegex(ValueError, "MAX_STEPS"):
            common.launch_contract(dict(good, params=dict(good["params"], MAX_STEPS=257)))
        common.load(self.case)
        size = (self.case / "case.json").stat().st_size + sum((self.case / name).stat().st_size for name in good["files"])
        raw_bytes = sum(field[0] for field in common.raw_schema(good).values()) * 4 * 4
        with mock.patch.object(common, "CAPTURE_LIMIT", size + raw_bytes + (32 << 20) - 1):
            with self.assertRaisesRegex(ValueError, "projected complete capture"):
                common.load(self.case)

    def test_export_freezes_schedule_and_checkpoint_masks(self):
        self.resident()
        original = dict(self.manifest)
        original.pop("family")
        original["residency"] = dict(schema=1, chunks=[2, 2])
        common.write_json(self.case / "case.json", original)
        contract = dict(launches=self.manifest["launches"], expectedarrays=self.expected,
                        zeromasks=common.load_npz(self.case / "zero_masks.npz"),
                        exactmasks=common.load_npz(self.case / "exact_masks.npz"),
                        exactfields=["stats", "lifecycle"], rtol=3e-4, atol=3e-4,
                        immutable_inputs=self.manifest["immutable_inputs"], scope="fake resident test only")
        out = self.base / "portable"
        result = exporter.freeze(self.case, out, oracle=lambda case: contract)
        self.assertEqual(result["schema"], 2)
        self.assertEqual(result["launches"], self.manifest["launches"])
        self.assertEqual(result["exact_mask_counts"], self.manifest["exact_mask_counts"])
        self.assertEqual(set(common.load_npz(out / "expected.npz")), set(self.expected))
        self.assertEqual((out / "expected.npz").read_bytes(), (self.case / "expected.npz").read_bytes())
        bad = dict(contract, launches=[dict(entrypoint="run", steps=4)])
        with self.assertRaisesRegex(ValueError, "schedules differ"):
            exporter.freeze(self.case, self.base / "bad", oracle=lambda case: bad)


class RolloutComparisonTests(unittest.TestCase):
    """Compare separately captured fake physical cases through strict audits."""

    setUp = WorkerTests.setUp
    arguments = WorkerTests.arguments
    resident = ResidentWorkerTests.resident

    def make_capture(self, name, chunks, variation=None, backend="hardware"):
        self.case = self.base / (name + "-case")
        self.manifest = make_case(self.case)
        self.inputs = common.load_npz(self.case / "inputs.npz")
        self.expected = common.load_npz(self.case / "expected.npz")
        symbols = []
        for index, (field, (_, dtype)) in enumerate(dict(self.manifest["inputs"], **self.manifest["outputs"]).items()):
            symbols.append(dict(id=index, name=field, kind="Var", type={"float32": "[*]f32", "uint32": "[*]u32", "uint16": "[*]u16"}[dtype]))
        symbols.append(dict(id=len(symbols), name="run", kind="Func", type="void", inputs=[]))
        common.write_json(self.build / "bin/out_rpc.json", dict(rpc_symbols=symbols))
        self.resident()
        for field in comparison.NUMERICAL:
            self.manifest["outputs"][field] = [2, "float32"]
            self.manifest["checks"][field] = dict(kind="approx", rtol=3e-4, atol=3e-4)
            for index in range(2):
                self.expected[f"launch{index}_" + field] = self.expected[f"launch{index}_h_out"].copy()
        masks = common.load_npz(self.case / "zero_masks.npz")
        exact = common.load_npz(self.case / "exact_masks.npz")
        if chunks == [4]:
            self.expected = {"launch0_" + key[len("launch1_"):]: value.copy()
                             for key, value in self.expected.items() if key.startswith("launch1_")}
            self.expected["launch0_stats"][:, 1] = 4
            self.expected["launch0_lifecycle"][:, 1] = 1
            masks = {"launch0_h_out": masks["launch1_h_out"]}
            exact = {"launch0_h_out": exact["launch1_h_out"]}
        self.manifest["launches"] = [dict(entrypoint="run", steps=n) for n in chunks]
        if variation == "state":
            field = f"launch{len(chunks)-1}_delta_h"
            self.expected[field][0, 0] = np.nextafter(self.expected[field][0, 0], np.float32(1))
        if variation == "graph":
            self.inputs["counts"][0, 0] = 0
        original_arrays = dict(edges=np.array([[0, 1]], np.uint32), weights=np.array([.25], np.float32))
        if variation == "original":
            original_arrays["edges"][0, 1] = 2
        if variation == "source":
            (self.case / "source/pe.csl").write_text("changed mock source, never compile")
            self.manifest["files"]["source/pe.csl"] = common.sha(self.case / "source/pe.csl")
        for filename, values in (("inputs.npz", self.inputs), ("expected.npz", self.expected),
                                 ("zero_masks.npz", masks), ("exact_masks.npz", exact), ("original.npz", original_arrays)):
            np.savez_compressed(self.case / filename, **values)
            self.manifest["files"][filename] = common.sha(self.case / filename)
        origin = dict(schema=1, files={"original.npz": common.sha(self.case / "original.npz")},
                      provenance=dict(dataset="mock-graph", model="other" if variation == "provenance" else "mock-weights"),
                      sources={"source/pe.csl": common.sha(self.case / "source/pe.csl")},
                      transport="mock-colored", schedule=dict(rounds=2), residency=dict(schema=1, chunks=chunks))
        common.write_json(self.case / "origin-case.json", origin)
        self.manifest["origin_files"] = {"original.npz": "original.npz"}
        self.manifest["origin_case_sha256"] = common.sha(self.case / "origin-case.json")
        self.manifest["files"]["origin-case.json"] = self.manifest["origin_case_sha256"]
        self.manifest["exact_mask_counts"] = {field: int(value.sum()) for field, value in exact.items()}
        common.write_json(self.case / "case.json", self.manifest)
        self.archive.write_bytes(tar_bytes([("case/" + filename, (self.case / filename).read_bytes())
            for filename in sorted(set(self.manifest["files"]) | {"case.json"})]))
        symbols = []
        for index, (field, (_, dtype)) in enumerate(dict(self.manifest["inputs"], **self.manifest["outputs"]).items()):
            symbols.append(dict(id=index, name=field, kind="Var", type={"float32": "[*]f32", "uint32": "[*]u32", "uint16": "[*]u16"}[dtype]))
        symbols.append(dict(id=len(symbols), name="run", kind="Func", type="void", inputs=[dict(type="u32")]))
        common.write_json(self.build / "bin/out_rpc.json", dict(rpc_symbols=symbols))
        (self.build / "bin/out_0_0.elf").write_bytes(minimal_elf(4356 if variation == "elf" else 4352))
        self.mode = "replay" if variation == "replay" else "normal"
        args = self.arguments(name, backend)
        self.assertEqual(worker.capture(args), 0)
        return args.out

    def test_complete_matched_rollouts_and_host_cli(self):
        first = self.make_capture("whole", [4])
        second = self.make_capture("split", [2, 2])
        report = comparison.compare(first, second)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["bit_equal_arrays"], list(comparison.NUMERICAL))
        self.assertEqual(report["chunks"], [[4], [2, 2]])
        self.assertTrue(report["identical_generated_elf_hashes"])
        self.assertEqual(report["intervals"][0]["per_pe_sum_cycles"], [1] * 4)
        self.assertEqual(report["intervals"][1]["per_pe_sum_cycles"], [2] * 4)
        calls = list(self.sdk_calls)
        output = self.base / "comparison.json"
        self.assertEqual(comparison.main([str(first), str(second), "--out", str(output)]), 0)
        self.assertEqual(self.sdk_calls, calls)
        with self.assertRaisesRegex(ValueError, "fresh comparison"):
            comparison.main([str(first), str(second), "--out", str(output)])

    def test_numerically_admitted_changed_final_bits_still_rejected(self):
        first = self.make_capture("whole", [4])
        second = self.make_capture("changed", [2, 2], "state")
        self.assertEqual(common.audit_report(second)["numerical_status"], "PASS")
        with self.assertRaisesRegex(ValueError, "final numerical state array bits: delta_h"):
            comparison.compare(first, second)

    def test_changed_source_graph_original_and_provenance_rejected(self):
        first = self.make_capture("whole", [4])
        for variation, message in (("source", "frozen CSL"), ("graph", "uploaded physical inputs"),
                                   ("original", "original graph"), ("provenance", "originating provenance")):
            with self.subTest(variation=variation):
                second = self.make_capture(variation, [2, 2], variation)
                self.assertEqual(common.audit_report(second)["numerical_status"], "PASS")
                with self.assertRaisesRegex(ValueError, message):
                    comparison.compare(first, second)

    def test_replayed_checkpoint_fails_strict_audit_before_comparison(self):
        first = self.make_capture("whole", [4])
        second = self.make_capture("replay", [2, 2], "replay")
        with self.assertRaisesRegex(ValueError, "both captures must pass strict numerical"):
            comparison.compare(first, second)

    def test_generated_elf_difference_is_reported_without_changing_numerical_result(self):
        first = self.make_capture("whole", [4])
        second = self.make_capture("different-elf", [2, 2], "elf")
        report = comparison.compare(first, second)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["identical_generated_elf_hashes"])
        self.assertTrue(report["identical_artifact_hashes"])

    def test_same_schedule_and_mixed_backend_rejected(self):
        first = self.make_capture("whole", [4])
        same = self.make_capture("same", [4])
        with self.assertRaisesRegex(ValueError, "different chunks"):
            comparison.compare(first, same)
        simulator = self.make_capture("simulator", [2, 2], backend="simulator")
        with self.assertRaisesRegex(ValueError, "same execution backend"):
            comparison.compare(first, simulator)


class LocalCompileBindingTests(unittest.TestCase):
    def test_local_compiler_workers_record_and_binding_without_sdk(self):
        try:
            from models.meshgraphnet.native_full import driver
        except ModuleNotFoundError:
            self.skipTest("local source driver is not part of standalone appliance export")
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            (case/"case.json").write_text("{}\n")
            manifest = dict(params=dict(C=32, R=3))
            for workers in (1, 4, 8):
                with self.subTest(workers=workers):
                    with mock.patch.object(driver, "load", return_value=manifest), \
                         mock.patch.object(driver, "check_abi"), \
                         mock.patch.object(driver.elf, "build_inventory", return_value={}), \
                         mock.patch.object(driver.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as compiler:
                        result = driver.compile_case(case, workers)
                    self.assertEqual(result["compile_workers"], workers)
                    self.assertIn(f"--max-parallelism={workers}", compiler.call_args.args[0])
                    driver.check_build_binding(case, manifest, result)
                    bad = dict(result, compile_workers=4 if workers != 4 else 8)
                    with self.assertRaisesRegex(ValueError, "binding"):
                        driver.check_build_binding(case, manifest, bad)
                    (case/"build.json").unlink()
            for workers in (True, 0, 2, 16):
                with self.assertRaises(ValueError):
                    driver.compile_command(case, manifest, workers)
            self.assertEqual(driver.elf.SCAN_BYTES, 1 << 30)


if __name__ == "__main__":
    unittest.main()
