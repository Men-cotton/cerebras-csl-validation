"""Fake appliance clients: no SDK import, submission, or numerical SIM claim."""
import hashlib
import io
import json
import logging
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_appliance as adapter


CONFIG = dict(kind="transfer", start="blocking", transfer="blocking", arm="rectangle",
              pause_at="none", pause_seconds=0.0, work=1, steps=3)


def archive_bytes(members):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, payload in members:
            item = tarfile.TarInfo(name)
            if isinstance(payload, tarfile.TarInfo):
                archive.addfile(payload)
            else:
                item.size = len(payload)
                archive.addfile(item, io.BytesIO(payload))
    return output.getvalue()


class ClockApplianceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "clock_source").mkdir()
        for name in ("layout.csl", "pe.csl"):
            (self.root / "clock_source" / name).write_text("// fixture " + name)
        for name in ("clock_probe.py", "clock_common.py", "clock_pmu.py", "clock_appliance.py"):
            (self.root / name).write_text("# fixture " + name)
        self.config_file = self.root / "condition.json"
        self.config_file.write_text(json.dumps({"config": CONFIG}))
        self.seen = {}

    def compile_record(self, backend="simulator", **overrides):
        seen = self.seen

        class Compiler:
            def __init__(self, **options):
                seen["compile_options"] = options

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def compile(self, source, layout, flags, destination):
                seen["compile"] = (source, layout, flags, destination)
                logging.warning("Job id: wsjob-compile-123, workflow id: flow-compile, remote log path: logs/compile")
                artifact = Path(destination) / "artifact.tar.gz"
                artifact.write_bytes(b"synthetic compile artifact")
                return str(artifact)

        values = dict(backend=backend, cols=2, rows=2, words=16, namespace="test-space",
                      fabric_dims=None, fabric_offsets="4,1", out=self.root / ("build-" + backend))
        values.update(overrides)
        args = SimpleNamespace(**values)
        self.assertEqual(adapter.compile_program(args, self.root, Compiler), 0)
        return args.out / "clock-build.json"

    def capture(self, manifest, request, status="pass", mutate=None, calls=None):
        # These bytes exercise controller integrity checks. The separate worker
        # tests own raw numerical admission and do not use this fake NPZ payload.
        raw = b"synthetic NPZ bytes for controller identity/integrity tests"
        calls = json.dumps(calls if calls is not None else [
            dict(index=0, api="stop", phase=phase, stage="teardown", detail={})
            for phase in ("enter", "exit")]).encode()
        meta = dict(schema=1, status=status, backend=manifest["backend"], params=manifest["params"],
                    source_sha256=manifest["source_sha256"], host_sha256=manifest["host_sha256"],
                    config=request["config"], run_id=request["run_id"],
                    manifest_sha256=request["manifest_sha256"],
                    raw_sha256=hashlib.sha256(raw).hexdigest(), calls_sha256=hashlib.sha256(calls).hexdigest())
        if mutate:
            mutate(meta)
        return archive_bytes([("capture/capture.json", json.dumps(meta).encode()),
                              ("capture/calls.json", calls), ("capture/raw.npz", raw)])

    def run_fake(self, record, backend="simulator", status="pass", exit_code=None, mutate=None,
                 missing=None, run_error=False, enter_error=False, exit_error=False,
                 constructor_error=False, out=None, suffixes=None, relative_returns=False,
                 numerical_error=False):
        seen, test = self.seen, self
        output = out or self.root / "result"
        seen["staged"] = {}
        seen["numerical_calls"] = []

        class Launcher:
            def __init__(self, artifact, **options):
                seen.update(artifact=artifact, launch_options=options)
                if constructor_error:
                    raise RuntimeError("constructor failure")

            def __enter__(self):
                if enter_error:
                    raise RuntimeError("enter failure")
                return self

            def __exit__(self, *args):
                if exit_error:
                    raise RuntimeError("exit failure")
                return False

            def stage(self, path):
                path = Path(path)
                seen["staged"][path.name] = path.read_bytes()

            def run(self, command):
                seen["command"] = command
                logging.warning("Job id: wsjob-launch-456, workflow id: flow-launch, remote log path: logs/launch")
                if run_error:
                    raise RuntimeError("transport interrupted")
                return None

            def download_artifact(self, remote, destination):
                if remote == missing:
                    return None
                manifest = json.loads(seen["staged"]["clock-build.json"])
                request = json.loads(seen["staged"]["clock-request.json"])
                code = exit_code if exit_code is not None else {
                    "pass": 0, "observed_numeric_fail": 1, "incomplete": 2}[status]
                payloads = {"host.log": b"worker log\n", "host-exit.txt": (str(code) + "\n").encode(),
                            "capture": test.capture(manifest, request, status, mutate)}
                if suffixes and remote in suffixes:
                    saved = Path(destination + suffixes[remote])
                    saved.write_bytes(payloads[remote])
                    return saved.name if relative_returns else str(saved)
                Path(destination).write_bytes(payloads[remote])
                return None  # SDK versions may create the file and return None.

        args = SimpleNamespace(artifact_record=record, backend=backend, config_file=self.config_file,
                               host_timeout=300, out=output)

        def numerical_hook(meta, calls, raw_bytes, manifest, request):
            seen["numerical_calls"].append((meta, calls, raw_bytes, manifest, request))
            if numerical_error:
                raise ValueError("raw numerical admission rejected")
            return dict(status=meta["status"], scope="synthetic controller fixture only")

        # Only replace raw/API numerical admission: archive bounds, hashes,
        # identity, manifest joins, status consistency and collection stay real.
        with mock.patch.object(adapter.common, "validate_capture_content", side_effect=numerical_hook):
            return adapter.run_program(args, self.root, Launcher), output

    def test_collection_reports_unavailable_pmu_separately_from_numerical_pass(self):
        path = self.compile_record()
        manifest = adapter.read_json(path)
        request = dict(config=dict(CONFIG,host_pmu=True),run_id='test-pmu',manifest_sha256=adapter.digest(path))
        errors = dict(stage='open',type='PermissionError',errno=1,message='denied')
        calls = [dict(pmu=dict(status='unavailable',scope='thread-self-user',error=errors))]
        capture = self.root/'pmu.tar.gz'
        capture.write_bytes(self.capture(manifest,request,calls=calls))
        with mock.patch.object(adapter.common,'validate_capture_content',return_value=dict(status='pass')):
            meta, receipt = adapter.validate_capture(capture,manifest,request)
        self.assertEqual(meta['status'],'pass')
        self.assertEqual(receipt['pmu'],dict(requested=True,samples=1,measured=0,unavailable=1,errors=[errors]))

    def test_compile_pins_source_host_adapter_params_and_exact_options(self):
        path = self.compile_record()
        manifest, artifact = adapter.load_manifest(path, "simulator", self.root)
        self.assertEqual(manifest["backend"], "appliance-sim")
        self.assertEqual(manifest["params"], dict(C=2, R=2, W=16))
        self.assertEqual(set(manifest["host_sha256"]), {"clock_probe.py", "clock_common.py", "clock_pmu.py"})
        self.assertEqual(manifest["launcher_sha256"], adapter.digest(self.root / "clock_appliance.py"))
        self.assertIn("--params=C:2,R:2,W:16", manifest["compiler_args"])
        self.assertIn("--fabric-dims=9,4", manifest["compiler_args"])
        self.assertEqual(artifact.read_bytes(), b"synthetic compile artifact")
        self.assertEqual(self.seen["compile_options"], dict(resource_cpu=1000, resource_mem=8 << 30,
                                                         disable_version_check=False, mgmt_namespace="test-space"))
        status = adapter.read_json(path.parent / "compile-status.json")
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["jobs"][0]["role"], "compile")
        self.assertEqual(status["jobs"][0]["job_id"], "wsjob-compile-123")

    def test_hardware_dimensions_and_override_are_explicit(self):
        first = self.compile_record("hardware")
        self.assertIn("--fabric-dims=762,1172", adapter.read_json(first)["compiler_args"])
        second = self.compile_record("hardware", fabric_dims="100,80", fabric_offsets="8,4",
                                     out=self.root / "build-custom")
        self.assertIn("--fabric-dims=100,80 --fabric-offsets=8,4", adapter.read_json(second)["compiler_args"])

    def test_matching_sources_and_artifact_are_required(self):
        path = self.compile_record()
        with self.assertRaisesRegex(ValueError, "backend differs"):
            adapter.load_manifest(path, "hardware", self.root)
        for file in ("clock_probe.py", "clock_common.py", "clock_pmu.py", "clock_appliance.py", "clock_source/pe.csl"):
            target = self.root / file
            original = target.read_bytes()
            target.write_bytes(original + b" changed")
            with self.subTest(file=file), self.assertRaisesRegex(ValueError, "differs"):
                adapter.load_manifest(path, "simulator", self.root)
            target.write_bytes(original)
        (path.parent / "artifact.tar.gz").write_bytes(b"replacement")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            adapter.load_manifest(path, "simulator", self.root)

    def test_compile_status_and_original_request_are_joined(self):
        path = self.compile_record()
        status_path, request_path = path.parent / "compile-status.json", path.parent / "request.json"
        saved_manifest, saved_status, saved_request = path.read_bytes(), status_path.read_bytes(), request_path.read_bytes()
        for key, value in (("status", "incomplete"), ("manifest_sha256", "wrong"),
                           ("compile_id", "other-compile"), ("backend", "hardware"),
                           ("accounting", {"status": "incomplete"})):
            status = json.loads(saved_status)
            status[key] = value
            adapter.write_json(status_path, status)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "compile"):
                adapter.load_manifest(path, "simulator", self.root)
            status_path.write_bytes(saved_status)
        # Updating the final-manifest hash cannot hide a changed compile request.
        manifest = json.loads(saved_manifest)
        manifest["params"] = dict(C=2, R=2, W=32)
        adapter.write_json(path, manifest)
        status = json.loads(saved_status)
        status["manifest_sha256"] = adapter.digest(path)
        adapter.write_json(status_path, status)
        with self.assertRaisesRegex(ValueError, "original compile request"):
            adapter.load_manifest(path, "simulator", self.root)
        path.write_bytes(saved_manifest)
        status_path.write_bytes(saved_status)
        original = json.loads(saved_request)
        original["compiler_args"] += " --changed"
        adapter.write_json(request_path, original)
        with self.assertRaisesRegex(ValueError, "original compile request"):
            adapter.load_manifest(path, "simulator", self.root)
        request_path.write_bytes(saved_request)
        status_path.unlink()
        with self.assertRaisesRegex(ValueError, "regular"):
            adapter.load_manifest(path, "simulator", self.root)

    def test_sim_stage_manifest_bytes_and_run_ids(self):
        path = self.compile_record()
        code, out = self.run_fake(path)
        self.assertEqual(code, 0)
        self.assertEqual(self.seen["staged"]["clock-build.json"], path.read_bytes())
        self.assertEqual(set(self.seen["staged"]), {"clock_probe.py", "clock_common.py", "clock_pmu.py", "clock-build.json", "clock-request.json"})
        self.assertIn("--backend appliance-sim", self.seen["command"])
        self.assertNotIn("%CMADDR%", self.seen["command"])
        self.assertTrue(self.seen["launch_options"]["simulator"])
        first = adapter.read_json(out / "clock-request.json")
        self.assertEqual(first["manifest_sha256"], adapter.digest(path))
        collection = adapter.read_json(out / "collection.json")
        self.assertEqual(collection["status"], "complete")
        self.assertEqual(collection["capture_validation"]["status"], "pass")
        self.assertEqual(len(self.seen["numerical_calls"]), 1)
        self.assertEqual(collection["jobs"][0]["job_id"], "wsjob-launch-456")
        self.assertEqual(set(collection["downloaded"]), {"host.log", "host-exit.txt", "capture"})
        self.run_fake(path, out=self.root / "result-again")
        second = adapter.read_json(self.root / "result-again/clock-request.json")
        self.assertNotEqual(first["run_id"], second["run_id"])

    def test_returned_archive_suffix_and_actual_exit_path_are_used(self):
        path = self.compile_record()
        for relative in (False, True):
            output = self.root / ("suffix-relative" if relative else "suffix-absolute")
            code, out = self.run_fake(path, out=output, suffixes={"capture": ".tar.gz", "host-exit.txt": ".saved"},
                                      relative_returns=relative)
            self.assertEqual(code, 0)
            summary = adapter.read_json(out / "collection.json")
            self.assertEqual(summary["downloaded"]["capture"]["path"], "capture.tar.gz")
            self.assertEqual(summary["downloaded"]["host-exit.txt"]["path"], "host-exit.txt.saved")
            self.assertTrue((out / "capture.tar.gz").is_file())
            self.assertFalse((out / "capture").exists())

    def test_raw_validator_failure_is_not_admitted(self):
        with self.assertRaisesRegex(ValueError, "collection_failed"):
            self.run_fake(self.compile_record(), numerical_error=True)
        summary = adapter.read_json(self.root / "result/collection.json")
        self.assertIn("raw numerical admission rejected", summary["errors"]["capture"])
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(len(self.seen["numerical_calls"]), 1)
        self.assertTrue((self.root / "result/capture").is_file())

    def test_hardware_numeric_failure_is_complete_and_kept(self):
        code, out = self.run_fake(self.compile_record("hardware"), backend="hardware", status="observed_numeric_fail")
        self.assertEqual(code, 1)
        self.assertIn("--cmaddr %CMADDR%", self.seen["command"])
        self.assertFalse(self.seen["launch_options"]["simulator"])
        self.assertEqual(adapter.read_json(out / "collection.json")["worker_status"], "observed_numeric_fail")

    def test_capture_identity_and_hash_mismatches_fail_closed(self):
        path = self.compile_record()
        changes = {"run_id": "wrong-run", "manifest_sha256": "wrong-manifest", "backend": "hardware",
                   "params": dict(C=1, R=1, W=1), "host_sha256": {}, "source_sha256": {},
                   "raw_sha256": "wrong-raw", "calls_sha256": "wrong-calls", "schema": 2}
        for index, (key, value) in enumerate(changes.items()):
            output = self.root / ("bad-" + str(index))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "collection_failed"):
                self.run_fake(path, mutate=lambda meta, k=key, v=value: meta.update({k: v}), out=output)
            summary = adapter.read_json(output / "collection.json")
            self.assertEqual(summary["status"], "incomplete")
            self.assertIn("capture", summary["errors"])
            self.assertTrue((output / "capture").is_file())

    def test_exit_disagreement_timeout_and_incomplete_are_not_pass(self):
        path = self.compile_record()
        for index, (status, code) in enumerate((("pass", 1), ("pass", 124), ("incomplete", 2))):
            output = self.root / ("exit-" + str(index))
            with self.subTest(status=status, code=code), self.assertRaises(ValueError):
                self.run_fake(path, status=status, exit_code=code, out=output)
            summary = adapter.read_json(output / "collection.json")
            self.assertEqual(summary["status"], "incomplete")
            self.assertTrue((output / "host-exit.txt").is_file())

    def test_missing_download_preserves_other_files(self):
        with self.assertRaisesRegex(ValueError, "collection_failed"):
            self.run_fake(self.compile_record(), missing="host.log")
        summary = adapter.read_json(self.root / "result/collection.json")
        self.assertEqual(set(summary["errors"]), {"host.log"})
        self.assertIn("capture", summary["downloaded"])

    def test_run_and_context_failures_leave_final_status(self):
        path = self.compile_record()
        for index, option in enumerate(("constructor_error", "enter_error", "run_error", "exit_error")):
            output = self.root / ("failure-" + str(index))
            with self.subTest(option=option), self.assertRaises(RuntimeError):
                self.run_fake(path, out=output, **{option: True})
            summary = adapter.read_json(output / "collection.json")
            self.assertEqual(summary["status"], "incomplete")
            self.assertIn("error", summary)
            if option in ("run_error", "exit_error"):
                self.assertIn("capture", summary["downloaded"])

    def test_compile_constructor_failure_leaves_status(self):
        class Compiler:
            def __init__(self, **kwargs):
                raise RuntimeError("compiler constructor failed")

        args = SimpleNamespace(backend="simulator", cols=2, rows=2, words=16, namespace=None,
                               fabric_dims=None, fabric_offsets="4,1", out=self.root / "bad-compile")
        with self.assertRaisesRegex(RuntimeError, "constructor failed"):
            adapter.compile_program(args, self.root, Compiler)
        summary = adapter.read_json(args.out / "compile-status.json")
        self.assertEqual(summary["status"], "incomplete")
        self.assertIn("constructor failed", summary["error"])

    def test_existing_output_and_ambiguous_config_are_rejected(self):
        path = self.compile_record()
        self.run_fake(path)
        with self.assertRaises(FileExistsError):
            self.run_fake(path)
        for bad in (CONFIG, {"config": CONFIG, "run_id": "user-supplied"}):
            self.config_file.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "only"):
                self.run_fake(path, out=self.root / "unused")
            self.assertFalse((self.root / "unused").exists())

    def test_timeout_and_fabric_inputs(self):
        for bad in (0, -1, 1801, True, "300; false"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                adapter.host_command("hardware", bad)
        self.assertIn("timeout --signal=INT --kill-after=15s 300s", adapter.host_command("hardware", 300))
        with self.assertRaises(ValueError):
            self.compile_record(fabric_dims="4,2")
        for bad in ("-1,2", "2,2; false", "2"):
            with self.subTest(value=bad), self.assertRaises(Exception):
                adapter.pair(bad)

    def test_archive_rejects_traversal_links_duplicates_and_missing_files(self):
        link = tarfile.TarInfo("capture/raw.npz")
        link.type, link.linkname = tarfile.SYMTYPE, "/tmp/not-read"
        cases = [
            [("capture/../escape", b"x")],
            [("/capture/capture.json", b"x")],
            [("capture/raw.npz", link)],
            [("capture/capture.json", b"{}"), ("capture/capture.json", b"{}")],
            [("capture/capture.json", b"{}")],
        ]
        for index, members in enumerate(cases):
            path = self.root / ("archive-" + str(index))
            path.write_bytes(archive_bytes(members))
            with self.subTest(index=index), self.assertRaises(ValueError):
                adapter.capture_payloads(path)

    def test_local_accounting_and_json_reject_symlinks(self):
        directory = self.root / "artifacts"
        directory.mkdir()
        (directory / "file").write_bytes(b"123")
        snapshot = adapter.accounting(directory)
        self.assertEqual((snapshot["entries"], snapshot["bytes"]), (1, 3))
        (directory / "link").symlink_to(directory / "file")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            adapter.accounting(directory)
        with self.assertRaisesRegex(ValueError, "regular"):
            adapter.bounded_bytes(directory / "link", 100)

    def test_download_paths_cannot_escape_or_alias_controller_files(self):
        directory = self.root / "downloads"
        directory.mkdir()
        requested = directory / "capture"
        requested.write_bytes(b"archive")
        self.assertEqual(adapter.downloaded_path(None, requested, 100), requested)
        outside = self.root / "outside"
        outside.write_bytes(b"outside")
        with self.assertRaisesRegex(ValueError, "outside"):
            adapter.downloaded_path(str(outside), requested, 100)
        link = directory / "capture.tar.gz"
        link.symlink_to(requested)
        with self.assertRaisesRegex(ValueError, "symlink"):
            adapter.downloaded_path(str(link), requested, 100)
        controller = directory / "clock-request.json"
        controller.write_text("{}")
        with self.assertRaisesRegex(ValueError, "controller record"):
            adapter.downloaded_path(str(controller), requested, 100)
        for invalid in ("", 123, {}):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "invalid path"):
                adapter.downloaded_path(invalid, requested, 100)

    def test_log_cap_is_explicit(self):
        path = self.root / "bounded.log"
        handler = adapter.BoundedLog(path)
        self.addCleanup(handler.close)
        handler.size = adapter.LOG_LIMIT - 1
        handler.emit(logging.LogRecord("test", logging.INFO, "", 0, "too large", (), None))
        self.assertTrue(handler.truncated)
        self.assertEqual(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
