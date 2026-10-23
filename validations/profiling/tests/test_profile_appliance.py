"""Single-action adapter tests with fake SDK compilation, staging and downloads."""
import io
import json
import logging
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

import profile_appliance as app


def archive_bytes(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, payload in files:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return stream.getvalue()


class AdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "source"
        (self.root / "source").mkdir(parents=True)
        for name in (*app.HOSTS, *app.SOURCES, "profile_appliance.py"):
            (self.root / name).write_text("fixture " + name)
        self.compile_calls, self.run_calls, self.options, self.staged = [], [], [], {}
        self.download_style, self.worker_exit = "relative", "0"
        self.launch_error = False
        self.capture_wrong_request = False
        self.missing_download = None
        self.bad_download = None
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
                parent.compile_calls.append((source, layout, flags, output))
                logging.warning("Job id: wsjob-compile-probe, workflow id: test")
                path = Path(output) / "artifact.bin"
                path.write_bytes(b"fake compiled artifact")
                return str(path)

        class Launcher:
            def __init__(self, artifact, **kwargs):
                parent.options.append(kwargs)
                parent.artifact = artifact

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def stage(self, filename):
                path = Path(filename)
                parent.staged[path.name] = path.read_bytes()

            def run(self, command):
                parent.run_calls.append(command)
                logging.warning("Job id: wsjob-runtime-probe, workflow id: test")
                if parent.launch_error:
                    raise RuntimeError("simulated launch failure")
                return "finished"

            def download_artifact(self, remote, requested):
                if remote == parent.missing_download:
                    raise RuntimeError("simulated missing download")
                path = Path(requested)
                if remote == parent.bad_download:
                    return str(path.parent / "request.json")
                if remote == "capture":
                    request_bytes = parent.staged["run-request.json"]
                    if parent.capture_wrong_request:
                        request_bytes += b" "
                    files = [("capture/capture.json", b"{}"), ("capture/raw.npz", b"raw fixture"),
                             ("capture/manifest.json", parent.staged["profile-build.json"]),
                             ("capture/request.json", request_bytes)]
                    files += [("capture/sources/" + name, parent.staged[name]) for name in app.HOSTS]
                    files += [("capture/sources/" + name, parent.staged[Path(name).name]) for name in app.SOURCES]
                    if parent.download_style != "none":
                        path = path.with_name("capture.tar.gz")
                    path.write_bytes(archive_bytes(files))
                elif remote == "host-exit.txt":
                    path.write_text(parent.worker_exit + "\n")
                else:
                    path.write_text("worker fixture\n")
                if parent.download_style == "none":
                    return None
                return str(path) if parent.download_style == "absolute" else path.name

        self.Compiler, self.Launcher = Compiler, Launcher
        self.audit = mock.patch.object(app.common, "audit_capture", side_effect=self.audit_capture).start()
        self.addCleanup(mock.patch.stopall)

    def audit_capture(self, folder):
        request = json.loads((folder / "request.json").read_bytes())
        manifest = json.loads((folder / "manifest.json").read_bytes())
        value = dict(status="pass", run_id=request["run_id"], flag=request["flag"],
                     backend=manifest["backend"], manifest_sha256=request["manifest_sha256"], invocations=[])
        value.update(self.audit_changes)
        return value

    def compile(self, backend="hardware"):
        args = app.parser().parse_args(["compile", "--backend", backend, "--out", str(self.base / "build"),
                                       "--namespace", "job-operator"])
        self.assertEqual(app.compile_program(args, self.root, self.Compiler), 0)
        return self.base / "build/profile-build.json"

    def args(self, output="result", backend="hardware", flag=1):
        return app.parser().parse_args(["run", "--backend", backend, "--out", str(self.base / output),
                                       "--artifact-record", str(self.base / "build/profile-build.json"),
                                       "--flag", str(flag), "--host-timeout", "300"])

    def test_compile_pins_sources_and_one_artifact_and_logs_job(self):
        build = self.compile()
        record = app.common.read_json(build)
        self.assertEqual(len(self.compile_calls), 1)
        self.assertEqual(record["params"], {"N": 4})
        self.assertIn("--fabric-dims=762,1172", record["compiler_args"])
        self.assertIn("--params=N:4", record["compiler_args"])
        self.assertEqual(set(record["host_sha256"]), set(app.HOSTS))
        self.assertEqual(set(record["source_sha256"]), set(app.SOURCES))
        self.assertFalse(self.options[0]["disable_version_check"])
        self.assertIn("Job id: wsjob-compile-probe", (build.parent / "client.log").read_text())
        self.assertEqual(app.common.read_json(build.parent / "compile-status.json")["status"], "complete")

    def test_compile_failure_retains_incomplete_status(self):
        class Failing(self.Compiler):
            def compile(self, *args):
                raise RuntimeError("compile failed")
        args = app.parser().parse_args(["compile", "--backend", "hardware", "--out", str(self.base / "failed")])
        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            app.compile_program(args, self.root, Failing)
        self.assertEqual(app.common.read_json(self.base / "failed/compile-status.json")["status"], "incomplete")
        self.assertFalse((self.base / "failed/profile-build.json").exists())

    def test_artifact_and_host_tampering_reject_before_launcher(self):
        build = self.compile()
        artifact = build.parent / "artifact.bin"
        original = artifact.read_bytes()
        artifact.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            app.run_program(self.args(), self.root, self.Launcher)
        artifact.write_bytes(original)
        (self.root / "profile_worker.py").write_text("changed worker")
        with self.assertRaisesRegex(ValueError, "source identity"):
            app.run_program(self.args(), self.root, self.Launcher)
        self.assertFalse(self.run_calls)
        self.assertFalse((self.base / "result").exists())

    def test_existing_output_not_reused(self):
        self.compile()
        output = self.base / "result"
        output.mkdir()
        (output / "marker").write_text("retained")
        with self.assertRaises(FileExistsError):
            app.run_program(self.args(), self.root, self.Launcher)
        self.assertEqual((output / "marker").read_text(), "retained")

    def test_run_stages_pinned_hosts_csl_and_records_and_honors_suffixes(self):
        self.compile()
        for index, style in enumerate(("relative", "absolute", "none")):
            with self.subTest(style=style):
                self.download_style = style
                args = self.args("result-" + str(index), flag=index % 2)
                self.assertEqual(app.run_program(args, self.root, self.Launcher), 0)
                self.assertEqual(set(self.staged), {*app.HOSTS, "layout.csl", "pe.csl",
                                                   "profile-build.json", "run-request.json"})
                summary = app.common.read_json(args.out / "collection.json")
                self.assertEqual(summary["status"], "complete")
                self.assertEqual(summary["audit"]["flag"], index % 2)
                self.assertIn("--cmaddr %CMADDR%", self.run_calls[-1])
                self.assertTrue((args.out / "capture-data/sources/source/pe.csl").is_file())
        self.assertEqual(len(self.run_calls), 3)
        self.assertEqual(len({app.common.read_json(self.base / f"result-{i}/run-request.json")["run_id"]
                              for i in range(3)}), 3)

    def test_simulator_uses_small_fabric_and_no_cmaddr(self):
        build = self.compile("simulator")
        self.assertIn("--fabric-dims=9,3", app.common.read_json(build)["compiler_args"])
        self.assertEqual(app.run_program(self.args(backend="simulator"), self.root, self.Launcher), 0)
        self.assertTrue(self.options[-1]["simulator"])
        self.assertNotIn("--cmaddr", self.run_calls[-1])

    def test_launch_failure_still_downloads_and_preserves_partial_state(self):
        self.compile()
        self.launch_error = True
        with self.assertRaisesRegex(RuntimeError, "partial downloads"):
            app.run_program(self.args(), self.root, self.Launcher)
        summary = app.common.read_json(self.base / "result/collection.json")
        self.assertEqual(set(summary["downloaded"]), {"host.log", "host-exit.txt", "capture"})
        self.assertEqual(summary["status"], "incomplete")
        self.audit.assert_not_called()

    def test_download_failure_or_collision_never_completes(self):
        self.compile()
        for field in ("missing_download", "bad_download"):
            with self.subTest(field=field):
                setattr(self, field, "host-exit.txt")
                args = self.args(field)
                with self.assertRaisesRegex(ValueError, "collection failed"):
                    app.run_program(args, self.root, self.Launcher)
                summary = app.common.read_json(args.out / "collection.json")
                self.assertIn("host-exit.txt", summary["errors"])
                self.assertEqual(summary["status"], "incomplete")
                setattr(self, field, None)

    def test_worker_timeout_and_wrong_request_do_not_pass(self):
        self.compile()
        self.worker_exit = "124"
        with self.assertRaisesRegex(ValueError, "worker failed or timed out"):
            app.run_program(self.args("timeout"), self.root, self.Launcher)
        self.worker_exit = "0"
        self.capture_wrong_request = True
        with self.assertRaisesRegex(ValueError, "captured request differs"):
            app.run_program(self.args("wrong-request"), self.root, self.Launcher)

    def test_audit_identity_and_numeric_status_bound_to_request(self):
        self.compile()
        for index, change in enumerate(({"run_id": "other"}, {"flag": 0}, {"backend": "simulator"},
                                        {"status": "numeric-fail"}, {"manifest_sha256": "other"})):
            self.audit_changes = change
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "capture admission differs"):
                app.run_program(self.args("bad-audit-" + str(index)), self.root, self.Launcher)

    def test_timeout_bounds_and_backend_command(self):
        for seconds in (0, -1, True, 1801):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                app.host_command("hardware", seconds)
        self.assertIn("timeout --signal=INT --kill-after=15s 300s", app.host_command("hardware", 300))


class ArchiveTests(unittest.TestCase):
    def test_traversal_duplicate_missing_and_symlink_members_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            examples = [archive_bytes([("capture/../escape", b"x")]),
                        archive_bytes([("capture/a", b"x"), ("capture/a", b"y")]),
                        archive_bytes([("capture/capture.json", b"{}")])]
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                member = tarfile.TarInfo("capture/link")
                member.type, member.linkname = tarfile.SYMTYPE, "/tmp/other"
                archive.addfile(member)
            examples.append(buffer.getvalue())
            for index, data in enumerate(examples):
                path = root / (str(index) + ".tar.gz")
                path.write_bytes(data)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    app.unpack_capture(path, root / ("unpacked-" + str(index)))
            self.assertFalse((root / "escape").exists())

    def test_download_escape_symlink_and_controller_collision_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            (root / "outside").write_text("outside")
            (output / "link").symlink_to(root / "outside")
            (output / "request.json").write_text("controller record")
            for returned in (str(root / "outside"), "link", "request.json"):
                with self.subTest(returned=returned), self.assertRaises(ValueError):
                    app.downloaded_path(returned, output / "capture", 1024)


if __name__ == "__main__":
    unittest.main()
