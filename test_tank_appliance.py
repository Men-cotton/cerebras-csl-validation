"""SDK client/controller mocks; these tests submit no cluster jobs."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import tank_appliance as appliance
import tank_campaign as campaign


class ApplianceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.driver = self.root / "driver"
        self.driver.mkdir()
        for name in appliance.SOURCES:
            (self.driver / name).write_text("# frozen test source: " + name + "\n")
        for owner in (appliance, campaign):
            patch = mock.patch.object(owner, "ROOT", self.driver)
            patch.start()
            self.addCleanup(patch.stop)
        self.package = self.root / "tank_cases"
        self.package.mkdir()
        files, cases = {}, {}
        for name in campaign.DEFAULT_CASES:
            folder = self.package / "cases" / name
            (folder / "source").mkdir(parents=True)
            params = dict(C=2, R=2, H=4, S=1, STEPS=2, NORM=0, L=4, E=8, W=394, MODE=2, T=4, K=6, Q=4)
            for relative, content in (("inputs.npz", b"mock inputs"), ("original.npz", b"mock graph"),
                                      ("mapping.npz", b"mock mapping"), ("sdk_runner.py", b"# original runner"),
                                      ("source/layout.csl", b"// layout"), ("source/pe.csl", b"// PE"),
                                      ("source/sparse_halo.csl", b"// halo"), ("source/no_halo.csl", b"// no halo")):
                (folder / relative).write_bytes(content)
            prepared = {str(path.relative_to(folder)): appliance.digest(path)
                        for path in folder.rglob("*") if path.is_file()}
            appliance.write_json(folder / "case.json", dict(params=params, grid=[2,2], files=prepared, sources={}, experiment="E192"))
            for path in folder.rglob("*"):
                if path.is_file():
                    files[str(path.relative_to(self.package))] = appliance.digest(path)
            cases[name] = dict(case_sha256=appliance.digest(folder / "case.json"), params=params)
        appliance.write_json(self.package / "manifest.json", dict(schema=1, files=files, cases=cases, source_commit="test-source"))

        def verify(package):
            manifest = appliance.read_json(Path(package) / "manifest.json")
            for name, value in manifest["files"].items():
                appliance.require(appliance.digest(Path(package) / name) == value, "mock package changed")
            return manifest

        patch = mock.patch.object(appliance.tank_audit, "verify_package", side_effect=verify)
        patch.start()
        self.addCleanup(patch.stop)

    def compile_args(self, backend="hardware"):
        return SimpleNamespace(package=self.package, case="topk-small", backend=backend,
                               out=self.root / ("build-" + backend), namespace="site-ns", fabric_dims=None, fabric_offsets="4,1")

    def compiler(self, seen, fail=False):
        class Compiler:
            def __init__(self, **options):
                seen["compile_options"] = options

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def compile(self, source, layout, flags, destination):
                seen.update(source=source, layout=layout, flags=flags)
                if fail:
                    raise RuntimeError("injected compile failure")
                path = Path(destination) / "artifact.tar"
                path.write_bytes(b"compiled test artifact")
                return str(path)
        return Compiler

    def build(self, backend="hardware"):
        seen = {}
        args = self.compile_args(backend)
        appliance.compile_program(args, compiler_type=self.compiler(seen))
        return args, seen

    def run_args(self, build_args):
        return SimpleNamespace(package=self.package, case=build_args.case, backend=build_args.backend,
                               artifact_record=build_args.out / "build.json", host_timeout=300, out=self.root / "run")

    def launcher(self, seen, fail=False, host_exit="0"):
        class Launcher:
            def __init__(self, artifact, **options):
                seen.update(artifact=artifact, launch_options=options, staged={}, downloads=[])

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def stage(self, path):
                seen["staged"][Path(path).name] = Path(path).read_bytes()

            def run(self, command):
                seen["command"] = command
                if fail:
                    raise RuntimeError("injected launcher failure")
                return "worker finished"

            def download_artifact(self, remote, destination):
                seen["downloads"].append(remote)
                path = Path(destination)
                if remote == "capture":
                    path = path.with_name(path.name + ".tar.gz")
                    payloads = {"raw.npz": b"mock raw", "execution.json": b'{"status":"COMPLETED"}',
                                "out.json": b"{}", "out_rpc.json": b"{}"}
                    with tarfile.open(path, "w:gz") as archive:
                        for name, value in payloads.items():
                            item = tarfile.TarInfo("capture/" + name)
                            item.size = len(value)
                            archive.addfile(item, io.BytesIO(value))
                else:
                    path.write_text(host_exit + "\n" if remote == "host-exit.txt" else "mock host log\n")
                return str(path)
        return Launcher

    def test_compile_preserves_params_and_hardware_identity(self):
        args, seen = self.build()
        manifest, artifact = appliance.load_build(args.out / "build.json", self.package, args.case, args.backend)
        self.assertEqual(manifest["params"]["Q"], 4)
        self.assertIn("--fabric-dims=762,1172", seen["flags"])
        self.assertIn("Q:4", seen["flags"])
        self.assertEqual(seen["layout"], "layout.csl")
        self.assertEqual(Path(seen["source"]), self.package / "cases/topk-small/source")
        self.assertEqual(seen["compile_options"]["mgmt_namespace"], "site-ns")
        self.assertEqual(manifest["artifact_sha256"], appliance.digest(artifact))
        with self.assertRaisesRegex(ValueError, "backend"):
            appliance.load_build(args.out / "build.json", self.package, args.case, "appliance-sim")
        artifact.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            appliance.load_build(args.out / "build.json", self.package, args.case, args.backend)

    def test_compile_failure_is_retained(self):
        args = self.compile_args()
        with self.assertRaisesRegex(RuntimeError, "compile failure"):
            appliance.compile_program(args, compiler_type=self.compiler({}, fail=True))
        record = appliance.read_json(args.out / "compile-status.json")
        self.assertEqual(record["status"], "incomplete")
        self.assertIn("error", record)
        self.assertFalse((args.out / "build.json").exists())

    def test_source_and_case_tampering_rejected(self):
        args, _ = self.build()
        (self.driver / "tank_worker.py").write_text("# modified worker")
        with self.assertRaisesRegex(ValueError, "identity"):
            appliance.load_build(args.out / "build.json", self.package, args.case, args.backend)
        (self.package / "cases/topk-small/inputs.npz").write_bytes(b"modified inputs")
        with self.assertRaisesRegex(ValueError, "package changed"):
            appliance.case_identity(self.package, args.case)

    def test_run_staging_suffix_capture_and_audit(self):
        build_args, _ = self.build()
        args, seen = self.run_args(build_args), {}
        with mock.patch.object(appliance.tank_audit, "audit_capture", return_value={"status": "PASS"}) as audit:
            record = appliance.run_program(args, launcher_type=self.launcher(seen))
        self.assertEqual(record["status"], "complete")
        self.assertEqual(set(seen["staged"]), {"tank_worker.py", "case.json", "inputs.npz", "request.json"})
        self.assertFalse(seen["launch_options"]["simulator"])
        self.assertIn("%CMADDR%", seen["command"])
        self.assertEqual(set(seen["downloads"]), {"host.log", "host-exit.txt", "capture"})
        self.assertEqual(record["downloaded"]["capture"]["path"], "capture-download.tar.gz")
        self.assertEqual((args.out / "capture/raw.npz").read_bytes(), b"mock raw")
        request = json.loads(seen["staged"]["request.json"])
        self.assertEqual(request, appliance.read_json(args.out / "request.json"))
        self.assertEqual(request["worker_sha256"], appliance.digest(self.driver / "tank_worker.py"))
        audit.assert_called_once_with(self.package, "topk-small", args.out / "capture", request)

    def test_launch_failure_still_downloads_and_keeps_capture(self):
        build_args, _ = self.build()
        args, seen = self.run_args(build_args), {}
        with mock.patch.object(appliance.tank_audit, "audit_capture") as audit:
            with self.assertRaisesRegex(ValueError, "launcher failed"):
                appliance.run_program(args, launcher_type=self.launcher(seen, fail=True))
        audit.assert_not_called()
        self.assertEqual(len(seen["downloads"]), 3)
        self.assertTrue((args.out / "capture/raw.npz").is_file())
        self.assertEqual(appliance.read_json(args.out / "collection.json")["status"], "incomplete")

    def test_simulator_command_and_timeout_failure(self):
        build_args, compile_seen = self.build("appliance-sim")
        self.assertIn("--fabric-dims=9,4", compile_seen["flags"])
        args, seen = self.run_args(build_args), {}
        with self.assertRaisesRegex(ValueError, "worker incomplete"):
            appliance.run_program(args, launcher_type=self.launcher(seen, host_exit="124"))
        self.assertTrue(seen["launch_options"]["simulator"])
        self.assertNotIn("--cmaddr", seen["command"])

    def test_capture_archive_paths_duplicates_links_and_missing_files(self):
        for index, kind in enumerate(("traversal", "duplicate", "link", "missing")):
            with self.subTest(kind=kind):
                path = self.root / (kind + ".tar.gz")
                with tarfile.open(path, "w:gz") as archive:
                    names = ["capture/raw.npz"] * (2 if kind == "duplicate" else 1)
                    if kind == "traversal":
                        names = ["capture/../escape"]
                    for name in names:
                        item = tarfile.TarInfo(name)
                        if kind == "link":
                            item.type, item.linkname = tarfile.SYMTYPE, "../../escape"
                            archive.addfile(item)
                        else:
                            item.size = 1
                            archive.addfile(item, io.BytesIO(b"x"))
                with self.assertRaises(ValueError):
                    appliance.unpack_capture(path, self.root / ("unpacked" + str(index)))
                self.assertFalse((self.root / ("unpacked" + str(index))).exists())

    def test_download_rejects_outside_record_collision_and_symlink(self):
        requested = self.root / "download"
        outside = self.root.parent / "outside"
        with self.assertRaises(ValueError):
            appliance.downloaded_path(str(outside), requested, 1024)
        (self.root / "collection.json").write_text("{}")
        with self.assertRaises(ValueError):
            appliance.downloaded_path("collection.json", requested, 1024)
        (self.root / "link").symlink_to(self.root / "collection.json")
        with self.assertRaises(ValueError):
            appliance.downloaded_path("link", requested, 1024)

    def prepared_campaign(self):
        args = campaign.parser().parse_args(["--package", str(self.package), "--out", str(self.root / "campaign"), "--plan-only"])
        return campaign.prepare(args)

    def test_campaign_plan_freezes_eight_builds_sixteen_runs_and_lock(self):
        root = self.prepared_campaign()
        plan, _ = campaign.load(root)
        self.assertEqual(len(plan["builds"]), 8)
        self.assertEqual(len(plan["runs"]), 16)
        self.assertEqual([item["case"] for item in plan["runs"][-3:]],
                         ["topk24", "halo-small", "topk-small"])
        with campaign.output_lock(root):
            with self.assertRaisesRegex(ValueError, "another controller"):
                with campaign.output_lock(root):
                    self.fail("second controller acquired the lock")
        (root / "driver/tank_worker.py").write_text("# modified frozen worker")
        with self.assertRaisesRegex(ValueError, "frozen controller"):
            campaign.load(root)

    def test_campaign_continues_failure_and_resume_does_not_retry(self):
        root = self.prepared_campaign()
        calls = []

        def runner(argv, log, timeout):
            calls.append(argv)
            log.write_text("mock client\n")
            if "run" in argv:
                folder=Path(argv[argv.index("--out")+1]); folder.mkdir()
                appliance.write_json(folder/"audit.json", dict(status="PASS"))
            return 1 if "compile" in argv and argv[argv.index("--case") + 1] == "topk-small" else 0

        with mock.patch.object(campaign, "validate_build", return_value={}), \
             mock.patch.object(campaign, "validate_run", return_value={"status": "PASS"}):
            self.assertEqual(campaign.run_campaign(root, runner), 2)
            count = len(calls)
            self.assertEqual(count, 22)
            self.assertEqual(campaign.run_campaign(root, runner), 2)
        self.assertEqual(len(calls), count)
        _, state = campaign.load(root)
        self.assertEqual(state["builds"]["topk-small"]["status"], "incomplete")
        self.assertEqual(state["runs"]["r001"]["status"], "skipped_dependency")
        self.assertEqual(sum(item["status"] == "pass" for item in state["runs"].values()), 14)

    def test_completed_campaign_keeps_cases_separate(self):
        root = self.prepared_campaign()
        def runner(argv, log, timeout):
            log.write_text('mock client\n')
            if 'run' in argv:
                folder=Path(argv[argv.index('--out')+1]); folder.mkdir()
                appliance.write_json(folder/'audit.json',dict(status='PASS'))
            return 0
        with mock.patch.object(campaign,'validate_build',return_value={}), \
             mock.patch.object(campaign,'validate_run',return_value={'status':'PASS'}):
            self.assertEqual(campaign.run_campaign(root,runner),0)
        report=appliance.read_json(root/'summary.json')['comparison']
        self.assertEqual(report['status'],'SEPARATE_CASES')
        self.assertEqual(len(report['reports']),16)

    def test_started_attempt_is_not_resubmitted(self):
        root = self.prepared_campaign()
        _, state = campaign.load(root)
        state["builds"]["topk-small"]["status"] = "started"
        appliance.write_json(root / "state.json", state)
        calls = []

        def validator(root, plan, entry):
            if entry["id"] == "topk-small":
                raise ValueError("no terminal build evidence")
            return {}

        def runner(argv, log, timeout):
            calls.append(argv)
            log.write_text("mock client\n")
            if "run" in argv:
                folder=Path(argv[argv.index("--out")+1]); folder.mkdir()
                appliance.write_json(folder/"audit.json", dict(status="PASS"))
            return 0

        with mock.patch.object(campaign, "validate_build", side_effect=validator), \
             mock.patch.object(campaign, "validate_run", return_value={"status": "PASS"}):
            campaign.run_campaign(root, runner)
        self.assertFalse(any(argv[argv.index("--case") + 1] == "topk-small" for argv in calls))
        self.assertEqual(campaign.load(root)[1]["builds"]["topk-small"]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
