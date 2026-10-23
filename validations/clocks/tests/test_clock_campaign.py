"""Controller failure, resume and packaging contracts; no SDK or tmux execution."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_campaign as campaign
import clock_common as common
import clock_report
import test_clock_report as report_fixtures


class FakeRunner:
    """Record durable admission state and return configured local client outcomes."""
    def __init__(self, root, outcomes=None):
        self.root = root
        self.outcomes = outcomes or {}
        self.calls = []

    def __call__(self, command, log_path, timeout):
        destination = Path(command[command.index("--out") + 1])
        role = "builds" if "compile" in command else "runs"
        state = common.read_json(self.root / "state.json")
        if state[role][destination.name]["status"] != "started":
            raise AssertionError("attempt was not durably marked started before submission")
        self.calls.append((role, destination.name, list(command), timeout))
        destination.mkdir(parents=True, exist_ok=False)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("offline fake client\n")
        outcome = self.outcomes.get(destination.name, 0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.serial = 0

    def prepare(self, profile="smoke", name=None):
        self.serial += 1
        root = self.base / (name or f"campaign-{self.serial}")
        arguments = campaign.parser().parse_args([
            "--out", str(root), "--profile", profile, "--plan-only",
            "--client-timeout", "17",
        ])
        self.assertEqual(campaign.prepare(arguments), root)
        return root

    def run_mocked(self, root, outcomes=None, invalid_analyses=(), invalid_builds=()):
        runner = FakeRunner(root, outcomes)
        records = dict(analyses=[], builds=[])

        def validate_build(output, plan, build):
            records["builds"].append(build["id"])
            if build["id"] in invalid_builds:
                raise ValueError("synthetic build integrity failure")
            return dict(params=build["params"])

        def analyze_run(directory, build, entry, plan):
            records["analyses"].append(entry["id"])
            if entry["id"] in invalid_analyses:
                raise ValueError("synthetic raw evidence missing or changed")
            status = "numeric_fail" if (outcomes or {}).get(entry["id"]) == 1 else "pass"
            return dict(status=status, run_id=str(uuid.uuid5(uuid.NAMESPACE_OID, entry["id"])))

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(campaign, "validate_build", side_effect=validate_build))
            stack.enter_context(mock.patch.object(campaign, "analyze_run", side_effect=analyze_run))
            reporting = stack.enter_context(mock.patch.object(campaign, "write_report", return_value={"errors": []}))
            bundling = stack.enter_context(mock.patch.object(campaign, "bundle", return_value=root.with_suffix(".zip")))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            code = campaign.run_campaign(root, command_runner=runner)
        self.assertEqual(reporting.call_count, 1)
        self.assertEqual(bundling.call_count, 1)
        return code, common.read_json(root / "state.json"), runner, records

    def test_failed_run_continues_without_retry(self):
        root = self.prepare()
        code, state, runner, _ = self.run_mocked(root, {"r001": RuntimeError("client failed")})
        self.assertEqual(code, 2)
        self.assertEqual(state["status"], "completed_with_failures")
        self.assertEqual(state["runs"]["r001"]["status"], "incomplete")
        self.assertTrue(state["runs"]["r001"]["remote_stop_unconfirmed"])
        self.assertEqual([state["runs"][f"r{i:03d}"]["status"] for i in (2, 3, 4)], ["pass"] * 3)
        self.assertEqual([name for role, name, _, _ in runner.calls if role == "runs"],
                         ["r001", "r002", "r003", "r004"])
        self.assertTrue(all(timeout == 17 for _, _, _, timeout in runner.calls))

    def test_compile_failure_skips_only_dependent_runs(self):
        root = self.prepare(profile="standard")
        plan = common.read_json(root / "plan.json")
        failed = plan["builds"][0]["id"]
        code, state, runner, _ = self.run_mocked(root, {failed: 2})
        self.assertEqual(code, 2)
        self.assertEqual(state["builds"][failed]["status"], "incomplete")
        self.assertTrue(all(s["status"] == "pass" for name, s in state["builds"].items() if name != failed))
        submitted = {name for role, name, _, _ in runner.calls if role == "runs"}
        independent = {r["id"] for r in plan["runs"] if r["build"] != failed}
        self.assertEqual(submitted, independent)
        self.assertTrue(independent)
        for entry in plan["runs"]:
            expected = "skipped_dependency" if entry["build"] == failed else "pass"
            self.assertEqual(state["runs"][entry["id"]]["status"], expected)

    def test_numeric_failure_remains_distinct_and_later_runs_execute(self):
        root = self.prepare()
        code, state, runner, _ = self.run_mocked(root, {"r001": 1})
        self.assertEqual(code, 2)
        self.assertEqual(state["runs"]["r001"]["status"], "numeric_fail")
        self.assertEqual(state["runs"]["r001"]["client_exit"], 1)
        self.assertEqual(state["runs"]["r004"]["status"], "pass")
        self.assertEqual(sum(role == "runs" for role, _, _, _ in runner.calls), 4)

    def test_started_unknown_is_not_resubmitted_and_pending_runs_continue(self):
        root = self.prepare()
        state = common.read_json(root / "state.json")
        for slot in state["builds"].values():
            slot["status"] = "pass"
        state["runs"]["r001"]["status"] = "started"
        campaign.atomic_json(root / "state.json", state)
        code, state, runner, records = self.run_mocked(root, invalid_analyses={"r001"})
        self.assertEqual(code, 2)
        self.assertEqual(state["runs"]["r001"]["status"], "unknown")
        self.assertTrue(state["runs"]["r001"]["remote_stop_unconfirmed"])
        self.assertEqual([name for _, name, _, _ in runner.calls], ["r002", "r003", "r004"])
        self.assertIn("r001", records["analyses"])

    def test_interrupted_build_is_not_resubmitted(self):
        root = self.prepare(profile="standard")
        state = common.read_json(root / "state.json")
        failed = next(iter(state["builds"]))
        state["builds"][failed]["status"] = "started"
        campaign.atomic_json(root / "state.json", state)
        code, state, runner, _ = self.run_mocked(root, invalid_builds={failed})
        self.assertEqual(code, 2)
        self.assertEqual(state["builds"][failed]["status"], "unknown")
        self.assertNotIn(failed, [name for _, name, _, _ in runner.calls])
        self.assertTrue(any(role == "runs" for role, _, _, _ in runner.calls))

    def test_completed_resume_revalidates_without_resubmission(self):
        root = self.prepare()
        self.assertEqual(self.run_mocked(root)[0], 0)
        code, state, runner, records = self.run_mocked(root)
        self.assertEqual(code, 0)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(runner.calls, [])
        self.assertEqual(records["analyses"], ["r001", "r002", "r003", "r004"])
        self.assertEqual(len(records["builds"]), 1)
        self.assertTrue(all("revalidated_at" in s for s in state["runs"].values()))

    def test_changed_completed_evidence_is_retained_as_incomplete(self):
        root = self.prepare()
        self.assertEqual(self.run_mocked(root)[0], 0)
        code, state, runner, _ = self.run_mocked(root, invalid_analyses={"r001"})
        self.assertEqual(code, 2)
        self.assertEqual(runner.calls, [])
        self.assertEqual(state["runs"]["r001"]["status"], "incomplete")
        self.assertEqual(state["runs"]["r004"]["status"], "pass")

    def test_keyboard_interrupt_preserves_attempt_and_finalizes_partial_results(self):
        root = self.prepare()
        code, state, runner, _ = self.run_mocked(root, {"r001": KeyboardInterrupt()})
        self.assertEqual(code, 130)
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(state["runs"]["r001"]["status"], "unknown")
        self.assertEqual(state["runs"]["r002"]["status"], "pending")
        self.assertEqual([name for role, name, _, _ in runner.calls if role == "runs"], ["r001"])

    def test_output_lock_rejects_a_second_controller(self):
        root = self.prepare()
        with campaign.output_lock(root):
            with self.assertRaisesRegex(ValueError, "another controller"):
                with campaign.output_lock(root):
                    self.fail("second output lock was admitted")

    def test_duplicate_foreground_preserves_launcher_and_state(self):
        root = self.prepare()
        campaign.atomic_json(root / "launcher.json", {"status": "running", "pid": 123})
        launcher_before = (root / "launcher.json").read_bytes()
        state_before = (root / "state.json").read_bytes()
        with campaign.output_lock(root), mock.patch.object(campaign, "run_campaign") as run:
            with self.assertRaisesRegex(ValueError, "another controller"):
                campaign.foreground(root)
        run.assert_not_called()
        self.assertEqual((root / "launcher.json").read_bytes(), launcher_before)
        self.assertEqual((root / "state.json").read_bytes(), state_before)
        self.assertFalse((root / "driver.log").exists())

    def test_foreground_final_bundle_failure_changes_exit_and_durable_status(self):
        root = self.prepare()
        with mock.patch.object(campaign, "run_campaign", return_value=0) as run:
            with mock.patch.object(campaign, "bundle", side_effect=OSError("synthetic ZIP write failure")):
                with redirect_stderr(io.StringIO()):
                    self.assertEqual(campaign.foreground(root), 2)
        run.assert_called_once_with(root, locked=True)
        for name in ("state.json", "launcher.json"):
            record = common.read_json(root / name)
            self.assertEqual(record["status"], "completed_with_failures")
            self.assertEqual(record["exit_code"], 2)
            self.assertIn("synthetic ZIP write failure", record["bundle_error"])

    def test_local_sim_parser_plan_and_driver_selection(self):
        root = self.base / "local-sim"
        arguments = campaign.parser().parse_args([
            "--out", str(root), "--profile", "smoke", "--backend", "local-sim", "--plan-only",
            "--compiler", "/mock/compiler", "--worker-python", "/mock/worker-python",
        ])
        self.assertEqual(campaign.prepare(arguments), root)
        plan = common.read_json(root / "plan.json")
        self.assertEqual(plan["backend"], "local-sim")
        self.assertEqual(len(plan["runs"]), 4)
        compile_argv = campaign.compile_command(root, plan, plan["builds"][0])
        run_argv = campaign.run_command_argv(root, plan, plan["runs"][0])
        self.assertEqual(compile_argv[1], str(root / "driver/clock_local.py"))
        self.assertEqual(compile_argv[compile_argv.index("--compiler") + 1], "/mock/compiler")
        self.assertEqual(run_argv[run_argv.index("--python") + 1], "/mock/worker-python")

    def test_enter_frozen_reexecutes_checked_driver_with_same_interpreter(self):
        root = self.prepare()
        newer = self.base / "updated-repository"
        newer.mkdir()
        (newer / "clock_campaign.py").write_text("different current repository source\n")
        interpreter = "/tmp/python with spaces"
        for mode in ("--resume", "--summarize"):
            with self.subTest(mode=mode), mock.patch.object(campaign, "ROOT", newer):
                with mock.patch.object(campaign.sys, "executable", interpreter):
                    with mock.patch.object(campaign.os, "execv") as execute:
                        campaign.enter_frozen(root, mode)
            expected = [interpreter, str(root / "driver/clock_campaign.py"), mode, str(root)]
            if mode == "--resume":
                expected.append("--foreground")
            execute.assert_called_once_with(interpreter, expected)

    def test_enter_frozen_rejects_changed_pins_before_exec(self):
        for kind in ("plan", "source"):
            root = self.prepare()
            path = root / "plan.json" if kind == "plan" else root / "driver/clock_probe.py"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.subTest(kind=kind), mock.patch.object(campaign.os, "execv") as execute:
                with self.assertRaisesRegex(ValueError, "frozen (plan|source) changed"):
                    campaign.enter_frozen(root, "--resume")
            execute.assert_not_called()

    def test_enter_frozen_already_in_frozen_driver_does_not_exec(self):
        root = self.prepare()
        with mock.patch.object(campaign, "ROOT", root / "driver"):
            with mock.patch.object(campaign.os, "execv") as execute:
                campaign.enter_frozen(root, "--resume")
                campaign.enter_frozen(root, "--summarize")
        execute.assert_not_called()

    def test_resume_rejects_source_and_plan_tampering(self):
        for kind in ("plan", "source"):
            with self.subTest(kind=kind):
                root = self.prepare()
                path = root / "plan.json" if kind == "plan" else root / "driver/clock_probe.py"
                with path.open("a") as stream:
                    stream.write("\n")
                arguments = campaign.parser().parse_args(["--resume", str(root), "--plan-only"])
                with self.assertRaisesRegex(ValueError, "frozen (plan|source) changed"):
                    campaign.prepare(arguments)

    def test_client_timeout_keeps_remote_termination_unknown_and_continues(self):
        root = self.prepare()
        code, state, _, _ = self.run_mocked(root, {
            "r001": TimeoutError("client wall timeout; remote job termination unconfirmed")})
        self.assertEqual(code, 2)
        self.assertTrue(state["runs"]["r001"]["remote_stop_unconfirmed"])
        self.assertIn("remote job termination unconfirmed", state["runs"]["r001"]["error"])
        self.assertEqual(state["runs"]["r004"]["status"], "pass")

    def test_execute_timeout_only_terminates_the_local_client(self):
        child = mock.Mock()
        child.poll.return_value = None
        child.wait.return_value = 0
        with mock.patch.object(campaign.subprocess, "Popen", return_value=child) as start:
            with mock.patch.object(campaign.time, "monotonic", side_effect=[0, 2]):
                with self.assertRaisesRegex(TimeoutError, "remote job termination unconfirmed"):
                    campaign.execute(["synthetic-client"], self.base / "client.log", 1)
        self.assertEqual(start.call_args.args, (["synthetic-client"],))
        self.assertNotIn("shell", start.call_args.kwargs)
        child.terminate.assert_called_once_with()
        child.wait.assert_called_once_with(timeout=15)
        child.kill.assert_not_called()

    def test_tmux_uses_private_socket_current_interpreter_and_quoted_command(self):
        root = self.prepare(name="campaign 'quote'; $(no-command)")
        interpreter = "/tmp/python 'quoted'; $(no-command)"
        response = SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(campaign.shutil, "which", return_value="/usr/bin/tmux"):
            with mock.patch.object(campaign.sys, "executable", interpreter):
                with mock.patch.object(campaign.subprocess, "run", return_value=response) as launch:
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(campaign.launch_tmux(root), 0)
        launch.assert_called_once()
        argv = launch.call_args.args[0]
        self.assertEqual(argv[:2], ["tmux", "-L"])
        self.assertRegex(argv[2], r"^clock-[0-9a-f]{16}$")
        self.assertEqual(argv[3:5], ["-f", "/dev/null"])
        self.assertEqual(argv[5:10], ["new-session", "-d", "-s", "clock", "-c"])
        self.assertEqual(argv[10], str(root / "driver"))
        self.assertEqual(argv[11:13], ["/bin/sh", "-c"])
        self.assertEqual(shlex.split(argv[13]), ["exec", interpreter,
            str(root / "driver/clock_campaign.py"), "--foreground", "--resume", str(root)])
        self.assertNotIn("env", launch.call_args.kwargs)
        request = common.read_json(root / "launcher.json")
        self.assertEqual(request["command"][0], interpreter)
        self.assertEqual(request["cwd"], str(root / "driver"))

    def test_existing_live_tmux_session_is_not_relaunched(self):
        root = self.prepare()
        campaign.atomic_json(root / "launcher.json", {"socket": "clock-existing"})
        response = SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(campaign.shutil, "which", return_value="/usr/bin/tmux"):
            with mock.patch.object(campaign.subprocess, "run", return_value=response) as launch:
                with self.assertRaisesRegex(ValueError, "existing campaign tmux session"):
                    campaign.launch_tmux(root)
        self.assertEqual(launch.call_args.args[0],
                         ["tmux", "-L", "clock-existing", "-f", "/dev/null", "has-session", "-t", "=clock"])


class BundleTests(unittest.TestCase):
    def fixture(self):
        fixture = report_fixtures.ReportTests(methodName="test_hardware_capture_admission")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.use_hardware()
        entry = fixture.entry("r001")
        directory = fixture.make_capture(entry)
        campaign.atomic_json(fixture.root / "plan.json", fixture.plan)
        campaign.atomic_json(fixture.root / "state.json", {"runs": {"r001": {"status": "pass"}}})
        self.addCleanup(lambda: fixture.root.with_suffix(".zip").unlink(missing_ok=True))
        return fixture, entry, directory

    def archive_names(self, archive, root):
        with zipfile.ZipFile(archive) as zipped:
            return {name.removeprefix(root.name + "/") for name in zipped.namelist()}

    def extracted_compact_fixture(self):
        fixture, entry, directory = self.fixture()
        for name in campaign.SOURCE_FILES:
            target = fixture.root / "driver" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((campaign.ROOT / name).read_bytes())
        analysis = clock_report.analyze_run(directory, fixture.build, entry, fixture.plan)
        # Actual appliance results retain the downloaded tarball, not this helper's staging directory.
        shutil.rmtree(directory / "capture")
        archive = campaign.bundle(fixture.root, fixture.plan)
        extracted = tempfile.TemporaryDirectory()
        self.addCleanup(extracted.cleanup)
        base = Path(extracted.name)
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                path = base / member.filename
                self.assertFalse(Path(member.filename).is_absolute())
                self.assertNotIn("..", Path(member.filename).parts)
                self.assertFalse(member.is_dir())
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(zipped.read(member))
        replay = base / fixture.root.name
        return fixture, entry, replay, analysis

    def test_compact_extraction_replay_and_rebundle_preserve_declared_omissions(self):
        fixture, entry, replay, expected = self.extracted_compact_fixture()
        inventory = campaign.verify_bundle(replay)
        omissions = inventory["omitted_originals"]
        self.assertEqual(omissions, ["runs/r001/capture.tar.gz"])
        run = replay / "runs/r001"
        build = replay / "builds/fixture"
        self.assertFalse((run / "capture.tar.gz").exists())
        self.assertEqual(clock_report.read_compact_run(run, build, entry, fixture.plan,
                                                      omitted_paths=omissions), expected)
        rebuilt = campaign.bundle(replay, fixture.plan, compact_omissions=omissions)
        second_inventory = campaign.verify_bundle(replay)
        self.assertEqual(second_inventory["omitted_originals"], omissions)
        self.assertNotIn("runs/r001/capture.tar.gz", self.archive_names(rebuilt, replay))
        self.assertIn("runs/r001/observed/raw.npz", self.archive_names(rebuilt, replay))
        self.assertEqual(clock_report.read_compact_run(run, build, entry, fixture.plan,
            omitted_paths=second_inventory["omitted_originals"]), expected)

    def test_compact_changed_raw_fails_inventory_and_independent_raw_binding(self):
        fixture, entry, replay, _ = self.extracted_compact_fixture()
        raw = replay / "runs/r001/observed/raw.npz"
        data = bytearray(raw.read_bytes())
        data[-1] ^= 1
        raw.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "bundle retained bytes differ"):
            campaign.verify_bundle(replay)
        # Even a rewritten outer inventory cannot substitute for raw-to-analysis validation.
        inventory = common.read_json(replay / "bundle.json")
        row = next(item for item in inventory["files"] if item["path"] == "runs/r001/observed/raw.npz")
        row["sha256"] = common.digest(raw)
        campaign.atomic_json(replay / "bundle.json", inventory)
        inventory = campaign.verify_bundle(replay)
        with self.assertRaisesRegex(ValueError, "retained capture hash"):
            clock_report.read_compact_run(replay / "runs/r001", replay / "builds/fixture",
                                         entry, fixture.plan, omitted_paths=inventory["omitted_originals"])

    def test_validated_capture_compacts_archive_but_preserves_raw_evidence(self):
        fixture, entry, directory = self.fixture()
        analysis = clock_report.analyze_run(directory, fixture.build, entry, fixture.plan)
        # The same completed record must support a later offline admission.
        self.assertEqual(clock_report.analyze_run(directory, fixture.build, entry, fixture.plan), analysis)
        archive = campaign.bundle(fixture.root, fixture.plan)
        names = self.archive_names(archive, fixture.root)
        self.assertNotIn("runs/r001/capture.tar.gz", names)
        for name in ("raw.npz", "capture.json", "calls.json"):
            self.assertIn("runs/r001/observed/" + name, names)
        self.assertIn("runs/r001/analysis.json", names)
        self.assertIn("builds/fixture/artifact.tar.gz", names)
        self.assertIn("plan.json", names)
        self.assertIn("state.json", names)
        self.assertIn("bundle.json", names)
        inventory = common.read_json(fixture.root / "bundle.json")
        self.assertIn("runs/r001/capture.tar.gz", inventory["omitted_originals"])
        self.assertTrue((directory / "capture.tar.gz").is_file())
        pins = {item["path"]: item for item in inventory["files"]}
        self.assertEqual(pins["runs/r001/observed/raw.npz"]["sha256"],
                         common.digest(directory / "observed/raw.npz"))

    def test_unvalidated_observed_files_do_not_justify_archive_omission(self):
        fixture, _, directory = self.fixture()
        observed = directory / "observed"
        observed.mkdir()
        for name in ("raw.npz", "capture.json", "calls.json"):
            (observed / name).write_bytes(b"not validated")
        archive = campaign.bundle(fixture.root, fixture.plan)
        names = self.archive_names(archive, fixture.root)
        self.assertIn("runs/r001/capture.tar.gz", names)
        self.assertNotIn("runs/r001/capture.tar.gz",
                         common.read_json(fixture.root / "bundle.json")["omitted_originals"])

    def test_bundle_rejects_symlink_directory_at_traversal_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            outside = Path(temporary) / "outside"
            root.mkdir()
            outside.mkdir()
            (outside / "not-campaign-data").write_bytes(b"synthetic outside data")
            (root / "runs").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                campaign.bundle(root, {"runs": []})
            self.assertFalse(root.with_suffix(".zip").exists())


if __name__ == "__main__":
    unittest.main()
