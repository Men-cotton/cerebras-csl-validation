"""Live-directory retry contracts with synthetic artifacts; no SDK or tmux."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_retry as retry
import clock_campaign as campaign
import clock_common as common
import clock_appliance as appliance
import test_clock_common as fixtures
import test_clock_report as report_fixtures
from test_clock_campaign import FakeRunner


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.old = self.base / "original"
        campaign.prepare(campaign.parser().parse_args(["--out", str(self.old), "--profile", "smoke", "--plan-only"]))
        self.plan = common.read_json(self.old / "plan.json")
        self.state = common.read_json(self.old / "state.json")
        self.build_id = self.plan["builds"][0]["id"]
        self.plan["builds"][0]["params"] = dict(fixtures.PARAMS)
        for entry in self.plan["runs"]:
            entry["config"] = common.validate_config({})
            campaign.atomic_json(self.old / "conditions" / (entry["id"]+".json"), dict(config=entry["config"]))
        self.build = self.old / "builds" / self.build_id
        self.build.mkdir()
        self.manifest = dict(schema=1, action="compile", compile_id=str(uuid.uuid4()), backend="hardware",
                             params=dict(fixtures.PARAMS), **common.source_identity(retry.ROOT),
                             launcher_sha256=common.digest(retry.ROOT / "clock_appliance.py"))
        campaign.atomic_json(self.build / "request.json", self.manifest)
        (self.build / "artifact.tar.gz").write_bytes(b"synthetic build artifact; never execute")
        self.manifest.update(artifact_path="artifact.tar.gz", artifact_sha256=common.digest(self.build / "artifact.tar.gz"))
        campaign.atomic_json(self.build / "clock-build.json", self.manifest)
        campaign.atomic_json(self.build / "compile-status.json", dict(status="complete", compile_id=self.manifest["compile_id"],
                             backend="hardware", manifest_sha256=common.digest(self.build / "clock-build.json"),
                             accounting=dict(status="complete")))
        self.state["builds"][self.build_id]["status"] = "pass"
        self.persist()

    def persist(self):
        campaign.atomic_json(self.old / "plan.json", self.plan)
        self.state["plan_sha256"] = common.digest(self.old / "plan.json")
        campaign.atomic_json(self.old / "state.json", self.state)

    def capture(self, index):
        harness = report_fixtures.ReportTests(methodName="runTest")
        harness.root, harness.build, harness.plan, harness.manifest = self.old, self.build, self.plan, self.manifest
        return harness.make_capture(self.plan["runs"][index])

    def prepare(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return retry.prepare_retry(self.old, **kwargs)

    def test_subset_uses_live_state_reuses_build_and_preserves_original(self):
        for name, status in zip(("r001", "r002", "r003", "r004"), ("pass", "numeric_fail", "unknown", "pending")):
            self.state["runs"][name]["status"] = status
        self.state["runs"]["r003"].update(remote_stop_unconfirmed=True, jobs=[dict(job_id="wsjob-old")])
        self.persist()
        before = {name: (self.old / name).read_bytes() for name in ("plan.json", "state.json")}
        new = self.prepare()
        self.assertEqual(new, self.base / "original-retry")
        plan, state = common.read_json(new / "plan.json"), common.read_json(new / "state.json")
        self.assertEqual([entry["id"] for entry in plan["runs"]], ["r003", "r004"])
        self.assertEqual(plan["runs"], self.plan["runs"][2:])
        self.assertNotEqual(plan["campaign_id"], self.plan["campaign_id"])
        self.assertEqual(str(uuid.UUID(plan["campaign_id"])), plan["campaign_id"])
        self.assertEqual(plan["retry_of"]["entries"]["r003"]["jobs"], [dict(job_id="wsjob-old")])
        self.assertTrue(plan["retry_of"]["entries"]["r003"]["remote_liveness_unknown"])
        self.assertEqual(state["runs"], {name: dict(status="pending") for name in ("r003", "r004")})
        self.assertEqual(list((new / "runs").iterdir()), [])
        self.assertEqual(state["builds"][self.build_id]["status"], "pass")
        self.assertEqual(plan["retry_of"]["builds"][self.build_id]["status"], "pass")
        self.assertEqual(plan["retry_of"]["builds"][self.build_id]["decision"], "reuse_validated_build")
        self.assertEqual((new / "builds" / self.build_id / "clock-build.json").read_bytes(), (self.build / "clock-build.json").read_bytes())
        self.assertEqual(plan["estimates"]["builds_to_compile"], 0)
        self.assertEqual(plan["estimates"]["selected_fresh_runtimes"], 2)
        self.assertEqual(plan["estimates"]["hardware_wall_seconds_low"], round(2*70.387, 3))
        for name, content in before.items():
            self.assertEqual((self.old / name).read_bytes(), content)

    def test_started_complete_capture_is_recovered_in_copy_only(self):
        directory = self.capture(0)
        self.state["runs"]["r001"]["status"] = "started"
        self.persist()
        new = self.prepare()
        decision = common.read_json(new / "plan.json")["retry_of"]["entries"]["r001"]
        self.assertEqual(decision["recovered"]["recovery"], "complete_original")
        self.assertEqual(decision["recovered"]["status"], "pass")
        self.assertNotIn("r001", common.read_json(new / "state.json")["runs"])
        self.assertFalse((directory / "observed").exists())
        self.assertFalse((directory / "analysis.json").exists())
        self.assertEqual(common.read_json(self.old / "state.json")["runs"]["r001"]["status"], "started")

    def test_terminal_raw_capture_without_final_collection_is_not_retried_or_promoted(self):
        directory = self.capture(1)
        collection = appliance.read_json(directory / "collection.json")
        collection["status"] = "incomplete"
        campaign.atomic_json(directory / "collection.json", collection)
        self.state["runs"]["r002"]["status"] = "incomplete"
        self.persist()
        new = self.prepare()
        outcome = common.read_json(new / "plan.json")["retry_of"]["entries"]["r002"]["recovered"]
        self.assertEqual(outcome["recovery"], "terminal_raw_capture")
        self.assertEqual(outcome["collection_state"], "unfinalized")
        self.assertFalse(outcome["collection_admitted"])
        self.assertNotIn("r002", common.read_json(new / "state.json")["runs"])
        self.assertEqual(appliance.read_json(directory / "collection.json"), collection)
        self.assertFalse((directory / "analysis.json").exists())

    def test_observed_numeric_failure_raw_is_also_excluded(self):
        directory = self.capture(1)
        collection = appliance.read_json(directory / "collection.json")
        collection["status"] = "incomplete"
        campaign.atomic_json(directory / "collection.json", collection)
        meta = common.read_json(directory / "capture/capture.json")
        meta["status"] = "observed_numeric_fail"
        with mock.patch.object(appliance, "validate_capture", return_value=(meta, dict(status="observed_numeric_fail"))):
            new = self.prepare()
        decision = common.read_json(new / "plan.json")["retry_of"]["entries"]["r002"]
        self.assertEqual(decision["recovered"]["status"], "observed_numeric_fail")
        self.assertNotIn("r002", common.read_json(new / "state.json")["runs"])

    def test_truncated_capture_without_download_inventory_remains_retry_eligible(self):
        directory = self.capture(1)
        collection = appliance.read_json(directory / "collection.json")
        collection.update(status="incomplete", downloaded={})
        campaign.atomic_json(directory / "collection.json", collection)
        shutil.rmtree(directory / "capture")
        archive = directory / "capture.tar.gz"
        archive.write_bytes(archive.read_bytes()[:10])
        self.state["runs"]["r002"]["status"] = "incomplete"
        self.persist()
        new = self.prepare()
        decision = common.read_json(new / "plan.json")["retry_of"]["entries"]["r002"]
        self.assertEqual(decision["decision"], "retry")
        self.assertIn("collection incomplete", decision["recovery_error"])
        self.assertRegex(decision["recovery_error"], "ReadError|EOFError|TarError")
        self.assertIn("r002", common.read_json(new / "state.json")["runs"])
        self.assertEqual(archive.stat().st_size, 10)

    def test_raw_recovery_refuses_wrong_original_request_binding(self):
        directory = self.capture(1)
        request = appliance.read_json(directory / "request.json")
        request["run_id"] = str(uuid.uuid4())
        campaign.atomic_json(directory / "request.json", request)
        with mock.patch.object(appliance, "validate_capture", side_effect=AssertionError("must reject request first")) as validate:
            new = self.prepare()
        validate.assert_not_called()
        self.assertIn("r002", common.read_json(new / "state.json")["runs"])

    def test_fresh_runtime_request_gets_new_uuid_before_any_sdk_submission(self):
        directory = self.old / "runs/r001"
        directory.mkdir()
        old_uuid = str(uuid.uuid4())
        campaign.atomic_json(directory / "clock-request.json", dict(run_id=old_uuid))
        self.state["runs"]["r001"]["status"] = "unknown"
        self.persist()
        new = self.prepare()
        class StopBeforeSubmission:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("synthetic stop before SDK submission")
        args = SimpleNamespace(artifact_record=new / "builds" / self.build_id / "clock-build.json",
                               backend="hardware", host_timeout=300, config_file=new / "conditions/r001.json",
                               out=new / "runs/r001")
        with mock.patch.object(appliance, "sdk_classes", side_effect=AssertionError("no SDK")):
            with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
                appliance.run_program(args, new / "driver", launcher_type=StopBeforeSubmission)
        request = appliance.read_json(new / "runs/r001/clock-request.json")
        self.assertNotEqual(request["run_id"], old_uuid)
        self.assertEqual(str(uuid.UUID(request["run_id"])), request["run_id"])
        self.assertEqual(appliance.read_json(directory / "clock-request.json"), dict(run_id=old_uuid))

    def test_invalid_build_stays_pending_in_fresh_directory(self):
        (self.build / "artifact.tar.gz").write_bytes(b"changed artifact")
        new = self.prepare()
        state = common.read_json(new / "state.json")
        self.assertEqual(state["builds"][self.build_id]["status"], "pending")
        self.assertFalse((new / "builds" / self.build_id).exists())
        self.assertEqual(common.read_json(new / "plan.json")["estimates"]["builds_to_compile"], 1)

    def test_skipped_dependency_retains_uncertain_compile_provenance(self):
        compile_status = appliance.read_json(self.build / "compile-status.json")
        compile_status["status"] = "incomplete"
        campaign.atomic_json(self.build / "compile-status.json", compile_status)
        for slot in self.state["runs"].values():
            slot["status"] = "pass"
        self.state["runs"]["r001"]["status"] = "skipped_dependency"
        for status in ("started", "unknown"):
            with self.subTest(status=status):
                original_jobs = [dict(role="compile", job_id="wsjob-state-"+status)]
                self.state["builds"][self.build_id].update(status=status, jobs=original_jobs,
                                                          remote_stop_unconfirmed=True)
                log = self.build / "client.log"
                log.write_text("Job id: wsjob-live-"+status+", workflow id: flow-old, remote log path: logs/compile\n")
                self.persist()
                old_state = (self.old / "state.json").read_bytes()
                output = io.StringIO()
                with redirect_stdout(output):
                    new = retry.prepare_retry(self.old, self.base / ("retry-"+status))
                plan = common.read_json(new / "plan.json")
                provenance = plan["retry_of"]["builds"][self.build_id]
                self.assertEqual(provenance["status"], status)
                self.assertEqual(provenance["jobs"], original_jobs)
                self.assertTrue(provenance["remote_stop_unconfirmed"])
                self.assertTrue(provenance["remote_liveness_unknown"])
                self.assertEqual(provenance["decision"], "recompile")
                self.assertEqual(provenance["locally_recorded_jobs"][0]["role"], "compile")
                self.assertEqual(provenance["locally_recorded_jobs"][0]["job_id"], "wsjob-live-"+status)
                self.assertIn(self.build_id, plan["retry_of"]["build_validation_errors"])
                self.assertNotIn("error", provenance)
                self.assertEqual(plan["retry_of"]["entries"]["r001"]["status"], "skipped_dependency")
                self.assertEqual(common.read_json(new / "state.json")["builds"][self.build_id], dict(status="pending"))
                self.assertFalse((new / "builds" / self.build_id).exists())
                self.assertIn("build:"+self.build_id, output.getvalue())
                self.assertEqual((self.old / "state.json").read_bytes(), old_state)
                self.assertEqual(appliance.read_json(self.build / "compile-status.json"), compile_status)

    def test_default_output_rejects_second_invocation_and_original_lock_is_respected(self):
        with campaign.output_lock(self.old):
            with self.assertRaisesRegex(ValueError, "another controller"):
                self.prepare()
        self.assertFalse((self.base / "original-retry").exists())
        new = self.prepare()
        before = (new / "plan.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare()
        self.assertEqual((new / "plan.json").read_bytes(), before)

    def test_source_change_is_rejected_even_when_old_plan_hashes_are_updated(self):
        path = self.old / "driver/clock_probe.py"
        path.write_bytes(path.read_bytes()+b"\n# changed\n")
        self.plan["source_sha256"]["clock_probe.py"] = common.digest(path)
        self.persist()
        with self.assertRaisesRegex(ValueError, "trusted repository sources differ"):
            self.prepare()
        self.assertFalse((self.base / "original-retry").exists())

    def test_copy_rejects_symlinks_and_byte_overflow(self):
        source = self.base / "copy-source"
        source.mkdir()
        (source / "link").symlink_to(self.build / "artifact.tar.gz")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            retry.copy_tree(source, self.base / "copy-link")
        (source / "link").unlink()
        (source / "large").write_bytes(b"0123456789")
        with self.assertRaisesRegex(ValueError, "byte bound"):
            retry.copy_tree(source, self.base / "copy-large", limit=5)

    def test_plan_only_never_launches_and_nothing_remaining_creates_no_campaign(self):
        with mock.patch.object(campaign, "launch_tmux", side_effect=AssertionError("must not launch")):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(retry.main([str(self.old), "--plan-only"]), 0)
        for slot in self.state["runs"].values():
            slot["status"] = "pass"
        self.persist()
        with mock.patch.object(campaign, "launch_tmux", side_effect=AssertionError("must not launch")):
            self.assertIsNone(self.prepare(out=self.base / "empty-retry", launch=True))
        self.assertFalse((self.base / "empty-retry").exists())

    def test_launch_delegates_to_existing_failure_continuation(self):
        self.state["runs"]["r001"]["status"] = "pass"
        self.state["runs"]["r002"]["status"] = "numeric_fail"
        self.persist()
        seen = {}
        def launch(root):
            runner = FakeRunner(root, {"r003": RuntimeError("synthetic failed client")})
            with mock.patch.object(campaign, "analyze_run", return_value=dict(status="pass", run_id=str(uuid.uuid4()))), \
                 mock.patch.object(campaign, "write_report", return_value={}), \
                 mock.patch.object(campaign, "bundle", return_value=root.with_suffix(".zip")), \
                 redirect_stderr(io.StringIO()):
                seen["code"] = campaign.run_campaign(root, command_runner=runner)
            seen["calls"] = runner.calls
            return 0  # tmux launch success does not imply worker success.
        with mock.patch.object(campaign, "launch_tmux", side_effect=launch) as launcher:
            new = self.prepare(launch=True)
        launcher.assert_called_once_with(new)
        self.assertEqual(seen["code"], 2)
        self.assertEqual([name for role, name, _, _ in seen["calls"] if role == "runs"], ["r003", "r004"])
        state = common.read_json(new / "state.json")
        self.assertEqual(state["runs"]["r003"]["status"], "incomplete")
        self.assertEqual(state["runs"]["r004"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
