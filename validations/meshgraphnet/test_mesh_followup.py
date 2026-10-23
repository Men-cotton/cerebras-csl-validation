"""Follow-up orchestration tests; mock clients submit no jobs or tmux sessions."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import mesh_appliance as appliance
import mesh_campaign as campaign
import mesh_followup as followup


class FollowupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        for name in followup.SOURCES:
            (self.source / name).write_text("# source fixture " + name + "\n")
        for module in (appliance, campaign, followup):
            patch = mock.patch.object(module, "ROOT", self.source)
            patch.start()
            self.addCleanup(patch.stop)
        self.packages = self.root / "mesh_followup_cases"
        for name, configuration in followup.COHORTS.items():
            package = self.packages / name
            package.mkdir(parents=True)
            files, cases = {}, {}
            for mode, case_name in enumerate(followup.CASES):
                folder = package / "cases" / case_name
                folder.mkdir(parents=True)
                params = dict(configuration, MODE=mode, STEPS=4, NORM=0, K=1 if mode == 0 else configuration["K"])
                case = dict(params=params, residency=dict(schema=1, chunks=[2, 2]), graph_scope="full", graph_limit=None,
                            original_graph_counts=dict(nodes=1896, edges=10908), partition="spatial",
                            census=dict(nodes=1896, directed_edges=10908))
                appliance.write_json(folder / "case.json", case)
                value = appliance.digest(folder / "case.json")
                files["cases/" + case_name + "/case.json"] = value
                cases[case_name] = dict(case_sha256=value, params=params)
            appliance.write_json(package / "manifest.json", dict(schema=1, files=files, cases=cases, source_commit="mock-source"))

        def verify(package):
            package = Path(package)
            manifest = appliance.read_json(package / "manifest.json")
            for name, value in manifest["files"].items():
                appliance.require(appliance.digest(package / name) == value, "fixture package changed")
            return manifest

        patch = mock.patch.object(appliance.mesh_audit, "verify_package", side_effect=verify)
        patch.start()
        self.addCleanup(patch.stop)

    def prepare(self):
        return followup.prepare(followup.parser().parse_args([
            "--package", str(self.packages), "--out", str(self.root / "followup"), "--plan-only", "--namespace", "site-ns"]))

    def rewrite_case(self, cohort, name, mutate):
        package = self.packages / cohort
        path = package / "cases" / name / "case.json"
        case = appliance.read_json(path)
        mutate(case)
        appliance.write_json(path, case)
        manifest = appliance.read_json(package / "manifest.json")
        manifest["files"]["cases/" + name + "/case.json"] = appliance.digest(path)
        manifest["cases"][name] = dict(case_sha256=appliance.digest(path), params=case["params"])
        appliance.write_json(package / "manifest.json", manifest)

    def mock_child_runner(self, child_calls, job_calls, fail_build=None, reject_started=None):
        def runner(argv, log, timeout):
            child = Path(argv[argv.index("--resume") + 1])
            child_calls.append(child.name)
            self.assertEqual(Path(argv[2]), child / "driver/mesh_campaign.py")
            self.assertIn("--foreground", argv)
            self.assertGreater(timeout, 0)
            log.write_text("mock foreground child\n")

            def client(command, child_log, client_timeout):
                case = command[command.index("--case") + 1]
                action = "compile" if "compile" in command else "run"
                job_calls.append((child.name, action, case))
                child_log.write_text("mock SDK client\n")
                if (child.name, case) == fail_build and action == "compile":
                    return 2
                if action == "run":
                    destination = Path(command[command.index("--out") + 1])
                    destination.mkdir()
                    appliance.write_json(destination / "audit.json", dict(status="PASS"))
                return 0

            def validate_build(root, plan, entry):
                if (root.name, entry["case"]) == reject_started:
                    raise ValueError("started attempt has no terminal evidence")
                return {}

            def compare(package, records):
                # A controller is allowed to compare only its own frozen package.
                self.assertEqual(Path(package), child / "mesh_cases")
                self.assertTrue(all(str(child / "runs") in record["capture_dir"] for record in records))
                return dict(status="PASS", comparisons=[], complete_transport_set=set(followup.CASES) <= {r["case_name"] for r in records})

            with mock.patch.object(campaign, "validate_build", side_effect=validate_build), \
                 mock.patch.object(campaign, "validate_run", return_value=dict(status="PASS")), \
                 mock.patch.object(campaign.mesh_audit, "compare_runs", side_effect=compare):
                return campaign.run_campaign(child, client)
        return runner

    def test_plan_only_prepares_two_disjoint_frozen_campaigns(self):
        with mock.patch.object(campaign, "execute", side_effect=AssertionError("SDK execution forbidden")), \
             mock.patch.object(followup.subprocess, "run", side_effect=AssertionError("tmux forbidden")):
            root = self.prepare()
        plan, state = followup.load(root)
        self.assertEqual(list(state["cohorts"]), list(followup.COHORTS))
        self.assertEqual(len(plan["source_sha256"]), 5)
        hashes = []
        for entry in plan["cohorts"]:
            child = root / entry["path"]
            child_plan, child_state = campaign.load(child)
            self.assertEqual([b["case"] for b in child_plan["builds"]], list(followup.CASES))
            self.assertEqual([r["case"] for r in child_plan["runs"]], list(followup.CASES) + list(reversed(followup.CASES)))
            self.assertEqual(child_plan["namespace"], "site-ns")
            self.assertEqual(child_state["status"], "pending")
            hashes.append(child_plan["package_manifest_sha256"])
        self.assertNotEqual(*hashes)

    def test_serial_failure_continues_and_comparisons_stay_in_cohort(self):
        root, child_calls, jobs = self.prepare(), [], []
        code = followup.run_followup(root, self.mock_child_runner(child_calls, jobs, ("h4-s2", "full-colored")))
        self.assertEqual(code, 2)
        self.assertEqual(child_calls, ["h4-s2", "h8-s2"])
        self.assertEqual([j[0] for j in jobs], ["h4-s2"] * 7 + ["h8-s2"] * 9)
        summary = appliance.read_json(root / "summary.json")
        self.assertFalse(summary["cohorts"]["h4-s2"]["summary"]["comparison"]["complete_transport_set"])
        self.assertTrue(summary["cohorts"]["h8-s2"]["summary"]["comparison"]["complete_transport_set"])
        self.assertEqual(summary["cohorts"]["h4-s2"]["state"]["status"], "completed_with_failures")
        self.assertEqual(summary["cohorts"]["h8-s2"]["state"]["status"], "complete")

    def test_resume_revalidates_without_reissuing_attempted_jobs(self):
        root, children, jobs = self.prepare(), [], []
        runner = self.mock_child_runner(children, jobs, ("h4-s2", "full-colored"))
        self.assertEqual(followup.run_followup(root, runner), 2)
        initial_jobs = list(jobs)
        # Checkout changes do not alter already-frozen drivers or packages.
        for name in followup.SOURCES:
            (self.source / name).write_text("# newer checkout\n")
        self.assertEqual(followup.run_followup(root, runner), 2)
        self.assertEqual(jobs, initial_jobs)
        self.assertEqual(children, ["h4-s2", "h8-s2", "h4-s2", "h8-s2"])
        self.assertTrue((root / "client-logs/h4-s2-002.log").is_file())

    def test_started_attempt_becomes_unknown_and_pending_jobs_continue(self):
        root, children, jobs = self.prepare(), [], []
        child = root / "cohorts/h4-s2"
        _, state = campaign.load(child)
        state["builds"]["full-broadcast"]["status"] = "started"
        appliance.write_json(child / "state.json", state)
        self.assertEqual(followup.run_followup(root, self.mock_child_runner(children, jobs, reject_started=("h4-s2", "full-broadcast"))), 2)
        self.assertNotIn(("h4-s2", "compile", "full-broadcast"), jobs)
        self.assertEqual(campaign.load(child)[1]["builds"]["full-broadcast"]["status"], "unknown")
        self.assertIn(("h8-s2", "run", "full-colored"), jobs)

    def test_missing_child_terminal_evidence_does_not_block_next_cohort(self):
        root, calls = self.prepare(), []
        def runner(argv, log, timeout):
            calls.append(Path(argv[-1]).name)
            log.write_text("client failed before terminal summary\n")
            return 2
        self.assertEqual(followup.run_followup(root, runner), 2)
        self.assertEqual(calls, list(followup.COHORTS))
        self.assertTrue(all(s["status"] == "unknown" for s in followup.load(root)[1]["cohorts"].values()))

    def test_busy_child_prevents_any_new_child_submission(self):
        root = self.prepare()
        with campaign.output_lock(root / "cohorts/h8-s2"), \
             mock.patch.object(followup, "child_command") as command:
            with self.assertRaisesRegex(ValueError, "another controller"):
                followup.run_followup(root)
        command.assert_not_called()

    def test_frozen_parent_or_child_mutation_is_rejected(self):
        root = self.prepare()
        path = root / "driver/mesh_followup.py"
        before = path.read_bytes()
        path.write_text("# changed parent\n")
        with self.assertRaisesRegex(ValueError, "frozen follow-up"):
            followup.load(root)
        path.write_bytes(before)
        (root / "cohorts/h8-s2/driver/mesh_worker.py").write_text("# changed child\n")
        with self.assertRaisesRegex(ValueError, "frozen controller"):
            followup.load(root)

    def test_changed_cohort_dimensions_and_mesh_prefix_are_rejected(self):
        self.rewrite_case("h8-s2", "full-colored", lambda c: c["params"].update(H=4))
        with self.assertRaisesRegex(ValueError, "cohort parameters"):
            self.prepare()
        self.rewrite_case("h8-s2", "full-colored", lambda c: c["params"].update(H=8))
        self.rewrite_case("h4-s2", "full-colored", lambda c: c.update(graph_scope="prefix", graph_limit=64))
        with self.assertRaisesRegex(ValueError, "full CylinderFlow"):
            self.prepare()

    def test_resume_executes_frozen_parent_before_loading_children(self):
        root = self.prepare()
        class ExecIntercept(Exception):
            pass
        with mock.patch.object(followup.os, "execv", side_effect=ExecIntercept) as execute, \
             mock.patch.object(followup, "prepare", side_effect=AssertionError("live prepare must not run")):
            with self.assertRaises(ExecIntercept):
                followup.main(["--resume", str(root), "--foreground"])
        self.assertEqual(Path(execute.call_args.args[1][2]), root / "driver/mesh_followup.py")
        self.assertIn("--foreground", execute.call_args.args[1])

    def test_resume_rejects_changed_frozen_parent_before_exec(self):
        root = self.prepare()
        (root / "driver/mesh_followup.py").write_text("# changed frozen parent\n")
        with mock.patch.object(followup.os, "execv") as execute:
            self.assertEqual(followup.main(["--resume", str(root), "--foreground"]), 2)
        execute.assert_not_called()

    def test_private_tmux_launch_uses_frozen_parent_and_detects_existing_session(self):
        root = self.prepare()
        with mock.patch.object(followup.shutil, "which", return_value="/usr/bin/tmux"), \
             mock.patch.object(followup.subprocess, "run", return_value=mock.Mock(returncode=0, stderr="")) as run:
            self.assertEqual(followup.launch_tmux(root), 0)
            command = run.call_args.args[0]
            self.assertEqual(command[:2], ["tmux", "-L"])
            self.assertTrue(command[2].startswith("mesh-followup-"))
            self.assertIn("new-session", command)
            self.assertIn(str(root / "driver/mesh_followup.py"), command[-1])
            with self.assertRaisesRegex(ValueError, "existing follow-up tmux"):
                followup.launch_tmux(root)
        self.assertEqual(run.call_count, 2)

    def test_parent_bundle_contains_child_zip_or_expanded_failed_child(self):
        root = self.prepare()
        child = root / "cohorts/h4-s2"
        state = appliance.read_json(root / "state.json")
        state["cohorts"]["h4-s2"]["status"] = "complete"
        appliance.write_json(root / "state.json", state)
        state = appliance.read_json(child / "state.json")
        state["status"] = "complete"
        appliance.write_json(child / "state.json", state)
        campaign.bundle(child)
        destination = followup.bundle(root)
        with zipfile.ZipFile(destination) as result:
            names = set(result.namelist())
        self.assertIn(root.name + "/cohorts/h4-s2.zip", names)
        self.assertNotIn(root.name + "/cohorts/h4-s2/plan.json", names)
        self.assertIn(root.name + "/cohorts/h8-s2/plan.json", names)
        self.assertIn(root.name + "/driver/mesh_followup.py", names)

        # New evidence invalidates a prior child ZIP even if the prior parent
        # status was complete. The next parent archive includes the new file.
        (child / "failure-evidence.txt").write_text("new interrupted resume evidence\n")
        destination = followup.bundle(root)
        with zipfile.ZipFile(destination) as result:
            names = set(result.namelist())
        self.assertIn(root.name + "/cohorts/h4-s2/failure-evidence.txt", names)
        self.assertIn(root.name + "/cohorts/h4-s2.zip", names)


if __name__ == "__main__":
    unittest.main()
