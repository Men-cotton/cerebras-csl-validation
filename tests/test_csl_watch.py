"""Offline controller tests: local Git remotes and fake appliance children only."""
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import csl_watch as watch


# Operator-supplied JSON shape, anonymized; running compile has no systems.
SUPPLIED_JOB = {
    "meta": {
        "createTime": "2026-01-01T19:09:27Z", "labels": {},
        "name": "wsjob-examplecompile", "namespace": "job-operator",
        "properties": {
            "cerebras/reconcile-operator-name": "job-operator-controller-example",
            "cerebras/semantic-version-appliance-client": "0.0.0",
            "cerebras/semantic-version-cluster-server": "0.0.0",
            "cerebras/semantic-version-job-operator": "0.0.0",
            "cerebras/version-appliance-client": "0.0.0-test-client",
            "cerebras/version-cluster-server": "0.0.0-test-server",
            "cerebras/version-job-operator": "0.0.0-test-server",
            "cerebras/version-user-sidecar": "",
            "cerebras/version-wsjob": "0.0.0-test-worker",
            "preferred-power-profiles": "",
        },
        "type": "job", "workflow": "wflow-example",
    },
    "spec": {
        "clientWorkdir": "", "priority": "P2 (299)", "type": "compile",
        "user": {"gid": "1000", "groups": [], "uid": "1000", "username": "operator"},
        "volumeMounts": [],
    },
    "status": {
        "completionTime": None,
        "conditions": [
            {"lastTransitionTime": "2026-01-01T19:09:27Z", "message": "job created", "type": "Created"},
            {"lastTransitionTime": "2026-01-01T19:09:28Z", "message": "job queueing to be scheduled", "type": "Queueing"},
            {"lastTransitionTime": "2026-01-01T19:09:28Z", "message": "job scheduled and waiting to be initialized", "type": "Scheduled"},
            {"lastTransitionTime": "2026-01-01T19:09:34Z", "message": "job initializing with image pulling/config update/...", "type": "Initializing"},
            {"lastTransitionTime": "2026-01-01T19:09:45Z", "message": "job started running", "type": "Running"},
        ],
        "errorReplicas": [], "executionTime": "2026-01-01T19:09:45Z",
        "lock": {"jobPendingStatus": {"allPendingDetails": [], "blockingJobs": {}, "pendingDetails": "", "pendingReason": ""}},
        "phase": "RUNNING", "systems": [],
    },
}


def request(ident="example", modes=("ok",)):
    return {"schema": 1, "id": ident, "description": "Offline fake SDK actions",
            "steps": [{"name": "step-" + str(i), "script": "fake_appliance.py",
                       "args": [mode, "{output}", "wsjob-step" + str(i)],
                       "kind": "appliance", "timeout_seconds": 1 if mode == "slow" else 10}
                      for i, mode in enumerate(modes)]}


class SchemaTests(unittest.TestCase):
    def test_operator_running_compile_is_busy_without_a_system_assignment(self):
        values = watch.parse_jobs({"items": [SUPPLIED_JOB]}, "job-operator", "operator")
        self.assertEqual(values[0]["phase"], "RUNNING")
        scheduler = watch.Scheduler({"namespace": "job-operator", "username": "operator"})
        with patch.object(scheduler, "call", return_value={"items": [SUPPLIED_JOB]}):
            self.assertFalse(scheduler.idle())

    def test_malformed_unknown_foreign_and_duplicate_jobs_fail_closed(self):
        invalid = [[], {}, {"items": None}, {"items": [None]}]
        for section, field, value in (("status", "phase", "NEW_PHASE"),
                                      ("meta", "namespace", "other"),
                                      ("meta", "name", "not-a-job")):
            item = copy.deepcopy(SUPPLIED_JOB)
            item[section][field] = value
            invalid.append({"items": [item]})
        item = copy.deepcopy(SUPPLIED_JOB)
        item["spec"]["user"]["username"] = "someone-else"
        invalid.extend(({"items": [item]}, {"items": [SUPPLIED_JOB, SUPPLIED_JOB]}))
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                watch.parse_jobs(value, "job-operator", "operator")

    def test_truncation_and_pagination_are_not_idle(self):
        for value in ({"items": [], "continue": "cursor"},
                      {"items": [], "metadata": {"continue": "cursor"}},
                      {"items": [], "nextPageToken": "cursor"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                watch.parse_jobs(value, "job-operator", "operator")
        with self.assertRaisesRegex(ValueError, "truncated"):
            watch.parse_jobs({"items": [SUPPLIED_JOB]}, "job-operator", "operator", maximum=1)

    def test_individual_lookup_requires_matching_identity_and_nonempty_result(self):
        scheduler = watch.Scheduler({"namespace": "job-operator", "username": "operator"})
        for value in ({"items": []}, SUPPLIED_JOB):
            with self.subTest(value=value), patch.object(scheduler, "call", return_value=value), self.assertRaises(ValueError):
                scheduler.job("wsjob-different")

    def test_only_complete_final_trailers_create_requests(self):
        good = "Subject\n\nALCF-Request: example\nALCF-Manifest: requests/example.json\n"
        self.assertEqual(watch.trailers(good), ("example", "requests/example.json"))
        self.assertIsNone(watch.trailers(good + "\nThis paragraph is ordinary prose.\n"))
        for message in ("ALCF-Request: example", good + "ALCF-Request: again\n",
                        good.replace("requests/example.json", "requests/../bad.json"),
                        good.replace("ALCF-Request: example", "ALCF-Request: $(bad)")):
            with self.subTest(message=message), self.assertRaises(ValueError):
                watch.trailers(message)

    def test_manifest_rejects_nested_campaigns_invalid_argv_and_duplicate_steps(self):
        original = request()
        self.assertIs(watch.manifest(original, ["fake_appliance.py"]), original)
        invalid = []
        for key, value in (("timeout_seconds", True), ("timeout_seconds", 0),
                           ("args", "shell text"), ("args", ["bad\0arg"]),
                           ("kind", "unknown"), ("script", "outside.py")):
            candidate = copy.deepcopy(original)
            candidate["steps"][0][key] = value
            invalid.append(candidate)
        candidate = copy.deepcopy(original)
        candidate["steps"].append(copy.deepcopy(candidate["steps"][0]))
        invalid.append(candidate)
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                watch.manifest(candidate, ["fake_appliance.py"])
        candidate = copy.deepcopy(original)
        candidate["steps"][0]["script"] = "tank_campaign.py"
        with self.assertRaisesRegex(ValueError, "nested legacy"):
            watch.manifest(candidate, ["tank_campaign.py"])

    def test_output_inventory_rejects_root_and_nested_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "output"
            output.mkdir()
            (output / "result.txt").write_text("retained result")
            root_link = base / "root-link"
            root_link.symlink_to(output, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "output root"):
                watch.files_under(root_link, 1024)
            (output / "linked-result").symlink_to(output / "result.txt")
            with self.assertRaisesRegex(ValueError, "symlink"):
                watch.files_under(output, 1024)

    def test_unreadable_output_scan_is_not_an_empty_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            def denied_walk(root, **kwargs):
                # os.walk suppresses this error unless the caller supplies onerror.
                kwargs["onerror"](PermissionError("cannot read output child"))
                yield root, [], []

            with patch.object(watch.os, "walk", side_effect=denied_walk):
                with self.assertRaisesRegex(PermissionError, "cannot read"):
                    watch.files_under(Path(temporary), 1024)

    def test_missing_and_dangling_output_roots_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "output root"):
                watch.files_under(root / "missing", 1024)
            (root / "dangling").symlink_to(root / "missing", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "output root"):
                watch.files_under(root / "dangling", 1024)

    def test_shipped_manifests_match_real_entrypoint_argument_parsers(self):
        # Import only parsers; neither handler nor appliance SDK is invoked.
        import importlib.util
        from csl_validation import catalog
        root = Path(__file__).resolve().parents[1]
        for name in ("native16/native16-baseline.json", "primitives/topk-small.json"):
            value = catalog.request_file(root, "requests/" + name)
            with tempfile.TemporaryDirectory() as temporary:
                for step in value["steps"]:
                    script = root / step["script"]
                    with patch.object(sys, "path", [str(script.parent)] + sys.path):
                        spec = importlib.util.spec_from_file_location("shipped_parser", script)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        args = module.parser().parse_args([a.replace("{output}", temporary) for a in step["args"]])
                        self.assertTrue(args.out.is_absolute())
                        self.assertEqual(args.backend, "hardware")
                        self.assertEqual(step["kind"], "appliance")
                        self.assertFalse(args.out.exists())

    def test_oversized_manifest_is_rejected_before_request_execution(self):
        value = request()
        value["description"] = "x" * (64 << 10)
        with self.assertRaisesRegex(ValueError, "64 KiB"):
            watch.manifest(value, ["fake_appliance.py"])


FAKE_SCRIPT = '''import pathlib, sys, time
mode, out, job = sys.argv[1:]
root = pathlib.Path(out)
root.mkdir(parents=True, exist_ok=True)
with (root / "invocations.txt").open("a") as stream:
    stream.write(job + "\\n")
if mode != "no-id":
    print("Job id: " + job, flush=True)
if mode == "slow":
    time.sleep(10)
(root / (job + ".json")).write_text('{"status":"PASS"}')
raise SystemExit(2 if mode == "fail" else 0)
'''


class FakeScheduler:
    def __init__(self):
        self.busy = False
        self.phases = {}
        self.lookups = []

    def idle(self):
        return not self.busy

    def job(self, name):
        self.lookups.append(name)
        return {"name": name, "namespace": "job-operator", "phase": self.phases.get(name, "SUCCEEDED")}


class GitControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source, self.remote, self.state = (self.base / name for name in ("source", "remote.git", "state"))
        self.source.mkdir()
        self.git("init", "--bare", str(self.remote))
        self.git("init", "-b", "main", cwd=self.source)
        (self.source / "fake_appliance.py").write_text(FAKE_SCRIPT)
        self.git("add", ".", cwd=self.source)
        self.git("commit", "-m", "Initial source, no execution request", cwd=self.source)
        self.baseline = self.git("rev-parse", "HEAD", cwd=self.source).strip()
        self.git("push", str(self.remote), "HEAD:refs/heads/main", cwd=self.source)
        self.state.mkdir()
        self.git("init", "--bare", str(self.state / "source.git"))
        self.config = {"schema": 1, "repo": str(self.remote), "branch": "main", "status_branch": "alcf-status",
                       "state_dir": str(self.state), "python": sys.executable, "namespace": "job-operator",
                       "username": "operator", "hostname": socket.gethostname(), "csctl": "NEVER-EXECUTE-CSCTL",
                       "idle_seconds": 0, "poll_seconds": 0.02, "output_bytes": 1 << 20,
                       "publish_status": False, "allowed_scripts": ["fake_appliance.py"],
                       "lock_file": str(self.base / "operator.lock")}
        watch.atomic_json(self.state / "ledger.json", {"schema": 1, "cursor": self.baseline, "last_check": 0,
                                                       "requests": {}, "order": []})
        self.scheduler = FakeScheduler()
        self.watcher = watch.Watcher(self.config, self.scheduler)

    @staticmethod
    def git(*args, cwd=None):
        env = dict(os.environ, GIT_AUTHOR_NAME="Offline test", GIT_AUTHOR_EMAIL="offline@example.invalid",
                   GIT_COMMITTER_NAME="Offline test", GIT_COMMITTER_EMAIL="offline@example.invalid",
                   GIT_ALLOW_PROTOCOL="file", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        result = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", *args], cwd=cwd, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15, check=True)
        return result.stdout

    def submit(self, value=None):
        value = value or request()
        folder = self.source / "requests"
        folder.mkdir(exist_ok=True)
        (folder / "example.json").write_text(json.dumps(value))
        self.git("add", ".", cwd=self.source)
        self.git("commit", "--allow-empty", "-m", "Request offline check\n\nALCF-Request: " + value["id"] +
                 "\nALCF-Manifest: requests/example.json", cwd=self.source)
        self.git("push", str(self.remote), "HEAD:refs/heads/main", cwd=self.source)
        return self.git("rev-parse", "HEAD", cwd=self.source).strip()

    def entry(self):
        return self.watcher.ledger["requests"]["example"]

    def invocations(self):
        path = self.state / "requests/example/output/invocations.txt"
        return path.read_text().splitlines() if path.exists() else []

    def complete(self, limit=12):
        for _ in range(limit):
            self.watcher.tick()
            entry = self.watcher.ledger["requests"].get("example", {})
            if entry.get("packaged") or entry.get("status") == "blocked":
                return entry
        self.fail("controller did not finish within bounded offline ticks")

    def test_two_steps_pin_source_and_package_results_without_resubmission(self):
        sha = self.submit(request(modes=("ok", "ok")))
        entry = self.complete()
        self.assertEqual(entry["status"], "passed")
        self.assertEqual(entry["sha"], sha)
        self.assertEqual(self.invocations(), ["wsjob-step0", "wsjob-step1"])
        self.assertEqual([step["status"] for step in entry["steps"]], ["passed", "passed"])
        self.assertEqual(watch.digest(entry["bundle"]), entry["bundle_sha256"])
        with zipfile.ZipFile(entry["bundle"]) as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn("output/wsjob-step1.json", archive.namelist())
            self.assertEqual(json.loads(archive.read("receipt.json"))["sha"], sha)
        restarted = watch.Watcher(self.config, self.scheduler)
        restarted.tick()
        self.assertEqual(self.invocations(), ["wsjob-step0", "wsjob-step1"])

    def test_busy_remote_blocks_discovery_clone_and_execution(self):
        self.submit()
        self.scheduler.busy = True
        self.watcher.tick()
        self.assertEqual(self.watcher.ledger["cursor"], self.baseline)
        self.assertEqual(self.watcher.ledger["requests"], {})
        self.assertFalse((self.state / "requests").exists())
        self.scheduler.busy = False
        self.watcher.tick()
        self.scheduler.busy = True
        self.watcher.tick()
        self.assertEqual(self.invocations(), [])

    def test_remote_running_after_client_exit_defers_second_step(self):
        self.submit(request(modes=("ok", "ok")))
        self.watcher.tick()
        self.watcher.tick()
        self.scheduler.phases["wsjob-step0"] = "RUNNING"
        self.watcher.tick()
        self.assertEqual(self.invocations(), ["wsjob-step0"])
        self.assertEqual(self.entry()["steps"][0]["status"], "awaiting")
        self.scheduler.phases["wsjob-step0"] = "SUCCEEDED"
        self.assertEqual(self.complete()["status"], "passed")
        self.assertEqual(self.invocations(), ["wsjob-step0", "wsjob-step1"])

    def test_nonzero_client_exit_records_failure_and_skips_remaining_step(self):
        self.submit(request(modes=("fail", "ok")))
        self.assertEqual(self.complete()["status"], "failed")
        self.assertEqual(self.invocations(), ["wsjob-step0"])
        self.watcher.tick()
        self.assertEqual(self.invocations(), ["wsjob-step0"])

    def test_failed_remote_job_blocks_following_action_even_after_client_success(self):
        self.submit(request(modes=("ok", "ok")))
        self.scheduler.phases["wsjob-step0"] = "FAILED"
        self.assertEqual(self.complete()["status"], "failed")
        self.assertEqual(self.invocations(), ["wsjob-step0"])

    def test_missing_job_id_blocks_without_guessing_completion(self):
        self.submit(request(modes=("no-id", "ok")))
        entry = self.complete()
        self.assertEqual(entry["status"], "blocked")
        self.assertIn("no job IDs", entry["error"])
        self.watcher.tick()
        self.assertEqual(self.invocations(), ["wsjob-step0"])

    def test_host_step_logging_remote_job_ids_is_blocked(self):
        value = request(modes=("ok", "ok"))
        value["steps"][0]["kind"] = "host"
        self.submit(value)
        entry = self.complete()
        self.assertEqual(entry["status"], "blocked")
        self.assertIn("host step submitted remote jobs", entry["error"])
        self.assertEqual(self.invocations(), ["wsjob-step0"])

    def test_timeout_retains_uncertainty_and_does_not_launch_next_step(self):
        self.submit(request(modes=("slow", "ok")))
        entry = self.complete()
        self.assertEqual(entry["status"], "blocked")
        self.assertIn("timeout", entry["error"])
        self.assertEqual(self.invocations(), ["wsjob-step0"])
        self.watcher.tick()
        self.assertEqual(self.invocations(), ["wsjob-step0"])

    def test_crash_after_durable_intent_never_retries(self):
        self.submit()
        self.watcher.tick()
        entry = self.entry()
        entry["steps"] = [{"name": "step-0", "status": "running", "jobs": []}]
        self.watcher.record(entry)
        restarted = watch.Watcher(self.config, self.scheduler)
        restarted.tick()
        self.assertEqual(restarted.ledger["requests"]["example"]["status"], "blocked")
        self.assertEqual(self.invocations(), [])

    def test_status_push_failure_retries_only_notification_on_separate_branch(self):
        request_sha = self.submit()
        entry = self.complete()
        self.config["publish_status"] = True
        real_git = watch.git

        def reject_push(args, *positional, **keywords):
            if args[0] == "push":
                raise ValueError("simulated status transport outage")
            return real_git(args, *positional, **keywords)

        with patch.object(watch, "git", side_effect=reject_push):
            self.watcher.tick()
        self.assertFalse(entry["notified"])
        self.assertEqual(self.invocations(), ["wsjob-step0"])
        self.watcher.tick()
        self.assertTrue(entry["notified"])
        self.assertEqual(self.invocations(), ["wsjob-step0"])
        status = json.loads(self.git("--git-dir", str(self.remote), "show", "alcf-status:results/example.json"))
        self.assertEqual(status["sha"], request_sha)
        self.assertEqual(status["bundle_sha256"], entry["bundle_sha256"])
        self.assertEqual(self.git("--git-dir", str(self.remote), "rev-parse", "main").strip(), request_sha)

    def test_force_push_behind_cursor_is_rejected(self):
        self.submit()
        self.watcher.discover()
        self.git("push", "--force", str(self.remote), self.baseline + ":refs/heads/main", cwd=self.source)
        with self.assertRaises(ValueError):
            self.watcher.discover()
        self.assertEqual(self.invocations(), [])

    def test_merged_request_is_discovered_and_dag_cursor_saved_once(self):
        self.git("checkout", "-b", "feature", cwd=self.source)
        folder = self.source / "requests"
        folder.mkdir()
        (folder / "example.json").write_text(json.dumps(request()))
        self.git("add", ".", cwd=self.source)
        self.git("commit", "-m", "Feature request\n\nALCF-Request: example\nALCF-Manifest: requests/example.json", cwd=self.source)
        request_sha = self.git("rev-parse", "HEAD", cwd=self.source).strip()
        self.git("checkout", "main", cwd=self.source)
        (self.source / "main-note.txt").write_text("independent main commit")
        self.git("add", ".", cwd=self.source)
        self.git("commit", "-m", "Main change", cwd=self.source)
        self.git("merge", "--no-ff", "feature", "-m", "Merge request branch", cwd=self.source)
        tip = self.git("rev-parse", "HEAD", cwd=self.source).strip()
        self.git("push", str(self.remote), "HEAD:refs/heads/main", cwd=self.source)
        with patch.object(self.watcher, "save", wraps=self.watcher.save) as save:
            self.watcher.discover()
        save.assert_called_once()
        self.assertEqual(self.entry()["sha"], request_sha)
        self.assertEqual(self.watcher.ledger["cursor"], tip)
        restarted = watch.Watcher(self.config, self.scheduler)
        restarted.discover()
        self.assertEqual(restarted.ledger["order"], ["example"])

    def test_invalid_discovery_batch_does_not_partially_advance_cursor(self):
        self.submit()
        self.submit()
        with self.assertRaisesRegex(ValueError, "reused request ID"):
            self.watcher.discover()
        self.assertEqual(self.watcher.ledger["cursor"], self.baseline)
        self.assertEqual(self.watcher.ledger["requests"], {})
        durable = watch.read_json(self.state / "ledger.json")
        self.assertEqual(durable["cursor"], self.baseline)
        self.assertEqual(durable["requests"], {})

    def test_publish_crash_after_file_write_retries_commit_and_verifies_remote(self):
        self.submit()
        entry = self.complete()
        self.config["publish_status"] = True
        real_git = watch.git

        def interrupt_before_add(args, *positional, **keywords):
            if args[0] == "add":
                raise KeyboardInterrupt("simulated process interruption")
            return real_git(args, *positional, **keywords)

        with patch.object(watch, "git", side_effect=interrupt_before_add):
            with self.assertRaises(KeyboardInterrupt):
                self.watcher.publish(entry)
        self.assertTrue((self.state / "status-git/results/example.json").is_file())
        self.assertFalse(entry["notified"])
        self.assertEqual(self.git("ls-remote", "--heads", str(self.remote), "refs/heads/alcf-status"), "")
        restarted = watch.Watcher(self.config, self.scheduler)
        recovered = restarted.ledger["requests"]["example"]
        restarted.publish(recovered)
        self.assertTrue(recovered["notified"])
        durable = watch.read_json(self.state / "ledger.json")
        self.assertTrue(durable["requests"]["example"]["notified"])
        remote_record = json.loads(self.git("--git-dir", str(self.remote), "show", "alcf-status:results/example.json"))
        self.assertEqual(remote_record["bundle_sha256"], entry["bundle_sha256"])
        self.assertEqual(self.invocations(), ["wsjob-step0"])

    def test_reused_request_id_is_rejected(self):
        self.submit()
        self.watcher.discover()
        self.submit()
        with self.assertRaisesRegex(ValueError, "reused request ID"):
            self.watcher.discover()
        self.assertEqual(self.watcher.ledger["order"], ["example"])

    def test_operator_lock_rejects_second_owner(self):
        with watch.operator_lock(self.config):
            with self.assertRaises(BlockingIOError):
                with watch.operator_lock(self.config):
                    self.fail("second controller obtained operator lock")
        with watch.operator_lock(self.config):
            pass

    def test_output_budget_cannot_exceed_collector_two_gib_bound(self):
        config = dict(self.config, idle_seconds=60, poll_seconds=1, output_bytes=2 << 30)
        self.assertIs(watch.validate_config(config), config)
        config["output_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "2 GiB"):
            watch.validate_config(config)

    def test_ledger_loader_and_writer_use_explicit_size_limit(self):
        ledger = copy.deepcopy(self.watcher.ledger)
        ledger["padding"] = "x" * 2048
        watch.atomic_json(self.state / "ledger.json", ledger)
        size = watch.encoded_size(ledger)
        with patch.object(watch, "LEDGER_LIMIT", size + 100):
            loaded = watch.Watcher(self.config, self.scheduler)
            self.assertEqual(loaded.ledger["padding"], ledger["padding"])
            loaded.save()
        original_bytes = (self.state / "ledger.json").read_bytes()
        with patch.object(watch, "LEDGER_LIMIT", size - 1):
            with self.assertRaises(ValueError):
                watch.Watcher(self.config, self.scheduler)
            with self.assertRaisesRegex(ValueError, "ledger size limit"):
                loaded.save()
        self.assertEqual((self.state / "ledger.json").read_bytes(), original_bytes)

    def test_admission_limit_preserves_cursor_before_discovery_and_clone(self):
        self.submit()
        original_bytes = (self.state / "ledger.json").read_bytes()
        with patch.object(watch, "ADMISSION_LIMIT", watch.encoded_size(self.watcher.ledger) + 10):
            with self.assertRaisesRegex(ValueError, "admission limit"):
                self.watcher.discover()
        self.assertEqual(self.watcher.ledger["cursor"], self.baseline)
        self.assertEqual((self.state / "ledger.json").read_bytes(), original_bytes)
        self.assertFalse((self.state / "requests").exists())
        self.watcher.discover()
        cursor = self.watcher.ledger["cursor"]
        with patch.object(watch, "ADMISSION_LIMIT", watch.encoded_size(self.watcher.ledger) - 1):
            with self.assertRaisesRegex(ValueError, "admission limit"):
                self.watcher.prepare(self.entry())
        self.assertFalse((self.state / "requests").exists())
        self.assertEqual(self.watcher.ledger["cursor"], cursor)
        self.assertEqual(self.invocations(), [])

    def test_startup_repairs_stale_public_status_while_holding_operator_lock(self):
        self.submit()
        self.watcher.discover()
        self.entry().update(status="blocked", error="test unresolved prior attempt")
        self.watcher.save()
        watch.atomic_json(self.state / "status.json", {"schema": 1, "cursor": self.baseline, "requests": []})
        config = dict(self.config, idle_seconds=60, poll_seconds=1)
        config_path = self.base / "config.json"
        watch.atomic_json(config_path, config)
        real_save = watch.Watcher.save

        def save_under_lock(controller):
            with self.assertRaises(BlockingIOError):
                with watch.operator_lock(config):
                    self.fail("startup repaired status without holding the operator lock")
            real_save(controller)

        with patch.object(watch.Watcher, "save", autospec=True, side_effect=save_under_lock), \
             patch.object(watch, "Scheduler", return_value=self.scheduler), \
             patch.object(watch.os, "umask"), patch.object(watch.signal, "signal"):
            self.assertEqual(watch.main(["once", "--config", str(config_path)]), 0)
        status = watch.read_json(self.state / "status.json")
        self.assertEqual(status["cursor"], self.watcher.ledger["cursor"])
        self.assertEqual(status["requests"][0]["status"], "blocked")
        self.assertEqual(self.invocations(), [])

    def test_failed_initialization_removes_only_new_state_and_can_retry(self):
        root, config_path = self.base / "new-state", self.base / "new-config.json"
        args = SimpleNamespace(state_dir=root, config=config_path, baseline=self.baseline,
                               repo=str(self.remote), branch="missing-branch", python=Path(sys.executable),
                               namespace="job-operator", publish_status=False)
        with patch.object(watch.Path, "home", return_value=self.base):
            with self.assertRaises(ValueError):
                watch.initialize(args)
            self.assertFalse(root.exists())
            self.assertFalse(config_path.exists())
            self.assertTrue(self.remote.is_dir())
            self.assertTrue(self.state.is_dir())
            args.branch = "main"
            watch.initialize(args)
        self.assertTrue((root / "ledger.json").is_file())
        self.assertEqual(watch.read_json(root / "ledger.json")["cursor"], self.baseline)
        self.assertTrue(config_path.is_file())
        self.assertFalse((root / "requests").exists())


if __name__ == "__main__":
    unittest.main()
