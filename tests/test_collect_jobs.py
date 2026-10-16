"""Read-only log discovery and mocked csctl collection tests."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collect_jobs


class CollectJobsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / "launch-hw-case.log"
        self.log.write_text("INFO:Job id: wsjob-example, workflow id: wflow-example, namespace: example, "
                            "remote log path: /remote/example\nWARNING wsjob=wsjob-example\n")

    def test_python_glob_and_exact_job_line(self):
        paths = collect_jobs.expand_logs([str(self.root / "launch-*.log"), str(self.log)])
        records = collect_jobs.scan_logs(paths)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["job_id"], "wsjob-example")
        self.assertEqual(records[0]["workflow_id"], "wflow-example")
        self.assertEqual(records[0]["remote_log_path"], "/remote/example")
        self.assertEqual(records[0]["role"], "launch")

    def test_default_does_not_call_csctl(self):
        with patch("subprocess.run", side_effect=AssertionError("unexpected external command")):
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(collect_jobs.main([str(self.log)]), 0)
        self.assertEqual(json.loads(output.getvalue())["jobs"][0]["job_id"], "wsjob-example")

    def test_saved_client_log_uses_request_action(self):
        client = self.root / "client.log"
        client.write_bytes(self.log.read_bytes())
        (self.root / "request.json").write_text('{"action":"run"}')
        self.assertEqual(collect_jobs.scan_logs([client])[0]["role"], "launch")

    def test_collect_only_get_and_export_in_fresh_directory(self):
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            if "get" in argv:
                kwargs["stdout"].write(b'{"status":{"systems":["system-example"]}}')
            else:
                (Path(kwargs["cwd"]) / "export.tar.gz").write_bytes(b"mock export")
            return SimpleNamespace(returncode=0)

        records = collect_jobs.scan_logs([self.log])
        result = collect_jobs.collect(records, self.root / "out", "example-space", 17, runner=runner)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["collections"][0]["systems"], ["system-example"])
        self.assertEqual(calls[0][0], ["csctl", "-n", "example-space", "get", "job", "wsjob-example", "-o", "json"])
        self.assertEqual(calls[1][0], ["csctl", "-n", "example-space", "log-export", "wsjob-example"])
        self.assertEqual(calls[0][1]["timeout"], 17)
        with self.assertRaises(FileExistsError):
            collect_jobs.collect(records, self.root / "out", None, 17, runner=runner)

    def test_unknown_systems_shape_not_interpreted(self):
        self.assertIsNone(collect_jobs.systems_from_job({"status": {"systems": {"name": "ambiguous"}}}))
        self.assertIsNone(collect_jobs.systems_from_job({"systems": ["wrong-level"]}))

    def test_timeout_preserves_partial_metadata_and_continues_export(self):
        calls = []

        def runner(argv, **kwargs):
            calls.append(argv)
            if "get" in argv:
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            return SimpleNamespace(returncode=4)

        result = collect_jobs.collect(collect_jobs.scan_logs([self.log]), self.root / "out", None, 2, runner=runner)
        self.assertEqual(result["status"], "collection_failed")
        self.assertEqual([item["status"] for item in result["collections"][0]["commands"]], ["timeout", "failed"])
        self.assertEqual(len(calls), 2)

    def test_job_id_rejects_option_or_shell_text(self):
        for value in ("--help", "wsjob-a;id", "wsjob-a/../../x"):
            with self.assertRaises(Exception):
                collect_jobs.job_id(value)


if __name__ == "__main__":
    unittest.main()
