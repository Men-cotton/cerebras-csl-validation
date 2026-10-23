import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import csl_fetch as fetch


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.bundle = b"a completed test bundle"
        self.entry = dict(id="case-01", sha="1" * 40, status="passed", packaged=True,
                          archive="case-01.zip", bundle_bytes=len(self.bundle),
                          bundle_sha256=hashlib.sha256(self.bundle).hexdigest())
        self.status = dict(schema=1, cursor="2" * 40, updated_at="2026-10-23T20:00:00+00:00",
                           requests=[self.entry])

    def payload(self):
        return json.dumps(self.status).encode()

    def fake_command(self, argv, **kwargs):
        self.assertIn("BatchMode=yes", argv)
        self.assertIn("ConnectTimeout=10", argv)
        if argv[0] == "scp":
            Path(argv[-1]).write_bytes(self.bundle)
            return b""
        if "check" in argv:
            return b""
        return self.payload()

    def test_collect_and_skip_verified_bundle(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(fetch, "run_command", side_effect=self.fake_command) as command:
            output = Path(name)
            _, ids = fetch.collect("alcf-results", "/home/user/state", output)
            self.assertEqual(ids, ["case-01"])
            self.assertEqual((output / "case-01.zip").read_bytes(), self.bundle)
            self.assertEqual((output / "case-01.zip.sha256").read_text(), self.entry["bundle_sha256"] + "  case-01.zip\n")
            self.assertEqual(command.call_args_list[0].args[0][-3:], ["-O", "check", "alcf-results"])
            command.reset_mock()
            self.assertEqual(fetch.collect("alcf-results", "/home/user/state", output)[1], [])
            self.assertFalse(any(c.args[0][0] == "scp" for c in command.call_args_list))

    def test_missing_master_prevents_status_transfer(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(fetch, "run_command", side_effect=ValueError("no master")) as command:
            with self.assertRaisesRegex(ValueError, "no master"):
                fetch.collect("alcf-results", "/home/user/state", Path(name))
            self.assertEqual(command.call_count, 1)

    def test_watch_stops_on_failure_without_sleep(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(fetch, "collect", side_effect=ValueError("lost transport")), mock.patch.object(fetch.time, "sleep") as sleep:
            self.assertEqual(fetch.main(["--host", "alcf-results", "--state-dir", "/home/user/state", "--out", name, "--watch"]), 2)
            sleep.assert_not_called()

    def test_bad_checksum_never_publishes_bundle(self):
        self.entry["bundle_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as name, mock.patch.object(fetch, "run_command", side_effect=self.fake_command):
            output = Path(name)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                fetch.collect("alcf-results", "/home/user/state", output)
            self.assertFalse((output / "case-01.zip").exists())
            self.assertFalse(list(output.glob(".download-*")))

    def test_existing_bad_or_symlink_bundle_is_not_overwritten(self):
        for symlink in (False, True):
            with self.subTest(symlink=symlink), tempfile.TemporaryDirectory() as name, mock.patch.object(fetch, "run_command", side_effect=self.fake_command):
                output = Path(name)
                target = output / "case-01.zip"
                if symlink:
                    target.symlink_to(output / "absent")
                else:
                    target.write_bytes(b"wrong existing data")
                with self.assertRaises(ValueError):
                    fetch.collect("alcf-results", "/home/user/state", output)
                self.assertTrue(target.is_symlink() if symlink else target.read_bytes() == b"wrong existing data")

    def test_changed_request_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as name, mock.patch.object(fetch, "run_command", side_effect=self.fake_command):
            output = Path(name)
            fetch.collect("alcf-results", "/home/user/state", output)
            self.entry["sha"] = "3" * 40
            with self.assertRaisesRegex(ValueError, "changed source SHA"):
                fetch.collect("alcf-results", "/home/user/state", output)

    def test_pending_requests_do_not_download(self):
        self.status["requests"] = [dict(id="pending", sha="1" * 40, status="pending")]
        with tempfile.TemporaryDirectory() as name, mock.patch.object(fetch, "run_command", side_effect=self.fake_command) as command:
            self.assertEqual(fetch.collect("alcf-results", "/home/user/state", Path(name))[1], [])
            self.assertFalse(any(c.args[0][0] == "scp" for c in command.call_args_list))

    def test_reject_malformed_status(self):
        for key, value in (("id", "../escape"), ("sha", "oops"), ("archive", "other.zip"),
                           ("bundle_bytes", True), ("bundle_bytes", fetch.BUNDLE_LIMIT + 1),
                           ("bundle_sha256", "no"), ("packaged", "yes"), ("status", "unknown")):
            with self.subTest(key=key, value=value):
                changed = json.loads(self.payload())
                changed["requests"][0][key] = value
                with self.assertRaises(ValueError):
                    fetch.parse_status(json.dumps(changed).encode())
        with self.assertRaises(ValueError):
            fetch.parse_status(b" " * (fetch.STATUS_LIMIT + 1))
        with self.assertRaises(ValueError):
            fetch.parse_status(b'{"schema":1,"schema":1}')
        self.status["requests"].append(self.entry)
        with self.assertRaises(ValueError):
            fetch.parse_status(self.payload())

    def test_reject_connection_injection(self):
        for host, path in (("-oProxyCommand=x", "/home/x"), ("ok;bad", "/home/x"),
                           ("ok", "/home/x;id"), ("ok", "/home/../x"),
                           ("ok", "/home//x"), ("ok", "/")):
            with self.subTest(host=host, path=path), self.assertRaises(ValueError):
                fetch.validate_connection(host, path)

    def test_subprocess_download_growth_limit(self):
        with tempfile.TemporaryDirectory() as name:
            partial = Path(name) / "oversized.partial"
            process = mock.Mock()
            process.poll.return_value = 0

            def fake_popen(argv, **kwargs):
                self.assertEqual(kwargs["stdin"], fetch.subprocess.DEVNULL)
                partial.write_bytes(b"larger than declared")
                return process

            with mock.patch.object(fetch.subprocess, "Popen", side_effect=fake_popen):
                with self.assertRaisesRegex(ValueError, "declared size"):
                    fetch.run_command(["scp", "fake", str(partial)], target=partial, target_limit=1)


if __name__ == "__main__":
    unittest.main()
