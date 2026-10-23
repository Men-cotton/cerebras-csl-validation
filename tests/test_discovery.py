"""Fail-closed request inventory and isolated host-test discovery regressions."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from csl_validation import catalog, checks


class RequestTraversalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.requests = self.root / "requests"
        self.requests.mkdir()

    def test_nested_regular_files_have_stable_order(self):
        nested = self.requests / "family" / "sweep"
        nested.mkdir(parents=True)
        (nested / "small.json").write_text("{}")
        (self.requests / "a.json").write_text("{}")
        (self.requests / "README.md").write_text("Not a request")
        self.assertEqual(catalog.request_paths(self.root),
                         [self.requests / "a.json", nested / "small.json"])

    def test_symlinked_and_broken_directories_are_rejected(self):
        target = self.root / "target"
        target.mkdir()
        (target / "hidden.json").write_text("{}")
        for destination in (target, self.root / "missing"):
            with self.subTest(destination=destination):
                link = self.requests / "family"
                link.symlink_to(destination, target_is_directory=True)
                try:
                    with self.assertRaisesRegex(ValueError, "symlink"):
                        catalog.requests(self.root)
                finally:
                    link.unlink()

    def test_symlinked_files_including_non_json_are_rejected(self):
        target = self.root / "target"
        target.write_text("{}")
        for name in ("request.json", "README.md"):
            with self.subTest(name=name):
                link = self.requests / name
                link.symlink_to(target)
                try:
                    with self.assertRaisesRegex(ValueError, "symlink"):
                        catalog.requests(self.root)
                finally:
                    link.unlink()

    def test_symlinked_inventory_root_is_rejected(self):
        moved = self.root / "moved"
        self.requests.rename(moved)
        self.requests.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            catalog.requests(self.root)

    def test_unreadable_subdirectory_does_not_return_partial_inventory(self):
        blocked = self.requests / "blocked"
        blocked.mkdir()
        (self.requests / "visible.json").write_text("{}")
        original = catalog.os.scandir

        def scan(path):
            if Path(path) == blocked:
                raise PermissionError("unreadable request directory")
            return original(path)

        with mock.patch.object(catalog.os, "scandir", side_effect=scan):
            with self.assertRaisesRegex(PermissionError, "unreadable"):
                catalog.request_paths(self.root)

    def test_entry_budget_counts_non_json_files(self):
        (self.requests / "README.md").write_text("Inventory also bounds ignored files")
        with mock.patch.object(catalog, "REQUEST_SCAN_ENTRIES", 0):
            with self.assertRaisesRegex(ValueError, "entry limit"):
                catalog.request_paths(self.root)

    def test_elapsed_budget_rejects_incomplete_scan(self):
        with mock.patch.object(catalog.time, "monotonic", side_effect=[0, 11]):
            with self.assertRaisesRegex(ValueError, "time limit"):
                catalog.request_paths(self.root)


class HostSuiteDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "tests").mkdir()

    def run_suite(self, root=None):
        return subprocess.run(checks.suite_command("tests"), cwd=root or self.root,
                              capture_output=True, text=True, timeout=10)

    def test_empty_suite_is_failure(self):
        result = self.run_suite()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("No tests discovered", result.stderr)

    def test_wrong_filename_cannot_make_suite_pass(self):
        (self.root / "tests" / "case.py").write_text(
            "import unittest\nclass Check(unittest.TestCase):\n"
            "    def test_fail(self): self.fail('not discovered')\n")
        self.assertEqual(self.run_suite().returncode, 2)

    def test_discovered_failure_is_propagated(self):
        (self.root / "tests" / "test_case.py").write_text(
            "import unittest\nclass Check(unittest.TestCase):\n"
            "    def test_fail(self): self.fail('observable failure')\n")
        result = self.run_suite()
        self.assertEqual(result.returncode, 1)
        self.assertIn("observable failure", result.stderr)

    def test_family_cwd_and_import_names_are_isolated(self):
        for name in ("first", "second"):
            with self.subTest(family=name):
                family = self.root / name
                (family / "tests").mkdir(parents=True)
                (family / "helper.py").write_text("VALUE = " + repr(name) + "\n")
                (family / "tests" / "test_case.py").write_text(
                    "import unittest\nfrom pathlib import Path\nimport helper\n"
                    "class Check(unittest.TestCase):\n"
                    "    def test_context(self):\n"
                    "        self.assertEqual(helper.VALUE, " + repr(name) + ")\n"
                    "        self.assertEqual(Path.cwd(), Path(__file__).resolve().parents[1])\n")
                result = self.run_suite(family)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Ran 1 test", result.stderr)

    def test_main_propagates_empty_suite_exit(self):
        family = (self.root, {"id": "demo", "tests": ["tests"]})
        with mock.patch.object(catalog, "ROOT", self.root), \
                mock.patch.object(catalog, "definitions", return_value=[family]), \
                mock.patch.object(catalog, "requests", return_value=[]), \
                mock.patch.object(checks.subprocess, "run", return_value=subprocess.CompletedProcess([], 2)) as run:
            self.assertEqual(checks.main(["--family", "demo"]), 2)
            run.assert_called_once_with(checks.suite_command("tests"), cwd=self.root)


if __name__ == "__main__":
    unittest.main()
