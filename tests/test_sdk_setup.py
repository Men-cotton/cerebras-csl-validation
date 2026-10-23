"""Offline SDK setup contracts; all uv processes and SDK modules are replaced."""
import contextlib
import copy
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from csl_validation import sdk_setup as setup


class SetupFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkout = self.root / "controller"
        self.checkout.mkdir()
        self.target = self.root / "fresh-sdk"
        self.uv = self.root / "uv"
        self.uv.write_text("#!/bin/sh\nexit 77\n")
        self.uv.chmod(0o700)
        self.calls = []
        self.failure = None
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(setup, "ROOT", self.checkout).start()
        mock.patch.object(setup.platform, "system", return_value="Linux").start()
        mock.patch.object(setup.platform, "machine", return_value="x86_64").start()
        self.command = mock.patch.object(setup, "command", side_effect=self.fake_command).start()

    def installed_details(self):
        return dict(executable=str(self.target / "bin/python"), prefix=str(self.target),
                    base_prefix="/base/python", python="3.11.13", implementation="CPython",
                    system="Linux", machine="x86_64", packages=dict(setup.profile()[1]),
                    imports=["numpy", "cerebras.appliance.logger",
                             "cerebras.sdk.client.SdkCompiler", "cerebras.sdk.client.SdkLauncher"])

    def fake_command(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if argv[1] == "venv":
            self.assertEqual(list(self.target.iterdir()), [])
            self.assertNotIn("--allow-existing", argv)
            self.assertNotIn("--clear", argv)
            if self.failure == "venv":
                raise ValueError("simulated venv failure")
            (self.target / "bin").mkdir()
            (self.target / "bin/python").write_text("fake interpreter")
            (self.target / "pyvenv.cfg").write_text("home = /base/python\n")
            return ""
        if argv[1:3] == ["pip", "sync"]:
            if self.failure == "sync":
                raise ValueError("simulated install failure")
            return ""
        if argv[1:3] == ["pip", "check"]:
            if self.failure == "check":
                raise ValueError("simulated incompatible dependency")
            return ""
        if argv[-1] == setup.PROBE:
            if self.failure == "probe":
                raise ValueError("simulated missing SDK import")
            return json.dumps(self.installed_details())
        if argv[1:] == ["--version"]:
            return "uv 0.7.13\n"
        self.assertIn("platform.python_implementation()", argv[-1])
        return "CPython 3.11.13\n"

    def create(self, **kwargs):
        return setup.create(self.target, uv=str(self.uv), **kwargs)

    def make_existing_environment(self):
        self.target.mkdir()
        (self.target / "bin").mkdir()
        (self.target / "bin/python").write_text("old Python")
        (self.target / "pyvenv.cfg").write_text("home = old\n")


class CreationTests(SetupFixture):
    def test_existing_environment_and_files_are_preserved_without_any_command(self):
        self.make_existing_environment()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.create()
        self.assertEqual((self.target / "bin/python").read_text(), "old Python")
        self.command.assert_not_called()
        file_target = self.root / "existing-file"
        file_target.write_bytes(b"retain me")
        with self.assertRaisesRegex(ValueError, "already exists"):
            setup.create(file_target, uv=str(self.uv))
        self.assertEqual(file_target.read_bytes(), b"retain me")
        self.command.assert_not_called()

    def test_existing_and_dangling_symlinks_are_preserved(self):
        old = self.root / "old-sdk"
        old.mkdir()
        marker = old / "marker"
        marker.write_bytes(b"unchanged")
        for destination in (old, self.root / "missing-sdk"):
            with self.subTest(destination=destination):
                self.target.symlink_to(destination, target_is_directory=True)
                try:
                    with self.assertRaisesRegex(ValueError, "symlink"):
                        self.create()
                    self.assertEqual(os.readlink(self.target), str(destination))
                    self.assertEqual(marker.read_bytes(), b"unchanged")
                    self.command.assert_not_called()
                finally:
                    self.target.unlink()

    def test_relative_checkout_and_nested_environment_targets_rejected(self):
        old = self.root / "old-sdk"
        old.mkdir()
        (old / "pyvenv.cfg").write_text("home = old\n")
        for target in (Path("relative-sdk"), self.checkout, self.checkout / "sdk", old / "new-sdk"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                setup.create(target, uv=str(self.uv))
        self.command.assert_not_called()
        self.assertFalse((old / "new-sdk").exists())

    def test_parent_symlink_cannot_hide_checkout_or_nested_environment(self):
        alias = self.root / "alias"
        alias.symlink_to(self.checkout, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside"):
            setup.create(alias / "sdk", uv=str(self.uv))
        self.command.assert_not_called()

    def test_new_environment_uses_explicit_target_hashed_lock_and_final_receipt(self):
        details = self.create(offline=True)
        self.assertEqual(details["status"], "ready")
        self.assertFalse(details["jobs_submitted"])
        self.assertEqual(json.loads((self.target / setup.RECEIPT).read_text()), details)
        self.assertEqual((self.target / "csl-sdk-requirements.txt").read_bytes(), setup.profile()[0])
        for argv, _ in self.calls:
            if argv[1:3] in (["pip", "sync"], ["pip", "check"]):
                self.assertEqual(argv[argv.index("--python") + 1], str(self.target / "bin/python"))
                self.assertIn("--no-config", argv)
        sync = next(argv for argv, _ in self.calls if argv[1:3] == ["pip", "sync"])
        self.assertIn("--require-hashes", sync)
        self.assertIn("--offline", sync)
        self.assertEqual(sync[sync.index("--only-binary") + 1], ":all:")
        probe = next(argv for argv, _ in self.calls if argv[-1] == setup.PROBE)
        for option in ("--no-project", "--offline", "-I", "-B"):
            self.assertIn(option, probe)

    def test_failures_retain_environment_but_never_mark_it_ready(self):
        for stage in ("venv", "sync", "check", "probe"):
            with self.subTest(stage=stage):
                self.target = self.root / ("failed-" + stage)
                self.failure = stage
                with self.assertRaisesRegex(ValueError, "simulated"):
                    self.create()
                receipt = json.loads((self.target / setup.RECEIPT).read_text())
                self.assertEqual(receipt["status"], "failed")
                self.assertFalse(receipt["jobs_submitted"])
                self.assertTrue(self.target.is_dir())
                calls_before = len(self.calls)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    self.create()
                self.assertEqual(len(self.calls), calls_before)

    def test_unsupported_host_rejected_before_creating_destination(self):
        with mock.patch.object(setup.platform, "machine", return_value="aarch64"):
            with self.assertRaisesRegex(ValueError, "Linux x86_64"):
                self.create()
        self.assertFalse(self.target.exists())
        self.command.assert_not_called()


class VerificationTests(SetupFixture):
    def setUp(self):
        super().setUp()
        self.make_existing_environment()

    def test_checks_all_transitive_lock_pins(self):
        original = self.installed_details()
        transitive = next(name for name in original["packages"]
                          if name not in {"cerebras-appliance", "cerebras-sdk", "numpy"})
        for replacement in (None, "0.0.0-wrong"):
            details = copy.deepcopy(original)
            if replacement is None:
                del details["packages"][transitive]
            else:
                details["packages"][transitive] = replacement
            with self.subTest(replacement=replacement), \
                    mock.patch.object(self, "installed_details", return_value=details), \
                    self.assertRaisesRegex(ValueError, "differs from lock: " + transitive):
                setup.verify(self.target, str(self.uv))
        self.assertFalse((self.target / setup.RECEIPT).exists())

    def test_wrong_python_platform_or_virtual_environment_is_rejected(self):
        original = self.installed_details()
        invalid = [dict(prefix="/another-env"), dict(base_prefix=str(self.target)),
                   dict(implementation="PyPy"), dict(python="3.12.1"),
                   dict(system="Darwin"), dict(machine="aarch64")]
        for change in invalid:
            details = dict(original, **change)
            with self.subTest(change=change), \
                    mock.patch.object(self, "installed_details", return_value=details), \
                    self.assertRaises(ValueError):
                setup.verify(self.target, str(self.uv))

    def test_non_venv_rejected_before_probe(self):
        (self.target / "pyvenv.cfg").unlink()
        with self.assertRaisesRegex(ValueError, "not a virtual environment"):
            setup.verify(self.target, str(self.uv))
        self.command.assert_not_called()


class IsolationTests(unittest.TestCase):
    def test_installer_environment_drops_uv_overrides_but_keeps_site_transport(self):
        hostile = dict(VIRTUAL_ENV="old", UV_PROJECT_ENVIRONMENT="old", PYTHONHOME="old",
                       PYTHONPATH="old", UV_PYTHON="old", UV_SYSTEM_PYTHON="1",
                       UV_NO_VERIFY_HASHES="1", UV_VENV_CLEAR="1", UV_CONSTRAINT="bad",
                       UV_PROJECT="old", UV_INDEX="wrong", UV_CONFIG_FILE="wrong",
                       UV_CACHE_DIR="cache", UV_PYTHON_INSTALL_DIR="interpreters", UV_NATIVE_TLS="1",
                       HTTPS_PROXY="site-proxy", SSL_CERT_FILE="site-cert", CS_CONFIG="site-config")
        with mock.patch.dict(os.environ, hostile, clear=True):
            actual = setup.setup_environment()
        for key in hostile:
            if key.startswith("UV_") and key not in {"UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_NATIVE_TLS"}:
                self.assertNotIn(key, actual)
        for key in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
            self.assertNotIn(key, actual)
        for key in ("HTTPS_PROXY", "SSL_CERT_FILE", "CS_CONFIG", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR"):
            self.assertEqual(actual[key], hostile[key])

    def test_import_probe_does_not_construct_sdk_clients(self):
        modules = {name: ModuleType(name) for name in
                   ("numpy", "cerebras", "cerebras.appliance", "cerebras.sdk", "cerebras.sdk.client")}
        modules["cerebras.appliance"].logger = object()
        compiler = mock.Mock(side_effect=AssertionError("compiler constructed"))
        launcher = mock.Mock(side_effect=AssertionError("launcher constructed"))
        modules["cerebras.sdk.client"].SdkCompiler = compiler
        modules["cerebras.sdk.client"].SdkLauncher = launcher
        distribution = SimpleNamespace(metadata={"Name": "cerebras_sdk"}, version="2.10.0")
        output = io.StringIO()
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(importlib.metadata, "distributions", return_value=[distribution]), \
                contextlib.redirect_stdout(output):
            exec(setup.PROBE, {})
        self.assertEqual(json.loads(output.getvalue())["packages"], {"cerebras-sdk": "2.10.0"})
        compiler.assert_not_called()
        launcher.assert_not_called()


if __name__ == "__main__":
    unittest.main()
