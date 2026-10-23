"""Offline uv execution contracts; no real SDK imports or network operations."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import venv

from csl_validation.sdk import resolve_uv, sdk_argv, sdk_environment


def snapshot(root):
    result = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            result[relative] = ("symlink", os.readlink(path))
        elif path.is_file():
            result[relative] = ("file", hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            result[relative] = ("directory",)
    return result


def process_live(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = Path("/proc") / str(pid) / "stat"
    try:
        # Killed orphan descendants may briefly remain as zombies until reaped.
        return status.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


class ArgumentTests(unittest.TestCase):
    def test_missing_uv_fails_before_execution(self):
        with mock.patch("csl_validation.sdk.shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "uv executable not found"):
                resolve_uv("absent-uv")

    def test_uv_path_and_sdk_interpreter_symlinks_are_not_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "real-uv"
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o700)
            alias = root / "uv"
            alias.symlink_to(executable)
            python = str(root / "sdk/bin/python")
            self.assertEqual(resolve_uv(alias), str(alias))
            argv = sdk_argv({"uv": str(alias), "python": python}, ["-B", "probe.py"])
            self.assertEqual(argv[0], str(alias))
            self.assertEqual(argv[argv.index("--python") + 1], python)
            self.assertEqual(argv[argv.index("--") + 1:], [python, "-B", "probe.py"])
            for flag in ("--no-project", "--no-sync", "--offline", "--no-python-downloads", "--no-config"):
                self.assertIn(flag, argv)

    def test_environment_removes_controller_paths_and_keeps_site_settings(self):
        original = dict(VIRTUAL_ENV="controller", UV_PROJECT_ENVIRONMENT="controller",
                        PYTHONHOME="wrong", PYTHONPATH="wrong", HTTPS_PROXY="site-proxy",
                        CS_CONFIG="site-config", OMP_NUM_THREADS="8")
        result = sdk_environment(original)
        for key in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONHOME", "PYTHONPATH"):
            self.assertNotIn(key, result)
        self.assertEqual(result["HTTPS_PROXY"], "site-proxy")
        self.assertEqual(result["CS_CONFIG"], "site-config")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertEqual(result[key], "1")
        self.assertEqual(original["OMP_NUM_THREADS"], "8")


class RealUvTests(unittest.TestCase):
    def setUp(self):
        self.uv = resolve_uv()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sdk = self.root / "sdk"
        self.clone = self.root / "clone"
        self.clone.mkdir()
        venv.EnvBuilder(with_pip=False, symlinks=True).create(self.sdk)
        self.python = str(self.sdk / "bin/python")
        self.config = dict(uv=self.uv, python=self.python)
        self.env = sdk_environment(dict(os.environ, UV_CACHE_DIR=str(self.root / "uv-cache")))
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"

    def wait_record(self, path, process):
        deadline = time.monotonic() + 10
        while not path.exists():
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                self.fail("uv child exited before readiness: " + repr((stdout, stderr)))
            if time.monotonic() > deadline:
                self.fail("uv child readiness timeout")
            time.sleep(0.05)
        return json.loads(path.read_text())

    def stop(self, process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)
        process.communicate()

    def test_selected_sdk_wins_without_syncing_or_modifying_clone(self):
        controller = self.root / "controller"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(controller)
        clone_env = self.clone / ".venv"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(clone_env)
        site = next((self.sdk / "lib").glob("python*/site-packages"))
        (site / "fake_sdk_marker.py").write_text("VALUE = 'selected-sdk'\n")
        (self.clone / "pyproject.toml").write_text(
            '[project]\nname="do-not-sync"\nversion="0.0.0"\n'
            'dependencies=["nonexistent-csl-probe-package==999999"]\n')
        (self.clone / "uv.lock").write_text("invalid lockfile: must not be read\n")
        (self.clone / "uv.toml").write_text("invalid configuration: must not be read\n")
        script = self.clone / "probe.py"
        script.write_text("import json,sys,fake_sdk_marker\n"
                          "print(json.dumps({'python':sys.executable,'prefix':sys.prefix,'marker':fake_sdk_marker.VALUE}))\n")
        base = dict(self.env, VIRTUAL_ENV=str(controller), UV_PROJECT_ENVIRONMENT=str(controller),
                    PYTHONHOME=str(controller), PYTHONPATH=str(controller))
        before_clone, before_sdk, before_controller = snapshot(self.clone), snapshot(self.sdk), snapshot(controller)
        result = subprocess.run(sdk_argv(self.config, ["-B", str(script)]), cwd=self.clone,
                                env=sdk_environment(base), capture_output=True, text=True, timeout=15, check=True)
        value = json.loads(result.stdout)
        self.assertEqual(value, dict(python=self.python, prefix=str(self.sdk), marker="selected-sdk"))
        self.assertEqual(snapshot(self.clone), before_clone)
        self.assertEqual(snapshot(self.sdk), before_sdk)
        self.assertEqual(snapshot(controller), before_controller)

    def test_uv_child_keeps_inherited_operator_lock(self):
        record = self.root / "ready.json"
        lock = self.root / "operator.lock"
        script = self.clone / "lock_probe.py"
        script.write_text("import json,os,pathlib,sys,time\n"
                          "fd=int(sys.argv[1]); os.fstat(fd)\n"
                          "pathlib.Path(sys.argv[2]).write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp()}))\n"
                          "time.sleep(60)\n")
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = None
        try:
            process = subprocess.Popen(sdk_argv(self.config, ["-B", str(script), str(fd), str(record)]),
                                       cwd=self.clone, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       process_group=0, pass_fds=(fd,))
            info = self.wait_record(record, process)
            self.assertEqual(info["pgid"], process.pid)
            os.close(fd)
            fd = None
            competing = os.open(lock, os.O_RDWR)
            try:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.stop(process)
                fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(competing)
        finally:
            if fd is not None:
                os.close(fd)
            if process is not None and process.poll() is None:
                self.stop(process)

    def test_watcher_timeout_terminates_uv_and_descendant_group(self):
        from csl_validation import watch
        record = self.root / "processes.json"
        script = self.clone / "descendant_probe.py"
        script.write_text("import json,os,pathlib,subprocess,sys,time\n"
                          "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
                          "pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'child':child.pid,'pgid':os.getpgrp()}))\n"
                          "time.sleep(60)\n")
        with self.assertRaisesRegex(ValueError, "timed out"):
            watch.command(sdk_argv(self.config, ["-B", str(script), str(record)]),
                          cwd=self.clone, env=self.env, seconds=2)
        info = json.loads(record.read_text())
        deadline = time.monotonic() + 3
        while any(process_live(info[key]) for key in ("pid", "child")) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(process_live(info["pid"]))
        self.assertFalse(process_live(info["child"]))


if __name__ == "__main__":
    unittest.main()
