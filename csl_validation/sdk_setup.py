"""Create a separate, pinned ALCF CSL client environment without submitting jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys

from .sdk import resolve_uv, sdk_argv, sdk_environment
from .watch import atomic_json, command, require

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "environments/alcf-csl/requirements.txt"
RECEIPT = "csl-sdk-setup.json"
PROBE = """import importlib.metadata as m, json, platform, sys
import numpy
from cerebras.appliance import logger
from cerebras.sdk.client import SdkCompiler, SdkLauncher
print(json.dumps({
    'executable': sys.executable, 'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
    'python': platform.python_version(), 'implementation': platform.python_implementation(),
    'system': platform.system(), 'machine': platform.machine(),
    'packages': {d.metadata['Name'].lower().replace('_', '-'): d.version
                 for d in m.distributions() if d.metadata['Name']},
    'imports': ['numpy', 'cerebras.appliance.logger',
                'cerebras.sdk.client.SdkCompiler', 'cerebras.sdk.client.SdkLauncher']
}))
"""


def setup_environment():
    env = sdk_environment()
    # Config discovery flags do not disable environment-variable overrides.
    # Preserve cache/interpreter storage and TLS trust, not installer targets,
    # constraints, hash bypasses, or existing-project selection.
    keep = {"UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_NATIVE_TLS"}
    for key in list(env):
        if key.startswith("UV_") and key not in keep:
            del env[key]
    return env


def target_path(value):
    path = Path(value).expanduser()
    require(path.is_absolute(), "SDK environment path must be absolute")
    require(not path.is_symlink(), "SDK environment cannot be a symlink")
    path = path.resolve()
    require(path != ROOT and not path.is_relative_to(ROOT), "SDK environment must be outside this checkout")
    require(not any((parent / "pyvenv.cfg").exists() for parent in path.parents),
            "SDK environment cannot be nested in another virtual environment")
    return path


def profile():
    data = PROFILE.read_bytes()
    pins = dict(re.findall(r"^([a-zA-Z0-9_.-]+)==([^\s\\;]+)", data.decode(), re.M))
    require(pins.get("cerebras-appliance") == "2.10.0" and
            pins.get("cerebras-sdk") == "2.10.0" and pins.get("numpy") == "1.25.0",
            "unexpected ALCF SDK profile")
    return data, pins


def verify(path, uv, env=None):
    env = setup_environment() if env is None else env
    require((path / "pyvenv.cfg").is_file(), "SDK path is not a virtual environment")
    interpreter = path / "bin/python"
    require(interpreter.is_file(), "SDK interpreter is missing")
    data, pins = profile()
    command([uv, "pip", "check", "--no-config", "--offline", "--no-python-downloads",
             "--python", str(interpreter)], env=env)
    details = json.loads(command(sdk_argv({"uv": uv, "python": str(interpreter)},
                                        ["-I", "-B", "-c", PROBE]), cwd=path, env=env))
    require(Path(details["prefix"]).resolve() == path and details["prefix"] != details["base_prefix"],
            "SDK interpreter did not use the selected virtual environment")
    require(details["implementation"] == "CPython" and details["python"].startswith("3.11."),
            "ALCF SDK profile requires CPython 3.11")
    require(details["system"] == "Linux" and details["machine"] == "x86_64",
            "ALCF SDK profile requires Linux x86_64")
    for name, version in pins.items():
        require(details["packages"].get(name) == version, "SDK package differs from lock: " + name)
    details.update(schema=1, profile="alcf-csl-2.10.0", status="ready",
                   requirements_sha256=hashlib.sha256(data).hexdigest(),
                   uv=command([uv, "--version"], env=env).strip(), jobs_submitted=False)
    return details


def create(path, python="3.11", uv=None, offline=False):
    path = target_path(path)
    require(not os.path.lexists(path), "SDK destination already exists; choose a new path")
    require(platform.system() == "Linux" and platform.machine() == "x86_64",
            "ALCF SDK profile requires Linux x86_64")
    uv, env = resolve_uv(uv), setup_environment()
    data, _ = profile()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve an empty directory atomically; concurrent creators fail here.
    # No --clear/--allow-existing: uv receives only our newly owned empty path.
    path.mkdir(mode=0o700)
    try:
        command([uv, "venv", "--no-project", "--no-config", "--no-python-downloads",
                 "--python", python, str(path)], env=env)
        interpreter = str(path / "bin/python")
        version = command(sdk_argv({"uv": uv, "python": interpreter}, ["-I", "-B", "-c",
                          "import platform; print(platform.python_implementation(), platform.python_version())"]), env=env)
        require(version.startswith("CPython 3.11."), "ALCF SDK profile requires CPython 3.11")
        lock = path / "csl-sdk-requirements.txt"
        lock.write_bytes(data)
        atomic_json(path / RECEIPT, dict(schema=1, status="installing", jobs_submitted=False))
        print("Installing pinned CSL SDK into " + str(path), flush=True)
        command([uv, "pip", "sync", "--no-config", "--no-python-downloads",
                 "--python", interpreter, "--require-hashes", "--only-binary", ":all:",
                 *(["--offline"] if offline else []), str(lock)], seconds=180, env=env)
        details = verify(path, uv, env)
        atomic_json(path / RECEIPT, details)
        return details
    except BaseException:
        # Retain the newly created environment for diagnosis; never remove an old one.
        atomic_json(path / RECEIPT, dict(schema=1, status="failed", jobs_submitted=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    new = sub.add_parser("create", help="create a new external SDK environment")
    new.add_argument("--path", required=True)
    new.add_argument("--python", default="3.11", help="installed CPython 3.11 or absolute interpreter path")
    new.add_argument("--offline", action="store_true", help="install only from uv's populated cache")
    check = sub.add_parser("check", help="check dependencies and imports without creating SDK clients")
    check.add_argument("--path", required=True)
    for action in (new, check):
        action.add_argument("--uv", help="explicit uv executable")
    args = parser.parse_args(argv)
    try:
        if args.action == "create":
            result = create(args.path, args.python, args.uv, args.offline)
        else:
            result = verify(target_path(args.path), resolve_uv(args.uv))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    sys.exit(main())
