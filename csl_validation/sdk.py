"""Run an existing site SDK environment through uv without synchronizing it."""
import os
from pathlib import Path
import shutil


def resolve_uv(value=None):
    """Pin uv's executable path without resolving a caller's symlink."""
    requested = "uv" if value is None else os.fspath(value)
    if not requested:
        raise ValueError("uv executable is empty")
    found = shutil.which(requested)
    if found is None:
        raise ValueError("uv executable not found or not executable: " + requested)
    path = Path(found).absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("uv executable is not a regular executable file: " + str(path))
    return str(path)


def sdk_argv(config, args):
    """Keep the SDK interpreter explicit even when another venv is active.

    The command is the interpreter, not a .py script interpreted by uv. This
    prevents a request script's inline dependency metadata from selecting or
    creating an environment. The venv's bin/python path must not be resolved
    to its base interpreter.
    """
    python = config["python"]
    return [resolve_uv(config.get("uv")), "run", "--no-project", "--no-sync",
            "--offline", "--no-python-downloads", "--no-config", "--python",
            python, "--", python, *args]


def sdk_environment(base=None):
    """Remove controller Python selection while retaining site connection data."""
    result = dict(os.environ if base is None else base)
    for key in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONHOME", "PYTHONPATH"):
        result.pop(key, None)
    result.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  NUMEXPR_NUM_THREADS="1")
    return result
