#!/usr/bin/env python3
"""Collect finalized CSL bundles over an existing authenticated SSH connection."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import tempfile
import time

STATUS_LIMIT = 4 << 20
BUNDLE_LIMIT = 4 << 30
NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
CHECKSUM = re.compile(r"[0-9a-f]{64}\Z")
HOST = re.compile(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
STATES = {"pending", "running", "blocked", "passed", "failed", "interrupted"}
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def regular(path):
    require(not path.is_symlink() and path.is_file(), "expected regular file: " + str(path))


def validate_connection(host, state_dir):
    require(isinstance(host, str) and HOST.fullmatch(host), "use an SSH alias or user@hostname")
    path = PurePosixPath(state_dir)
    require(re.fullmatch(r"/[A-Za-z0-9_./-]+", state_dir) and
            str(path) == state_dir and ".." not in path.parts and path != PurePosixPath("/"),
            "state directory must be a simple absolute remote path without '..'")


def no_duplicate_keys(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key")
        value[key] = item
    return value


def timestamp(value):
    require(isinstance(value, str) and len(value) <= 64, "timestamp string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "timestamp requires timezone")


def parse_status(payload):
    require(len(payload) <= STATUS_LIMIT, "status exceeds 4 MiB")
    value = json.loads(payload, object_pairs_hook=no_duplicate_keys)
    require(isinstance(value, dict) and type(value.get("schema")) is int and value["schema"] == 1,
            "unsupported status schema")
    require(isinstance(value.get("cursor"), str) and SHA.fullmatch(value["cursor"]), "status cursor")
    timestamp(value.get("updated_at"))
    entries = value.get("requests")
    require(isinstance(entries, list) and len(entries) <= 10000, "status request list")
    seen = set()
    for entry in entries:
        require(isinstance(entry, dict), "request object")
        ident = entry.get("id")
        require(isinstance(ident, str) and NAME.fullmatch(ident) and ident not in seen, "invalid or duplicate request ID")
        seen.add(ident)
        require(isinstance(entry.get("sha"), str) and SHA.fullmatch(entry["sha"]), "request SHA")
        require(isinstance(entry.get("status"), str) and entry["status"] in STATES, "request status")
        require(type(entry.get("packaged", False)) is bool, "packaged boolean")
        if "finished_at" in entry:
            timestamp(entry["finished_at"])
        if "error" in entry:
            require(isinstance(entry["error"], str) and len(entry["error"]) <= 4096, "request error")
        if entry.get("packaged"):
            require(entry["status"] in {"passed", "failed", "interrupted"}, "nonterminal packaged request")
            require(entry.get("archive") == ident + ".zip", "archive must match request ID")
            require(isinstance(entry.get("bundle_sha256"), str) and CHECKSUM.fullmatch(entry["bundle_sha256"]), "bundle checksum")
            require(type(entry.get("bundle_bytes")) is int and 1 <= entry["bundle_bytes"] <= BUNDLE_LIMIT, "bundle size bound")
        else:
            require(not any(key in entry for key in ("archive", "bundle_sha256", "bundle_bytes")), "unfinalized bundle metadata")
    return value


def run_command(argv, *, seconds=30, limit=STATUS_LIMIT + 65536, target=None, target_limit=None):
    """Bound output/time and, for SCP, partial-file growth. Never prompt for MFA."""
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        started = time.monotonic()
        try:
            while True:
                rc = process.poll()
                require(time.monotonic() - started < seconds, "transfer command timed out")
                require(os.fstat(stdout.fileno()).st_size + os.fstat(stderr.fileno()).st_size <= limit, "transfer command output limit")
                if target is not None and target.exists():
                    regular(target)
                    require(target.stat().st_size <= target_limit, "download exceeds declared size")
                if rc is not None:
                    break
                time.sleep(0.1)
            stderr.seek(0)
            detail = stderr.read(4096).decode(errors="replace")
            require(rc == 0, "SSH/SCP failed; polling stopped. Reauthenticate manually if needed. " + detail)
            stdout.seek(0)
            return stdout.read(limit + 1)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=2)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=2)


def atomic_bytes(path, payload):
    if path.exists() or path.is_symlink():
        regular(path)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def same_bundle(path, entry):
    regular(path)
    require(path.stat().st_size == entry["bundle_bytes"] and digest(path) == entry["bundle_sha256"],
            "local/downloaded bundle identity mismatch: " + path.name)


def collect(host, state_dir, output):
    validate_connection(host, state_dir)
    # Do not silently fall back to a fresh authentication attempt when no master exists.
    run_command(["ssh", *SSH_OPTIONS, "-O", "check", host], limit=65536)
    payload = run_command(["ssh", *SSH_OPTIONS, host,
                           "head -c " + str(STATUS_LIMIT + 1) + " " + state_dir + "/status.json"])
    status = parse_status(payload)
    snapshot = output / "status.json"
    if snapshot.exists() or snapshot.is_symlink():
        regular(snapshot)
        require(snapshot.stat().st_size <= STATUS_LIMIT, "existing status size")
        previous = {entry["id"]: entry for entry in parse_status(snapshot.read_bytes())["requests"]}
        for entry in status["requests"]:
            old = previous.get(entry["id"])
            if old:
                require(old["sha"] == entry["sha"], "request ID changed source SHA")
                if old.get("packaged"):
                    require(all(old[key] == entry.get(key) for key in ("packaged", "archive", "bundle_bytes", "bundle_sha256")),
                            "finalized request changed")
    downloaded = []
    for entry in status["requests"]:
        if not entry.get("packaged"):
            continue
        target = output / entry["archive"]
        checksum = output / (entry["archive"] + ".sha256")
        checksum_text = (entry["bundle_sha256"] + "  " + entry["archive"] + "\n").encode()
        if checksum.exists() or checksum.is_symlink():
            regular(checksum)
            require(checksum.stat().st_size == len(checksum_text) and checksum.read_bytes() == checksum_text,
                    "local checksum collision")
        if target.exists() or target.is_symlink():
            same_bundle(target, entry)
        else:
            with tempfile.TemporaryDirectory(prefix=".download-", dir=output) as work:
                partial = Path(work) / (entry["archive"] + ".partial")
                run_command(["ssh", *SSH_OPTIONS, "-O", "check", host], limit=65536)
                run_command(["scp", *SSH_OPTIONS, host + ":" + state_dir + "/exports/" + entry["archive"], str(partial)],
                            seconds=1800, limit=65536, target=partial, target_limit=entry["bundle_bytes"])
                same_bundle(partial, entry)
                with partial.open("rb") as stream:
                    os.fsync(stream.fileno())
                require(not target.exists() and not target.is_symlink(), "bundle destination appeared during transfer")
                os.replace(partial, target)
                downloaded.append(entry["id"])
        atomic_bytes(checksum, checksum_text)
    # A local published snapshot only advertises bundles already verified locally.
    atomic_bytes(snapshot, (json.dumps(status, indent=2, sort_keys=True) + "\n").encode())
    return status, downloaded


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    require(output.absolute() == output.resolve() and output.is_dir(), "output directory cannot contain symlinks")
    descriptor = os.open(output / ".fetch.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="SSH alias with ControlPath, or user@hostname")
    parser.add_argument("--state-dir", required=True, help="absolute remote path on shared home")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        validate_connection(args.host, args.state_dir)
        require(10 <= args.interval <= 86400, "poll interval must be 10..86400 seconds")
        output = args.out.absolute()
        with output_lock(output):
            while True:
                status, downloaded = collect(args.host, args.state_dir, output)
                print(json.dumps(dict(updated_at=status["updated_at"], downloaded=downloaded,
                                      requests=[dict(id=e["id"], status=e["status"]) for e in status["requests"]])), flush=True)
                if not args.watch:
                    return 0
                time.sleep(args.interval)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print("error: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
