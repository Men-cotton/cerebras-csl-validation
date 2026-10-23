#!/usr/bin/env python3
"""SHA-pinned, serial CSL requests. Standard library; SDK stays in child Python."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile

SCHEMA = 1
NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
JOB = re.compile(r"\bJob id:\s*(wsjob-[A-Za-z0-9-]+)")
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED", "CANCELED"}
ACTIVE = {"CREATED", "QUEUED", "QUEUEING", "PENDING", "SCHEDULED", "INITIALIZING", "RUNNING", "EXECUTING", "FINALIZING", "CANCELLING", "CANCELING"}
DEFAULT_SCRIPTS = ["appliance.py", "tank_appliance.py", "mesh_appliance.py", "clock_appliance.py"]
LEDGER_LIMIT = 32 << 20
ADMISSION_LIMIT = 16 << 20


def encoded_size(value):
    return len((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def require(test, message):
    if not test:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path, limit=4 << 20):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, "invalid/oversize JSON: " + str(path))
    return json.loads(path.read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def command(argv, cwd=None, seconds=30, env=None, limit=8 << 20):
    """Bound command output and wall time, including Git and csctl outages."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=err, process_group=0)
        started = time.monotonic()
        try:
            while process.poll() is None:
                require(time.monotonic() - started < seconds, "command timed out: " + argv[0])
                require(os.fstat(out.fileno()).st_size + os.fstat(err.fileno()).st_size <= limit, "command output limit")
                time.sleep(0.05)
            require(os.fstat(out.fileno()).st_size + os.fstat(err.fileno()).st_size <= limit, "command output limit")
            out.seek(0)
            err.seek(0)
            stdout, stderr = out.read(), err.read()
            require(process.returncode == 0, argv[0] + " failed: " + stderr.decode(errors="replace")[-1000:])
            return stdout.decode()
        finally:
            if process.poll() is None:
                stop_group(process)


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # Descendants may remain even after the direct child exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def git(args, cwd=None, seconds=120):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="/bin/false")
    return command(["git", "-c", "core.hooksPath=/dev/null", *args], cwd, seconds, env)


def contained(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), "expected relative path")
    require(".." not in Path(relative).parts and not relative.startswith("-"), "unsafe relative path")
    root = Path(root).resolve()
    path = root / relative
    require(path.resolve().is_relative_to(root) and path.resolve() == path, "symlink or path escape")
    return path


def manifest(value, allowed):
    require(isinstance(value, dict) and value.get("schema") == SCHEMA, "manifest schema")
    require(encoded_size(value) <= 64 << 10, "manifest exceeds 64 KiB")
    require(isinstance(value.get("id"), str) and NAME.fullmatch(value["id"]), "request ID")
    steps = value.get("steps")
    require(isinstance(steps, list) and 1 <= len(steps) <= 64, "step count")
    names = set()
    for step in steps:
        require(isinstance(step, dict) and isinstance(step.get("name"), str) and NAME.fullmatch(step["name"]), "step name")
        require(step["name"] not in names, "duplicate step name")
        names.add(step["name"])
        require(step.get("script") in allowed, "script not in operator allowlist")
        require(step.get("kind") in {"appliance", "host"}, "step kind")
        require(type(step.get("timeout_seconds")) is int and 1 <= step["timeout_seconds"] <= 86400, "step timeout")
        args = step.get("args")
        require(isinstance(args, list) and len(args) <= 128 and all(isinstance(a, str) and len(a) <= 4096 and "\0" not in a for a in args), "step argv")
        require(not step["script"].endswith(("_campaign.py", "_followup.py")), "nested legacy campaigns are not single-action steps")
    return value


def trailers(message):
    # Only the final paragraph is a trailer block; prose is never executable.
    block = message.strip().split("\n\n")[-1]
    found = {}
    for line in block.splitlines():
        if line.startswith(("ALCF-Request:", "ALCF-Manifest:")):
            key, value = line.split(":", 1)
            require(key not in found, "duplicate ALCF trailer")
            found[key] = value.strip()
    if not found:
        return None
    require(set(found) == {"ALCF-Request", "ALCF-Manifest"}, "incomplete ALCF trailers")
    require(NAME.fullmatch(found["ALCF-Request"]), "invalid trailer request ID")
    path = found["ALCF-Manifest"]
    require(re.fullmatch(r"requests/[a-zA-Z0-9_./-]+\.json", path) and ".." not in Path(path).parts, "invalid manifest path")
    return found["ALCF-Request"], path


def job_record(value, namespace, username):
    require(isinstance(value, dict), "invalid job object")
    meta, spec, status = value.get("meta", {}), value.get("spec", {}), value.get("status", {})
    require(isinstance(meta, dict) and isinstance(spec, dict) and isinstance(status, dict), "job sections")
    name, ns, phase = meta.get("name"), meta.get("namespace"), status.get("phase")
    require(isinstance(name, str) and len(name) <= 128 and re.fullmatch(r"wsjob-[A-Za-z0-9-]+", name), "job name")
    require(ns == namespace, "unexpected job namespace")
    require(isinstance(spec.get("user"), dict) and spec["user"].get("username") == username, "unexpected job owner")
    require(phase in TERMINAL | ACTIVE, "unknown job phase: " + str(phase))
    return {"name": name, "namespace": ns, "phase": phase}


def parse_jobs(value, namespace, username, maximum=5000):
    require(isinstance(value, dict) and isinstance(value.get("items"), list), "expected csctl items array")
    require(len(value["items"]) < maximum, "job list may be truncated")
    require(not any(value.get(k) for k in ("continue", "next", "nextPageToken")), "paginated job list")
    metadata = value.get("metadata", {})
    require(isinstance(metadata, dict) and not metadata.get("continue"), "paginated job list metadata")
    result = [job_record(item, namespace, username) for item in value["items"]]
    require(len({(item["namespace"], item["name"]) for item in result}) == len(result), "duplicate job identity")
    return result


class Scheduler:
    def __init__(self, config):
        self.config = config

    def call(self, args):
        return json.loads(command([self.config["csctl"], "-n", self.config["namespace"], "get", *args, "-o", "json"], seconds=20))

    def jobs(self):
        return parse_jobs(self.call(["jobs", "--my-jobs", "--max-jobs", "5000"]), self.config["namespace"], self.config["username"])

    def idle(self):
        return all(job["phase"] in TERMINAL for job in self.jobs())

    def job(self, name):
        result = self.call(["job", name])
        if isinstance(result, dict) and "items" in result:
            values = parse_jobs(result, self.config["namespace"], self.config["username"])
            require(len(values) == 1, "individual job not found")
            item = values[0]
        else:
            item = job_record(result, self.config["namespace"], self.config["username"])
        require(item["name"] == name, "individual job identity mismatch")
        return item


def files_under(root, byte_limit, entry_limit=20000):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "invalid output root")
    def scan_error(error):
        raise error
    result, count, total, start = [], 0, 0, time.monotonic()
    for current, directories, files in os.walk(root, followlinks=False, onerror=scan_error):
        for name in directories + files:
            count += 1
            require(count <= entry_limit and time.monotonic() - start < 15, "output inventory bound")
            path = Path(current) / name
            require(not path.is_symlink(), "output symlink")
            if path.is_file():
                total += path.stat().st_size
                require(total <= byte_limit, "output byte limit")
                result.append(path)
            else:
                require(path.is_dir(), "nonregular output")
    return sorted(result)


def job_ids(root, limit):
    result = set()
    log_bytes = 0
    for path in files_under(root, limit):
        if path.suffix != ".log":
            continue
        log_bytes += path.stat().st_size
        require(log_bytes <= 128 << 20, "log scan limit")
        result.update(JOB.findall(path.read_text(errors="replace")))
    require(len(result) <= 2048 and all(len(name) <= 128 for name in result), "job ID inventory limit")
    return sorted(result)


@contextmanager
def operator_lock(config):
    path = Path(config["lock_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    finally:
        os.close(fd)


class Watcher:
    def __init__(self, config, scheduler=None, lock_fd=None):
        self.config = config
        self.root = Path(config["state_dir"])
        self.cache = self.root / "source.git"
        self.ledger = read_json(self.root / "ledger.json", LEDGER_LIMIT)
        self.scheduler = scheduler or Scheduler(config)
        self.lock_fd = lock_fd

    def save(self):
        require(encoded_size(self.ledger) <= LEDGER_LIMIT, "ledger size limit; preserve state for operator recovery")
        atomic_json(self.root / "ledger.json", self.ledger)
        records = []
        for ident in self.ledger["order"]:
            entry = self.ledger["requests"][ident]
            public = {key: entry[key] for key in ("id", "sha", "status", "finished_at", "bundle_sha256", "bundle_bytes", "packaged") if key in entry}
            if entry.get("packaged"):
                public["archive"] = Path(entry["bundle"]).name
            records.append(public)
        atomic_json(self.root / "status.json", dict(schema=SCHEMA, cursor=self.ledger["cursor"], updated_at=now(), requests=records))

    def record(self, entry):
        atomic_json(self.root / "requests" / entry["id"] / "receipt.json", entry)
        self.save()

    def fetch(self):
        git(["--git-dir", str(self.cache), "fetch", "--no-tags", self.config["repo"], "+refs/heads/" + self.config["branch"] + ":refs/heads/watched"])
        return git(["--git-dir", str(self.cache), "rev-parse", "refs/heads/watched"]).strip()

    def discover(self):
        tip = self.fetch()
        cursor = self.ledger["cursor"]
        git(["--git-dir", str(self.cache), "merge-base", "--is-ancestor", cursor, tip])
        commits = git(["--git-dir", str(self.cache), "rev-list", "--topo-order", "--reverse", cursor + ".." + tip]).splitlines()
        require(len(commits) <= 1000, "discovery commit limit")
        pending = {}
        for sha in commits:
            info = trailers(git(["--git-dir", str(self.cache), "show", "-s", "--format=%B", sha]))
            if info:
                ident, path = info
                require(ident not in self.ledger["requests"] and ident not in pending, "reused request ID: " + ident)
                value = json.loads(git(["--git-dir", str(self.cache), "show", sha + ":" + path]))
                manifest(value, self.config["allowed_scripts"])
                require(value["id"] == ident, "trailer/manifest ID mismatch")
                entry = dict(id=ident, sha=sha, manifest_path=path, manifest=value,
                             status="pending", created_at=now(), steps=[], notified=False)
                pending[ident] = entry
        # A DAG traversal is not an ancestry chain. Commit queue and tip atomically.
        projected = dict(self.ledger, requests=dict(self.ledger["requests"], **pending), order=self.ledger["order"] + list(pending))
        require(len(projected["order"]) <= 1024 and encoded_size(projected) <= ADMISSION_LIMIT,
                "metadata admission limit; archive completed controller state before admitting more requests")
        self.ledger["requests"].update(pending)
        self.ledger["order"].extend(pending)
        self.ledger["cursor"] = tip
        self.ledger["last_check"] = time.time()
        self.save()

    def prepare(self, entry):
        require(encoded_size(self.ledger) <= ADMISSION_LIMIT, "metadata admission limit; retain state and inspect pending requests before rotating controller")
        require(shutil.disk_usage(self.root).free >= (1 << 30), "less than 1 GiB free; collect/clean completed artifacts")
        files_under(self.root, self.config.get("state_bytes", 32 << 30), entry_limit=100000)
        root = self.root / "requests" / entry["id"]
        root.mkdir(parents=True, exist_ok=True)
        src = root / "src"
        # Preparation interrupted before launch is safe to resume only if its clone is valid.
        if not src.exists():
            git(["clone", "--no-hardlinks", "--no-checkout", str(self.cache), str(src)])
        git(["checkout", "--detach", entry["sha"]], src)
        require(git(["status", "--porcelain"], src) == "", "request clone is dirty")
        source_manifest = contained(src, entry["manifest_path"])
        require(read_json(source_manifest) == entry["manifest"], "manifest changed during clone")
        for step in entry["manifest"]["steps"]:
            require(contained(src, step["script"]).is_file(), "script missing")
        (root / "output").mkdir(exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        atomic_json(root / "request.json", entry["manifest"])
        environment = command([self.config["python"], "-c", "import sys,json,importlib.metadata as m; names=('numpy','cerebras_appliance','cerebras_sdk'); installed={d.metadata['Name'].lower().replace('-','_'):d.version for d in m.distributions() if d.metadata['Name']}; print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':{n:installed.get(n) for n in names}}))"])
        atomic_json(root / "environment.json", json.loads(environment))
        entry.update(status="running", manifest_sha256=digest(source_manifest), started_at=now())
        self.record(entry)

    def start_step(self, entry, step):
        root = self.root / "requests" / entry["id"]
        src, output = root / "src", root / "output"
        require(git(["rev-parse", "HEAD"], src).strip() == entry["sha"], "clone revision changed")
        require(git(["diff", "--name-only", "HEAD"], src) == "", "tracked source modified")
        before = job_ids(output, self.config["output_bytes"])
        argv = [self.config["python"], "-B", str(contained(src, step["script"]))]
        argv += [arg.replace("{output}", str(output)) for arg in step["args"]]
        require(encoded_size(argv) <= 64 << 10, "expanded step argv exceeds 64 KiB")
        record = dict(name=step["name"], status="running", argv=argv, started_at=now(), jobs=[])
        entry["steps"].append(record)
        self.record(entry)  # Intent is durable BEFORE Popen.
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   CSL_REQUEST_ID=entry["id"], CSL_SOURCE_SHA=entry["sha"], CSL_OUTPUT_DIR=str(output))
        log = root / "logs" / (step["name"] + ".log")
        process = None
        try:
            with log.open("xb") as stream:
                process = subprocess.Popen(argv, cwd=src, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                           process_group=0, pass_fds=(() if self.lock_fd is None else (self.lock_fd,)))
                record["pid"] = process.pid
                self.record(entry)
                began, checked = time.monotonic(), 0
                while process.poll() is None:
                    require(time.monotonic() - began < step["timeout_seconds"], "client timeout; remote stop unconfirmed")
                    require(log.stat().st_size <= 64 << 20, "client log limit")
                    if time.monotonic() - checked >= self.config["poll_seconds"]:
                        observed = sorted((set(job_ids(output, self.config["output_bytes"])) - set(before)) | set(JOB.findall(log.read_text(errors="replace"))))
                        require(len(observed) <= 32 and all(len(name) <= 128 for name in observed), "single-action job ID limit")
                        record["jobs"] = observed
                        self.record(entry)
                        checked = time.monotonic()
                    time.sleep(0.2)
                record["exit_code"] = process.returncode
            # Background descendants violate the foreground single-action contract.
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError("child descendants remain after foreground exit")
            require(log.stat().st_size <= 64 << 20, "client log limit")
            observed = sorted((set(job_ids(output, self.config["output_bytes"])) - set(before)) | set(JOB.findall(log.read_text(errors="replace"))))
            require(len(observed) <= 32 and all(len(name) <= 128 for name in observed), "single-action job ID limit")
            record["jobs"] = observed
            require(step["kind"] != "appliance" or record["jobs"], "no job IDs captured; submission outcome unknown")
            require(step["kind"] != "host" or not record["jobs"], "host step submitted remote jobs")
            record.update(status="awaiting", client_finished_at=now())
        except (Exception, KeyboardInterrupt) as error:
            if process is not None:
                stop_group(process)
            record.update(status="blocked", error=str(error) or "interrupted")
            entry.update(status="blocked", error=record["error"])
        finally:
            self.record(entry)

    def reconcile(self, entry):
        record = entry["steps"][-1]
        # A previous process may have submitted work before disappearing. Never retry it.
        if record["status"] == "running":
            entry.update(status="blocked", error="interrupted step has no durable exit receipt; inspect and resolve")
            record["status"] = "blocked"
            self.record(entry)
            return False
        outcomes = [self.scheduler.job(name) for name in record["jobs"]]
        record["remote"] = outcomes
        self.record(entry)
        if any(job["phase"] not in TERMINAL for job in outcomes) or not self.scheduler.idle():
            return False
        passed = record["exit_code"] == 0 and all(job["phase"] == "SUCCEEDED" for job in outcomes)
        record.update(status="passed" if passed else "failed", finished_at=now())
        if not passed:
            entry["status"] = "failed"
        self.record(entry)
        return True

    def finish(self, entry):
        root = self.root / "requests" / entry["id"]
        exports = self.root / "exports"
        exports.mkdir(exist_ok=True)
        target = exports / (entry["id"] + ".zip")
        pending = exports / (entry["id"] + ".zip.partial")
        entry["finished_at"] = entry.get("finished_at", now())
        atomic_json(root / "receipt.json", entry)
        paths = [root / "request.json", root / "receipt.json", root / "environment.json"]
        paths += files_under(root / "output", self.config["output_bytes"])
        paths += files_under(root / "logs", 128 << 20)
        # A restart during packaging recreates only the archive, never the calculation.
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in paths:
                archive.write(path, path.relative_to(root))
        with pending.open("rb") as stream:
            os.fsync(stream.fileno())
        checksum = digest(pending)
        require(pending.stat().st_size <= 4 << 30, "archive exceeds collector size limit")
        os.replace(pending, target)
        checksum_path = exports / (entry["id"] + ".zip.sha256")
        checksum_path.write_text(checksum + "  " + target.name + "\n")
        entry.update(bundle=str(target), bundle_sha256=checksum, bundle_bytes=target.stat().st_size, packaged=True)
        self.record(entry)

    def publish(self, entry):
        if entry.get("notified") or not entry.get("packaged") or not self.config["publish_status"]:
            return
        directory = self.root / "status-git"
        if not directory.exists():
            directory.mkdir()
            git(["init", "-b", self.config["status_branch"]], directory)
        branch = self.config["status_branch"]
        refs = git(["ls-remote", "--heads", self.config["repo"], "refs/heads/" + branch]).strip()
        if refs:
            git(["fetch", "--no-tags", self.config["repo"], "refs/heads/" + branch], directory)
            git(["reset", "--hard", "FETCH_HEAD"], directory)
        record = {key: entry[key] for key in ("id", "sha", "status", "finished_at", "bundle_sha256", "bundle_bytes")}
        record["archive"] = Path(entry["bundle"]).name
        destination = directory / "results" / (entry["id"] + ".json")
        if destination.exists():
            require(read_json(destination) == record, "conflicting published result")
        else:
            atomic_json(destination, record)
        git(["add", "--", "results/" + destination.name], directory)
        if git(["diff", "--cached", "--name-only"], directory).strip():
            git(["-c", "user.name=CSL validation watcher", "-c", "user.email=csl-validation@localhost", "commit", "-m", "Record " + entry["id"] + " " + entry["status"]], directory)
        # Normal push, never force. Re-fetch and reapply on the next notification retry.
        git(["push", self.config["repo"], "HEAD:refs/heads/" + branch], directory)
        git(["fetch", "--no-tags", self.config["repo"], "refs/heads/" + branch], directory)
        require(json.loads(git(["show", "FETCH_HEAD:results/" + destination.name], directory)) == record, "published result not verified")
        entry["notified"] = True
        self.record(entry)

    def tick(self):
        for ident in self.ledger["order"]:
            entry = self.ledger["requests"][ident]
            if entry["status"] in {"passed", "failed", "interrupted"}:
                if not entry.get("packaged"):
                    self.finish(entry)
                try:
                    self.publish(entry)
                except (ValueError, OSError) as error:
                    print("Notification pending:", str(error), flush=True)
        for ident in self.ledger["order"]:
            entry = self.ledger["requests"][ident]
            if entry["status"] == "blocked":
                print("BLOCKED", ident, entry.get("error"), flush=True)
                return
            if entry["status"] == "running":
                if entry["steps"] and entry["steps"][-1]["status"] in {"running", "awaiting"}:
                    if not self.reconcile(entry):
                        return
                    if entry["status"] == "failed":
                        self.finish(entry)
                        self.ledger["last_check"] = 0
                        self.save()
                        return
                if len(entry["steps"]) == len(entry["manifest"]["steps"]):
                    entry["status"] = "passed"
                    self.finish(entry)
                    self.ledger["last_check"] = 0
                    self.save()
                    return
                if self.scheduler.idle():
                    self.start_step(entry, entry["manifest"]["steps"][len(entry["steps"])])
                return
        # No active request. Check remote work BEFORE touching the source refs.
        if not self.scheduler.idle():
            print("WAIT: own remote jobs remain active", flush=True)
            return
        if time.time() - self.ledger["last_check"] >= self.config["idle_seconds"]:
            self.discover()
        for ident in self.ledger["order"]:
            entry = self.ledger["requests"][ident]
            if entry["status"] == "pending":
                self.prepare(entry)
                return

    def resolve(self, ident, reason):
        require(ident in self.ledger["requests"] and reason.strip(), "request and reason required")
        entry = self.ledger["requests"][ident]
        require(entry["status"] in {"running", "blocked"}, "only interrupted/blocked attempts can be resolved")
        for record in entry["steps"]:
            pid = record.get("pid")
            if pid:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError("recorded local PID still exists; inspect before resolving")
            for name in record.get("jobs", []):
                require(self.scheduler.job(name)["phase"] in TERMINAL, "remote job still active")
        require(self.scheduler.idle(), "own remote jobs remain active")
        entry.update(status="interrupted", resolution=reason, resolved_at=now())
        self.record(entry)
        self.finish(entry)
        self.ledger["last_check"] = 0
        self.save()


def load_config(path):
    return validate_config(read_json(path))


def validate_config(value):
    require(value.get("schema") == SCHEMA, "config schema")
    require(value["hostname"] == socket.gethostname(), "controller belongs to another user node")
    require(value["branch"] != value["status_branch"], "status branch must differ from request branch")
    require(NAME.fullmatch(value["namespace"]) and value["username"], "namespace/username")
    require(value["idle_seconds"] >= 60 and value["poll_seconds"] >= 1, "poll bounds")
    require(1 << 20 <= value["output_bytes"] <= 2 << 30, "output budget must be 1 MiB..2 GiB")
    require(value.get("state_bytes", 32 << 30) >= value["output_bytes"], "total state budget")
    require(Path(value["python"]).is_absolute() and Path(value["python"]).is_file(), "absolute existing Python required")
    for field in ("state_dir", "lock_file"):
        require(Path(value[field]).is_absolute() and Path(value[field]).resolve() == Path(value[field]), "absolute nonsymlink " + field)
    require(value["repo"] and not value["repo"].startswith("-"), "repository URL")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", value["branch"]) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", value["status_branch"]), "branch names")
    require(isinstance(value["allowed_scripts"], list) and value["allowed_scripts"], "script allowlist")
    return value


def initialize(args):
    import getpass
    root = args.state_dir.absolute()
    require(root == root.resolve() and not root.exists(), "state directory must be new and nonsymlink")
    require(not args.config.exists(), "configuration already exists")
    require(SHA.fullmatch(args.baseline), "baseline must be an explicit full commit SHA")
    config = dict(schema=SCHEMA, repo=args.repo, branch=args.branch, status_branch="alcf-status", state_dir=str(root),
                  python=str(args.python.absolute()), namespace=args.namespace, username=getpass.getuser(),
                  hostname=socket.gethostname(), csctl="csctl", idle_seconds=3600, poll_seconds=60,
                  output_bytes=2 << 30, publish_status=args.publish_status, allowed_scripts=DEFAULT_SCRIPTS,
                  state_bytes=32 << 30, lock_file=str(Path.home() / ".csl-validation.lock"))
    # Reject invalid configuration before mutating the remote or running any experiment.
    validate_config(config)
    root.mkdir(parents=True)
    try:
        git(["init", "--bare", str(root / "source.git")])
        atomic_json(root / "ledger.json", dict(schema=SCHEMA, cursor=args.baseline, last_check=0, requests={}, order=[]))
        watcher = Watcher(config)
        tip = watcher.fetch()
        git(["--git-dir", str(watcher.cache), "merge-base", "--is-ancestor", args.baseline, tip])
        watcher.save()
        atomic_json(args.config, config)
    except BaseException:
        # This directory was created by this call and has never run a request.
        shutil.rmtree(root)
        raise
    print("Initialized. Baseline excluded:", args.baseline)
    print("No jobs submitted. Review config:", args.config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    init = commands.add_parser("init")
    init.add_argument("--config", required=True, type=Path)
    init.add_argument("--repo", required=True)
    init.add_argument("--branch", default="main")
    init.add_argument("--state-dir", required=True, type=Path)
    init.add_argument("--python", required=True, type=Path)
    init.add_argument("--baseline", required=True)
    init.add_argument("--namespace", default="job-operator")
    init.add_argument("--publish-status", action="store_true")
    for name in ("run", "once", "status", "resolve"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True, type=Path)
        if name == "resolve":
            sub.add_argument("--request", required=True)
            sub.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        if args.action == "init":
            initialize(args)
            return 0
        config = load_config(args.config)
        if args.action == "status":
            ledger = read_json(Path(config["state_dir"]) / "ledger.json", LEDGER_LIMIT)
            print(json.dumps(dict(cursor=ledger["cursor"], requests=[{k: ledger["requests"][i].get(k) for k in ("id", "sha", "status", "bundle", "notified", "error")} for i in ledger["order"]]), indent=2))
            return 0
        with operator_lock(config) as fd:
            watcher = Watcher(config, lock_fd=fd)
            watcher.save()  # Repair a crash between the ledger and public-status writes.
            if args.action == "resolve":
                watcher.resolve(args.request, args.reason)
                return 0
            signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            while True:
                def progress():
                    return [(i, e["status"], len(e["steps"]), e["steps"][-1]["status"] if e["steps"] else None) for i, e in watcher.ledger["requests"].items()]
                before = progress()
                try:
                    watcher.tick()
                except (ValueError, OSError, KeyError, TypeError) as error:
                    print("WAIT/ERROR:", str(error), flush=True)
                    if args.action == "once":
                        return 2
                if args.action == "once":
                    return 0
                # Process completed actions immediately; wait only when state did not advance.
                if before == progress():
                    time.sleep(min(config["poll_seconds"], config["idle_seconds"]))
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, KeyError, TypeError) as error:
        print("error:", str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
