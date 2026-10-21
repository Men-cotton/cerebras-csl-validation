#!/usr/bin/env python3
"""Compile/run command-clock probes with the ordinary appliance client.

The run --config-file contains {"config": {...}}. Each invocation requires a
fresh output directory. CLI backend simulator means the appliance simulator;
the worker records appliance-sim. This module submits no jobs on import.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import tarfile
import time
import uuid

import clock_common as common

ROOT = Path(__file__).resolve().parent
JSON_LIMIT = 2 << 20
LOG_LIMIT = 16 << 20
CAPTURE_LIMITS = {"capture.json": JSON_LIMIT, "calls.json": JSON_LIMIT, "raw.npz": 32 << 20}
TREE_LIMIT = 1 << 30
JOB = re.compile(r"\bJob id:\s*(wsjob-[A-Za-z0-9-]+)")
WORKFLOW = re.compile(r"\bworkflow id:\s*(\S+?)(?:,|\s|$)")
REMOTE = re.compile(r"\bremote log path:\s*(\S+)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded_bytes(path, limit):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "not a regular file: " + str(path))
    require(path.stat().st_size <= limit, "oversized file: " + str(path))
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    require(len(data) <= limit, "file grew beyond bound: " + str(path))
    return data


def digest(path, limit=TREE_LIMIT):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "not a regular file: " + str(path))
    require(path.stat().st_size <= limit, "oversized hash input: " + str(path))
    result, total = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            total += len(chunk)
            require(total <= limit, "hash input grew beyond bound: " + str(path))
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(len(data) <= JSON_LIMIT, "JSON output exceeds bound")
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    require(not path.is_symlink() and not temporary.exists() and not temporary.is_symlink(),
            "unsafe JSON output path")
    with temporary.open("xb") as handle:
        handle.write(data)
    temporary.replace(path)


def read_json(path):
    value = json.loads(bounded_bytes(path, JSON_LIMIT))
    require(isinstance(value, dict), "JSON object required")
    return value


def new_directory(path):
    path = Path(path).absolute()
    require(path == path.resolve(), "output path must not traverse symlinks")
    path.mkdir(parents=True, exist_ok=False)
    return path


def accounting(path):
    """Bounded local snapshot; does not claim to limit a remote running job."""
    start, entries, size, pending = time.monotonic(), 0, 0, [Path(path)]
    while pending:
        with os.scandir(pending.pop()) as children:
            for child in children:
                entries += 1
                require(entries <= 4096 and time.monotonic() - start <= 5,
                        "local artifact accounting incomplete: scan bound")
                require(not child.is_symlink(), "local artifact accounting rejects symlinks")
                if child.is_dir(follow_symlinks=False):
                    pending.append(Path(child.path))
                else:
                    require(child.is_file(follow_symlinks=False), "nonregular local artifact")
                    size += child.stat(follow_symlinks=False).st_size
                    require(size <= TREE_LIMIT, "local artifacts exceed 1 GiB")
    return dict(status="complete", bytes=size, entries=entries,
                scope="local output snapshot; no remote or live growth enforcement")


def identity(root):
    return dict(common.source_identity(root), launcher_sha256=digest(root / "clock_appliance.py"))


def worker_backend(backend):
    require(backend in ("simulator", "hardware"), "unsupported appliance backend")
    return "appliance-sim" if backend == "simulator" else "hardware"


def sdk_options(namespace):
    result = dict(resource_cpu=1000, resource_mem=8 << 30, disable_version_check=False)
    if namespace:
        result["mgmt_namespace"] = namespace
    return result


def sdk_classes():
    from cerebras.appliance import logger  # noqa: F401; enable SDK job/version messages
    from cerebras.sdk.client import SdkCompiler, SdkLauncher
    return SdkCompiler, SdkLauncher


class BoundedLog(logging.Handler):
    def __init__(self, path):
        super().__init__()
        self.handle_file = Path(path).open("xb")
        self.size, self.truncated = 0, False
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))

    def emit(self, record):
        data = (self.format(record) + "\n").encode(errors="replace")
        if self.size + len(data) > LOG_LIMIT:
            self.truncated = True
            return
        self.handle_file.write(data)
        self.handle_file.flush()
        self.size += len(data)

    def close(self):
        self.handle_file.close()
        super().close()


@contextmanager
def client_log(destination):
    handler = BoundedLog(destination / "client.log")
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        yield handler
    finally:
        root_logger.removeHandler(handler)
        handler.close()


def local_jobs(destination, role):
    path = destination / "client.log"
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(bounded_bytes(path, LOG_LIMIT).decode(errors="replace").splitlines(), 1):
        match = JOB.search(line)
        if match:
            require(len(line) <= 4096 and len(records) < 256, "job identity log exceeds record bound")
            workflow, remote = WORKFLOW.search(line), REMOTE.search(line)
            records.append(dict(role=role, log="client.log", line=number, job_id=match.group(1),
                                workflow_id=workflow.group(1) if workflow else None,
                                remote_log_path=remote.group(1) if remote else None))
    return records


def finalize(destination, summary, started, log, role):
    summary["wall_seconds"] = time.monotonic() - started
    summary["client_log_truncated"] = bool(log and log.truncated)
    try:
        summary["jobs"] = local_jobs(destination, role)
        summary["accounting"] = accounting(destination)
    except Exception as error:
        summary["accounting"] = dict(status="incomplete", error=repr(error))
        summary["status"] = "incomplete"
    summary["job_identity_scope"] = "local client log only; missing IDs/versions/systems remain unknown"
    write_json(destination / ("collection.json" if role == "launch" else "compile-status.json"), summary)


def pair(value):
    try:
        first, second = map(int, value.split(","))
    except (ValueError, AttributeError) as error:
        raise argparse.ArgumentTypeError("expected integer X,Y") from error
    if min(first, second) < 0:
        raise argparse.ArgumentTypeError("X,Y must be nonnegative")
    return f"{first},{second}"


def compile_program(args, root=ROOT, compiler_type=None):
    params = common.validate_params(dict(C=args.cols, R=args.rows, W=args.words))
    pinned = identity(root)
    backend = worker_backend(args.backend)
    dims = args.fabric_dims or (f"{params['C'] + 7},{params['R'] + 2}"
                               if args.backend == "simulator" else "762,1172")
    width, height = map(int, pair(dims).split(","))
    x, y = map(int, pair(args.fabric_offsets).split(","))
    require(x + params["C"] <= width and y + params["R"] <= height, "application exceeds fabric")
    flags = (f"--arch=wse3 --fabric-dims={width},{height} --fabric-offsets={x},{y} "
             f"--params=C:{params['C']},R:{params['R']},W:{params['W']} "
             "--max-parallelism=1 --memcpy --channels=1 "
             "--width-west-buf=0 --width-east-buf=0 -o out")
    destination = new_directory(args.out)
    manifest = dict(schema=1, action="compile", compile_id=str(uuid.uuid4()), backend=backend,
                    params=params, compiler_args=flags, namespace=args.namespace,
                    sdk_options=sdk_options(args.namespace), **pinned)
    write_json(destination / "request.json", manifest)
    summary = dict(status="incomplete", compile_id=manifest["compile_id"], backend=backend)
    write_json(destination / "compile-status.json", summary)
    started, log = time.monotonic(), None
    try:
        with client_log(destination) as log:
            cls = compiler_type or sdk_classes()[0]
            with cls(**sdk_options(args.namespace)) as compiler:
                returned = Path(compiler.compile(str(root / "clock_source"), "layout.csl", flags,
                                                 str(destination)))
                require(not returned.is_symlink(), "compiler artifact is a symlink")
                artifact = returned.resolve()
            require(artifact.parent == destination, "compiler artifact outside requested directory")
            require(artifact.name not in ("request.json", "clock-build.json", "compile-status.json", "client.log"),
                    "compiler artifact collides with controller record")
            require(pinned == identity(root), "sources changed during compilation")
            manifest.update(artifact_path=artifact.name, artifact_sha256=digest(artifact))
            accounting(destination)
            write_json(destination / "clock-build.json", manifest)
            summary.update(status="complete", manifest_sha256=digest(destination / "clock-build.json"))
    except BaseException as error:
        summary["error"] = repr(error)
        raise
    finally:
        finalize(destination, summary, started, log, "compile")
    require(summary["status"] == "complete", "compile accounting incomplete")
    print(destination / "clock-build.json", flush=True)
    return 0


def load_manifest(path, backend, root=ROOT):
    path = Path(path)
    manifest_bytes = bounded_bytes(path, JSON_LIMIT)
    manifest = json.loads(manifest_bytes)
    require(isinstance(manifest, dict), "build manifest must be an object")
    require(type(manifest.get("schema")) is int and manifest["schema"] == 1 and
            manifest.get("action") == "compile", "invalid build manifest")
    require(manifest.get("backend") == worker_backend(backend), "build backend differs")
    require(common.validate_params(manifest["params"]) == manifest["params"], "noncanonical build params")
    original = read_json(path.parent / "request.json")
    status = read_json(path.parent / "compile-status.json")
    accounted = status.get("accounting")
    require(status.get("status") == "complete" and isinstance(accounted, dict) and accounted.get("status") == "complete",
            "compile did not complete with artifact accounting")
    require(status.get("manifest_sha256") == hashlib.sha256(manifest_bytes).hexdigest(),
            "compile status manifest hash mismatch")
    require(isinstance(manifest.get("compile_id"), str) and bool(manifest["compile_id"]), "missing compile ID")
    for key in ("compile_id", "backend"):
        require(status.get(key) == manifest[key], "compile status " + key + " differs")
    requested = {key: value for key, value in manifest.items() if key not in ("artifact_path", "artifact_sha256")}
    require(original == requested, "original compile request differs from final manifest pins")
    for name, value in identity(root).items():
        require(manifest.get(name) == value, "compiled " + name + " differs from current source")
    name = manifest.get("artifact_path")
    require(isinstance(name, str) and name not in ("", ".", "..") and Path(name).name == name,
            "artifact_path must be a filename")
    artifact = path.parent / name
    require(digest(artifact) == manifest.get("artifact_sha256"), "artifact hash mismatch")
    return manifest, artifact


def host_command(backend, seconds):
    require(isinstance(seconds, int) and not isinstance(seconds, bool) and 1 <= seconds <= 1800,
            "host timeout must be 1..1800 seconds")
    words = ["cs_python", "clock_probe.py", "--name", "out", "--out", "capture", "--backend",
             worker_backend(backend), "--config-file", "clock-request.json", "--manifest", "clock-build.json"]
    if backend == "hardware":
        words.extend(["--cmaddr", "%CMADDR%"])
    command = shlex.join(words)
    return (f"if timeout --signal=INT --kill-after=15s {seconds}s {command} "
            "> host.log 2>&1; then code=0; else code=$?; fi; "
            "printf '%s\\n' \"$code\" > host-exit.txt")


def capture_payloads(path):
    path = Path(path)
    require(not path.is_symlink(), "capture must not be a symlink")
    if path.is_dir():
        return {name: bounded_bytes(path / name, limit) for name, limit in CAPTURE_LIMITS.items()}
    require(path.is_file() and path.stat().st_size <= 64 << 20, "capture archive exceeds bound")
    payloads, names, expanded, count = {}, set(), 0, 0
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            count += 1
            expanded += member.size
            require(count <= 256 and expanded <= 64 << 20, "capture archive expanded bound")
            name = PurePosixPath(member.name)
            require(not name.is_absolute() and ".." not in name.parts and name.parts and name.parts[0] == "capture",
                    "unsafe capture archive path")
            require(str(name) not in names, "duplicate capture member")
            names.add(str(name))
            require(member.isdir() or member.isfile(), "capture archive contains link or special file")
            if member.isdir():
                continue
            relative = str(name.relative_to("capture"))
            if relative in CAPTURE_LIMITS:
                require(member.size <= CAPTURE_LIMITS[relative], "oversized capture member")
                with archive.extractfile(member) as handle:
                    data = handle.read(CAPTURE_LIMITS[relative] + 1)
                require(len(data) == member.size, "capture member size mismatch")
                payloads[relative] = data
    require(set(payloads) == set(CAPTURE_LIMITS), "capture missing required members")
    return payloads


def validate_capture(path, manifest, request):
    """Join capture identity, integrity, API completion and raw numerical checks."""
    payloads = capture_payloads(path)
    meta, calls = json.loads(payloads["capture.json"]), json.loads(payloads["calls.json"])
    require(isinstance(meta, dict), "capture metadata must be an object")
    require(type(meta.get("schema")) is int, "capture schema must be an integer")
    for key, expected in (("schema", 1), ("backend", manifest["backend"]), ("params", manifest["params"]),
                          ("config", request["config"]), ("run_id", request["run_id"]),
                          ("manifest_sha256", request["manifest_sha256"]),
                          ("host_sha256", manifest["host_sha256"]),
                          ("source_sha256", manifest["source_sha256"])):
        require(meta.get(key) == expected, "capture " + key + " differs from request")
    require(common.validate_config(meta["config"]) == meta["config"], "noncanonical capture config")
    for name, key in (("raw.npz", "raw_sha256"), ("calls.json", "calls_sha256")):
        require(hashlib.sha256(payloads[name]).hexdigest() == meta.get(key), "capture hash mismatch: " + name)
    require(meta.get("status") in ("pass", "observed_numeric_fail", "incomplete"), "unknown capture status")
    validation = common.validate_capture_content(meta, calls, payloads["raw.npz"], manifest, request)
    require(isinstance(validation, dict) and validation.get("status") == meta["status"],
            "raw validation status differs from capture")
    return meta, validation


def downloaded_path(returned, requested, limit):
    """Honor an SDK-added suffix while confining reads to the requested folder."""
    requested = Path(requested).absolute()
    require(returned is None or isinstance(returned, str) and bool(returned),
            "download returned an invalid path")
    candidate = requested if returned is None else Path(returned)
    if not candidate.is_absolute():
        candidate = requested.parent / candidate
    require(not candidate.is_symlink(), "download returned a symlink")
    saved = candidate.resolve()
    require(saved.parent == requested.parent.resolve(), "download returned a path outside destination")
    require(saved.name not in ("request.json", "clock-build.json", "clock-request.json", "collection.json",
                               "client.log", "launcher-response.txt"), "download collides with controller record")
    require(saved.is_file() and saved.stat().st_size <= limit, "missing, nonregular or oversized download")
    return saved


def run_program(args, root=ROOT, launcher_type=None):
    manifest, artifact = load_manifest(args.artifact_record, args.backend, root)
    supplied = read_json(args.config_file)
    require(set(supplied) == {"config"}, "config file must contain only {'config': {...}}")
    config = common.validate_config(supplied["config"])
    command = host_command(args.backend, args.host_timeout)
    destination = new_directory(args.out)
    # Copy the exact manifest bytes, including its artifact hash, before staging.
    manifest_bytes = bounded_bytes(args.artifact_record, JSON_LIMIT)
    require(json.loads(manifest_bytes) == manifest, "manifest changed during admission")
    compile_status = read_json(Path(args.artifact_record).parent / "compile-status.json")
    require(hashlib.sha256(manifest_bytes).hexdigest() == compile_status.get("manifest_sha256"),
            "manifest bytes changed after compile-status admission")
    (destination / "clock-build.json").write_bytes(manifest_bytes)
    request = dict(config=config, run_id=str(uuid.uuid4()), manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
    write_json(destination / "clock-request.json", request)
    write_json(destination / "request.json", dict(action="run", backend=manifest["backend"], command=command,
                                                   artifact_sha256=manifest["artifact_sha256"], **request))
    summary = dict(status="incomplete", run_id=request["run_id"], manifest_sha256=request["manifest_sha256"],
                   downloaded={}, errors={},
                   validation_scope="identity, file integrity, exact API sequence and raw numerical admission")
    write_json(destination / "collection.json", summary)
    started, log, meta, pending_error = time.monotonic(), None, None, None
    try:
        with client_log(destination) as log:
            cls = launcher_type or sdk_classes()[1]
            with cls(str(artifact), simulator=args.backend == "simulator", **sdk_options(manifest.get("namespace"))) as launcher:
                try:
                    for name, pin in manifest["host_sha256"].items():
                        require(Path(name).name == name, "worker host must have a flat staging filename")
                        require(digest(root / name) == pin, "host changed before staging: " + name)
                        launcher.stage(str(root / name))
                    for name in ("clock-build.json", "clock-request.json"):
                        launcher.stage(str(destination / name))
                    require(all(manifest.get(key) == value for key, value in identity(root).items()),
                            "source or adapter changed during staging")
                    require(digest(destination / "clock-build.json") == request["manifest_sha256"],
                            "staged build manifest changed")
                    require(read_json(destination / "clock-request.json") == request, "staged request changed")
                    require(digest(artifact) == manifest["artifact_sha256"], "artifact changed before launch")
                    response = launcher.run(command)
                    text = "" if response is None else str(response)
                    require(len(text.encode()) <= LOG_LIMIT, "launcher response exceeds bound")
                    (destination / "launcher-response.txt").write_text(text)
                except BaseException as error:
                    pending_error = error
                    summary["launch_error"] = repr(error)
                finally:
                    for remote, limit in (("host.log", LOG_LIMIT), ("host-exit.txt", 128), ("capture", 64 << 20)):
                        try:
                            requested = destination / remote
                            returned = launcher.download_artifact(remote, str(requested))
                            saved = downloaded_path(returned, requested, limit)
                            require(saved.name not in {item["path"] for item in summary["downloaded"].values()},
                                    "multiple downloads returned the same path")
                            summary["downloaded"][remote] = dict(path=saved.name, sha256=digest(saved, limit), bytes=saved.stat().st_size)
                            if remote == "capture":
                                meta, validation = validate_capture(saved, manifest, request)
                                summary["capture_validation"] = validation
                            accounting(destination)
                        except Exception as error:
                            summary["errors"][remote] = repr(error)
                    write_json(destination / "collection.json", summary)
        if pending_error is not None:
            raise RuntimeError("launcher failed; downloaded files retained") from pending_error
        require(not summary["errors"], "collection_failed; inspect collection.json")
        exit_text = bounded_bytes(destination / summary["downloaded"]["host-exit.txt"]["path"], 128).decode().strip()
        require(exit_text in ("0", "1", "2"), "worker exit indicates timeout or execution failure: " + exit_text)
        expected_status = {"0": "pass", "1": "observed_numeric_fail", "2": "incomplete"}[exit_text]
        require(meta is not None and meta["status"] == expected_status, "worker exit/status mismatch")
        summary.update(status="complete" if exit_text in ("0", "1") else "incomplete",
                       worker_status=meta["status"], host_exit=int(exit_text))
    except BaseException as error:
        summary["error"] = repr(error)
        raise
    finally:
        finalize(destination, summary, started, log, "launch")
    require(summary["status"] == "complete", "incomplete worker or artifact accounting; inspect collection.json")
    print(json.dumps(dict(status=summary["worker_status"], host_exit=summary["host_exit"],
                          run_id=request["run_id"], out=str(destination)), sort_keys=True), flush=True)
    return summary["host_exit"]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="action", required=True)
    for action in ("compile", "run"):
        command = sub.add_parser(action)
        command.add_argument("backend", choices=("simulator", "hardware"))
        command.add_argument("--out", type=Path, required=True, help="fresh local output directory")
        if action == "compile":
            command.add_argument("--cols", type=int, default=2)
            command.add_argument("--rows", type=int, default=2)
            command.add_argument("--words", type=int, default=16)
            command.add_argument("--fabric-dims", type=pair,
                                 help="default SIM C+7,R+2; hardware 762,1172 (ALCF WSE-3)")
            command.add_argument("--fabric-offsets", type=pair, default="4,1")
            command.add_argument("--namespace", help="existing site namespace")
        else:
            command.add_argument("--artifact-record", type=Path, required=True, help="compiled clock-build.json")
            command.add_argument("--config-file", type=Path, required=True, help="JSON object with one config key")
            command.add_argument("--host-timeout", type=int, default=300)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        return compile_program(args) if args.action == "compile" else run_program(args)
    except Exception as error:
        print("error: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
