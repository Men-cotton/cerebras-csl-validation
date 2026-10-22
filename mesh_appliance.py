#!/usr/bin/env python3
"""Compile and capture immutable MeshGraphNet cases through the appliance SDK."""
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

import mesh_audit

ROOT = Path(__file__).resolve().parent
SOURCES = ("mesh_appliance.py", "mesh_campaign.py", "mesh_audit.py", "mesh_worker.py")
JSON_LIMIT = 8 << 20
LOG_LIMIT = 16 << 20
TREE_LIMIT = 1 << 30
CAPTURE_LIMIT = 128 << 20
CAPTURE_FILES = {"raw.npz": 96 << 20, "execution.json": JSON_LIMIT,
                 "out.json": JSON_LIMIT, "out_rpc.json": JSON_LIMIT}
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
JOB = re.compile(r"\bJob id:\s*(wsjob-[A-Za-z0-9-]+)")


def require(value, message):
    if not value:
        raise ValueError(message)


def bounded(path, limit=JSON_LIMIT):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
            "bounded regular file required: " + str(path))
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    require(len(value) <= limit, "file grew beyond bound")
    return value


def digest(path, limit=TREE_LIMIT):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
            "bounded hash input required: " + str(path))
    total, result = 0, hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            total += len(block)
            require(total <= limit, "hash input grew beyond bound")
            result.update(block)
    return result.hexdigest()


def read_json(path):
    result = json.loads(bounded(path))
    require(isinstance(result, dict), "JSON object required")
    return result


def write_json(path, value):
    path = Path(path)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(len(data) <= JSON_LIMIT, "JSON output bound")
    temporary = path.with_name(path.name + ".tmp")
    require(not path.is_symlink() and not temporary.exists() and not temporary.is_symlink(), "unsafe JSON path")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def regular_directory(path):
    path = Path(path).absolute()
    require(path == path.resolve() and path.is_dir(), "directory must not traverse symlinks")
    return path


def new_directory(path):
    path = Path(path).absolute()
    require(path == path.resolve(), "output must not traverse symlinks")
    path.mkdir(parents=True, exist_ok=False)
    return path


def inventory(directory, byte_limit=TREE_LIMIT, entry_limit=4096):
    root = regular_directory(directory)
    started, pending, total, entries, result = time.monotonic(), [root], 0, 0, {}
    while pending:
        with os.scandir(pending.pop()) as children:
            for child in children:
                entries += 1
                require(entries <= entry_limit and time.monotonic() - started <= 10, "inventory scan bound")
                require(not child.is_symlink(), "inventory rejects symlinks")
                if child.is_dir(follow_symlinks=False):
                    pending.append(Path(child.path))
                    continue
                require(child.is_file(follow_symlinks=False), "inventory requires regular files")
                size = child.stat(follow_symlinks=False).st_size
                total += size
                require(total <= byte_limit, "inventory byte bound")
                result[str(Path(child.path).relative_to(root))] = size
    return result


def source_identity():
    return {name: digest(ROOT / name, JSON_LIMIT) for name in SOURCES}


def case_identity(package, name):
    package = regular_directory(package)
    require(isinstance(name, str) and NAME.fullmatch(name), "case name")
    manifest = mesh_audit.verify_package(package)
    require(manifest.get("schema") == 1 and name in manifest["cases"], "case absent from package")
    folder = regular_directory(package / "cases" / name)
    case = read_json(folder / "case.json")
    case_hash = digest(folder / "case.json")
    require(manifest["cases"][name]["case_sha256"] == case_hash and
            manifest["cases"][name]["params"] == case["params"], "case identity differs")
    return package, folder, case, dict(case_name=name, case_sha256=case_hash,
                                      package_manifest_sha256=digest(package / "manifest.json"))


def pair(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9]+,[0-9]+", value), "expected nonnegative X,Y")
    return tuple(map(int, value.split(",")))


def sdk_options(namespace):
    result = dict(resource_cpu=1000, resource_mem=8 << 30, disable_version_check=False)
    if namespace is not None:
        require(isinstance(namespace, str) and 0 < len(namespace) <= 255 and
                all(ord(c) >= 32 for c in namespace), "namespace")
        result["mgmt_namespace"] = namespace
    return result


def sdk_classes():
    from cerebras.appliance import logger  # noqa: F401; SDK job/version logging
    from cerebras.sdk.client import SdkCompiler, SdkLauncher
    return SdkCompiler, SdkLauncher


class BoundedLog(logging.Handler):
    def __init__(self, path):
        super().__init__()
        self.stream, self.size, self.truncated = Path(path).open("xb"), 0, False
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))

    def emit(self, record):
        data = (self.format(record) + "\n").encode(errors="replace")
        if self.size + len(data) > LOG_LIMIT:
            self.truncated = True
            return
        self.stream.write(data)
        self.stream.flush()
        self.size += len(data)

    def close(self):
        self.stream.close()
        super().close()


@contextmanager
def client_log(destination):
    handler = BoundedLog(destination / "client.log")
    logging.getLogger().addHandler(handler)
    try:
        yield handler
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def finalize(destination, summary, start, log, filename):
    summary.update(wall_seconds=time.monotonic() - start,
                   client_log_truncated=bool(log and log.truncated))
    try:
        text = bounded(destination / "client.log", LOG_LIMIT).decode(errors="replace")
        jobs = list(dict.fromkeys(JOB.findall(text)))
        require(len(jobs) <= 256, "job identity count bound")
        summary["job_ids"] = jobs
        files = inventory(destination)
        summary["accounting"] = dict(status="complete", bytes=sum(files.values()), files=len(files),
                                      scope="local snapshot; no remote live growth bound")
    except Exception as error:
        summary.update(status="incomplete", accounting=dict(status="incomplete", error=str(error)))
    write_json(destination / filename, summary)


def compile_program(args, compiler_type=None):
    package, case_dir, case, identity = case_identity(args.package, args.case)
    require(args.backend in ("hardware", "appliance-sim"), "appliance backend")
    p = case["params"]
    require(p and all(isinstance(k, str) and re.fullmatch(r"[A-Z][A-Z0-9_]*", k) and
                      type(v) is int and 0 <= v <= 65535 for k, v in p.items()), "compiler parameters")
    dims = pair(args.fabric_dims or ("762,1172" if args.backend == "hardware" else f"{p['C']+7},{p['R']+2}"))
    offsets = pair(args.fabric_offsets)
    require(min(dims) > 0 and offsets[0] + p["C"] <= dims[0] and offsets[1] + p["R"] <= dims[1],
            "application exceeds fabric")
    flags = (f"--arch=wse3 --fabric-dims={dims[0]},{dims[1]} --fabric-offsets={offsets[0]},{offsets[1]} "
             "--memcpy --channels=1 --max-parallelism=1 --params=" +
             ",".join(f"{k}:{v}" for k, v in sorted(p.items())) + " -o out")
    pins = source_identity()
    manifest = dict(schema=1, backend=args.backend, compile_id=str(uuid.uuid4()), params=p,
                    compiler_args=flags, namespace=args.namespace, source_sha256=pins,
                    sdk_options=sdk_options(args.namespace), **identity)
    destination = new_directory(args.out)
    write_json(destination / "compile-request.json", manifest)
    status = dict(status="incomplete", compile_id=manifest["compile_id"], **identity)
    write_json(destination / "compile-status.json", status)
    start, log = time.monotonic(), None
    try:
        with client_log(destination) as log:
            cls = compiler_type or sdk_classes()[0]
            with cls(**sdk_options(args.namespace)) as compiler:
                returned = Path(compiler.compile(str(case_dir / "source"), "layout.csl", flags, str(destination)))
                require(not returned.is_symlink(), "compiler returned symlink")
                artifact = returned.resolve()
            require(artifact.parent == destination and artifact.name not in
                    ("build.json", "compile-request.json", "compile-status.json", "client.log"), "artifact destination")
            require(source_identity() == pins and case_identity(package, args.case)[3] == identity,
                    "source/package changed during compilation")
            manifest.update(artifact_path=artifact.name, artifact_sha256=digest(artifact))
            write_json(destination / "build.json", manifest)
            status.update(status="complete", manifest_sha256=digest(destination / "build.json"))
    except BaseException as error:
        status["error"] = repr(error)
        raise
    finally:
        finalize(destination, status, start, log, "compile-status.json")
    require(status["status"] == "complete", "compile accounting incomplete")
    return manifest


def load_build(path, package, case_name, backend):
    path = Path(path).absolute()
    require(path == path.resolve(), "build path symlink traversal")
    manifest, status = read_json(path), read_json(path.parent / "compile-status.json")
    _, _, case, identity = case_identity(package, case_name)
    require(manifest.get("schema") == 1 and manifest.get("backend") == backend and
            manifest.get("params") == case["params"], "build backend/parameters")
    require(all(manifest.get(k) == v for k, v in identity.items()) and
            manifest.get("source_sha256") == source_identity(), "build identity differs")
    require(status.get("status") == "complete" and status.get("accounting", {}).get("status") == "complete" and
            status.get("manifest_sha256") == digest(path) and status.get("compile_id") == manifest.get("compile_id"),
            "compile did not complete with matching manifest")
    require(read_json(path.parent / "compile-request.json") ==
            {k: v for k, v in manifest.items() if k not in ("artifact_path", "artifact_sha256")}, "compile request differs")
    name = manifest.get("artifact_path")
    require(isinstance(name, str) and name not in ("", ".", "..") and Path(name).name == name, "artifact filename")
    artifact = path.parent / name
    require(digest(artifact) == manifest.get("artifact_sha256"), "artifact hash differs")
    return manifest, artifact


def host_command(backend, timeout):
    require(backend in ("hardware", "appliance-sim"), "appliance backend")
    require(type(timeout) is int and 1 <= timeout <= 1800, "worker timeout must be 1..1800 seconds")
    words = ["cs_python", "mesh_worker.py", "--name", "out", "--case", ".", "--out", "capture",
             "--request", "request.json", "--backend", backend]
    if backend == "hardware":
        words += ["--cmaddr", "%CMADDR%"]
    return (f"if timeout --signal=INT --kill-after=15s {timeout}s {shlex.join(words)} > host.log 2>&1; "
            "then code=0; else code=$?; fi; printf '%s\\n' \"$code\" > host-exit.txt")


def downloaded_path(returned, requested, limit):
    requested = Path(requested).absolute()
    require(returned is None or isinstance(returned, str) and bool(returned), "download return path")
    candidate = requested if returned is None else Path(returned)
    if not candidate.is_absolute():
        candidate = requested.parent / candidate
    require(not candidate.is_symlink(), "download symlink")
    saved = candidate.resolve()
    require(saved.parent == requested.parent and
            saved.name not in ("request.json", "launch-request.json", "collection.json", "build.json", "client.log",
                               "launcher-response.txt", "audit.json"),
            "download outside destination or record collision")
    require(saved.is_file() and saved.stat().st_size <= limit, "download missing or oversized")
    return saved


def unpack_capture(archive_path, destination):
    require(Path(archive_path).stat().st_size <= CAPTURE_LIMIT, "capture archive bound")
    payloads, seen, total, entries = {}, set(), 0, 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            total += member.size
            entries += 1
            require(total <= CAPTURE_LIMIT and entries <= 64, "expanded capture bound")
            name = PurePosixPath(member.name)
            require(not name.is_absolute() and ".." not in name.parts and name.parts and
                    name.parts[0] == "capture" and str(name) not in seen, "unsafe/duplicate capture member")
            seen.add(str(name))
            require(member.isdir() or member.isfile(), "capture contains link/special file")
            if member.isdir():
                require(str(name) == "capture", "unexpected capture directory")
                continue
            relative = str(name.relative_to("capture"))
            require(relative in CAPTURE_FILES and member.size <= CAPTURE_FILES[relative], "unexpected/oversized capture file")
            with archive.extractfile(member) as stream:
                value = stream.read(CAPTURE_FILES[relative] + 1)
            require(len(value) == member.size, "capture member size differs")
            payloads[relative] = value
    require(set(payloads) == set(CAPTURE_FILES), "capture missing required files")
    destination = new_directory(destination)
    for name, value in payloads.items():
        (destination / name).write_bytes(value)
    return destination


def run_program(args, launcher_type=None):
    manifest, artifact = load_build(args.artifact_record, args.package, args.case, args.backend)
    package, case_dir, _, identity = case_identity(args.package, args.case)
    command = host_command(args.backend, args.host_timeout)
    request = dict(schema=1, run_id=str(uuid.uuid4()), backend=args.backend, **identity,
                   artifact_sha256=manifest["artifact_sha256"], worker_sha256=digest(ROOT / "mesh_worker.py"))
    destination = new_directory(args.out)
    write_json(destination / "request.json", request)
    write_json(destination / "launch-request.json", dict(command=command, build_sha256=digest(args.artifact_record),
                                                        request_sha256=digest(destination / "request.json")))
    (destination / "build.json").write_bytes(bounded(args.artifact_record))
    status = dict(status="incomplete", request=request, downloaded={}, errors={})
    write_json(destination / "collection.json", status)
    start, log, pending_error = time.monotonic(), None, None
    try:
        with client_log(destination) as log:
            cls = launcher_type or sdk_classes()[1]
            with cls(str(artifact), simulator=args.backend == "appliance-sim", **sdk_options(manifest["namespace"])) as launcher:
                try:
                    for path in (ROOT / "mesh_worker.py", case_dir / "case.json", case_dir / "inputs.npz", destination / "request.json"):
                        launcher.stage(str(path))
                    load_build(args.artifact_record, package, args.case, args.backend)
                    require(read_json(destination / "request.json") == request, "request changed during staging")
                    response = "" if (value := launcher.run(command)) is None else str(value)
                    require(len(response.encode()) <= LOG_LIMIT, "launcher response bound")
                    (destination / "launcher-response.txt").write_text(response)
                except BaseException as error:
                    pending_error = error
                    status["launch_error"] = repr(error)
                finally:
                    for remote, local, limit in (("host.log", "host.log", LOG_LIMIT),
                                                 ("host-exit.txt", "host-exit.txt", 128),
                                                 ("capture", "capture-download", CAPTURE_LIMIT)):
                        try:
                            saved = downloaded_path(launcher.download_artifact(remote, str(destination / local)),
                                                    destination / local, limit)
                            require(saved.name not in {v["path"] for v in status["downloaded"].values()}, "download collision")
                            status["downloaded"][remote] = dict(path=saved.name, sha256=digest(saved, limit), bytes=saved.stat().st_size)
                        except Exception as error:
                            status["errors"][remote] = repr(error)
                    write_json(destination / "collection.json", status)
        if "capture" in status["downloaded"]:
            capture = unpack_capture(destination / status["downloaded"]["capture"]["path"], destination / "capture")
            execution = read_json(capture / "execution.json")
            status["worker_status"] = execution.get("status")
        require(pending_error is None, "launcher failed; downloaded evidence retained: " + repr(pending_error))
        require(not status["errors"], "artifact collection incomplete")
        exit_text = bounded(destination / status["downloaded"]["host-exit.txt"]["path"], 128).decode().strip()
        require(exit_text == "0" and status.get("worker_status") == "COMPLETED", "worker incomplete; inspect retained capture")
        report = mesh_audit.audit_capture(package, args.case, capture, request)
        require(report.get("status") == "PASS", "capture audit did not pass")
        write_json(destination / "audit.json", report)
        status.update(status="complete", audit_sha256=digest(destination / "audit.json"), host_exit=0)
    except BaseException as error:
        status["error"] = repr(error)
        raise
    finally:
        finalize(destination, status, start, log, "collection.json")
    require(status["status"] == "complete", "collection accounting incomplete")
    return status


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("compile", "run"):
        command = sub.add_parser(action)
        command.add_argument("--backend", choices=("hardware", "appliance-sim"), default="hardware")
        command.add_argument("--package", type=Path, default=ROOT / "mesh_cases")
        command.add_argument("--case", required=True)
        command.add_argument("--out", type=Path, required=True)
        if action == "compile":
            command.add_argument("--namespace")
            command.add_argument("--fabric-dims")
            command.add_argument("--fabric-offsets", default="4,1")
        else:
            command.add_argument("--artifact-record", type=Path, required=True)
            command.add_argument("--host-timeout", type=int, default=300)
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        compile_program(args) if args.action == "compile" else run_program(args)
        print(json.dumps(dict(status="complete", action=args.action, out=str(args.out)), sort_keys=True))
        return 0
    except Exception as error:
        print("error: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
