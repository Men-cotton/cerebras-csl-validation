"""One compile or runtime action for the two-PE timestamp feasibility probe."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
import shlex
import sys
import tarfile
import uuid

import profile_common as common

ROOT = Path(__file__).resolve().parent
HOSTS = ("profile_worker.py", "profile_common.py", "oracle.py")
SOURCES = ("source/layout.csl", "source/pe.csl")
LOG_LIMIT = 16 << 20
CAPTURE_LIMIT = 32 << 20
JSON_LIMIT = 2 << 20


def bounded(path, limit):
    path = Path(path)
    common.require(path.is_file() and not path.is_symlink(), "regular file required: " + str(path))
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    common.require(len(value) <= limit, "file exceeds bound: " + str(path))
    return value


def digest(path, limit=1 << 30):
    path = Path(path)
    common.require(path.is_file() and not path.is_symlink(), "regular file required")
    result, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            size += len(chunk)
            common.require(size <= limit, "artifact exceeds bound")
            result.update(chunk)
    return result.hexdigest()


def fresh(path):
    path = Path(path).absolute()
    common.require(path == path.resolve(), "output path traverses symlinks")
    path.mkdir(parents=True, exist_ok=False)
    return path


def identity(root):
    return dict(source_sha256={name: digest(root / name) for name in SOURCES},
                host_sha256={name: digest(root / name) for name in HOSTS},
                adapter_sha256=digest(root / "profile_appliance.py"))


def sdk_classes():
    from cerebras.appliance import logger  # noqa: F401; enable SDK job-ID logging
    from cerebras.sdk.client import SdkCompiler, SdkLauncher
    return SdkCompiler, SdkLauncher


def sdk_options(namespace):
    value = dict(resource_cpu=1000, resource_mem=8 << 30, disable_version_check=False)
    if namespace:
        value["mgmt_namespace"] = namespace
    return value


class BoundedLog(logging.Handler):
    def __init__(self, path):
        super().__init__()
        self.stream = path.open("xb")
        self.size, self.truncated = 0, False
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
    log = BoundedLog(destination / "client.log")
    root = logging.getLogger()
    root.addHandler(log)
    try:
        yield log
    finally:
        root.removeHandler(log)
        log.close()


def compile_program(args, root=ROOT, compiler_type=None):
    common.require(args.backend in ("hardware", "simulator"), "backend")
    pinned = identity(root)
    dimensions = "762,1172" if args.backend == "hardware" else "9,3"
    flags = (f"--arch=wse3 --fabric-dims={dimensions} --fabric-offsets=4,1 --params=N:4 "
             "--max-parallelism=1 --memcpy --channels=1 "
             "--width-west-buf=0 --width-east-buf=0 -o out")
    destination = fresh(args.out)
    record = dict(schema=1, action="compile", backend=args.backend, params={"N": 4},
                  compile_id=str(uuid.uuid4()), compiler_args=flags, namespace=args.namespace, **pinned)
    common.write_json(destination / "request.json", record)
    status = dict(status="incomplete", compile_id=record["compile_id"])
    common.write_json(destination / "compile-status.json", status)
    log = None
    try:
        with client_log(destination) as log:
            cls = compiler_type or sdk_classes()[0]
            with cls(**sdk_options(args.namespace)) as compiler:
                returned = Path(compiler.compile(str(root / "source"), "layout.csl", flags, str(destination)))
                common.require(not returned.is_symlink(), "artifact symlink")
                artifact = returned.resolve()
        common.require(artifact.parent == destination and artifact.name not in
                       {"request.json", "profile-build.json", "compile-status.json", "client.log"},
                       "artifact path outside destination or record collision")
        common.require(not log.truncated, "client log truncated")
        common.require(pinned == identity(root), "source changed during compile")
        record.update(artifact_path=artifact.name, artifact_sha256=digest(artifact))
        common.write_json(destination / "profile-build.json", record)
        status.update(status="complete", manifest_sha256=digest(destination / "profile-build.json"))
    except BaseException as error:
        status["error"] = repr(error)
        raise
    finally:
        common.write_json(destination / "compile-status.json", status)
    print(destination / "profile-build.json", flush=True)
    return 0


def load_build(path, backend, root=ROOT):
    path = Path(path)
    raw = bounded(path, JSON_LIMIT)
    record = json.loads(raw)
    common.require(record.get("schema") == 1 and record.get("action") == "compile" and
                   record.get("params") == {"N": 4} and record.get("backend") == backend, "compile contract")
    status = common.read_json(path.parent / "compile-status.json")
    common.require(status.get("status") == "complete" and status.get("compile_id") == record.get("compile_id") and
                   status.get("manifest_sha256") == hashlib.sha256(raw).hexdigest(), "compile status identity")
    original = common.read_json(path.parent / "request.json")
    common.require(original == {key: value for key, value in record.items()
                                if key not in ("artifact_path", "artifact_sha256")}, "original compile request")
    common.require(all(record.get(key) == value for key, value in identity(root).items()), "compiled source identity")
    name = record.get("artifact_path")
    common.require(isinstance(name, str) and name not in ("", ".", "..") and Path(name).name == name, "artifact filename")
    artifact = path.parent / name
    common.require(digest(artifact) == record.get("artifact_sha256"), "artifact hash")
    return record, artifact, raw


def host_command(backend, seconds):
    common.require(type(seconds) is int and 1 <= seconds <= 1800, "host timeout must be 1..1800")
    words = ["cs_python", "profile_worker.py", "--name", "out", "--out", "capture",
             "--manifest", "profile-build.json", "--request", "run-request.json", "--backend", backend]
    if backend == "hardware":
        words += ["--cmaddr", "%CMADDR%"]
    return (f"if timeout --signal=INT --kill-after=15s {seconds}s {shlex.join(words)} "
            "> host.log 2>&1; then code=0; else code=$?; fi; "
            "printf '%s\\n' \"$code\" > host-exit.txt")


def downloaded_path(returned, requested, limit):
    requested = Path(requested).absolute()
    common.require(returned is None or isinstance(returned, str) and returned, "invalid SDK download path")
    path = requested if returned is None else Path(returned)
    if not path.is_absolute():
        path = requested.parent / path
    common.require(not path.is_symlink(), "download symlink")
    path = path.resolve()
    common.require(path.parent == requested.parent and path.name not in
                   {"request.json", "profile-build.json", "run-request.json", "collection.json",
                    "client.log", "launcher-response.txt", "capture-data"}, "download path escapes or collides")
    bounded(path, limit)
    return path


def unpack_capture(archive_path, destination):
    """Extract only bounded regular files below capture/, never tar links."""
    bounded(archive_path, CAPTURE_LIMIT)
    destination.mkdir(exist_ok=False)
    names, total, count = set(), 0, 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            count += 1
            total += member.size
            name = PurePosixPath(member.name)
            common.require(count <= 128 and 0 <= member.size <= CAPTURE_LIMIT and total <= CAPTURE_LIMIT,
                           "capture archive bound")
            common.require(not name.is_absolute() and ".." not in name.parts and name.parts and
                           name.parts[0] == "capture" and str(name) not in names, "capture archive path")
            names.add(str(name))
            common.require(member.isdir() or member.isfile(), "capture archive link or special file")
            relative = name.relative_to("capture")
            if not relative.parts:
                common.require(member.isdir(), "capture root must be a directory")
                continue
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                common.require(member.size <= 8 << 20, "capture member bound")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream:
                    payload = stream.read((8 << 20) + 1)
                common.require(len(payload) == member.size, "capture member length")
                with target.open("xb") as stream:
                    stream.write(payload)
    for name in ("capture.json", "raw.npz", "manifest.json", "request.json"):
        common.require((destination / name).is_file(), "missing capture member: " + name)
    return destination


def run_program(args, root=ROOT, launcher_type=None):
    common.require(type(args.flag) is int and args.flag in (0, 1), "timestamp flag must be 0 or 1")
    record, artifact, manifest_bytes = load_build(args.artifact_record, args.backend, root)
    command = host_command(args.backend, args.host_timeout)
    destination = fresh(args.out)
    (destination / "profile-build.json").write_bytes(manifest_bytes)
    request = dict(schema=1, run_id=str(uuid.uuid4()), flag=args.flag,
                   manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
    common.write_json(destination / "run-request.json", request)
    request_bytes = bounded(destination / "run-request.json", JSON_LIMIT)
    common.write_json(destination / "request.json", dict(action="run", backend=args.backend, command=command, **request))
    summary = dict(status="incomplete", run_id=request["run_id"], flag=args.flag, downloaded={}, errors={})
    common.write_json(destination / "collection.json", summary)
    log, pending = None, None
    try:
        with client_log(destination) as log:
            cls = launcher_type or sdk_classes()[1]
            with cls(str(artifact), simulator=args.backend == "simulator", **sdk_options(record.get("namespace"))) as launcher:
                try:
                    for name in HOSTS:
                        common.require(digest(root / name) == record["host_sha256"][name], "host changed before staging")
                        launcher.stage(str(root / name))
                    for name in SOURCES:
                        common.require(digest(root / name) == record["source_sha256"][name], "CSL changed before staging")
                        launcher.stage(str(root / name))
                    for name in ("profile-build.json", "run-request.json"):
                        launcher.stage(str(destination / name))
                    common.require(all(record.get(key) == value for key, value in identity(root).items()), "source changed during staging")
                    common.require(digest(artifact) == record["artifact_sha256"], "artifact changed during staging")
                    common.require(bounded(destination / "profile-build.json", JSON_LIMIT) == manifest_bytes and
                                   bounded(destination / "run-request.json", JSON_LIMIT) == request_bytes,
                                   "staged manifest or request changed")
                    response = str(launcher.run(command) or "")
                    common.require(len(response.encode()) <= LOG_LIMIT, "launcher response bound")
                    (destination / "launcher-response.txt").write_text(response)
                except BaseException as error:
                    pending = error
                    summary["launch_error"] = repr(error)
                finally:
                    for remote, limit in (("host.log", LOG_LIMIT), ("host-exit.txt", 128), ("capture", CAPTURE_LIMIT)):
                        try:
                            requested = destination / remote
                            saved = downloaded_path(launcher.download_artifact(remote, str(requested)), requested, limit)
                            common.require(saved.name not in {item["path"] for item in summary["downloaded"].values()}, "download paths alias")
                            summary["downloaded"][remote] = dict(path=saved.name, sha256=digest(saved, limit))
                        except Exception as error:
                            summary["errors"][remote] = repr(error)
        common.require(not log.truncated, "client log truncated")
        if pending is not None:
            raise RuntimeError("launcher failed; partial downloads retained") from pending
        common.require(not summary["errors"], "collection failed; inspect collection.json")
        exit_path = destination / summary["downloaded"]["host-exit.txt"]["path"]
        common.require(bounded(exit_path, 128).decode().strip() == "0", "worker failed or timed out")
        captured = unpack_capture(destination / summary["downloaded"]["capture"]["path"], destination / "capture-data")
        common.require(bounded(captured / "manifest.json", JSON_LIMIT) == manifest_bytes, "captured manifest differs")
        common.require(bounded(captured / "request.json", JSON_LIMIT) == request_bytes, "captured request differs")
        audited = common.audit_capture(captured)
        for key, value in dict(status="pass", run_id=request["run_id"], flag=args.flag,
                               backend=args.backend, manifest_sha256=request["manifest_sha256"]).items():
            common.require(audited.get(key) == value, "capture admission differs: " + key)
        summary.update(status="complete", host_exit=0, capture_path="capture-data", audit=audited)
    except BaseException as error:
        summary["error"] = repr(error)
        raise
    finally:
        common.write_json(destination / "collection.json", summary)
    print(json.dumps(dict(status="pass", run_id=request["run_id"], out=str(destination))), flush=True)
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="action", required=True)
    for action in ("compile", "run"):
        child = sub.add_parser(action)
        child.add_argument("--backend", choices=("hardware", "simulator"), required=True)
        child.add_argument("--out", type=Path, required=True)
        if action == "compile":
            child.add_argument("--namespace", default="job-operator")
        else:
            child.add_argument("--artifact-record", type=Path, required=True)
            child.add_argument("--flag", type=int, choices=(0, 1), required=True)
            child.add_argument("--host-timeout", type=int, default=300)
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
