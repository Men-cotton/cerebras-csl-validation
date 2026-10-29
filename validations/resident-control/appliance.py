"""One compile or runtime job for a frozen eight-step grouped GNN control case.

Collection and job logging follow the existing CSL validation profiling adapter.
No campaign controller, retry loop, optimizer, PyTorch or environment mutation.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import logging
from pathlib import Path
import re
import shlex
import sys
import tarfile
import uuid

try:
    from . import saved_audit as common
except ImportError:
    import saved_audit as common

ROOT = Path(__file__).resolve().parent
HOSTS = common.HOST_FILES
LOG_LIMIT = 16 << 20
ARTIFACT_LIMIT = 1 << 30


def fresh(path):
    path = Path(path).absolute()
    common.require(path == path.resolve(), "output path traverses symlinks")
    path.mkdir(parents=True, exist_ok=False)
    return path


def identity(root, case_name):
    root = Path(root).absolute()
    common.require(re.fullmatch("[a-z][a-z0-9_-]{0,63}", case_name) is not None, "case name")
    package = common.read_json(root/"package.json")
    common.require(package.get("schema") == 1 and package.get("family") == "resident-control", "package contract")
    case = root/"cases"/case_name
    manifest = common.load(case)
    case_hash = common.sha(case/"case.json")
    common.require(package.get("cases", {}).get(case_name) == case_hash, "package case identity")
    hosts = {name: common.sha(root/name) for name in HOSTS}
    common.require(package.get("host_sha256") == hosts, "package host source identity")
    return dict(package_sha256=common.sha(root/"package.json"), host_sha256=hosts,
                case_name=case_name, case_sha256=case_hash, case_files=manifest["files"],
                params=manifest["params"], scope=manifest["scope"])


def sdk_classes():
    from cerebras.appliance import logger  # noqa: F401; keep appliance job IDs visible
    from cerebras.sdk.client import SdkCompiler, SdkLauncher
    return SdkCompiler, SdkLauncher


def sdk_options(namespace, workers=1):
    common.require(isinstance(namespace, str) and re.fullmatch("[a-z0-9][a-z0-9-]*", namespace), "explicit namespace")
    return dict(resource_cpu=common.compile_workers(workers)*1000, resource_mem=8 << 30, disable_version_check=False,
                mgmt_namespace=namespace)


class BoundedLog(logging.Handler):
    def __init__(self, path):
        super().__init__()
        self.stream = path.open("xb")
        self.size, self.truncated = 0, False
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))

    def emit(self, record):
        content = (self.format(record)+"\n").encode(errors="replace")
        if self.size+len(content) > LOG_LIMIT:
            self.truncated = True
            return
        self.stream.write(content)
        self.stream.flush()
        self.size += len(content)

    def close(self):
        self.stream.close()
        super().close()


@contextmanager
def client_log(destination):
    log = BoundedLog(destination/"client.log")
    logger = logging.getLogger()
    logger.addHandler(log)
    try:
        yield log
    finally:
        logger.removeHandler(log)
        log.close()


def compile_program(args, root=ROOT, compiler_type=None):
    pinned = identity(root, args.case)
    p = pinned["params"]
    common.require(args.backend in ("hardware", "simulator"), "backend")
    workers = common.compile_workers(args.compile_workers)
    options = sdk_options(args.namespace, workers)
    if args.backend == "hardware":
        common.require(p["C"]+7 <= 762 and p["R"]+2 <= 1172, "hardware fabric bounds")
    flags = common.compiler_arguments(p, args.backend, workers)
    destination = fresh(args.out)
    case = Path(root)/"cases"/args.case
    bundle = destination/"case.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name in sorted(set(common.load(case)["files"]) | {"case.json"}):
            archive.add(common.local_file(case, name), arcname="case/"+name, recursive=False)
    record = dict(schema=1, action="compile", backend=args.backend, compile_id=str(uuid.uuid4()),
                  compiler_args=flags, namespace=args.namespace, compile_workers=workers,
                  compile_resource_cpu=options["resource_cpu"],
                  case_archive_sha256=common.sha(bundle), **pinned)
    common.write_json(destination/"request.json", record)
    status = dict(status="incomplete", compile_id=record["compile_id"])
    common.write_json(destination/"compile-status.json", status)
    try:
        with client_log(destination) as log:
            cls = compiler_type or sdk_classes()[0]
            with cls(**options) as compiler:
                returned = Path(compiler.compile(str(case/"source"), "layout.csl", flags, str(destination)))
                common.require(not returned.is_symlink(), "compiler artifact symlink")
                artifact = returned.resolve()
        common.require(artifact.parent == destination and artifact.name not in
                       {"request.json", "native-build.json", "compile-status.json", "client.log", "case.tar.gz"},
                       "compiler artifact escapes or collides")
        common.require(not log.truncated, "client log truncated")
        common.require(pinned == identity(root, args.case), "case or host changed during compile")
        common.require(common.sha(bundle) == record["case_archive_sha256"], "case archive changed during compile")
        record.update(artifact_path=artifact.name, artifact_sha256=common.sha(artifact, ARTIFACT_LIMIT))
        common.write_json(destination/"native-build.json", record)
        status.update(status="complete", manifest_sha256=common.sha(destination/"native-build.json"))
    except BaseException as error:
        status["error"] = repr(error)
        raise
    finally:
        common.write_json(destination/"compile-status.json", status)
    print(destination/"native-build.json", flush=True)
    return 0


def load_build(path, backend, case_name, root=ROOT):
    path = Path(path).absolute()
    raw = common.bounded(path, common.JSON_LIMIT)
    record = json.loads(raw)
    common.require(record.get("schema") == 1 and record.get("action") == "compile"
                   and record.get("backend") == backend and record.get("case_name") == case_name, "compile contract")
    common.compile_binding(record)
    status = common.read_json(path.parent/"compile-status.json")
    common.require(status.get("status") == "complete" and status.get("compile_id") == record.get("compile_id")
                   and status.get("manifest_sha256") == hashlib.sha256(raw).hexdigest(), "compile status identity")
    original = common.read_json(path.parent/"request.json")
    common.require(original == {k: v for k, v in record.items() if k not in ("artifact_path", "artifact_sha256")},
                   "compile request identity")
    common.require(all(record.get(k) == v for k, v in identity(root, case_name).items()), "compiled source identity")
    name = record.get("artifact_path")
    common.require(isinstance(name, str) and name not in ("", ".", "..") and Path(name).name == name, "artifact filename")
    artifact = path.parent/name
    common.require(common.sha(artifact, ARTIFACT_LIMIT) == record["artifact_sha256"], "artifact hash")
    common.require(common.sha(path.parent/"case.tar.gz") == record["case_archive_sha256"], "case archive hash")
    return record, artifact, raw


def host_command(backend, seconds):
    common.require(type(seconds) is int and 1 <= seconds <= 1800, "host timeout must be 1..1800")
    common.require(backend in ("hardware", "simulator"), "backend")
    words = ["cs_python", "appliance_worker.py", "--build", "out", "--out", "capture",
             "--case-archive", "case.tar.gz", "--manifest", "native-build.json",
             "--request", "run-request.json", "--backend", backend]
    if backend == "hardware":
        words += ["--cmaddr", "%CMADDR%"]
    return (f"if timeout --signal=INT --kill-after=15s {seconds}s {shlex.join(words)} "
            "> host.log 2>&1; then code=0; else code=$?; fi; "
            "printf '%s\\n' \"$code\" > host-exit.txt")


def downloaded_path(returned, requested, limit):
    requested = Path(requested).absolute()
    common.require(returned is None or isinstance(returned, str) and returned, "invalid download path")
    path = requested if returned is None else Path(returned)
    if not path.is_absolute():
        path = requested.parent/path
    common.require(not path.is_symlink(), "download symlink")
    path = path.resolve()
    common.require(path.parent == requested.parent and path.name not in
                   {"request.json", "native-build.json", "run-request.json", "collection.json",
                    "client.log", "launcher-response.txt", "capture-data"}, "download escapes or collides")
    common.bounded(path, limit)
    return path


def run_program(args, root=ROOT, launcher_type=None):
    common.arm(args.arm)
    record, artifact, manifest_bytes = load_build(args.artifact_record, args.backend, args.case, root)
    command = host_command(args.backend, args.host_timeout)
    destination = fresh(args.out)
    (destination/"native-build.json").write_bytes(manifest_bytes)
    request = dict(schema=1, run_id=str(uuid.uuid4()), case_name=args.case,
                   manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), arm=args.arm)
    common.write_json(destination/"run-request.json", request)
    request_bytes = common.bounded(destination/"run-request.json", common.JSON_LIMIT)
    common.write_json(destination/"request.json", dict(action="run", backend=args.backend, command=command, **request))
    summary = dict(status="incomplete", run_id=request["run_id"], downloaded={}, errors={})
    common.write_json(destination/"collection.json", summary)
    pending = None
    try:
        with client_log(destination) as log:
            cls = launcher_type or sdk_classes()[1]
            with cls(str(artifact), simulator=args.backend == "simulator", **sdk_options(record["namespace"])) as launcher:
                try:
                    for name in HOSTS:
                        common.require(common.sha(Path(root)/name) == record["host_sha256"][name], "host changed before staging")
                        launcher.stage(str(Path(root)/name))
                    for path in (Path(args.artifact_record).parent/"case.tar.gz",
                                 destination/"native-build.json", destination/"run-request.json"):
                        launcher.stage(str(path))
                    common.require(all(record.get(k) == v for k, v in identity(root, args.case).items()), "source changed during staging")
                    common.require(common.sha(artifact, ARTIFACT_LIMIT) == record["artifact_sha256"] and
                                   common.sha(Path(args.artifact_record).parent/"case.tar.gz") == record["case_archive_sha256"],
                                   "artifact or case changed during staging")
                    common.require(common.bounded(destination/"native-build.json", common.JSON_LIMIT) == manifest_bytes and
                                   common.bounded(destination/"run-request.json", common.JSON_LIMIT) == request_bytes,
                                   "request changed during staging")
                    response = str(launcher.run(command) or "")
                    common.require(len(response.encode()) <= LOG_LIMIT, "launcher response bound")
                    (destination/"launcher-response.txt").write_text(response)
                except BaseException as error:
                    pending = error
                    summary["launch_error"] = repr(error)
                finally:
                    for remote, limit in (("host.log", LOG_LIMIT), ("host-exit.txt", 128), ("capture", common.CAPTURE_LIMIT)):
                        try:
                            requested = destination/remote
                            saved = downloaded_path(launcher.download_artifact(remote, str(requested)), requested, limit)
                            common.require(saved.name not in {x["path"] for x in summary["downloaded"].values()}, "download aliases")
                            summary["downloaded"][remote] = dict(path=saved.name, sha256=common.sha(saved, limit),
                                                                  bytes=saved.stat().st_size, retained=True)
                        except Exception as error:
                            summary["errors"][remote] = repr(error)
        common.require(not log.truncated, "client log truncated")
        if pending is not None:
            raise RuntimeError("launcher failed; partial downloads retained") from pending
        common.require(not summary["errors"], "collection failed")
        exit_path = destination/summary["downloaded"]["host-exit.txt"]["path"]
        common.require(common.bounded(exit_path, 128).decode().strip() == "0", "worker failed or timed out")
        capture = common.unpack_archive(destination/summary["downloaded"]["capture"]["path"],
            destination/"capture-data", "capture", common.CAPTURE_LIMIT,
            ("capture.json", "manifest.json", "request.json", "runtime/raw.npz", "runtime/execution.json"))
        common.require(common.bounded(capture/"manifest.json", common.JSON_LIMIT) == manifest_bytes, "captured manifest differs")
        common.require(common.bounded(capture/"request.json", common.JSON_LIMIT) == request_bytes, "captured request differs")
        audited = common.admit_capture(capture)
        for key, value in dict(status="PASS", run_id=request["run_id"], backend=args.backend,
                               manifest_sha256=request["manifest_sha256"]).items():
            common.require(audited.get(key) == value, "capture admission differs: " + key)
        summary.update(status="complete", execution_status="PASS", numerical_status="NOT_RUN",
                       host_exit=0, capture_path="capture-data", execution_admission=audited)
        # The admitted tree contains every byte needed for offline replay. Keep
        # the transport receipt, but do not store the archive a second time.
        transport = summary["downloaded"]["capture"]
        transport.update(extracted_path="capture-data", retention="remove-after-admission")
        common.write_json(destination/"collection.json", summary)
        try:
            (destination/transport["path"]).unlink()
        except OSError as error:
            transport["cleanup_error"] = str(error)[:4096]
        else:
            transport.update(retained=False, retention="verified-extraction")
    except BaseException as error:
        summary["error"] = repr(error)
        raise
    finally:
        common.write_json(destination/"collection.json", summary)
    print(json.dumps(dict(status="PASS", execution_status="PASS", numerical_status="NOT_RUN",
                          out=str(destination), run_id=request["run_id"])), flush=True)
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="action", required=True)
    for action in ("compile", "run"):
        child = sub.add_parser(action)
        child.add_argument("--case", required=True)
        child.add_argument("--backend", choices=("hardware", "simulator"), required=True)
        child.add_argument("--out", type=Path, required=True)
        if action == "compile":
            child.add_argument("--namespace", default="job-operator")
            child.add_argument("--compile-workers", type=int, choices=(1, 4, 8), default=1)
        else:
            child.add_argument("--arm", choices=common.ARMS, required=True)
            child.add_argument("--artifact-record", type=Path, required=True)
            child.add_argument("--host-timeout", type=int, default=300)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        return compile_program(args) if args.action == "compile" else run_program(args)
    except Exception as error:
        print("error: "+str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
