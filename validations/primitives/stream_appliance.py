#!/usr/bin/env python3
"""Registered compile/run/audit adapter for the bounded E210 stream pilot."""
import argparse
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import time
import uuid

if __package__:
    from . import chunks, halo, stream_contract as contract, stream_worker as worker
else:
    import chunks
    import halo
    import stream_contract as contract
    import stream_worker as worker

require = contract.require
ARCHIVE_LIMIT = 64 << 20
METADATA_FILES = {"case.json", "request.json", "build.json", "out.json", "out_rpc.json", "plan.json", "execution.json"}


def identity(package, case):
    """The registered case check used by request construction and execution."""
    return contract.package_identity(package, case)[3]


def sdk_classes():
    from cerebras.appliance import logger  # noqa: F401
    from cerebras.sdk.client import SdkCompiler, SdkLauncher
    return SdkCompiler, SdkLauncher


def sdk_options(namespace):
    require(namespace is None or isinstance(namespace, str) and 0 < len(namespace) <= 255 and
            all(ord(c) >= 32 for c in namespace), "namespace")
    result = dict(resource_cpu=1000, resource_mem=8 << 30, disable_version_check=False)
    if namespace is not None:
        result["mgmt_namespace"] = namespace
    return result


@contextmanager
def client_log(destination):
    class BoundedLog(logging.Handler):
        def __init__(self):
            super().__init__()
            self.stream, self.size, self.truncated = (destination/"client.log").open("xb"), 0, False
        def emit(self, record):
            data = (self.format(record)+"\n").encode(errors="replace")
            if self.size+len(data) > contract.LOG_LIMIT:
                self.truncated = True
                return
            self.stream.write(data)
            self.stream.flush()
            self.size += len(data)
    handler = BoundedLog()
    logging.getLogger().addHandler(handler)
    try:
        yield handler
    finally:
        logging.getLogger().removeHandler(handler)
        handler.stream.close()


def finalize(destination, status, started, filename):
    status["wall_seconds"] = time.monotonic()-started
    try:
        files = contract.inventory(destination)
        status["accounting"] = dict(status="complete", files=len(files), bytes=sum(files.values()),
                                    scope="bounded local snapshot; remote growth is not measured")
    except Exception as error:
        status.update(status="incomplete", accounting=dict(status="incomplete", error=repr(error)))
    contract.write_json(destination/filename, status)



def artifact_metadata(artifact, case):
    """Read only bounded compiler metadata from the SDK's compiled tar artifact."""
    require(artifact.stat().st_size <= contract.TREE_LIMIT, "compiled archive bound")
    result, prefixes, seen, total, count, started = {}, set(), set(), 0, 0, time.monotonic()
    with tarfile.open(artifact, "r|gz") as archive:
        for member in archive:
            count += 1
            total += member.size
            name = PurePosixPath(member.name)
            require(count <= 10000 and total <= contract.TREE_LIMIT and time.monotonic()-started <= 10,
                    "compiled archive scan bound")
            require(not name.is_absolute() and ".." not in name.parts and str(name) not in seen and
                    (member.isdir() or member.isfile()), "compiled archive member")
            seen.add(str(name))
            # SDK artifacts also contain out/east/out.json and out/west/out.json.
            # Only the application's root metadata belongs to its exported ABI.
            key, prefix = None, None
            if name.parts[-2:] == ("out", "out.json"):
                key, prefix = "out.json", name.parts[:-2]
            elif name.parts[-3:] == ("out", "bin", "out_rpc.json"):
                key, prefix = "out_rpc.json", name.parts[:-3]
            if key is not None:
                prefixes.add(prefix)
                require(member.isfile() and key not in result and member.size <= contract.JSON_LIMIT,
                        "unique bounded compiled metadata")
                with archive.extractfile(member) as stream:
                    value = stream.read(contract.JSON_LIMIT+1)
                require(len(value) == member.size, "compiled metadata extent")
                result[key] = value
    require(set(result) == {"out.json", "out_rpc.json"} and len(prefixes) == 1,
            "compiled archive missing ABI metadata or mixed compiler roots")
    contract.check_abi(case, json.loads(result["out.json"]), json.loads(result["out_rpc.json"]))
    return {k: chunks.digest(v) for k, v in result.items()}


def compile_program(args, compiler_type=None):
    package, folder, case, pin = contract.package_identity(args.package, args.case)
    backend = args.backend
    require(backend in ("local-sim", "hardware", "appliance-sim"), "compile backend")
    dims = (762, 1172) if backend == "hardware" else (10, 5)
    flags = (f"--arch=wse3 --fabric-dims={dims[0]},{dims[1]} --fabric-offsets=4,1 "
             "--memcpy --channels=1 --max-parallelism=1 --params=height:3,width:3 -o out")
    sources = contract.source_identity(contract.APP_FILES)
    record = dict(schema=2, compile_id=str(uuid.uuid4()), backend=backend, params=case["params"],
                  source_sha256=sources, worker_sha256=contract.worker_digest(), compiler_args=flags,
                  namespace=args.namespace, sdk_options=sdk_options(args.namespace), **pin)
    destination = contract.new_directory(args.out)
    contract.write_json(destination/"compile-request.json", record)
    status, start = dict(status="incomplete", compile_id=record["compile_id"]), time.monotonic()
    try:
        if backend == "local-sim":
            compiler = Path(args.compiler).resolve()
            require(compiler.is_file(), "local compiler wrapper")
            command = [str(compiler), str(folder/"source/layout.csl"), *shlex.split(flags)]
            # This command belongs inside the caller's global budget-runner slot.
            with (destination/"compiler.log").open("xb") as log:
                subprocess.run(command, cwd=destination, stdout=log, stderr=subprocess.STDOUT, check=True,
                               env=dict(os.environ, CSL_IMPORT_PATH=str(folder/"source")))
            build = destination/"out"
            contract.inventory(build)
            names = [n for n in contract.inventory(build) if n.endswith(".elf") or Path(n).name in ("out.json", "out_rpc.json")]
            inventory = dict(files={n: halo.file_hash(build/n) for n in names})
            halo.verify_build(build, inventory)
            contract.check_abi(case, contract.read_json(build/"out.json"), contract.read_json(build/"bin/out_rpc.json"))
            record.update(artifact_path="out", artifact_sha256=chunks.digest(chunks.encoded(inventory)), inventory=inventory,
                          local_compiler_sha256=halo.file_hash(compiler, contract.JSON_LIMIT),
                          compiled_metadata_sha256={"out.json": inventory["files"]["out.json"],
                                                    "out_rpc.json": inventory["files"]["bin/out_rpc.json"]})
        else:
            with client_log(destination) as log:
                cls = compiler_type or sdk_classes()[0]
                with cls(**sdk_options(args.namespace)) as compiler:
                    returned = Path(compiler.compile(str(folder/"source"), "layout.csl", flags, str(destination)))
                    require(not returned.is_symlink(), "compiler artifact symlink")
                    artifact = returned.resolve()
                require(artifact.parent == destination and artifact.name not in
                        ("build.json", "compile-request.json", "compile-status.json", "client.log"), "compiler artifact path")
                record.update(artifact_path=artifact.name, artifact_sha256=halo.file_hash(artifact),
                              compiled_metadata_sha256=artifact_metadata(artifact, case))
                status["client_log_truncated"] = log.truncated
        require(contract.source_identity(contract.APP_FILES) == sources and identity(package, args.case) == pin,
                "source/package changed during compilation")
        contract.write_json(destination/"build.json", record)
        status.update(status="complete", manifest_sha256=halo.file_hash(destination/"build.json"))
    except BaseException as error:
        status["error"] = repr(error)
        raise
    finally:
        finalize(destination, status, start, "compile-status.json")
    require(status["status"] == "complete", "compile accounting incomplete")
    return record


def load_build(path, package, case_name, backend):
    path = Path(path).absolute()
    require(path == path.resolve(), "compiled record path")
    record = contract.read_json(path)
    status = contract.read_json(path.parent/"compile-status.json")
    _, _, case, pin = contract.package_identity(package, case_name)
    require(record.get("schema") == 2 and record.get("backend") == backend and record.get("params") == case["params"]
            and all(record.get(k) == v for k, v in pin.items()) and
            record.get("source_sha256") == contract.source_identity(contract.APP_FILES)
            and record.get("worker_sha256") == contract.worker_digest(), "compiled source/case/backend identity")
    require(status.get("status") == "complete" and status.get("accounting", {}).get("status") == "complete"
            and status.get("manifest_sha256") == halo.file_hash(path) and status.get("compile_id") == record["compile_id"],
            "complete compile receipt")
    extras = {"artifact_path", "artifact_sha256", "inventory", "local_compiler_sha256", "compiled_metadata_sha256"}
    require(contract.read_json(path.parent/"compile-request.json") == {k: v for k, v in record.items() if k not in extras},
            "compiled request identity")
    name = record["artifact_path"]
    require(isinstance(name, str) and Path(name).name == name and name not in ("", ".", ".."), "compiled artifact name")
    artifact = path.parent/name
    if backend == "local-sim":
        halo.verify_build(artifact, record["inventory"])
        require(chunks.digest(chunks.encoded(record["inventory"])) == record["artifact_sha256"], "compiled inventory identity")
    else:
        require(halo.file_hash(artifact) == record["artifact_sha256"] and
                artifact_metadata(artifact, case) == record["compiled_metadata_sha256"], "compiled artifact/ABI identity")
    return record, artifact


def host_command(backend, timeout):
    require(backend in ("hardware", "appliance-sim") and type(timeout) is int and 1 <= timeout <= 1800,
            "worker backend/timeout")
    words = ["cs_python", "stream_worker.py", "--name", "out", "--case", "case.json", "--request", "request.json",
             "--build-record", "build.json", "--out", "capture", "--backend", backend]
    if backend == "hardware":
        words += ["--cmaddr", "%CMADDR%"]
    return (f"if timeout --signal=INT --kill-after=15s {timeout}s {shlex.join(words)} > host.log 2>&1; "
            "then code=0; else code=$?; fi; printf '%s\\n' \"$code\" > host-exit.txt")


def download(launcher, remote, destination, limit):
    require(not destination.exists() and not destination.is_symlink(), "fresh download target")
    returned = launcher.download_artifact(remote, str(destination))
    require(returned is None or isinstance(returned, str) and bool(returned), "download return path")
    path = destination if returned is None else Path(returned)
    if not path.is_absolute():
        path = destination.parent/path
    require(not path.is_symlink() and path.resolve().parent == destination.parent, "download path outside destination")
    path = path.resolve()
    require(path.name in {destination.name, destination.name+".tar.gz", destination.name+".tgz"}, "download filename")
    require(path.is_file() and path.stat().st_size <= limit, "download byte bound")
    return path


def unpack_capture(archive_path, destination, plan):
    """Stream known members to fresh files; retain complete/partial files on error.

    The SDK's established directory-download contract supplies a gzip tar. No
    nested named-file download behavior or complete payload dictionary is needed.
    """
    admission = chunks.validate_plan(plan)
    require(Path(archive_path).stat().st_size <= ARCHIVE_LIMIT, "compressed capture bound")
    expected = {name: (contract.JSON_LIMIT, False) for name in METADATA_FILES}
    expected["chunks/capture.json"] = (contract.JSON_LIMIT, False)
    for descriptor in chunks.iter_plan(plan):
        expected["chunks/"+descriptor["file"]] = (descriptor["offset"]+descriptor["bytes"], True)
    expanded_limit = admission["payload_bytes"]+len(METADATA_FILES | {"chunks/capture.json"})*contract.JSON_LIMIT
    require(expanded_limit <= ARCHIVE_LIMIT, "pilot expanded archive bound")
    destination = contract.new_directory(destination)
    (destination/"chunks").mkdir()
    seen, total, entries, peak = set(), 0, 0, 0
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            entries += 1
            require(entries <= len(expected)+2, "archive entry bound")
            name = PurePosixPath(member.name)
            require(not name.is_absolute() and ".." not in name.parts and name.parts and name.parts[0] == "capture"
                    and str(name) not in seen, "unsafe/duplicate archive path")
            seen.add(str(name))
            require(member.isdir() or member.isfile(), "archive link/special file")
            if member.isdir():
                require(str(name) in ("capture", "capture/chunks"), "unexpected archive directory")
                continue
            relative = str(name.relative_to("capture"))
            require(relative in expected, "unexpected capture member")
            bound, exact = expected[relative]
            require(0 <= member.size <= bound and (not exact or member.size == bound), "capture member extent")
            total += member.size
            require(total <= expanded_limit, "expanded archive byte bound")
            remaining = member.size
            with archive.extractfile(member) as source, (destination/relative).open("xb") as target:
                while remaining:
                    block = source.read(min(1 << 20, remaining))
                    require(bool(block), "truncated capture member")
                    target.write(block)
                    remaining -= len(block)
                    peak = max(peak, len(block))
    require({"capture/"+n for n in expected} <= seen, "capture missing required files")
    return dict(payload_bytes=admission["payload_bytes"], extracted_bytes=total, peak_copy_bytes=peak, files=len(expected))


def audit_program(args):
    _, folder, _, pin = contract.package_identity(args.package, args.case)
    root = chunks.regular_root(args.capture)
    request_bytes, build_bytes = halo.bounded(root/"request.json"), halo.bounded(root/"build.json")
    request = json.loads(request_bytes)
    require(all(request[k] == value for k, value in pin.items()), "audit package identity")
    record = json.loads(build_bytes)
    require(record.get("source_sha256") == contract.source_identity(contract.APP_FILES), "audit adapter source identity")
    report = worker.audit_capture(root/"capture", halo.bounded(folder/"case.json"), request_bytes, build_bytes)
    collection = contract.read_json(root/"collection.json")
    require(collection.get("status") == "complete" and collection.get("accounting", {}).get("status") == "complete"
            and collection.get("request") == request,
            "collection did not complete")
    require(halo.bounded(root/"host-exit.txt", 128).strip() == b"0", "worker exit failure")
    contract.write_json(args.out, report)
    return report


def run_program(args, launcher_type=None, sdk_factory=None):
    record, artifact = load_build(args.artifact_record, args.package, args.case, args.backend)
    package, folder, case, pin = contract.package_identity(args.package, args.case)
    case_bytes, build_bytes = halo.bounded(folder/"case.json"), halo.bounded(args.artifact_record)
    request = contract.request(case_bytes, pin, args.backend, record["artifact_sha256"], chunks.digest(build_bytes))
    request_bytes = chunks.encoded(request)
    plan = contract.make_plan(case, contract.plan_identity(request_bytes))
    destination = contract.new_directory(args.out)
    (destination/"request.json").write_bytes(request_bytes)
    (destination/"build.json").write_bytes(build_bytes)
    status, start = dict(status="incomplete", downloaded={}, errors={}, request=request), time.monotonic()
    contract.write_json(destination/"collection.json", status)
    try:
        if args.backend == "local-sim":
            worker_args = argparse.Namespace(name=artifact, case=folder/"case.json", request=destination/"request.json",
                                             build_record=destination/"build.json", out=destination/"capture",
                                             backend=args.backend, cmaddr=None)
            execution = worker.capture(worker_args, sdk_factory)
            (destination/"host-exit.txt").write_text("0\n" if execution["execution_status"] == "PASS" else "2\n")
        else:
            command = host_command(args.backend, args.host_timeout)
            contract.write_json(destination/"launch-request.json", dict(command=command, request_sha256=chunks.digest(request_bytes),
                                                                      build_record_sha256=chunks.digest(build_bytes)))
            pending = None
            with client_log(destination) as log:
                cls = launcher_type or sdk_classes()[1]
                with cls(str(artifact), simulator=args.backend == "appliance-sim", **sdk_options(record["namespace"])) as launcher:
                    try:
                        for path in [contract.ROOT/n for n in contract.WORKER_FILES]+[folder/"case.json", destination/"request.json", destination/"build.json"]:
                            launcher.stage(str(path))
                        load_build(args.artifact_record, package, args.case, args.backend)
                        require(halo.bounded(destination/"request.json") == request_bytes, "staged request changed")
                        response = launcher.run(command)
                        text = "" if response is None else str(response)
                        require(len(text.encode()) <= contract.LOG_LIMIT, "launcher response bound")
                        (destination/"launcher-response.txt").write_text(text)
                    except BaseException as error:
                        pending = error
                        status["launch_error"] = repr(error)
                    finally:
                        for remote, name, limit in (("host.log", "host.log", contract.LOG_LIMIT),
                                                    ("host-exit.txt", "host-exit.txt", 128),
                                                    ("capture", "capture-download", ARCHIVE_LIMIT)):
                            try:
                                saved = download(launcher, remote, destination/name, limit)
                                status["downloaded"][remote] = dict(path=saved.name, bytes=saved.stat().st_size,
                                                                   sha256=halo.file_hash(saved, limit))
                            except Exception as error:
                                status["errors"][remote] = repr(error)
                        contract.write_json(destination/"collection.json", status)
                status["client_log_truncated"] = log.truncated
            if "capture" in status["downloaded"]:
                status["extraction"] = unpack_capture(destination/status["downloaded"]["capture"]["path"], destination/"capture", plan)
            require(pending is None and not status["errors"], "launch/collection incomplete; retained partial evidence")
        require(halo.bounded(destination/"host-exit.txt", 128).strip() == b"0", "worker failed")
        execution = contract.read_json(destination/"capture/execution.json")
        require(execution.get("execution_status") == "PASS", "worker execution incomplete")
        # Execution collection is distinct from strict numerical auditing. The
        # next registered audit step consumes this complete captured request.
        require(execution.get("identity") == plan["identity"], "collected execution identity")
        chunks.audit(destination/"capture/chunks", plan, execution["capture"]["manifest_sha256"])
        status.update(status="complete", execution_status="PASS", numerical_status="NOT_RUN")
    except BaseException as error:
        status["error"] = repr(error)
        raise
    finally:
        finalize(destination, status, start, "collection.json")
    require(status["status"] == "complete", "collection accounting incomplete")
    return status


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("compile", "run", "audit"):
        command = commands.add_parser(action)
        command.add_argument("--package", type=Path, required=True)
        command.add_argument("--case", default=contract.CASE_NAME)
        command.add_argument("--out", type=Path, required=True)
        if action == "audit":
            command.add_argument("--capture", type=Path, required=True)
        else:
            command.add_argument("--backend", choices=("local-sim", "hardware", "appliance-sim"), required=True)
        if action == "compile":
            command.add_argument("--namespace")
            command.add_argument("--compiler", type=Path)
        elif action == "run":
            command.add_argument("--artifact-record", type=Path, required=True)
            command.add_argument("--host-timeout", type=int, default=240)
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    {"compile": compile_program, "run": run_program, "audit": audit_program}[args.action](args)
    print(json.dumps(dict(status="complete", action=args.action, out=str(args.out))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
