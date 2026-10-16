#!/usr/bin/env python3
"""Compile or run one native16 condition through the Cerebras appliance client.

Each invocation needs a new --out directory. SDK imports happen only after
argument and source checks. Run exits 0 for pass, 1 for a numerical mismatch or
an unmatched payload representation,
and 2 for setup, transport, collection, or incomplete execution.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import logging
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parent
VARIANTS = {"baseline": None, "control": 0, "events": 1, "payload": 2}
MAX_JSON = 2 << 20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path, limit=1 << 30):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    require(path.stat().st_size <= limit, f"file exceeds {limit} bytes: {path}")
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_JSON,
            f"not a bounded JSON file: {path}")
    return json.loads(path.read_text())


def new_directory(path):
    path = Path(path).absolute()
    require(path == path.resolve(), f"output path must not traverse symlinks: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_identity(root, variant):
    directory = "source" if variant == "baseline" else "diagnostics"
    hosts = ["run.py"] if variant == "baseline" else ["run.py", "diagnostic.py"]
    require(set((root / directory).rglob("*.csl")) ==
            {root / directory / name for name in ("layout.csl", "pe.csl")},
            "source directory must contain exactly layout.csl and pe.csl")
    return {
        "source_dir": directory,
        "source_sha256": {str(Path(directory) / name): digest(root / directory / name)
                          for name in ("layout.csl", "pe.csl")},
        "host_sha256": {name: digest(root / name) for name in hosts},
    }


def parse_case(value):
    try:
        off, length, mode = map(int, value.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("case must be offset:length:mode") from error
    if not (0 <= off <= 31 and 0 <= length <= 16 and off + length <= 32 and mode in (0, 1)):
        raise argparse.ArgumentTypeError("case requires offset 0..31, length 0..16, offset+length <=32, mode 0 or 1")
    return f"{off}:{length}:{mode}"


def parse_offsets(value):
    try:
        x, y = map(int, value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("offsets must be nonnegative x,y") from error
    if min(x, y) < 0:
        raise argparse.ArgumentTypeError("offsets must be nonnegative x,y")
    return f"{x},{y}"


def sdk_options(namespace):
    options = dict(resource_cpu=1000, resource_mem=8 << 30, disable_version_check=False)
    if namespace:
        options["mgmt_namespace"] = namespace
    return options


def sdk_classes():
    # The logger import enables the appliance SDK's INFO job/version messages.
    from cerebras.appliance import logger  # noqa: F401
    from cerebras.sdk.client import SdkCompiler, SdkLauncher
    return SdkCompiler, SdkLauncher


@contextmanager
def client_log(destination):
    handler = logging.FileHandler(destination / "client.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        handler.close()


def compile_program(args, root=ROOT, compiler_type=None):
    identity = source_identity(root, args.variant)
    dims = "9,3" if args.backend == "simulator" else "762,1172"
    offsets = getattr(args, "fabric_offsets", "4,1")
    x, y = map(int, offsets.split(","))
    width, height = map(int, dims.split(","))
    require(x + 2 <= width and y + 1 <= height, "2x1 application exceeds fabric dimensions")
    destination = new_directory(args.out)
    flags = (f"--arch=wse3 --fabric-dims={dims} --fabric-offsets={offsets} "
             "--max-parallelism=1 --memcpy --channels=1 "
             "--width-west-buf=0 --width-east-buf=0 -o out")
    if args.variant != "baseline":
        flags += f" --params=OBSERVE:{VARIANTS[args.variant]}"
    record = dict(schema=1, action="compile", backend=args.backend, variant=args.variant,
                  compiler_args=flags, namespace=args.namespace, **identity)
    write_json(destination / "request.json", record)
    if compiler_type is None:
        compiler_type = sdk_classes()[0]
    with client_log(destination), compiler_type(**sdk_options(args.namespace)) as compiler:
        artifact = Path(compiler.compile(str(root / identity["source_dir"]),
                                        "layout.csl", flags, str(destination))).resolve()
    require(artifact.parent == destination, "compiler returned an artifact outside the requested directory")
    require(identity == source_identity(root, args.variant), "source or host changed while compiling")
    record.update(artifact_path=artifact.name, artifact_sha256=digest(artifact))
    write_json(destination / "artifact.json", record)
    print(destination / "artifact.json", flush=True)
    return 0


def load_record(path, backend, variant, root=ROOT):
    path = Path(path).resolve()
    record = read_json(path)
    require(record.get("schema") == 1, "artifact record schema is not supported; compile with this utility")
    require(record.get("backend") == backend, "artifact backend differs from requested backend")
    require(record.get("variant") == variant, "artifact variant differs from requested variant")
    identity = source_identity(root, variant)
    for key, value in identity.items():
        require(record.get(key) == value, f"compiled {key} differs from current files; use the compiled revision or recompile")
    name = record["artifact_path"]
    require(isinstance(name, str) and Path(name).name == name, "artifact_path must be a filename beside artifact.json")
    artifact = path.parent / name
    require(digest(artifact) == record["artifact_sha256"], "compile artifact hash mismatch")
    return record, artifact


def host_command(backend, variant, case, seconds):
    host = "run.py" if variant == "baseline" else "diagnostic.py"
    words = ["cs_python", host, "--name", "out", "--out", "capture", "--case", case,
             "--backend", backend]
    if variant != "baseline":
        words.extend(["--variant", variant])
    if backend == "hardware":
        words.extend(["--cmaddr", "%CMADDR%"])
    command = shlex.join(words)
    return (f"if timeout --signal=INT --kill-after=15s {seconds}s {command} "
            "> host.log 2>&1; then code=0; else code=$?; fi; "
            "printf '%s\\n' \"$code\" > host-exit.txt")


def capture_metadata(path):
    from inspect_result import read_capture, validate_result
    meta, calls, _ = read_capture(path)
    if meta.get("status") in ("pass", "observed_numeric_fail", "observed_payload_difference"):
        validate_result(meta, calls)
    return meta


def run_program(args, root=ROOT, launcher_type=None):
    record, artifact = load_record(args.artifact_record, args.backend, args.variant, root)
    destination = new_directory(args.out)
    command = host_command(args.backend, args.variant, args.case, args.host_timeout)
    request = dict(action="run", backend=args.backend, variant=args.variant, case=args.case,
                   command=command, compile_record=record)
    write_json(destination / "request.json", request)
    collection = dict(status="incomplete", downloaded={}, errors={})
    run_error = None
    meta = None
    if launcher_type is None:
        launcher_type = sdk_classes()[1]
    with client_log(destination), launcher_type(str(artifact), simulator=args.backend == "simulator",
                       **sdk_options(record.get("namespace"))) as launcher:
        try:
            for name, pin in record["host_sha256"].items():
                require(digest(root / name) == pin, f"host changed before staging: {name}")
                launcher.stage(str(root / name))
            require(source_identity(root, args.variant)["host_sha256"] == record["host_sha256"],
                    "host changed during staging")
            reply = launcher.run(command)
            (destination / "launcher-response.txt").write_text("" if reply is None else str(reply))
        except Exception as error:
            run_error = error
            (destination / "launcher-error.txt").write_text(repr(error) + "\n")
        finally:
            for remote in ("host.log", "host-exit.txt", "capture"):
                saved = destination / remote
                try:
                    launcher.download_artifact(remote, str(saved))
                    require(saved.is_file() and not saved.is_symlink(), f"download did not create {saved}")
                    if remote == "capture":
                        meta = capture_metadata(saved)
                    collection["downloaded"][remote] = str(saved)
                    print("downloaded", saved, flush=True)
                except Exception as error:
                    collection["errors"][remote] = repr(error)
                    (destination / (remote + "-download-error.txt")).write_text(repr(error) + "\n")
            collection["status"] = "collection_failed" if collection["errors"] else "complete"
            write_json(destination / "collection.json", collection)
    require(not collection["errors"], "collection_failed; inspect collection.json and launcher-error.txt")
    if run_error is not None:
        raise RuntimeError("launcher failed; collected files retained") from run_error
    exit_text = (destination / "host-exit.txt").read_text().strip()
    require(exit_text in ("0", "1"), f"host exit {exit_text!r}; inspect logs and partial capture")
    expected_config = dict(zip(("offset", "length", "mode"), map(int, args.case.split(":"))))
    require(meta.get("backend") == args.backend and meta.get("config") == expected_config,
            "capture backend/config does not match launch request")
    host = "run.py" if args.variant == "baseline" else "diagnostic.py"
    require(meta.get("host_sha256") == record["host_sha256"][host], "capture host hash differs from staged host")
    if args.variant != "baseline":
        require(meta.get("variant") == args.variant, "capture variant differs from launch request")
        require(meta.get("baseline_host_sha256") == record["host_sha256"]["run.py"],
                "capture imported baseline host hash differs from staged run.py")
    status = meta.get("status")
    require((exit_text == "0" and status == "pass") or
            (exit_text == "1" and status in ("observed_numeric_fail", "observed_payload_difference")),
            f"host exit/status disagree or execution is incomplete: {exit_text}/{status}")
    print(json.dumps(dict(status=status, host_exit=int(exit_text), out=str(destination)), sort_keys=True))
    return int(exit_text)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="action", required=True)
    for name in ("compile", "run"):
        command = commands.add_parser(name)
        command.add_argument("backend", choices=("simulator", "hardware"))
        command.add_argument("--variant", choices=tuple(VARIANTS), default="baseline")
        command.add_argument("--out", type=Path, required=True, help="new local output directory")
        if name == "compile":
            command.add_argument("--namespace", help="site-assigned namespace; omit for existing site default")
            command.add_argument("--fabric-offsets", type=parse_offsets, default="4,1",
                                 help="application x,y; compiler checks memcpy placement constraints")
        else:
            command.add_argument("--artifact-record", type=Path, required=True)
            command.add_argument("--case", type=parse_case, required=True)
            command.add_argument("--host-timeout", type=int, default=300, help="worker host timeout in seconds (1..1800)")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        if args.action == "run":
            require(1 <= args.host_timeout <= 1800, "host timeout must be in 1..1800 seconds")
            return run_program(args)
        return compile_program(args)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
