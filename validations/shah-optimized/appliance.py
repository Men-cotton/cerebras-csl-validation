#!/usr/bin/env python3
"""One foreground compile, run, or offline audit of frozen optimized SpMM."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import logging
from pathlib import Path
import shlex
import sys
import uuid

from compiled import archive_inventory

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ("layout.csl", "pe.csl")
HOST_FILES = ("driver.py", "oracle.py", "worker.py", "compiled.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path, limit=256 << 20):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    require(path.stat().st_size <= limit, f"file exceeds {limit} bytes: {path}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path):
    digest(path, 2 << 20)
    return json.loads(Path(path).read_text())


def new_directory(path):
    path = Path(path).absolute()
    require(path == path.resolve(), "output path traverses symlinks")
    path.mkdir(parents=True, exist_ok=False)
    return path


def source_identity(root):
    require(set((root / "source").rglob("*.csl")) ==
            {root / "source" / name for name in SOURCE_FILES}, "unexpected CSL source inventory")
    return dict(source_sha256={name: digest(root / "source" / name) for name in SOURCE_FILES},
                host_sha256={name: digest(root / name) for name in HOST_FILES},
                audit_sha256=digest(root / "saved_audit.py"))


def admitted_config(args):
    params = {name: getattr(args, name) for name in ("R", "F", "V", "Q", "B")}
    require(all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in params.values()),
            "all dimensions must be positive integers")
    R, F, V, Q, B = [params[k] for k in ("R", "F", "V", "Q", "B")]
    require(Q & (Q - 1) == 0, "Q must be a power of two")
    require(V >= 2 and max(V, Q, R + B, F + 1) <= 32760, "prototype i16/fixture bounds")
    require(args.channels in (1, 2, 4), "only channels 1, 2 and 4 are admitted")
    require(args.channels <= R + B, "more channels than application rows")
    require(F + 8 <= 762 and R + B + 2 <= 1172, "application exceeds hardware fabric")
    # Keep this initial mechanism campaign's host captures and input generation bounded.
    require(R * V * F <= 1 << 20 and B * Q * F <= 1 << 20 and R * B * Q <= 1 << 20,
            "case exceeds initial host/capture budget")
    require(args.workload in ("adversary", "random", "balanced", "paired-skew"), "unknown workload")
    require(isinstance(args.seed, int) and args.seed >= 0 and isinstance(args.degree, int) and args.degree > 0,
            "invalid seed or degree")
    if args.workload in ("balanced", "paired-skew"):
        require(not args.empty, "controlled and empty are mutually exclusive")
        require(args.workload != "paired-skew" or R % 2 == 0, "paired-skew needs even R")
        factor = 2 if args.workload == "paired-skew" else 1
        require(B * Q * R * args.degree <= 1_000_000 and
                2 * B * Q * R * (factor * args.degree + 1) <= 8_000_000,
                "controlled fixture stream bound exceeded")
    else:
        require((B * Q + 4) * args.degree <= 1_000_000 and
                2 * R * ((B * Q + 4) * args.degree + B * Q) <= 8_000_000,
                "random fixture stream bound exceeded")
        require(not (args.workload == "random" and args.empty), "random and empty are mutually exclusive")
    return dict(params=params, channels=args.channels, empty=bool(args.empty),
                workload=args.workload, seed=args.seed, degree=args.degree)


def sdk_options(namespace):
    options = dict(resource_cpu=1000, resource_mem=8 << 30, disable_version_check=False)
    if namespace:
        options["mgmt_namespace"] = namespace
    return options


def sdk_classes():
    from cerebras.appliance import logger  # noqa: F401; enables job-ID logging
    from cerebras.sdk.client import SdkCompiler, SdkLauncher
    return SdkCompiler, SdkLauncher


@contextmanager
def client_log(destination):
    handler = logging.FileHandler(destination / "client.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        yield
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def compile_program(args, root=ROOT, compiler_type=None):
    config = admitted_config(args)
    identity = source_identity(root)
    destination = new_directory(args.out)
    p = config["params"]
    dims = "762,1172" if args.backend == "hardware" else f"{p['F'] + 8},{p['R'] + p['B'] + 2}"
    params = ",".join(f"{key}:{value}" for key, value in p.items())
    params += ",MEMCPYH2D_DATA_1_ID:0,MEMCPYD2H_DATA_1_ID:1"
    flags = (f"--arch=wse3 --fabric-dims={dims} --fabric-offsets=4,1 "
             f"--params={params} --memcpy --channels={config['channels']} --max-parallelism=1 -o out")
    record = dict(schema=1, backend=args.backend, namespace=args.namespace,
                  compiler_args=flags, **config, **identity)
    write_json(destination / "request.json", record)
    try:
        if compiler_type is None:
            compiler_type = sdk_classes()[0]
        with client_log(destination), compiler_type(**sdk_options(args.namespace)) as compiler:
            artifact = Path(compiler.compile(str(root / "source"), "layout.csl", flags,
                                            str(destination))).resolve()
        require(artifact.parent == destination, "artifact outside compile output")
        require(identity == source_identity(root), "source changed during compilation")
        record.update(artifact_path=artifact.name, artifact_sha256=digest(artifact),
                      compiled_inventory=archive_inventory(artifact))
        write_json(destination / "build.json", record)
    except Exception as error:
        write_json(destination / "compile-error.json", dict(error=repr(error)))
        raise
    print(destination / "build.json", flush=True)
    return 0


def load_record(path, backend, root=ROOT):
    path = Path(path).resolve()
    record = read_json(path)
    require(record.get("schema") == 1 and record.get("backend") == backend, "build schema/backend mismatch")
    for key, value in source_identity(root).items():
        require(record.get(key) == value, f"compiled {key} changed")
    name = record.get("artifact_path")
    require(isinstance(name, str) and Path(name).name == name, "artifact must be adjacent filename")
    artifact = path.parent / name
    require(digest(artifact) == record.get("artifact_sha256"), "compile artifact changed")
    require(archive_inventory(artifact) == record.get("compiled_inventory"), "compiled archive inventory mismatch")
    return record, artifact


def host_command(backend, seconds):
    require(isinstance(seconds, int) and 1 <= seconds <= 86400, "worker timeout must be 1..86400s")
    words = ["cs_python", "worker.py", "--record", "case.json", "--name", "out", "--out", "capture"]
    if backend == "hardware":
        words += ["--cmaddr", "%CMADDR%"]
    return (f"if timeout --signal=INT --kill-after=15s {seconds}s {shlex.join(words)} "
            "> host.log 2>&1; then code=0; else code=$?; fi; "
            "printf '%s\\n' \"$code\" > host-exit.txt")


def downloaded_path(returned, requested, limit):
    requested = Path(requested).absolute()
    require(returned is None or isinstance(returned, str) and returned, "invalid returned download path")
    path = requested if returned is None else Path(returned)
    if not path.is_absolute():
        path = requested.parent / path
    require(not path.is_symlink(), "download symlink")
    path = path.resolve()
    require(path.parent == requested.parent and path.name not in
            {"request.json", "case.json", "collection.json", "client.log", "launcher-response.txt",
             "launcher-error.txt"}, "download escapes or collides")
    digest(path, limit)
    return path


def run_program(args, root=ROOT, launcher_type=None):
    record, artifact = load_record(args.artifact_record, args.backend, root)
    command = host_command(args.backend, args.host_timeout)
    destination = new_directory(args.out)
    require(args.output_mode in ("buffered", "per-chunk"), "invalid output mode")
    run_id = str(uuid.uuid4())
    request = dict(command=command, compile_record=record, output_mode=args.output_mode, run_id=run_id)
    write_json(destination / "request.json", request)
    write_json(destination / "case.json", dict(compile_record=record, output_mode=args.output_mode, run_id=run_id))
    collection = dict(run_id=run_id, downloaded={}, errors={})
    run_error = None
    if launcher_type is None:
        launcher_type = sdk_classes()[1]
    with client_log(destination), launcher_type(str(artifact), simulator=args.backend == "simulator",
                                               **sdk_options(record.get("namespace"))) as launcher:
        try:
            for name, pin in record["host_sha256"].items():
                require(digest(root / name) == pin, "host changed before staging")
                launcher.stage(str(root / name))
            for name, pin in record["source_sha256"].items():
                require(digest(root / "source" / name) == pin, "CSL changed before staging")
                launcher.stage(str(root / "source" / name))
            launcher.stage(str(destination / "case.json"))
            require(source_identity(root) == {key: record[key] for key in source_identity(root)},
                    "source changed during staging")
            reply = launcher.run(command)
            (destination / "launcher-response.txt").write_text("" if reply is None else str(reply))
        except Exception as error:
            run_error = error
            (destination / "launcher-error.txt").write_text(repr(error) + "\n")
        finally:
            for remote, limit in (("host.log", 8 << 20), ("host-exit.txt", 128), ("capture", 64 << 20)):
                requested = destination / remote
                try:
                    saved = downloaded_path(launcher.download_artifact(remote, str(requested)), requested, limit)
                    require(saved.name not in {item["path"] for item in collection["downloaded"].values()},
                            "download aliases another artifact")
                    collection["downloaded"][remote] = dict(path=saved.name, sha256=digest(saved, limit))
                    print("downloaded", saved, flush=True)
                except Exception as error:
                    collection["errors"][remote] = repr(error)
            collection["status"] = "collection_failed" if collection["errors"] else "complete"
            write_json(destination / "collection.json", collection)
    require(not collection["errors"], "collection failed; partial artifacts retained")
    if run_error is not None:
        raise RuntimeError("launcher failed; artifacts retained") from run_error
    exit_path = destination / collection["downloaded"]["host-exit.txt"]["path"]
    require(exit_path.read_text().strip() == "0", "worker did not exit successfully")
    # The separate host audit step remains mandatory for numerical acceptance.
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for action in ("compile", "run"):
        c = sub.add_parser(action)
        c.add_argument("--backend", choices=("hardware", "simulator"), default="hardware")
        c.add_argument("--out", type=Path, required=True)
        if action == "compile":
            c.add_argument("--namespace", default="job-operator")
            for name, default in (("R", 2), ("F", 2), ("V", 3), ("Q", 2), ("B", 3)):
                c.add_argument("--" + name, type=int, default=default)
            c.add_argument("--channels", type=int, choices=(1, 2, 4), default=4)
            c.add_argument("--empty", action="store_true")
            c.add_argument("--workload", choices=("adversary", "random", "balanced", "paired-skew"), default="adversary")
            c.add_argument("--seed", type=int, default=7)
            c.add_argument("--degree", type=int, default=8)
        else:
            c.add_argument("--artifact-record", type=Path, required=True)
            c.add_argument("--host-timeout", type=int, default=86400)
            c.add_argument("--output-mode", choices=("buffered", "per-chunk"), default="buffered")
    c = sub.add_parser("audit")
    c.add_argument("--capture", type=Path, required=True, help="run output directory")
    c.add_argument("--out", type=Path, required=True, help="new audit JSON filename")
    c.add_argument("--artifact-record", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        if args.action == "compile":
            return compile_program(args)
        if args.action == "run":
            return run_program(args)
        from saved_audit import audit
        require(not args.out.exists(), "fresh audit output required")
        result = audit(args.capture, args.artifact_record, ROOT)
        write_json(args.out, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
