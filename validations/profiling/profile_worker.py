"""Bounded two-PE timestamp capture for local SIM or one appliance runtime."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np
import profile_common as common

ROOT = Path(__file__).resolve().parent


def sdk():
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    return SdkRuntime, dict(streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                           order=MemcpyOrder.ROW_MAJOR, nonblock=False)


def validate_inputs(args, root=ROOT):
    manifest = common.read_json(args.manifest)
    request = common.read_json(args.request)
    common.require(set(request) == {"schema", "run_id", "flag", "manifest_sha256"} and
                   request["schema"] == 1, "run request schema")
    common.require(str(uuid.UUID(request["run_id"])) == request["run_id"], "canonical run UUID")
    common.require(type(request["flag"]) is int and request["flag"] in (0, 1), "only flags 0/1")
    common.require(request["manifest_sha256"] == common.digest(args.manifest), "manifest identity")
    common.require(manifest["schema"] == 1 and manifest["params"] == {"N": 4}, "N4 build contract")
    common.require(args.backend == manifest["backend"] and
                   args.backend in ("hardware", "simulator", "local-sim"), "backend identity")
    common.require((args.backend == "hardware") == bool(args.cmaddr), "hardware address only")
    common.require(set(manifest["host_sha256"]) == {"profile_worker.py", "profile_common.py", "oracle.py"},
                   "host inventory")
    common.require(set(manifest["source_sha256"]) == {"source/layout.csl", "source/pe.csl"}, "CSL inventory")
    for name, pin in manifest["host_sha256"].items():
        common.require(common.digest(root / name) == pin, "host identity: " + name)
    # SdkLauncher stages each file at the remote workdir's top level.
    for name, pin in manifest["source_sha256"].items():
        location = root / name if args.backend == "local-sim" else root / Path(name).name
        common.require(common.digest(location) == pin, "CSL identity: " + name)
    build = Path(args.name).resolve()
    compiled = common.read_json(build / "out.json")
    common.compiled_n4(compiled["params"])
    symbols = common.read_json(build / "bin/out_rpc.json")["rpc_symbols"]
    common.require(len(symbols) == 12 and len({s["name"] for s in symbols}) == 12 and
                   len({s["id"] for s in symbols}) == 12, "unique complete RPC ABI")
    abi = {s["name"]: (s["kind"], s["type"], [a["type"] for a in s.get("inputs", [])]) for s in symbols}
    expected = {name: ("Var", "[*]u32", []) for name in common.shape(4)}
    expected.update(prepare=("Func", "void", []), run_probe=("Func", "void", []))
    common.require(abi == expected, "compiled RPC types")
    return manifest, request, build, {s["name"]: s["id"] for s in symbols if s["kind"] == "Var"}


def capture(args, sdk_factory=sdk, root=ROOT):
    manifest, request, build, compiled_ids = validate_inputs(args, root)
    destination = Path(args.out).absolute()
    common.require(destination == destination.resolve(), "capture path cannot traverse symlinks")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "sources/source").mkdir(parents=True)
    for name in manifest["host_sha256"]:
        shutil.copyfile(root / name, destination / "sources" / name)
    for name in manifest["source_sha256"]:
        source = root / name if args.backend == "local-sim" else root / Path(name).name
        shutil.copyfile(source, destination / "sources" / name)
    shutil.copyfile(args.manifest, destination / "manifest.json")
    shutil.copyfile(args.request, destination / "request.json")
    shutil.copyfile(build / "out.json", destination / "out.json")
    shutil.copyfile(build / "bin/out_rpc.json", destination / "out_rpc.json")
    packages = {}
    for name in ("numpy", "cerebras-sdk", "cerebras-appliance"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unknown"
    record = dict(schema=1, status="incomplete", backend=args.backend, n=4, flag=request["flag"],
                  run_id=request["run_id"], manifest_sha256=common.digest(args.manifest),
                  request_sha256=common.digest(args.request), host_sha256=manifest["host_sha256"],
                  source_sha256=manifest["source_sha256"], python=sys.version, packages=packages,
                  compiled_metadata_sha256={name: common.digest(destination / name)
                                            for name in ("out.json", "out_rpc.json")},
                  clock_scope="same-PE intervals; no cross-PE clock subtraction")
    common.write_json(destination / "capture.json", record)
    raw, journal, runtime, failure = {}, [], None, None
    previous = Path.cwd()

    def call(name, function, *positional, **keyword):
        row = dict(name=name, status="started")
        journal.append(row)
        started = time.monotonic()
        print("BEGIN " + name, flush=True)
        try:
            result = function(*positional, **keyword)
            row["status"] = "complete"
            return result
        except BaseException as error:
            row.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            row["wall_seconds"] = time.monotonic() - started

    try:
        runtime_type, options = sdk_factory()
        os.chdir(destination)
        runtime_options = {"cmaddr": args.cmaddr} if args.backend == "hardware" else {
            "simfab_numthreads": 1, "suppress_simfab_trace": True}

        def construct():
            nonlocal runtime
            runtime = runtime_type(str(build), **runtime_options)

        call("construct", construct)
        ids = {name: call("id:" + name, runtime.get_id, name) for name in common.shape(4)}
        common.require(ids == compiled_ids, "runtime symbol IDs")
        call("load", runtime.load)
        call("run", runtime.run)
        for invocation in range(3):
            inputs = common.fixture(4, request["flag"], invocation)
            for name, count in common.shape(4).items():
                data = np.ascontiguousarray(inputs[name], dtype=np.uint32)
                raw[f"input{invocation}_{name}"] = data.copy()
                call(f"in{invocation}:" + name, runtime.memcpy_h2d,
                     ids[name], data.ravel(), 0, 0, 2, 1, count, **options)
            call(f"prepare{invocation}", runtime.launch, "prepare", nonblock=False)
            call(f"launch{invocation}", runtime.launch, "run_probe", nonblock=False)
            for name, count in common.shape(4).items():
                data = np.zeros((2, count), np.uint32)
                call(f"out{invocation}:" + name, runtime.memcpy_d2h,
                     data.ravel(), ids[name], 0, 0, 2, 1, count, **options)
                raw[f"output{invocation}_{name}"] = data
    except BaseException as error:
        failure = type(error).__name__ + ": " + str(error)
    finally:
        if runtime is not None:
            try:
                call("stop", runtime.stop)
            except BaseException as error:
                failure = (failure + "; " if failure else "") + "stop: " + str(error)
        os.chdir(previous)
        np.savez(destination / "raw.npz", **raw)
        record.update(api=journal, raw_sha256=common.digest(destination / "raw.npz"), error=failure)
        if failure is None:
            try:
                common.validate_raw(raw, 4, request["flag"])
                common.require([row["name"] for row in journal] == common.expected_calls(4), "native API sequence")
                record["status"] = "pass"
            except (ValueError, KeyError) as error:
                failure = str(error)
                record.update(status="invalid", error=failure)
        common.write_json(destination / "capture.json", record)
    common.require(failure is None, failure or "capture failed")
    common.write_json(destination / "audit.json", common.audit_capture(destination))
    print("PASS: bounded phase capture " + request["run_id"], flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--backend", choices=("hardware", "simulator", "local-sim"), required=True)
    parser.add_argument("--cmaddr")
    args = parser.parse_args(argv)
    try:
        return capture(args)
    except (ValueError, OSError, KeyError) as error:
        print("error: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
