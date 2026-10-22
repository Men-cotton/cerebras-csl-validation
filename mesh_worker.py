#!/usr/bin/env python3
"""Capture the frozen resident MeshGraphNet SDK sequence on an appliance worker.

COMPLETED denotes a closed SDK capture. Numerical admission runs offline.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time
import uuid
import zipfile

import numpy as np


JSON_LIMIT = 2 << 20
ARRAY_LIMIT = 64 << 20
BACKENDS = ("hardware", "appliance-sim", "local-sim")
REQUEST_KEYS = {"schema", "run_id", "backend", "case_name", "case_sha256",
                "package_manifest_sha256", "artifact_sha256", "worker_sha256"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded_bytes(path, limit):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular file required: " + path.name)
    require(path.stat().st_size <= limit, "file exceeds bound: " + path.name)
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    require(len(payload) <= limit, "file grew beyond bound: " + path.name)
    return payload


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    require(len(payload) <= JSON_LIMIT, "JSON output exceeds bound")
    Path(path).write_bytes(payload)


def check_abi(manifest, out_json, rpc_json):
    """Validate decoded compiler metadata and return the variable ID mapping."""
    p = manifest["params"]
    compiled = out_json["params"]
    require(isinstance(compiled, dict) and set(compiled) == set(p), "compiled parameter inventory")
    require(all(type(v) in (str, int) for v in compiled.values()), "compiled parameter type")
    require({k: int(v) for k, v in compiled.items()} == p, "compiled parameters differ")
    inputs, outputs = manifest["inputs"], manifest["outputs"]
    require(isinstance(inputs, dict) and isinstance(outputs, dict) and not (set(inputs) & set(outputs)),
            "distinct input/output symbols")
    types = {"float32": "[*]f32", "uint32": "[*]u32", "uint16": "[*]u16"}
    expected = {}
    for name, spec in dict(inputs, **outputs).items():
        require(isinstance(spec, (list, tuple)) and len(spec) == 2 and
                type(spec[0]) is int and spec[0] > 0 and spec[1] in types, "symbol schema: " + name)
        expected[name] = ("Var", types[spec[1]])
    expected["run"] = ("Func", "void")
    symbols = rpc_json["rpc_symbols"]
    require(isinstance(symbols, list) and len(symbols) == len(expected), "compiled symbol count")
    require(len({s["name"] for s in symbols}) == len(symbols), "duplicate compiled symbol")
    require(all(type(s.get("id")) is int and s["id"] >= 0 for s in symbols), "compiled symbol ID")
    require(len({s["id"] for s in symbols}) == len(symbols), "duplicate compiled symbol ID")
    require({s["name"]: (s["kind"], s["type"]) for s in symbols} == expected,
            "compiled symbol ABI differs")
    entry = next(s for s in symbols if s["name"] == "run")
    require(entry.get("inputs") == [{"name": "steps", "type": "u16"}],
            "compiled run argument must be a single u16 step count")
    return {s["name"]: s["id"] for s in symbols if s["kind"] == "Var"}


def validate_request(request, backend, case_bytes, worker_bytes):
    require(isinstance(request, dict) and set(request) == REQUEST_KEYS, "request fields")
    require(type(request["schema"]) is int and request["schema"] == 1, "request schema")
    require(str(uuid.UUID(request["run_id"])) == request["run_id"], "canonical run UUID")
    require(request["backend"] == backend and backend in BACKENDS, "request backend")
    name = request["case_name"]
    require(isinstance(name, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)),
            "case name")
    for key in ("case_sha256", "package_manifest_sha256", "artifact_sha256", "worker_sha256"):
        require(isinstance(request[key], str) and bool(re.fullmatch(r"[0-9a-f]{64}", request[key])),
                "SHA256 field: " + key)
    require(request["case_sha256"] == sha(case_bytes), "case hash mismatch")
    require(request["worker_sha256"] == sha(worker_bytes), "worker hash mismatch")


def load_arrays(path, manifest):
    payload = bounded_bytes(path, ARRAY_LIMIT)
    require(sha(payload) == manifest["files"]["inputs.npz"], "input hash mismatch")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) <= 64 and sum(e.file_size for e in entries) <= ARRAY_LIMIT,
                "input NPZ expanded bound")
        require(len({e.filename for e in entries}) == len(entries), "duplicate NPZ member")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    require(set(arrays) == set(manifest["inputs"]), "input array inventory")
    p = manifest["params"]
    for name, value in arrays.items():
        count, dtype = manifest["inputs"][name]
        require(value.shape == (p["C"] * p["R"], count) and value.dtype == np.dtype(dtype),
                "input shape/dtype: " + name)
        require(bool(np.isfinite(value).all()), "nonfinite input: " + name)
    require(sha(bounded_bytes(path, ARRAY_LIMIT)) == sha(payload), "input changed during admission")
    return arrays


def sdk_classes():
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    return SdkRuntime, MemcpyDataType, MemcpyOrder


def sdk_identity(runtime_type):
    """Record available package/module facts without guessing a release number."""
    result = {"packages": {}, "modules": []}
    for package in ("cerebras_sdk", "cerebras_appliance", "numpy"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except Exception:
            result["packages"][package] = "unknown"
    for name in sorted({runtime_type.__module__, "cerebras.sdk", "cerebras.sdk.runtime"}):
        module = sys.modules.get(name)
        if module is None:
            continue
        record = {"module": name}
        version = vars(module).get("__version__")
        record["version"] = version if isinstance(version, str) else "unknown"
        filename = vars(module).get("__file__")
        if isinstance(filename, str):
            record["file_name"] = Path(filename).name
            try:
                record["file_sha256"] = sha(bounded_bytes(Path(filename).resolve(), ARRAY_LIMIT))
            except Exception as error:
                record["fingerprint_error"] = type(error).__name__
        result["modules"].append(record)
    return result


def capture(args, sdk_factory=sdk_classes, worker_path=None):
    case, build, destination = Path(args.case).resolve(), Path(args.name).resolve(), Path(args.out).absolute()
    require(destination == destination.resolve(), "output path traverses a symlink")
    destination.mkdir(parents=True, exist_ok=False)
    raw, api, timings, launches, runner = {}, [], [], [], None
    result = dict(status="FAILED")
    manifest, request_bytes = None, None
    start = time.monotonic()
    write_json(destination / "execution.json", result)

    def error_text(error):
        message = type(error).__name__ + ": " + str(error)
        if args.cmaddr:
            message = message.replace(args.cmaddr, "<cmaddr>")
        return message[:8192]

    def call(label, fn, *positional, **keywords):
        began = time.monotonic()
        print(json.dumps(dict(api=label, event="begin", elapsed_seconds=began-start)), flush=True)
        value = fn(*positional, **keywords)
        api.append(label)
        timings.append(dict(api=label, wall_seconds=time.monotonic()-began))
        print(json.dumps(dict(api=label, event="end", wall_seconds=timings[-1]["wall_seconds"])), flush=True)
        return value

    try:
        require(args.backend in BACKENDS and ((args.backend == "hardware") == bool(args.cmaddr)),
                "hardware requires cmaddr; simulator has none")
        request_bytes = bounded_bytes(args.request, JSON_LIMIT)
        result["request_sha256"] = sha(request_bytes)
        request = json.loads(request_bytes)
        case_bytes = bounded_bytes(case / "case.json", JSON_LIMIT)
        worker_bytes = bounded_bytes(worker_path or __file__, JSON_LIMIT)
        validate_request(request, args.backend, case_bytes, worker_bytes)
        result.update({key: request[key] for key in REQUEST_KEYS if key != "schema"})
        manifest = json.loads(case_bytes)
        p = manifest["params"]
        require(p.get("MODE") in (0, 1, 2), "resident transport mode")
        residency = manifest["residency"]
        require(isinstance(residency, dict) and residency.get("schema") == 1, "resident schema")
        chunks = residency["chunks"]
        require(isinstance(chunks, list) and bool(chunks) and
                all(type(s) is int and 0 < s <= 65535 for s in chunks) and sum(chunks) == p["STEPS"],
                "positive chunks must sum to STEPS")
        arrays = load_arrays(case / "inputs.npz", manifest)
        pes = p["C"] * p["R"]
        # All readbacks use 32-bit host carriers, including u16 device clocks.
        input_words = sum(spec[0] for spec in manifest["inputs"].values())
        output_words = sum(spec[0] for spec in manifest["outputs"].values())
        velocity_words = manifest["inputs"]["velocity"][0]
        raw_words = 20 + 2*input_words - velocity_words + len(chunks)*(output_words + velocity_words)
        require(4*pes*raw_words <= ARRAY_LIMIT - JSON_LIMIT, "planned raw capture bound")
        metadata = {"out.json": bounded_bytes(build / "out.json", JSON_LIMIT),
                    "out_rpc.json": bounded_bytes(build / "bin/out_rpc.json", JSON_LIMIT)}
        compiled_ids = check_abi(manifest, json.loads(metadata["out.json"]), json.loads(metadata["out_rpc.json"]))
        for name, payload in metadata.items():
            (destination / name).write_bytes(payload)
        result["compiled_sha256"] = {name: sha(payload) for name, payload in metadata.items()}
        runtime_type, datatype, order = sdk_factory()
        result["sdk_identity"] = sdk_identity(runtime_type)
        options = dict(cmaddr=args.cmaddr)
        if args.backend != "hardware":
            options.update(simfab_numthreads=1, suppress_simfab_trace=True)
        runner = call("construct", runtime_type, str(build), **options)
        schema = dict(manifest["inputs"], **manifest["outputs"])
        ids = {name: call("get_id:"+name, runner.get_id, name) for name in schema}
        require(ids == compiled_ids, "runtime symbol IDs differ from compiled ABI")
        call("load", runner.load)
        call("run", runner.run)

        def opts(dtype):
            return dict(streaming=False, nonblock=False, order=order.ROW_MAJOR,
                        data_type=datatype.MEMCPY_16BIT if dtype == "uint16" else datatype.MEMCPY_32BIT)

        def read(label, name):
            count, dtype = schema[name]
            value = np.full(p["C"]*p["R"]*count, np.nan if dtype == "float32" else 0xdead,
                            np.float32 if dtype == "float32" else np.uint32)
            try:
                call("d2h:"+label, runner.memcpy_d2h, value, ids[name], 0, 0, p["C"], p["R"], count,
                     **opts(dtype))
            finally:
                raw[label] = value.reshape(p["C"]*p["R"], count).copy()

        for name, value in arrays.items():
            call("h2d:"+name, runner.memcpy_h2d, ids[name], value.ravel(), 0, 0, p["C"], p["R"],
                 value.shape[1], **opts(str(value.dtype)))
        read("initial_stats", "stats")
        read("initial_lifecycle", "lifecycle")
        for name in arrays:
            read("before_"+name, name)
        for index, steps in enumerate(chunks):
            call("launch:run", runner.launch, "run", np.uint16(steps), nonblock=False)
            launches.append(dict(entrypoint="run", steps=int(steps)))
            for name in manifest["outputs"]:
                read("chunk%d_%s" % (index, name), name)
            read("chunk%d_velocity" % index, "velocity")
        for name in arrays:
            if name != "velocity":
                read("after_"+name, name)
    except BaseException as error:
        result["error"] = error_text(error)
    finally:
        try:
            if runner is not None:
                call("stop", runner.stop)
        except BaseException as error:
            result["stop_error"] = error_text(error)
        result.update(api=api, api_timings=timings, wall_seconds=time.monotonic()-start, launches=launches)
        try:
            if request_bytes is not None:
                require(bounded_bytes(args.request, JSON_LIMIT) == request_bytes, "request changed during capture")
        except BaseException as error:
            result["integrity_error"] = error_text(error)
        try:
            require(sum(value.nbytes for value in raw.values()) <= ARRAY_LIMIT, "raw capture bound")
            np.savez(destination / "raw.npz", **raw)
            result["raw_sha256"] = sha(bounded_bytes(destination / "raw.npz", ARRAY_LIMIT))
        except BaseException as error:
            result["capture_error"] = error_text(error)
        if not any(key in result for key in ("error", "stop_error", "integrity_error", "capture_error")):
            result["status"] = "COMPLETED"
        write_json(destination / "execution.json", result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--name", type=Path, default=Path("out"))
    result.add_argument("--case", type=Path, default=Path("."))
    result.add_argument("--out", type=Path, default=Path("capture"))
    result.add_argument("--request", type=Path, default=Path("request.json"))
    result.add_argument("--backend", choices=BACKENDS, required=True)
    result.add_argument("--cmaddr")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return 0 if capture(args)["status"] == "COMPLETED" else 2
    except BaseException as error:
        # The capture may fail before its fresh output directory is admitted.
        print("worker failed: " + type(error).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
