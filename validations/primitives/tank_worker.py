#!/usr/bin/env python3
"""Capture a bounded declarative SDK program; numerical validation stays offline."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid
import zipfile

import numpy as np

JSON_LIMIT = 4 << 20
ARRAY_LIMIT = 96 << 20
BACKENDS = ("hardware", "appliance-sim", "local-sim")
REQUEST_KEYS = {"schema", "run_id", "backend", "case_name", "case_sha256", "package_manifest_sha256", "artifact_sha256", "worker_sha256"}
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded_bytes(path, limit):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, "bounded regular file: " + path.name)
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    require(len(payload) <= limit, "input grew beyond bound")
    return payload


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(len(payload) <= JSON_LIMIT, "JSON bound")
    Path(path).write_bytes(payload)


def arrays(path):
    """Bound NPY headers before constructing arrays; NaN/Inf bits are legitimate guards."""
    payload = bounded_bytes(path, ARRAY_LIMIT)
    result = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        require(len(entries) <= 256 and len({e.filename for e in entries}) == len(entries) and
                sum(e.file_size for e in entries) <= ARRAY_LIMIT, "NPZ inventory/expanded bound")
        for entry in entries:
            require(entry.filename.endswith(".npy") and Path(entry.filename).name == entry.filename and
                    not entry.flag_bits & 1, "NPZ member")
            name = entry.filename[:-4]
            require(NAME.fullmatch(name), "array name")
            data = archive.read(entry)
            stream = io.BytesIO(data)
            version = np.lib.format.read_magic(stream)
            require(version in ((1, 0), (2, 0)), "NPY version")
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(stream, max_header_size=16384)
            require(dtype in (np.dtype("float32"), np.dtype("uint32")) and not fortran and
                    len(shape) == 2 and all(type(n) is int and 0 < n <= ARRAY_LIMIT for n in shape), "NPY schema")
            size = math.prod(shape) * dtype.itemsize
            require(size == len(data) - stream.tell() and size <= ARRAY_LIMIT, "NPY byte bound")
            result[name] = np.frombuffer(data, dtype=dtype, count=math.prod(shape), offset=stream.tell()).reshape(shape)
    return result


def validate_manifest(case):
    require(case.get("schema") == 1 and case.get("kind") in ("topk", "halo"), "case schema/kind")
    grid, symbols, functions = case["grid"], case["symbols"], case["functions"]
    require(isinstance(grid, list) and len(grid) == 2 and all(type(v) is int and 2 <= v <= 64 for v in grid), "2..64 PE axes")
    require(isinstance(case["params"], dict) and 1 <= len(case["params"]) <= 16 and
            all(NAME.fullmatch(k) and type(v) is int and 0 <= v <= 65535 for k, v in case["params"].items()), "compile parameters")
    require(isinstance(symbols, dict) and 1 <= len(symbols) <= 32 and isinstance(functions, dict) and
            1 <= len(functions) <= 8 and not set(symbols) & set(functions), "symbol/function inventory")
    for name, spec in symbols.items():
        require(NAME.fullmatch(name) and isinstance(spec, list) and len(spec) == 2 and
                type(spec[0]) is int and 0 < spec[0] <= 16384 and spec[1] in ("float32", "uint32", "uint16"), "symbol schema")
    for name, arguments in functions.items():
        require(NAME.fullmatch(name) and isinstance(arguments, list) and len(arguments) <= 8 and
                all(set(a) == {"name", "type"} and NAME.fullmatch(a["name"]) and a["type"] in ("i16", "u16") for a in arguments), "function schema")
    operations = case["operations"]
    require(isinstance(operations, list) and 1 <= len(operations) <= 256, "operation count")
    inputs, outputs, raw_bytes = {}, {}, 0
    for operation in operations:
        op, name = operation["op"], operation["symbol"]
        if op == "launch":
            require(set(operation) == {"op", "symbol", "args"} and name in functions and
                    isinstance(operation["args"], list) and len(operation["args"]) == len(functions[name]), "launch schema")
            for value, argument in zip(operation["args"], functions[name]):
                lower, upper = (-32768, 32767) if argument["type"] == "i16" else (0, 65535)
                require(type(value) is int and lower <= value <= upper, "typed RPC argument")
            continue
        require(op in ("h2d", "d2h") and name in symbols and
                set(operation) <= {"op", "symbol", "array", "rect", "clock"} and
                {"op", "symbol", "array", "rect"} <= set(operation), "transfer schema")
        key, rectangle = operation["array"], operation["rect"]
        require(isinstance(key, str) and NAME.fullmatch(key) and isinstance(rectangle, list) and len(rectangle) == 4 and
                all(type(v) is int for v in rectangle), "transfer array/rectangle")
        x, y, width, height = rectangle
        require(0 <= x < grid[0] and 0 <= y < grid[1] and width > 0 and height > 0 and
                x + width <= grid[0] and y + height <= grid[1], "transfer exceeds grid")
        count, dtype = symbols[name]
        shape = (width * height, count)
        host_dtype = "float32" if dtype == "float32" else "uint32"
        if op == "h2d":
            require("clock" not in operation, "input clock flag")
            require(key not in inputs or inputs[key] == (shape, host_dtype), "reused input schema differs")
            inputs[key] = (shape, host_dtype)
        else:
            require(key not in outputs, "duplicate output key")
            require("clock" not in operation or operation["clock"] is True and dtype == "uint16" and count == 6, "clock carrier schema")
            outputs[key] = (shape, host_dtype)
            raw_bytes += width * height * count * 4
    require(raw_bytes <= ARRAY_LIMIT - JSON_LIMIT and outputs and
            sum(math.prod(shape) * 4 for shape, _ in inputs.values()) <= ARRAY_LIMIT - JSON_LIMIT, "capture/input allocation bound")
    require(set(case.get("validations", {})) == {"restore"} and isinstance(case["validations"]["restore"], list), "validation schema")
    for pair in case["validations"]["restore"]:
        require(isinstance(pair, list) and len(pair) == 2 and all(n in outputs for n in pair) and outputs[pair[0]] == outputs[pair[1]], "restoration pair")
    return inputs, outputs


def check_abi(case, out_json, rpc_json):
    validate_manifest(case)
    compiled = out_json["params"]
    require(isinstance(compiled, dict) and set(compiled) == set(case["params"]) and
            all(type(v) in (str, int) for v in compiled.values()) and
            {k: int(v) for k, v in compiled.items()} == case["params"], "compiled parameters")
    types = {"float32": "[*]f32", "uint32": "[*]u32", "uint16": "[*]u16"}
    wanted = {name: ("Var", types[spec[1]]) for name, spec in case["symbols"].items()}
    wanted.update({name: ("Func", "void") for name in case["functions"]})
    symbols = rpc_json["rpc_symbols"]
    require(isinstance(symbols, list) and len(symbols) == len(wanted) and
            len({s["name"] for s in symbols}) == len(symbols) and
            {s["name"]: (s["kind"], s["type"]) for s in symbols} == wanted, "compiled symbol inventory/types")
    variables = [s for s in symbols if s["kind"] == "Var"]
    require(all(type(s.get("id")) is int and s["id"] >= 0 for s in variables) and
            len({s["id"] for s in variables}) == len(variables), "distinct compiled variable IDs")
    for symbol in symbols:
        if symbol["kind"] == "Func":
            require(symbol.get("inputs", []) == case["functions"][symbol["name"]], "compiled RPC argument ABI")
    return {s["name"]: s["id"] for s in variables}


def validate_request(request, backend, case_bytes, worker_bytes):
    require(isinstance(request, dict) and set(request) == REQUEST_KEYS and type(request["schema"]) is int and request["schema"] == 1, "request schema")
    require(str(uuid.UUID(request["run_id"])) == request["run_id"] and request["backend"] == backend and backend in BACKENDS, "request UUID/backend")
    require(isinstance(request["case_name"], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", request["case_name"]), "case name")
    for key in ("case_sha256", "package_manifest_sha256", "artifact_sha256", "worker_sha256"):
        require(isinstance(request[key], str) and re.fullmatch(r"[0-9a-f]{64}", request[key]), "request hash")
    require(request["case_sha256"] == sha(case_bytes) and request["worker_sha256"] == sha(worker_bytes), "request case/worker identity")


def operation_record(operation, symbols, ids):
    if operation["op"] == "launch":
        return dict(operation)
    _, dtype = symbols[operation["symbol"]]
    return dict(operation, symbol_id=ids[operation["symbol"]], length=symbols[operation["symbol"]][0],
                host_dtype="float32" if dtype == "float32" else "uint32", memcpy_bits=16 if dtype == "uint16" else 32,
                streaming=False, nonblock=False, order="ROW_MAJOR")


def sdk_classes():
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    return SdkRuntime, MemcpyDataType, MemcpyOrder


def sdk_identity(runtime_type):
    result = dict(packages={}, runtime_module=runtime_type.__module__)
    for package in ("cerebras_sdk", "cerebras_appliance", "numpy"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except Exception:
            result["packages"][package] = "unknown"
    module = sys.modules.get(runtime_type.__module__)
    filename = getattr(module, "__file__", None)
    if isinstance(filename, str):
        result["runtime_file"] = Path(filename).name
        try:
            result["runtime_sha256"] = sha(bounded_bytes(Path(filename).resolve(), ARRAY_LIMIT))
        except Exception as error:
            result["runtime_hash_error"] = type(error).__name__
    return result


def capture(args, sdk_factory=sdk_classes, worker_path=None):
    case_dir, build = Path(args.case).resolve(), Path(args.name).resolve()
    destination = Path(args.out).absolute()
    require(destination == destination.resolve(), "output path symlink")
    destination.mkdir(parents=True, exist_ok=False)
    result, raw, api, timings, completed, runner = dict(status="FAILED"), {}, [], [], [], None
    request_bytes, case_bytes, input_bytes = None, None, None
    start = time.monotonic()
    write_json(destination / "execution.json", result)

    def error_text(error):
        value = type(error).__name__ + ": " + str(error)
        return value.replace(args.cmaddr, "<cmaddr>")[:8192] if args.cmaddr else value[:8192]

    def call(label, function, *positional, _cleanup=False, **keywords):
        began = time.monotonic()
        log_error = None
        try:
            print(json.dumps(dict(api=label, event="begin", elapsed_seconds=began-start)), flush=True)
        except BaseException as error:
            if not _cleanup:
                raise
            log_error = error
        value = function(*positional, **keywords)
        api.append(label)
        timings.append(dict(api=label, wall_seconds=time.monotonic()-began))
        try:
            print(json.dumps(dict(api=label, event="end", wall_seconds=timings[-1]["wall_seconds"])), flush=True)
        except BaseException as error:
            if not _cleanup:
                raise
            log_error = log_error or error
        if log_error is not None:
            raise log_error
        return value

    try:
        require(args.backend in BACKENDS and ((args.backend == "hardware") == bool(args.cmaddr)), "hardware requires cmaddr")
        request_bytes = bounded_bytes(args.request, JSON_LIMIT)
        case_bytes = bounded_bytes(case_dir / "case.json", JSON_LIMIT)
        request = json.loads(request_bytes)
        validate_request(request, args.backend, case_bytes, bounded_bytes(worker_path or __file__, JSON_LIMIT))
        result.update({key: request[key] for key in REQUEST_KEYS if key != "schema"})
        result["request_sha256"] = sha(request_bytes)
        case = json.loads(case_bytes)
        input_schema, _ = validate_manifest(case)
        input_bytes = bounded_bytes(case_dir / "inputs.npz", ARRAY_LIMIT)
        require(sha(input_bytes) == case["files"]["inputs.npz"], "input file identity")
        inputs = arrays(case_dir / "inputs.npz")
        require(set(inputs) == set(input_schema), "input inventory")
        for name, (shape, dtype) in input_schema.items():
            require(inputs[name].shape == shape and inputs[name].dtype == np.dtype(dtype), "input shape/type")
        metadata = {"out.json": bounded_bytes(build / "out.json", JSON_LIMIT),
                    "out_rpc.json": bounded_bytes(build / "bin/out_rpc.json", JSON_LIMIT)}
        ids = check_abi(case, json.loads(metadata["out.json"]), json.loads(metadata["out_rpc.json"]))
        for name, payload in metadata.items():
            (destination / name).write_bytes(payload)
        result["compiled_sha256"] = {n: sha(v) for n, v in metadata.items()}
        runtime_type, datatype, order = sdk_factory()
        result["sdk_identity"] = sdk_identity(runtime_type)
        options = dict(cmaddr=args.cmaddr)
        if args.backend != "hardware":
            options.update(simfab_numthreads=1, suppress_simfab_trace=True)
        result["runtime_options"] = dict(cmaddr="<provided>" if args.cmaddr else None,
                                          **({} if args.backend == "hardware" else dict(simfab_numthreads=1, suppress_simfab_trace=True)))
        def construct(*positional, **keywords):
            nonlocal runner
            runner = runtime_type(*positional, **keywords)
            return runner
        call("construct", construct, str(build), **options)
        observed_ids = {name: call("get_id:" + name, runner.get_id, name) for name in case["symbols"]}
        require(observed_ids == ids, "runtime IDs differ from compiled ABI")
        result["symbol_ids"] = ids
        call("load", runner.load)
        call("run", runner.run)
        for index, operation in enumerate(case["operations"]):
            op, name = operation["op"], operation["symbol"]
            label = "%03d:%s:%s" % (index, op, name)
            if op == "launch":
                values = [(np.int16 if argument["type"] == "i16" else np.uint16)(value)
                          for argument, value in zip(case["functions"][name], operation["args"])]
                call(label, runner.launch, name, *values, nonblock=False)
            else:
                key, (x, y, width, height) = operation["array"], operation["rect"]
                count, dtype = case["symbols"][name]
                flags = dict(streaming=False, nonblock=False, order=order.ROW_MAJOR,
                             data_type=datatype.MEMCPY_16BIT if dtype == "uint16" else datatype.MEMCPY_32BIT)
                if op == "h2d":
                    value = inputs[key].copy().ravel()
                    call(label, runner.memcpy_h2d, ids[name], value, x, y, width, height, count, **flags)
                else:
                    value = np.full(width * height * count, 0xdeadbeef, np.uint32)
                    if dtype == "float32":
                        value = value.view(np.float32)
                    try:
                        call(label, runner.memcpy_d2h, value, ids[name], x, y, width, height, count, **flags)
                    finally:
                        raw[key] = value.reshape(width * height, count).copy()
            completed.append(operation_record(operation, case["symbols"], ids))
    except BaseException as error:
        result["error"] = error_text(error)
    finally:
        try:
            if runner is not None:
                call("stop", runner.stop, _cleanup=True)
        except BaseException as error:
            result["stop_error"] = error_text(error)
        result.update(api=api, api_timings=timings, operations=completed, wall_seconds=time.monotonic()-start)
        try:
            if request_bytes is not None:
                require(bounded_bytes(args.request, JSON_LIMIT) == request_bytes, "request changed during capture")
            if case_bytes is not None:
                require(bounded_bytes(case_dir / "case.json", JSON_LIMIT) == case_bytes, "case changed during capture")
            if input_bytes is not None:
                require(bounded_bytes(case_dir / "inputs.npz", ARRAY_LIMIT) == input_bytes, "input changed during capture")
        except BaseException as error:
            result["integrity_error"] = error_text(error)
        try:
            require(sum(v.nbytes for v in raw.values()) <= ARRAY_LIMIT, "raw byte bound")
            np.savez(destination / "raw.npz", **raw)
            result["raw_sha256"] = sha(bounded_bytes(destination / "raw.npz", ARRAY_LIMIT))
        except BaseException as error:
            result["capture_error"] = error_text(error)
        if not any(k in result for k in ("error", "stop_error", "integrity_error", "capture_error")):
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


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        raise SystemExit(0 if capture(args)["status"] == "COMPLETED" else 2)
    except Exception as error:
        print("worker failed: " + type(error).__name__, file=sys.stderr)
        raise SystemExit(2)
