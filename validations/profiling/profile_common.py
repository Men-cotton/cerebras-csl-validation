"""Portable TANK29 capture admission and work accounting; no SDK imports."""
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import uuid
import zipfile

import numpy as np
import oracle

BASE_NAMES = ("source", "received", "control", "stats", "clocks", "scratch_mirror")
HOST_NAMES = {"profile_worker.py", "profile_common.py", "oracle.py"}
SOURCE_NAMES = {"source/layout.csl", "source/pe.csl"}
BACKENDS = {"hardware", "simulator", "local-sim"}
PHASES = ("arm_to_rx_callback", "rx_callback_to_validation_end",
          "validation_end_to_pre_ack", "pre_ack_to_ack_callback")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compiled_n4(params):
    require(isinstance(params, dict) and set(params) == {"N"} and
            ((type(params["N"]) is int and params["N"] == 4) or
             (type(params["N"]) is str and params["N"] == "4")), "compiled parameters")


def digest(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 8 << 20,
            "bounded regular file: " + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1 << 20,
            "bounded regular JSON: " + str(path))
    return json.loads(path.read_text())


def write_json(path, value):
    path = Path(path)
    require(not path.is_symlink() and (not path.exists() or path.is_file()), "regular JSON destination")
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    require(len(encoded) <= 1 << 20, "JSON size limit")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def local_file(folder, name):
    relative = PurePosixPath(name)
    require(isinstance(name, str) and name and not relative.is_absolute() and
            ".." not in relative.parts and str(relative) == name, "relative capture path")
    path = folder / name
    require(path.resolve() == path and path.resolve().is_relative_to(folder), "capture symlink or path escape")
    return path


def shape(n=4):
    require(type(n) is int and n == 4, "N4 feasibility profile only")
    return dict(source=6*n+3, received=2*n+3, control=2, stats=32, clocks=12,
                scratch_mirror=2*n+3, profile_control=1, profile_records=25,
                profile_status=4, perf_samples=12)


def fixture(n, flag, invocation):
    shape(n)
    require(type(flag) is int and flag in (0, 1), "profiling flag must be 0 or 1")
    value = oracle.fixture(0, n, invocation, 1)
    value.update(profile_control=[[flag], [flag]], profile_records=[[0]*25 for _ in range(2)],
                 profile_status=[[0]*4 for _ in range(2)], perf_samples=[[0]*12 for _ in range(2)])
    return value


def u48(words):
    require(len(words) == 3 and all(type(word) is int and 0 <= word < 65536 for word in words),
            "u48 limbs")
    return sum(word << (16*i) for i, word in enumerate(words))


def delta(start, end):
    value = (end-start) & ((1 << 48)-1)
    require(0 < value < 1 << 47, "positive unambiguous modular interval")
    return value


def raw_shapes(n):
    return {f"{direction}{i}_{name}": (2, count) for i in range(3)
            for direction in ("input", "output") for name, count in shape(n).items()}


def load_raw(path, n):
    """Validate every NPY header/extent before NumPy can allocate an array."""
    digest(path)
    expected = raw_shapes(n)
    raw = {}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require(len(members) == len(expected) and
                {item.filename for item in members} == {name + ".npy" for name in expected},
                "complete unique raw archive members")
        for item in members:
            require(not item.is_dir() and 0 < item.file_size <= 4096, "bounded NPY member")
            content = archive.read(item)
            stream = io.BytesIO(content)
            version = np.lib.format.read_magic(stream)
            require(version in ((1, 0), (2, 0)), "supported NPY header")
            reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                      else np.lib.format.read_array_header_2_0)
            actual_shape, fortran, dtype = reader(stream)
            name = item.filename[:-4]
            require(actual_shape == expected[name] and dtype == np.dtype(np.uint32) and not fortran,
                    "NPY shape/type/order")
            require(len(content)-stream.tell() == math.prod(actual_shape)*4, "exact NPY body extent")
            raw[name] = np.load(io.BytesIO(content), allow_pickle=False)
    return raw


def expected_calls(n=4):
    names = ["construct"] + ["id:" + name for name in shape(n)] + ["load", "run"]
    for i in range(3):
        names += ([f"in{i}:" + name for name in shape(n)] + [f"prepare{i}", f"launch{i}"] +
                  [f"out{i}:" + name for name in shape(n)])
    return names + ["stop"]


def validate_raw(raw, n, flag, oracle_module=None):
    reference = oracle if oracle_module is None else oracle_module
    require(type(flag) is int and flag in (0, 1), "profiling flag must be 0 or 1")
    require(set(raw) == set(raw_shapes(n)), "complete three-invocation arrays")
    rows = []
    for i in range(3):
        wanted = fixture(n, flag, i)
        actual = {}
        for name, count in shape(n).items():
            for direction in ("input", "output"):
                array = raw[f"{direction}{i}_{name}"]
                require(array.shape == (2, count) and array.dtype == np.uint32, "array shape/type")
            require(raw[f"input{i}_{name}"].tobytes() == np.asarray(wanted[name], np.uint32).tobytes(),
                    "input bytes: " + name)
            actual[name] = raw[f"output{i}_{name}"].tolist()
        sender = reference.audit({name: actual[name] for name in BASE_NAMES}, 0, n, i, 1)
        require(actual["profile_control"] == [[flag], [flag]], "profiling flag retained")
        require(actual["profile_status"] == [[i+1, 0, 0, 0], [i+1, 5 if flag else 0, 0, 0]],
                "generation/count/overflow/error")
        require(actual["profile_records"][0] == [0]*25, "sender receiver-record slots remain zero")
        require(actual["perf_samples"] == [[0]*12, [0]*12], "raw counters remain disabled")
        events, spans = [], []
        if flag:
            records = actual["profile_records"][1]
            for index in range(5):
                record = records[5*index:5*index+5]
                require(record[:2] == [i+1, index+1], "ordered generation and event ID")
                events.append(u48(record[2:]))
            spans = [delta(events[j], events[j+1]) for j in range(4)]
            require(delta(events[0], events[-1]) == sum(spans), "unambiguous complete receiver interval")
        else:
            require(actual["profile_records"][1] == [0]*25, "disabled observer slots remain zero")
        rows.append(dict(invocation=i+1, input_case="ABA"[i], sender=sender,
                         receiver_event_timestamps=events,
                         receiver_phase_cycles=dict(zip(PHASES, spans))))
    work = reference.costs(0, n, 1)
    extra_per_pe = 4*(1+25+4+12)
    work["base_exported_array_bytes_per_pe"] = work["exported_array_bytes_per_pe"]
    work["exported_array_bytes_per_pe"] = sum(shape(n).values())*4
    work["arrays_bytes_per_pe"] += extra_per_pe + 6  # Additional private u16 timestamp triple.
    work["arrays_scope"] = "Exported arrays plus physical scratch, two private timestamp triples and token; scalar state, descriptors, pointers, code and stack excluded."
    work.update(useful_flops=0, profiling_export_bytes_per_pe=extra_per_pe,
                exported_bytes_per_pe=sum(shape(n).values())*4,
                h2d_bytes_per_invocation=2*sum(shape(n).values())*4,
                d2h_bytes_per_invocation=2*sum(shape(n).values())*4,
                h2d_bytes_per_capture=6*sum(shape(n).values())*4,
                d2h_bytes_per_capture=6*sum(shape(n).values())*4,
                receiver_timestamp_reads_per_invocation=5 if flag else 0,
                scope="Source logical work/endpoint traffic bound to complete oracle witnesses; not hardware instruction or stall counters.")
    return dict(status="pass", n=n, flag=flag, arrays=len(raw), invocations=rows,
                work=work, independent_runtimes=1, nested_invocations=3)


def identity(value, names, label):
    require(isinstance(value, dict) and set(value) == names and
            all(isinstance(item, str) and SHA256.fullmatch(item) for item in value.values()), label)
    return value


def audit_capture(folder):
    folder = Path(folder).absolute()
    require(folder.is_dir() and folder.resolve() == folder, "regular capture directory")
    record = read_json(local_file(folder, "capture.json"))
    manifest = read_json(local_file(folder, "manifest.json"))
    request = read_json(local_file(folder, "request.json"))
    require(record.get("status") == "pass" and record.get("n") == 4, "complete N4 capture")
    require(record.get("backend") in BACKENDS and manifest.get("backend") == record["backend"], "capture backend")
    require(type(manifest.get("schema")) is int and type(request.get("schema")) is int and
            manifest.get("schema") == request.get("schema") == 1 and manifest.get("params") == {"N": 4} and
            type(manifest["params"]["N"]) is int,
            "manifest/request schema and parameters")
    run_id = record.get("run_id")
    require(isinstance(run_id, str), "runtime UUID")
    require(str(uuid.UUID(run_id)) == run_id and request.get("run_id") == run_id, "runtime UUID binding")
    flag = record.get("flag")
    require(type(flag) is int and flag in (0, 1) and type(request.get("flag")) is int and request["flag"] == flag,
            "request profiling flag binding")
    manifest_hash = digest(local_file(folder, "manifest.json"))
    require(record.get("manifest_sha256") == request.get("manifest_sha256") == manifest_hash,
            "manifest identity binding")
    require(record.get("request_sha256") == digest(local_file(folder, "request.json")), "request identity binding")
    for label, names in (("host_sha256", HOST_NAMES), ("source_sha256", SOURCE_NAMES)):
        pins = identity(manifest.get(label), names, label)
        require(record.get(label) == pins, "capture " + label)
        require(all(digest(local_file(folder, "sources/" + name)) == value for name, value in pins.items()),
                "frozen " + label)
    raw_path = local_file(folder, "raw.npz")
    require(record.get("raw_sha256") == digest(raw_path), "raw identity")
    calls = record.get("api")
    require(isinstance(calls, list) and all(isinstance(row, dict) for row in calls), "API records")
    require([row.get("name") for row in calls] == expected_calls(4) and
            all(row.get("status") == "complete" for row in calls), "closed exact API sequence")
    if "expected_api" in record:
        require(record["expected_api"] == expected_calls(4), "declared API sequence")
    pins = identity(record.get("compiled_metadata_sha256"), {"out.json", "out_rpc.json"}, "compiled metadata")
    require(all(digest(local_file(folder, name)) == value for name, value in pins.items()), "compiled metadata identity")
    compiled = read_json(local_file(folder, "out.json"))
    compiled_n4(compiled["params"])
    symbols = read_json(local_file(folder, "out_rpc.json"))["rpc_symbols"]
    require(len(symbols) == 12 and len({symbol["name"] for symbol in symbols}) == 12 and
            len({symbol["id"] for symbol in symbols}) == 12, "unique complete RPC ABI")
    abi = {symbol["name"]: (symbol["kind"], symbol["type"], [arg["type"] for arg in symbol.get("inputs", [])])
           for symbol in symbols}
    expected = {name: ("Var", "[*]u32", []) for name in shape(4)}
    expected.update(prepare=("Func", "void", []), run_probe=("Func", "void", []))
    require(abi == expected, "compiled RPC types")
    result = validate_raw(load_raw(raw_path, 4), 4, flag)
    result.update(run_id=run_id, backend=record["backend"], manifest_sha256=manifest_hash,
                  request_sha256=record["request_sha256"], raw_sha256=record["raw_sha256"],
                  host_sha256=manifest["host_sha256"], source_sha256=manifest["source_sha256"],
                  native_calls=len(calls))
    return result
