"""Bounded portable native GNN execution admission and local numerical audit."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile
import tempfile
import time
import zipfile
import numpy as np

JSON_LIMIT = 4 << 20
INVENTORY_JSON_LIMIT = 16 << 20
ELF_TOTAL_LIMIT = 1 << 30
CASE_LIMIT = 128 << 20
CAPTURE_LIMIT = 256 << 20
CASE_REQUIRED = {"inputs.npz", "expected.npz", "zero_masks.npz", "origin-case.json", "source/layout.csl", "source/pe.csl"}
HOST_FILES = ("appliance.py", "appliance_worker.py", "saved_audit.py", "sdk_runner.py")

def require(ok, message):
    if not ok:
        raise ValueError(message)


def bounded(path, limit):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular file required: " + str(path))
    with path.open("rb") as stream:
        payload = stream.read(limit+1)
    require(len(payload) <= limit, "file exceeds bound: " + str(path))
    return payload


def sha(path, limit=CASE_LIMIT):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular file required: " + str(path))
    result, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            size += len(chunk)
            require(size <= limit, "file exceeds hash bound")
            result.update(chunk)
    return result.hexdigest()


def read_json(path, *, limit=JSON_LIMIT):
    require(type(limit) is int and 0 < limit <= INVENTORY_JSON_LIMIT, "JSON read limit")
    return json.loads(bounded(path, limit))


def write_json(path, value, *, limit=JSON_LIMIT):
    require(type(limit) is int and 0 < limit <= INVENTORY_JSON_LIMIT, "JSON write limit")
    path = Path(path)
    require(not path.is_symlink() and (not path.exists() or path.is_file()), "regular JSON destination")
    content = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    require(len(content) <= limit, "JSON output bound")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="."+path.name, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def local_file(folder, name):
    require(isinstance(name, str) and name, "nonempty relative path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == name,
            "canonical relative path")
    folder = Path(folder).absolute()
    path = folder/name
    require(path.resolve() == path and path.is_relative_to(folder), "symlink or path escape")
    return path


def load_npz(path, limit=CASE_LIMIT):
    """Bound ZIP members and validate NPY extents before NumPy allocation."""
    bounded(path, limit)
    result, total = {}, 0
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) <= 256, "NPZ entry bound")
        for item in entries:
            name = item.filename
            require(name.endswith(".npy") and "/" not in name and "\\" not in name
                    and name[:-4] not in result and not item.is_dir(), "NPZ unique flat members")
            total += item.file_size
            require(0 <= item.file_size <= limit and total <= limit, "NPZ expanded byte bound")
            payload = archive.read(item)
            stream = io.BytesIO(payload)
            version = np.lib.format.read_magic(stream)
            require(version in ((1, 0), (2, 0)), "NPY version")
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(stream)
            require(not fortran and not dtype.hasobject and dtype.kind in "fiu"
                    and dtype.itemsize in (2, 4, 8), "NPY type/order")
            require(len(payload)-stream.tell() == math.prod(shape)*dtype.itemsize, "exact NPY extent")
            result[name[:-4]] = np.load(io.BytesIO(payload), allow_pickle=False)
    return result


def check_abi(build, manifest):
    build = Path(build)
    actual = read_json(build/"out.json")["params"]
    require({k: int(v) for k, v in actual.items()} == manifest["params"], "compiled parameter ABI")
    symbols = read_json(build/"bin/out_rpc.json")["rpc_symbols"]
    expected = {name: ("Var", {"float32": "[*]f32", "uint32": "[*]u32", "uint16": "[*]u16"}[dtype], ())
                for name, (_, dtype) in dict(manifest["inputs"], **manifest["outputs"]).items()}
    expected["run"] = ("Func", "void", ())
    require(len(symbols) == len(expected) and len({s["name"] for s in symbols}) == len(symbols)
            and len({s["id"] for s in symbols}) == len(symbols), "unique complete compiled symbols")
    require({s["name"]: (s["kind"], s["type"], tuple(arg["type"] for arg in s.get("inputs", [])))
             for s in symbols} == expected, "compiled symbol ABI")


def compile_workers(value):
    require(type(value) is int and value in (1, 4, 8), "compile workers must be 1, 4 or 8")
    return value


def compiler_arguments(params, backend, workers):
    workers = compile_workers(workers)
    require(backend in ("hardware", "simulator"), "compiler backend")
    dimensions = "762,1172" if backend == "hardware" else f"{params['C']+7},{params['R']+2}"
    return (f"--arch=wse3 --fabric-dims={dimensions} --fabric-offsets=4,1 --params="
            + ",".join(f"{k}:{v}" for k, v in sorted(params.items()))
            + f" --max-parallelism={workers} --memcpy --channels=1 -o out")


def compile_binding(record):
    workers = compile_workers(record.get("compile_workers"))
    require(type(record.get("compile_resource_cpu")) is int
            and record["compile_resource_cpu"] == workers * 1000, "compiler CPU resource binding")
    require(record.get("compiler_args") == compiler_arguments(record["params"], record["backend"], workers),
            "compiler worker/argument binding")
    return workers


def unpack_archive(path, destination, prefix, limit, required=()):
    bounded(path, limit)
    destination = Path(destination).absolute()
    require(destination == destination.resolve(), "archive destination symlink")
    destination.mkdir(parents=True, exist_ok=False)
    names, total, count = set(), 0, 0
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            count += 1
            total += member.size
            name = PurePosixPath(member.name)
            require(count <= 256 and 0 <= member.size <= limit and total <= limit, "archive byte/entry bound")
            require(not name.is_absolute() and ".." not in name.parts and name.parts
                    and name.parts[0] == prefix and str(name) not in names, "archive path or duplicate")
            require(member.isdir() or member.isfile(), "archive link or special file")
            names.add(str(name))
            relative = name.relative_to(prefix)
            if not relative.parts:
                require(member.isdir(), "archive root directory")
                continue
            target = local_file(destination, str(relative))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream:
                    payload = stream.read(limit+1)
                require(len(payload) == member.size, "archive member extent")
                with target.open("xb") as stream:
                    stream.write(payload)
    for name in required:
        require(local_file(destination, name).is_file(), "missing archive member: " + name)
    return destination


def _schema(value, label):
    require(isinstance(value, dict) and 1 <= len(value) <= 64, label + " schema size")
    for name, field in value.items():
        require(isinstance(name, str) and re.fullmatch("[a-z][a-z0-9_]{0,63}", name), "symbol name")
        require(isinstance(field, list) and len(field) == 2 and type(field[0]) is int
                and 1 <= field[0] <= 32767 and field[1] in ("float32", "uint32", "uint16"), label + " field schema")


def tolerance(value):
    require(isinstance(value, dict) and set(value) == {"rtol", "atol"}
            and all(type(value[n]) in (int, float) and math.isfinite(value[n])
                    and 0 <= value[n] <= 1e-3 for n in value), "bounded numerical tolerance")
    return {name: float(value[name]) for name in ("rtol", "atol")}


def mask_arrays(masks, expected, label, *, float_outputs=False):
    require(isinstance(masks, dict) and set(masks) <= set(expected), label + " field inventory")
    for name, value in masks.items():
        require(isinstance(value, np.ndarray) and value.dtype == np.dtype("uint32")
                and value.shape == expected[name].shape and np.all((value == 0) | (value == 1)), label + " ABI")
        if float_outputs:
            require(expected[name].dtype == np.dtype("float32"), label + " float output")
    return masks


def exact_masks(case, manifest, expected):
    declared = "exact_mask_counts" in manifest
    require(declared == ("exact_masks.npz" in manifest["files"]), "exact-mask frozen file declaration")
    if not declared:
        return {}
    counts = manifest["exact_mask_counts"]
    require(isinstance(counts, dict) and set(counts) <= set(expected)
            and all(type(count) is int and count >= 0 for count in counts.values()), "exact-mask coverage declaration")
    masks = mask_arrays(load_npz(Path(case) / "exact_masks.npz"), expected, "exact-mask", float_outputs=True)
    require(set(masks) == set(counts) and all(int(value.sum(dtype=np.uint64)) == counts[name]
            for name, value in masks.items()), "exact-mask coverage")
    return masks


def load(case):
    case = Path(case).absolute()
    manifest = read_json(local_file(case, "case.json"))
    require(manifest.get("schema") == 1 and manifest.get("family") == "native-gnn", "native case identity")
    params = manifest.get("params")
    require(isinstance(params, dict) and {"C", "R", "H"} <= set(params)
            and len(params) <= 32 and all(isinstance(k, str) and re.fullmatch("[A-Z][A-Z0-9_]*", k)
                and type(v) is int and 0 <= v <= 32767 for k, v in params.items()), "bounded parameters")
    require(params["H"] == 128 and 2 <= params["C"] <= 755 and 2 <= params["R"] <= 1170
            and params["C"] * params["R"] <= 32767, "native H128 fabric bounds")
    _schema(manifest.get("inputs"), "input")
    _schema(manifest.get("outputs"), "output")
    inputs, outputs = manifest["inputs"], manifest["outputs"]
    require(all(field[1] in ("float32", "uint32") for field in inputs.values()), "native inputs use 32-bit carriers")
    require(all(inputs[n] == outputs[n] for n in set(inputs) & set(outputs)), "inout ABI")
    require(outputs.get("clocks") == [6, "uint16"] and outputs.get("stats", [0, ""])[1] == "uint32", "clock/stats ABI")
    require(manifest.get("launch") == {"entrypoint": "run", "arguments": []}, "single zero-argument launch")
    immutable = manifest.get("immutable_inputs")
    require(isinstance(immutable, list) and len(immutable) == len(set(immutable))
            and set(immutable) <= set(inputs) and set(inputs) - set(immutable) <= set(outputs), "input mutation contract")
    checks = manifest.get("checks")
    require(isinstance(checks, dict) and set(checks) == set(outputs), "complete numerical contract")
    for name, check in checks.items():
        require(isinstance(check, dict), "field check")
        kind = check.get("kind")
        if name == "clocks":
            require(check == {"kind": "clock"}, "clock check")
        elif outputs[name][1] != "float32":
            require(check == {"kind": "exact"}, "integer fields require exact checks")
        elif kind == "exact":
            require(set(check) == {"kind"}, "exact check fields")
        else:
            require(kind == "approx" and set(check) == {"kind", "rtol", "atol"}, "bounded numerical tolerance")
            tolerance({n: check[n] for n in ("rtol", "atol")})
    require(isinstance(manifest.get("scope"), str) and 1 <= len(manifest["scope"]) <= 8192, "explicit scope")
    files = manifest.get("files")
    require(isinstance(files, dict) and CASE_REQUIRED <= set(files) and len(files) <= 128, "frozen case inventory")
    total = (case / "case.json").stat().st_size
    for name, digest in files.items():
        require(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest), "SHA256 encoding")
        path = local_file(case, name)
        require(sha(path) == digest, "frozen file changed: " + name)
        total += path.stat().st_size
        require(total <= CASE_LIMIT, "case total byte bound")
    require(manifest.get("origin_case_sha256") == sha(case / "origin-case.json"), "origin manifest identity")
    origin = read_json(case / "origin-case.json")
    mapping = manifest.get("origin_files")
    require(isinstance(origin.get("files"), dict) and isinstance(mapping, dict)
            and set(mapping) == set(origin["files"]) and len(set(mapping.values())) == len(mapping), "origin file mapping")
    require(all(files.get(mapping[name]) == digest for name, digest in origin["files"].items()), "origin frozen bytes")
    actual = load_npz(case / "inputs.npz")
    require(set(actual) == set(inputs), "input array inventory")
    pes = params["C"] * params["R"]
    for name, (count, dtype) in inputs.items():
        require(actual[name].shape == (pes, count) and actual[name].dtype == np.dtype(dtype), "input array ABI: " + name)
    expected = load_npz(case / "expected.npz")
    require(set(expected) == set(outputs) - {"clocks"}, "expected array inventory")
    for name, value in expected.items():
        count, dtype = outputs[name]
        require(value.shape == (pes, count) and value.dtype == np.dtype("uint32" if dtype == "uint16" else dtype)
                and np.isfinite(value).all(), "expected finite array ABI: " + name)
    masks = mask_arrays(load_npz(case / "zero_masks.npz"), expected, "zero-mask")
    for name, value in masks.items():
        require(np.all(expected[name][value != 0] == 0), "zero-mask oracle consistency")
    exact_masks(case, manifest, expected)
    raw_words = (2 * sum(field[0] for field in inputs.values())
                 + sum(field[0] for field in outputs.values()) + outputs["stats"][0]) * pes
    require(total + raw_words * 4 + (32 << 20) <= CAPTURE_LIMIT,
            "projected complete capture exceeds admission limit")
    return manifest


def api_sequence(manifest):
    return (["construct"] + ["get_id:" + n for n in dict(manifest["inputs"], **manifest["outputs"])]
            + ["load", "run"] + ["h2d:" + n for n in manifest["inputs"]]
            + ["d2h:before_" + n for n in manifest["inputs"]]
            + ["d2h:initial_stats", "launch:run"]
            + ["d2h:" + n for n in manifest["outputs"]]
            + ["d2h:after_" + n for n in manifest["inputs"]] + ["stop"])


def execution_arrays(case, output):
    manifest = load(case)
    execution = read_json(Path(output) / "execution.json")
    require(execution.get("status") == "COMPLETED" and "error" not in execution
            and "stop_error" not in execution and execution.get("backend") in ("hardware", "simulator"), "completed execution backend")
    require(execution.get("case_sha256") == sha(Path(case) / "case.json")
            and execution.get("raw_sha256") == sha(Path(output) / "raw.npz", CAPTURE_LIMIT), "bound execution arrays")
    wanted_calls = api_sequence(manifest)
    require(execution.get("api") == wanted_calls, "exact SDK API sequence")
    timings = execution.get("api_timings", [])
    require([entry.get("api") for entry in timings] == wanted_calls
            and all(type(entry.get("wall_seconds")) in (int, float)
                    and math.isfinite(entry["wall_seconds"]) and entry["wall_seconds"] >= 0 for entry in timings), "complete SDK timing inventory")
    raw = load_npz(Path(output) / "raw.npz", CAPTURE_LIMIT)
    schema = dict(manifest["outputs"], initial_stats=manifest["outputs"]["stats"])
    for name, field in manifest["inputs"].items():
        schema["before_" + name] = field
        schema["after_" + name] = field
    require(set(raw) == set(schema), "complete raw capture inventory")
    pes = manifest["params"]["C"] * manifest["params"]["R"]
    for name, (count, dtype) in schema.items():
        require(raw[name].shape == (pes, count)
                and raw[name].dtype == np.dtype("uint32" if dtype == "uint16" else dtype), "capture ABI: " + name)
    # Deliberately no value, finiteness, counter or clock checks here.
    return manifest, execution, raw


def admit_execution(case, output):
    manifest, execution, raw = execution_arrays(case, output)
    return dict(status="PASS", execution_status="PASS", numerical_status="NOT_RUN",
                backend=execution["backend"], case_sha256=execution["case_sha256"],
                raw_sha256=execution["raw_sha256"], arrays=len(raw), params=manifest["params"],
                claim="SDK calls and bound capture complete; numerical correctness not checked")


def _elf_storage(payload):
    require(len(payload) >= 16 and payload[:4] == b"\x7fELF" and payload[4] in (1, 2)
            and payload[5] in (1, 2), "generated ELF header")
    wide, endian = payload[4] == 2, "<" if payload[5] == 1 else ">"
    header_fmt = endian + ("HHIQQQIHHHHHH" if wide else "HHIIIIIHHHHHH")
    require(len(payload) >= 16 + struct.calcsize(header_fmt), "generated ELF header extent")
    header = struct.unpack_from(header_fmt, payload, 16)
    offset, size, count = header[5], header[10], header[11]
    section_fmt = endian + ("IIQQQQIIQQ" if wide else "IIIIIIIIII")
    require(0 < count <= 4096 and size >= struct.calcsize(section_fmt)
            and offset >= 16 + struct.calcsize(header_fmt) and offset + size * count <= len(payload), "generated ELF section extent")
    ranges, outside = [], []
    for index in range(count):
        section = struct.unpack_from(section_fmt, payload, offset + index * size)
        require(section[1] == 8 or section[5] == 0 or section[4] + section[5] <= len(payload), "ELF section payload")
        if section[2] & 2 and section[5]:
            target = ranges if section[3] < 48 * 1024 else outside
            target.append([section[3], section[3] + section[5]])
    return dict(static_sram_end=max((end for _, end in ranges), default=0),
                sram_ranges=ranges, outside_sram_ranges=outside)


def compiled_inventory(build):
    """Read only generated kernels, never compiler/runtime binary internals."""
    build = Path(build).absolute()
    require(build == build.resolve(), "compiled directory symlink")
    start, total, entries = time.monotonic(), 0, 0
    inventory = {}
    def scan_error(error):
        raise error
    for directory, subdirs, files in os.walk(build, followlinks=False, onerror=scan_error):
        entries += len(subdirs) + len(files)
        require(entries <= 20000 and time.monotonic() - start <= 30, "compiled inventory scan bound")
        for name in subdirs:
            require(not (Path(directory) / name).is_symlink(), "compiled symlink directory")
        for name in sorted(files):
            require(time.monotonic() - start <= 30, "compiled inventory deadline")
            path = Path(directory) / name
            require(not path.is_symlink(), "compiled symlink file")
            if not name.endswith(".elf"):
                continue
            payload = bounded(path, 32 << 20)
            total += len(payload)
            require(total <= ELF_TOTAL_LIMIT, "compiled ELF total byte bound")
            relative = str(path.relative_to(build))
            inventory[relative] = dict(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), **_elf_storage(payload))
    require(inventory and len(inventory) <= 16384, "generated ELF inventory required")
    require(time.monotonic() - start <= 30, "compiled inventory deadline")
    return dict(files=inventory, artifact_bytes=total,
                max_static_sram_end=max(row["static_sram_end"] for row in inventory.values()),
                scope="worker-observed generated ELF hashes and allocated low-address sections before launch; stack/runtime high-water unmeasured")


def admit_compiled_inventory(inventory):
    files = inventory.get("files")
    require(isinstance(files, dict) and 0 < len(files) <= 16384, "compiled ELF inventory")
    require(all(isinstance(row, dict) and type(row.get("bytes")) is int and 0 < row["bytes"] <= 32 << 20
                and isinstance(row.get("sha256"), str) and re.fullmatch("[0-9a-f]{64}", row["sha256"])
                for row in files.values()), "compiled ELF entry bounds")
    total = sum(row["bytes"] for row in files.values())
    require(total <= ELF_TOTAL_LIMIT and inventory.get("artifact_bytes") == total, "compiled ELF total byte bound")
    ends = [row.get("static_sram_end") for row in files.values()]
    require(all(type(end) is int and end >= 0 for end in ends), "compiled SRAM endpoints")
    require(inventory.get("max_static_sram_end") == max(ends), "compiled SRAM maximum")
    require(max(ends) <= 44 * 1024, "compiled static SRAM exceeds 44 KiB admission")


def admit_capture(folder):
    folder = Path(folder).absolute()
    record, build, request = (read_json(folder / n) for n in ("capture.json", "manifest.json", "request.json"))
    require(record.get("status") == "COMPLETED" and record.get("backend") == build.get("backend")
            and build.get("backend") in ("hardware", "simulator"), "completed capture backend")
    require(record.get("run_id") == request.get("run_id") and
            record.get("manifest_sha256") == request.get("manifest_sha256") == sha(folder / "manifest.json"), "capture manifest binding")
    require(record.get("request_sha256") == sha(folder / "request.json") and request.get("case_name") == build.get("case_name"), "capture request binding")
    compile_binding(build)
    require(set(build.get("host_sha256", {})) == set(HOST_FILES), "captured host inventory")
    for name, digest in build["host_sha256"].items():
        require(sha(local_file(folder / "host", name)) == digest, "captured host source")
    manifest = load(folder / "case")
    require(sha(folder / "case/case.json") == build.get("case_sha256")
            and manifest["params"] == build.get("params") and manifest["files"] == build.get("case_files"), "captured case identity")
    check_abi(folder / "build", manifest)
    metadata = record.get("compiled_metadata_sha256", {})
    require(set(metadata) == {"out.json", "bin/out_rpc.json", "inventory.json"}, "compiled capture metadata inventory")
    for name, digest in metadata.items():
        require(sha(folder / "build" / name) == digest, "compiled metadata bytes")
    inventory = read_json(folder / "build/inventory.json", limit=INVENTORY_JSON_LIMIT)
    admit_compiled_inventory(inventory)
    result = admit_execution(folder / "case", folder / "runtime")
    require(result["backend"] == build["backend"], "runtime backend identity")
    result.update(run_id=request["run_id"], manifest_sha256=sha(folder / "manifest.json"),
                  request_sha256=sha(folder / "request.json"), artifact_sha256=build["artifact_sha256"],
                  host_sha256=build["host_sha256"],
                  compiled_inventory={name: inventory[name] for name in ("artifact_bytes", "max_static_sram_end", "scope")},
                  compiled_elf_files=len(inventory["files"]), compiled_inventory_sha256=metadata["inventory.json"],
                  compile_workers=build["compile_workers"], compile_resource_cpu=build["compile_resource_cpu"],
                  worker_status=record["status"])
    return result


def audit(case, output):
    manifest, execution, raw = execution_arrays(case, output)
    expected = load_npz(Path(case) / "expected.npz")
    inputs = load_npz(Path(case) / "inputs.npz")
    masks = load_npz(Path(case) / "zero_masks.npz")
    bit_masks = exact_masks(case, manifest, expected)
    np.testing.assert_array_equal(raw["initial_stats"], 0, err_msg="initial stats")
    for name, value in inputs.items():
        np.testing.assert_array_equal(raw["before_" + name].view(np.uint32), value.view(np.uint32), err_msg="initial input " + name)
        final = value if name in manifest["immutable_inputs"] else raw[name]
        np.testing.assert_array_equal(raw["after_" + name].view(np.uint32), final.view(np.uint32), err_msg="final input " + name)
    errors = {}
    for name, wanted in expected.items():
        observed, check = raw[name], manifest["checks"][name]
        require(np.isfinite(observed).all(), "finite output: " + name)
        if check["kind"] == "exact":
            np.testing.assert_array_equal(observed.view(np.uint32), wanted.view(np.uint32), err_msg=name)
        else:
            np.testing.assert_allclose(observed, wanted, rtol=check["rtol"], atol=check["atol"], err_msg=name)
        if name in masks:
            np.testing.assert_array_equal(observed[masks[name] != 0], 0, err_msg="exact padding " + name)
        if name in bit_masks:
            selected = bit_masks[name] != 0
            np.testing.assert_array_equal(observed[selected].view(np.uint32), wanted[selected].view(np.uint32),
                                          err_msg="exact masked values " + name)
        errors[name] = float(np.max(np.abs(observed.astype(np.float64) - wanted)))
    clocks = raw["clocks"].astype(np.uint64)
    require(np.all(clocks < 65536), "clock carrier")
    begin = clocks[:, 0] + (clocks[:, 1] << 16) + (clocks[:, 2] << 32)
    end = clocks[:, 3] + (clocks[:, 4] << 16) + (clocks[:, 5] << 32)
    spans = (end - begin) & ((1 << 48) - 1)
    require(np.all((spans > 0) & (spans < 1 << 47)), "valid PE-local clock intervals")
    return dict(status="PASS", numerical_status="PASS", backend=execution["backend"], errors=errors,
                local_cycles_min=int(spans.min()), local_cycles_max=int(spans.max()),
                scope=manifest["scope"], tolerance_contract=manifest["checks"],
                claim="frozen native GNN numerical agreement; PE-local intervals, no global makespan or trained accuracy claim")


def audit_report(folder):
    admission = admit_capture(folder)
    try:
        result = audit(Path(folder) / "case", Path(folder) / "runtime")
    except (AssertionError, ValueError) as error:
        return dict(admission, status="FAIL", numerical_status="FAIL", numerical_error=str(error)[:65536],
                    claim="capture complete; strict local numerical checks failed")
    return dict(admission, **result)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit a native GNN capture locally without SDK access.")
    parser.add_argument("capture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists() and not args.out.is_symlink(), "fresh report required")
    try:
        if args.capture.is_dir():
            report = audit_report(args.capture)
        else:
            with tempfile.TemporaryDirectory(prefix="native-gnn-audit-") as temporary:
                folder = unpack_archive(args.capture, Path(temporary) / "capture", "capture", CAPTURE_LIMIT)
                report = audit_report(folder)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile, tarfile.TarError) as error:
        report = dict(status="FAIL", execution_status="FAIL", numerical_status="NOT_RUN",
                      error=str(error)[:65536], claim="capture completeness or identity check failed")
    write_json(args.out, report)
    print(json.dumps(dict(status=report["status"], execution_status=report["execution_status"],
                          numerical_status=report["numerical_status"], report=str(args.out))))
    return 0 if report["numerical_status"] == "PASS" else 1 if report["execution_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
