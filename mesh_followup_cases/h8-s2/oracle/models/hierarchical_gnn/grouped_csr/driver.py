#!/usr/bin/env python3
"""Bounded compile and local simulator driver for grouped Cora CSR inference.

Invoke each public command through tools/run_with_budget.py. The native
``simulate`` command launches only the SDK capture in its Python 3.7 container,
then independently audits the complete arrays in the native interpreter.
ELF inspection reads section headers; no instruction decoding is performed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import time
import uuid
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARAMETERS = ("C", "R", "K", "L", "FW", "H", "O", "T", "MX", "MG")
INPUT_FILES = frozenset(("arrays.npz", "reference.npz", "node_ids.npy"))
RUNTIME = {"cmaddr": None, "simfab_numthreads": 1, "suppress_simfab_trace": True}
RUNTIME_ELFS = frozenset(("build/generated/MEMCPY_XY_ROUTES.elf",
                          "build/generated/coord.elf", "build/generated/default.elf"))
PROFILE = "h064-grouped-csr-v2-prelaunch"
SCAN_BYTES = 256 * 1024**2
SCAN_ENTRIES = 10000
SCAN_SECONDS = 10


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular file required: " + str(path))
    before = path.stat()
    require(before.st_size <= SCAN_BYTES, "hash file byte bound")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), "file changed during hash")
    return result.hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 16*1024**2,
            "JSON regular-file/byte bound")
    with path.open() as stream:
        return json.load(stream)


def save_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def source_hashes():
    paths = [HERE / name for name in ("driver.py", "oracle.py", "layout.csl", "pe.csl")]
    paths += [HERE.parent / name for name in ("partition.py", "experiment.py")]
    paths += [ROOT / name for name in ("cslc", "cs_python", "sdk_debug_shell")]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def validate_params(p):
    require(isinstance(p, dict) and set(p) == set(PARAMETERS), "compile parameter inventory")
    require(all(type(v) is int and 0 < v <= 65535 for v in p.values()), "u16 compile parameters")
    require(p["T"] in (4, 8, 16), "feature tile must be four, eight or sixteen")
    require(p["C"] >= 2 and p["R"] >= 2, "collective rectangle minimum is 2x2")
    # All CSL array/descriptor indices in this ABI are u16.
    extents = (p["K"]+1, p["FW"]*p["H"], p["H"]*p["O"],
               p["R"]*p["L"]+1, p["L"]*p["H"], p["L"]*p["O"],
               p["K"]*p["T"], p["L"]*p["T"])
    require(max(extents) <= 65535, "CSL array/descriptor extent overflows u16")
    require(p["K"]*p["T"] <= 32767, "collective panel extent exceeds i16")
    require(p["C"]*p["R"] <= 4096, "bounded local validation supports at most 4096 PEs")
    require(declared_bytes(p) <= 44*1024, "declared state exceeds 44KiB precompile gate")
    return p


def declared_bytes(p):
    """Uniform input/output/scratch arrays plus four u16 state variables."""
    indexes = p["K"]+1+p["MX"]+p["R"]*p["L"]+1+p["MG"]+p["L"]
    floats = (p["MX"]+p["FW"]*p["H"]+p["H"]*p["O"]+p["MG"]+
              p["L"]*p["H"]+p["L"]*p["O"]+3*p["K"]*p["T"]+p["L"]*p["T"])
    return 2*indexes+4*floats+8*4+6*2+8


def input_schema(p):
    validate_params(p)
    return {"x_ptr": (p["K"]+1, "uint16"), "x_col": (p["MX"], "uint16"),
            "x_val": (p["MX"], "float32"), "w0": (p["FW"]*p["H"], "float32"),
            "w1": (p["H"]*p["O"], "float32"),
            "g_ptr": (p["R"]*p["L"]+1, "uint16"), "g_src": (p["MG"], "uint16"),
            "g_val": (p["MG"], "float32"), "slots": (p["L"], "uint16")}


def output_schema(p):
    return {"pre": (p["L"]*p["H"], "float32"),
            "logits": (p["L"]*p["O"], "float32"),
            "panel_sum": (p["K"]*p["T"], "float32"),
            "stats": (8, "uint32"), "clocks": (6, "uint16")}


def capture_schema(p):
    # Host diagnostic captures are not additional device exports.
    return dict(output_schema(p), upload_tail=(p["L"], "uint32"),
                initial_stats=(8, "uint32"))


def load_npz(path, expected):
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) == len(expected) and
                {entry.filename for entry in entries} == {name+".npy" for name in expected},
                "NPZ member inventory")
        require(sum(entry.file_size for entry in entries) <= 128*1024**2,
                "NPZ expanded byte bound")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(expected), "NPZ array inventory")
        return {name: archive[name].copy() for name in expected}


def validate_arrays(p, arrays, node_ids, reference):
    import numpy as np
    schema, pes = input_schema(p), p["C"]*p["R"]
    require(set(arrays) == set(schema), "input array inventory")
    for name, (count, dtype) in schema.items():
        array = arrays[name]
        require(array.dtype == np.dtype(dtype) and array.shape == (pes, count),
                "input dtype/shape mismatch: " + name)
        require(np.isfinite(array).all(), "nonfinite input: " + name)
    require(node_ids.shape == (pes, p["L"]) and node_ids.dtype.kind == "i",
            "node ownership dtype/shape")
    require(set(reference) == {"pre", "logits"}, "reference tensor inventory")
    nodes = reference["pre"].shape[0]
    require(reference["pre"].shape == (nodes, p["H"]) and
            reference["logits"].shape == (nodes, p["O"]) and nodes > 0,
            "reference tensor shape")
    require(all(a.dtype.kind == "f" and np.isfinite(a).all() for a in reference.values()),
            "reference dtype/finiteness")
    require(np.all((node_ids >= -1) & (node_ids < nodes)), "node ownership index range")
    valid = node_ids >= 0
    require(np.array_equal(np.sort(node_ids[valid]), np.arange(nodes)),
            "node ownership must cover every node exactly once")
    require(np.array_equal(arrays["slots"] != 65535, valid) and
            np.all(arrays["slots"][valid] < p["K"]), "local slot ownership/padding")
    require(np.all(arrays["w1"] == arrays["w1"][0]), "W1 replicas differ")
    for row in range(p["R"]):
        row_ids = node_ids[row*p["C"]:(row+1)*p["C"]]
        group = np.sort(row_ids[row_ids >= 0])
        require(len(group) <= p["K"], "group node capacity")
        for column in range(p["C"]):
            pe = row*p["C"]+column
            active = valid[pe]
            require(np.all(arrays["slots"][pe, active] < len(group)) and
                    np.array_equal(group[arrays["slots"][pe, active]], node_ids[pe, active]),
                    "group slot does not match node ownership")
    for prefix, capacity, index, bound in (("x", p["MX"], "x_col", p["FW"]),
                                           ("g", p["MG"], "g_src", p["K"])):
        ptr = arrays[prefix+"_ptr"].astype(np.int64)
        require(np.all(ptr[:, 0] == 0) and np.all(np.diff(ptr, axis=1) >= 0) and
                np.all(ptr[:, -1] <= capacity), "CSR pointer range/order: " + prefix)
        for pe in range(pes):
            active = int(ptr[pe, -1])
            require(np.all(arrays[index][pe, :active] < bound), "CSR active index: " + prefix)
            require(np.all(arrays[index][pe, active:] == 0) and
                    np.all(arrays[prefix+"_val"][pe, active:] == 0), "CSR padding: " + prefix)


def load_case(case):
    import numpy as np
    case = Path(case).resolve(strict=True)
    require(case.is_dir(), "case directory required")
    manifest = read_json(case/"case.json")
    require(manifest.get("source_sha256") == source_hashes(), "case source hashes differ")
    require(isinstance(manifest.get("metadata"), dict), "case metadata required")
    p = validate_params(manifest.get("params"))
    require(isinstance(manifest.get("files"), dict) and set(manifest["files"]) == INPUT_FILES,
            "case file inventory")
    before = {name: digest(case/name) for name in INPUT_FILES}
    require(manifest["files"] == before, "case input hash mismatch")
    arrays = load_npz(case/"arrays.npz", input_schema(p))
    reference = load_npz(case/"reference.npz", ("pre", "logits"))
    node_ids = np.load(case/"node_ids.npy", allow_pickle=False)
    validate_arrays(p, arrays, node_ids, reference)
    require(before == {name: digest(case/name) for name in INPUT_FILES} and
            manifest["source_sha256"] == source_hashes(), "case changed during loading")
    return manifest, arrays, node_ids, reference


def compile_command(case, p):
    validate_params(p)
    return [str(ROOT/"cslc"), str(HERE/"layout.csl"), "--arch=wse3",
            "--fabric-dims=%d,%d" % (p["C"]+7, p["R"]+2), "--fabric-offsets=4,1",
            "--params="+",".join("%s:%d" % (name, p[name]) for name in PARAMETERS),
            "--memcpy", "--channels=1", "--width-west-buf=0", "--width-east-buf=0",
            "--max-parallelism=1", "-o="+str(case/"build")]


def build_files(case):
    """Fail closed while walking the complete artifact tree, including helpers."""
    build = case/"build"
    require(build.is_dir() and not build.is_symlink(), "regular build directory required")
    deadline, entries, total = time.monotonic()+SCAN_SECONDS, 0, 0
    pending, files = [build], []
    while pending:
        require(time.monotonic() < deadline, "build scan deadline")
        with os.scandir(pending.pop()) as iterator:
            for entry in iterator:
                entries += 1
                require(entries <= SCAN_ENTRIES and time.monotonic() < deadline,
                        "build scan entry/deadline bound")
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                else:
                    require(stat.S_ISREG(info.st_mode), "nonregular/symlink build artifact")
                    total += info.st_size
                    require(total <= SCAN_BYTES, "build scan byte bound")
                    files.append(Path(entry.path))
    require(time.monotonic() < deadline, "incomplete build scan")
    return sorted(files), {"entries": entries, "visible_bytes": total, "complete": True,
                           "limits": {"seconds": SCAN_SECONDS, "entries": SCAN_ENTRIES,
                                      "visible_bytes": SCAN_BYTES}}


def elf_sections(path):
    """Parse bounded ELF section headers, never instructions or disassembly."""
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 2*1024**2,
            "ELF regular-file/byte bound")
    data = path.read_bytes()
    require(len(data) >= 52 and data[:4] == b"\x7fELF" and data[4] in (1, 2) and
            data[5] in (1, 2) and data[6] == 1, "ELF format")
    endian, wide = "<" if data[5] == 1 else ">", data[4] == 2
    headfmt = endian+("HHIQQQIHHHHHH" if wide else "HHIIIIIHHHHHH")
    require(len(data) >= 16+struct.calcsize(headfmt), "ELF header bound")
    header = struct.unpack_from(headfmt, data, 16)
    offset, size, count = header[5], header[10], header[11]
    fmt = endian+("IIQQQQIIQQ" if wide else "IIIIIIIIII")
    require(0 < count <= 4096 and size >= struct.calcsize(fmt) and offset >= 16+struct.calcsize(headfmt)
            and offset+size*count <= len(data), "ELF section table bound")
    sections = [struct.unpack_from(fmt, data, offset+i*size) for i in range(count)]
    require(all(s[1] == 8 or s[5] == 0 or s[4]+s[5] <= len(data) for s in sections),
            "ELF section payload bound")
    require(hashlib.sha256(data).hexdigest() == digest(path), "ELF changed during inspection")
    return data, sections


def elf_storage(path):
    _, sections = elf_sections(path)
    low, outside = [], []
    for section in sections:
        if section[2] & 2 and section[5]:
            target = low if section[3] < 48*1024 else outside
            target.append([section[3], section[3]+section[5]])
    endpoint = max((end for _, end in low), default=0)
    require(0 < endpoint <= 44*1024, "44KiB static SRAM gate")
    return {"sram_ranges": low, "outside_sram_ranges": outside, "static_sram_end": endpoint,
            "scope": "allocated low-address ELF sections; excludes stack and runtime high-water"}


def runtime_elf_identity(path):
    data, sections = elf_sections(path)
    require(all(section[2] & 2 == 0 for section in sections), "runtime metadata contains SHF_ALLOC")
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "no_shf_alloc": True, "scope": "SDK-added metadata, not a compiled executable"}


def build_inventory(case, compiled=None):
    files, scan = build_files(case)
    deadline = time.monotonic()+30
    observed = {str(path.relative_to(case)): path for path in files if path.suffix == ".elf"}
    if compiled is None:
        require(not (set(observed) & RUNTIME_ELFS), "runtime metadata present before execution")
        names, additions = set(observed), set()
    else:
        require(isinstance(compiled, dict), "compiled inventory schema")
        names = {name for name in compiled if name.endswith(".elf")}
        require(set(compiled) == names | {"build/out.json", "build/bin/out_rpc.json"}
                and not (names & RUNTIME_ELFS), "compiled inventory schema")
        require(names <= set(observed), "compiled ELF missing")
        additions = set(observed)-names
        require(not additions or additions == RUNTIME_ELFS, "unknown or partial SDK ELF additions")
    require(0 < len(names) <= 10000, "missing/excessive complete ELF inventory")
    paths = {name: observed[name] for name in names}
    for name in ("build/out.json", "build/bin/out_rpc.json"):
        require(case/name in files, "compiled metadata missing")
        paths[name] = case/name
    hashes, memory = {}, {}
    for name, path in sorted(paths.items()):
        require(time.monotonic() < deadline, "ELF inventory inspection deadline")
        hashes[name] = digest(path)
        if name.endswith(".elf"):
            memory[name] = elf_storage(path)
    if compiled is not None:
        require(hashes == compiled, "compiled artifact changed")
    scan["runtime_generated_elf"] = {name: runtime_elf_identity(observed[name]) for name in sorted(additions)}
    require(time.monotonic() < deadline and
            hashes == {name: digest(path) for name, path in sorted(paths.items())},
            "artifact changed during inventory/deadline")
    require(time.monotonic() < deadline, "artifact inspection exceeded deadline")
    return {"artifacts_sha256": hashes, "artifact_bytes": sum(path.stat().st_size for path in paths.values()),
            "elf_storage": memory, "current_scan": scan,
            "max_static_sram_end": max(item["static_sram_end"] for item in memory.values())}


def check_abi(case, p):
    actual = read_json(case/"build/out.json").get("params")
    require(isinstance(actual, dict) and set(actual) == set(p), "compiled parameter inventory")
    for name, value in p.items():
        got = actual[name]
        if isinstance(got, str) and got.isdecimal():
            got = int(got)
        require(type(got) is int and got == value, "compiled parameter mismatch: "+name)
    symbols = read_json(case/"build/bin/out_rpc.json").get("rpc_symbols")
    require(isinstance(symbols, list) and all(isinstance(symbol, dict) for symbol in symbols), "RPC schema")
    for name, (_, dtype) in dict(input_schema(p), **output_schema(p)).items():
        matches = [symbol for symbol in symbols if symbol.get("name") == name]
        pointer = {"uint16": "[*]u16", "uint32": "[*]u32", "float32": "[*]f32"}[dtype]
        require(len(matches) == 1 and matches[0].get("kind") == "Var" and
                matches[0].get("type") == pointer, "RPC variable mismatch: "+name)
    matches = [symbol for symbol in symbols if symbol.get("name") == "f_run"]
    require(len(matches) == 1 and matches[0].get("kind") == "Func" and
            matches[0].get("type") == "void" and matches[0].get("inputs", []) == [], "RPC launch ABI")


def compile_case(case):
    case = Path(case).resolve(strict=True)
    manifest, _, _, _ = load_case(case)
    require(not (case/"build").exists(), "fresh build required")
    request = {"profile": PROFILE, "case": str(case), "case_sha256": digest(case/"case.json"),
               "source_sha256": source_hashes(), "command": compile_command(case, manifest["params"]),
               "cwd": str(ROOT)}
    save_json(case/"compile-request.json", request)
    result, start = dict(request, status="FAILED"), time.monotonic()
    try:
        completed = subprocess.run(request["command"], cwd=str(ROOT), check=False)
        result["returncode"] = completed.returncode
        require(completed.returncode == 0, "compiler failed")
        load_case(case)
        check_abi(case, manifest["params"])
        result.update(build_inventory(case))
        result["status"] = "PASS"
    except Exception as error:
        result["error"] = type(error).__name__+": "+str(error)
        raise
    finally:
        result["wall_seconds"] = time.monotonic()-start
        result["request_sha256"] = digest(case/"compile-request.json")
        save_json(case/"build.json", result)
    return result


def validate_build(case):
    manifest, arrays, node_ids, reference = load_case(case)
    request = {"profile": PROFILE, "case": str(case), "case_sha256": digest(case/"case.json"),
               "source_sha256": source_hashes(), "command": compile_command(case, manifest["params"]),
               "cwd": str(ROOT)}
    build = read_json(case/"build.json")
    require(read_json(case/"compile-request.json") == request and
            all(build.get(key) == value for key, value in request.items()) and
            build.get("request_sha256") == digest(case/"compile-request.json") and
            build.get("status") == "PASS" and build.get("returncode") == 0 and "error" not in build,
            "build/source/completion provenance mismatch")
    check_abi(case, manifest["params"])
    current = build_inventory(case, build.get("artifacts_sha256"))
    require(all(current[key] == build.get(key) for key in
                ("artifacts_sha256", "artifact_bytes", "elf_storage", "max_static_sram_end")),
            "compiled artifact storage differs")
    return manifest, arrays, node_ids, reference, current


def expected_stats(p, arrays):
    import numpy as np
    tiles = (p["H"]+p["T"]-1)//p["T"] + (p["O"]+p["T"]-1)//p["T"]
    result = np.zeros((p["C"]*p["R"], 8), dtype=np.uint64)
    result[:, :4] = (0x600D, tiles, tiles, p["R"]*tiles)
    result[:, 4] = arrays["x_ptr"][:, -1].astype(np.uint64)*p["H"]
    result[:, 5] = arrays["g_ptr"][:, -1].astype(np.uint64)*(p["H"]+p["O"])
    result[:, 6] = np.count_nonzero(arrays["slots"] != 65535, axis=1)*p["H"]*p["O"]
    result[:, 7] = tiles
    require(np.all(result <= 0xffffffff), "expected stats overflow u32")
    return result.astype(np.uint32)


def expected_final_panel(p, arrays, node_ids, reference):
    """Final output-feature panel before graph aggregation, replicated per row."""
    import numpy as np
    start = ((p["O"]-1)//p["T"])*p["T"]
    active = p["O"]-start
    weights = arrays["w1"][0].reshape(p["H"], p["O"]).astype(np.float64)
    projected = np.maximum(reference["pre"].astype(np.float64), 0) @ weights[:, start:]
    panel = np.zeros((p["C"]*p["R"], p["K"], p["T"]), dtype=np.float64)
    for row in range(p["R"]):
        local = node_ids[row*p["C"]:(row+1)*p["C"]]
        group = np.sort(local[local >= 0])
        panel[row*p["C"]:(row+1)*p["C"], :len(group), :active] = projected[group]
    return panel.reshape(p["C"]*p["R"], p["K"]*p["T"])


def audit_arrays(p, arrays, node_ids, reference, raw):
    import numpy as np
    validate_arrays(p, arrays, node_ids, reference)
    schema, pes = capture_schema(p), p["C"]*p["R"]
    require(set(raw) == set(schema), "raw output inventory")
    for name, (count, dtype) in schema.items():
        require(raw[name].shape == (pes, count) and raw[name].dtype == np.dtype(dtype),
                "raw output dtype/shape: "+name)
        require(np.isfinite(raw[name]).all(), "nonfinite raw output: "+name)
    require(np.array_equal(raw["upload_tail"] & 65535, arrays["slots"].astype(np.uint32)),
            "prelaunch last-upload witness")
    require(np.all(raw["initial_stats"] == 0), "prelaunch initial counters")
    require(np.array_equal(raw["stats"], expected_stats(p, arrays)), "completion/protocol/arithmetic counters")
    valid, errors = node_ids >= 0, {}
    for name, width in (("pre", p["H"]), ("logits", p["O"])):
        tensor = raw[name].reshape(pes, p["L"], width)
        require(np.all(tensor[~valid] == 0), "nonzero padded output: "+name)
        observed = tensor[valid].astype(np.float64)
        expected = reference[name][node_ids[valid]].astype(np.float64)
        difference = np.abs(observed-expected)
        errors[name] = {"max_abs": float(difference.max()),
                        "max_scaled": float(np.max(difference/(2e-5+2e-4*np.abs(expected))))}
        require(np.allclose(observed, expected, rtol=2e-4, atol=2e-5), "FP64 reference mismatch: "+name)
    panel = expected_final_panel(p, arrays, node_ids, reference)
    require(np.allclose(raw["panel_sum"].astype(np.float64), panel, rtol=2e-4, atol=2e-5),
            "final source panel differs from FP64 reference")
    panel_view = raw["panel_sum"].reshape(p["R"], p["C"], p["K"], p["T"])
    active = (p["O"]-1) % p["T"]+1
    require(np.all(panel_view[:, :, :, active:] == 0), "nonzero panel feature padding")
    for row in range(p["R"]):
        count = np.count_nonzero(node_ids[row*p["C"]:(row+1)*p["C"]] >= 0)
        require(np.all(panel_view[row, :, count:] == 0), "nonzero panel vertex padding")
    errors["panel_sum"] = {"max_abs": float(np.max(np.abs(raw["panel_sum"].astype(np.float64)-panel)))}
    clocks = raw["clocks"].astype(np.uint64)
    starts = clocks[:, 0] | (clocks[:, 1] << np.uint64(16)) | (clocks[:, 2] << np.uint64(32))
    ends = clocks[:, 3] | (clocks[:, 4] << np.uint64(16)) | (clocks[:, 5] << np.uint64(32))
    elapsed = (ends-starts) & np.uint64((1 << 48)-1)
    require(np.all(elapsed > 0) and np.all(elapsed < (1 << 47)), "invalid local clock interval")
    return {"status": "PASS", "nodes": int(valid.sum()), "errors": errors,
            "local_interval_min": int(elapsed.min()), "local_interval_max": int(elapsed.max()),
            "local_intervals": elapsed.tolist(), "atol": 2e-5, "rtol": 2e-4,
            "scope": "full local WSE-3 simulator inference; PE-local intervals, not global or hardware timing"}


def api_names(p):
    return (["construct"]+["get_id:"+name for name in dict(input_schema(p), **output_schema(p))]+
            ["load", "run"]+["h2d:"+name for name in input_schema(p)]+
            ["d2h:upload_tail", "d2h:initial_stats", "launch:f_run", "d2h:stats"]+
            ["d2h:"+name for name in output_schema(p) if name != "stats"]+["stop"])


def execute_sdk(case, output, bindings=None):
    """Capture only; native simulate_case performs the numerical decision."""
    import numpy as np
    manifest, inputs, _, _, inventory = validate_build(case)
    p, pes = manifest["params"], manifest["params"]["C"]*manifest["params"]["R"]
    request = read_json(output/"request.json")
    require(request.get("case") == str(case) and request.get("output") == str(output) and
            request.get("case_sha256") == digest(case/"case.json") and
            request.get("build_sha256") == digest(case/"build.json") and
            request.get("source_sha256") == source_hashes() and request.get("runtime") == RUNTIME,
            "runtime request provenance")
    report = dict(request, status="FAILED", api=[], api_seconds={},
                  pre_runtime_inventory=inventory["current_scan"])
    raw, runner, previous, start = {}, None, Path.cwd(), time.monotonic()
    with (output/"api.jsonl").open("x") as receipt:
        def invoke(operation, function, *args, **kwargs):
            item = {"op": operation, "event": "enter"}
            report["api"].append(item)
            receipt.write(json.dumps(item, sort_keys=True)+"\n")
            receipt.flush()
            call_start = time.monotonic()
            try:
                value = function(*args, **kwargs)
            finally:
                report["api_seconds"][operation] = time.monotonic()-call_start
            item = {"op": operation, "event": "exit"}
            report["api"].append(item)
            receipt.write(json.dumps(item, sort_keys=True)+"\n")
            receipt.flush()
            return value
        try:
            os.chdir(output)
            if bindings is None:
                from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
                bindings = (SdkRuntime, MemcpyDataType, MemcpyOrder)
            Runtime, Types, Order = bindings
            runner = invoke("construct", Runtime, str(case/"build"), **RUNTIME)
            symbols = dict(input_schema(p), **output_schema(p))
            ids = {name: invoke("get_id:"+name, runner.get_id, name) for name in symbols}
            invoke("load", runner.load)
            invoke("run", runner.run)
            for name, (count, dtype) in input_schema(p).items():
                payload = inputs[name].astype(np.uint32) if dtype == "uint16" else inputs[name].copy()
                invoke("h2d:"+name, runner.memcpy_h2d, ids[name], payload.ravel(order="C"),
                       0, 0, p["C"], p["R"], count, streaming=False,
                       data_type=Types.MEMCPY_16BIT if dtype == "uint16" else Types.MEMCPY_32BIT,
                       order=Order.ROW_MAJOR, nonblock=False)
            # Verify the last uploaded symbol, then read initialized counters.
            # This is an observed SDK ordering witness, not proof that every
            # independent input has drained. It changes the launch timing.
            for label, symbol, count, bits in (("upload_tail", "slots", p["L"], 16),
                                                ("initial_stats", "stats", 8, 32)):
                payload = ((inputs["slots"].astype(np.uint32).ravel() ^ 65535) if bits == 16
                           else np.full(pes*count, 0xffffffff, dtype=np.uint32))
                invoke("d2h:"+label, runner.memcpy_d2h, payload, ids[symbol],
                       0, 0, p["C"], p["R"], count, streaming=False,
                       data_type=Types.MEMCPY_16BIT if bits == 16 else Types.MEMCPY_32BIT,
                       order=Order.ROW_MAJOR, nonblock=False)
                raw[label] = payload.reshape(pes, count).copy()
                if label == "upload_tail":
                    require(np.array_equal(raw[label] & 65535, inputs["slots"].astype(np.uint32)),
                            "prelaunch last-upload witness")
                else:
                    require(np.all(raw[label] == 0), "prelaunch initial counters")
            report["setup_until_prelaunch_complete_seconds"] = time.monotonic()-start
            launch_start = time.monotonic()
            invoke("launch:f_run", runner.launch, "f_run", nonblock=False)
            order = ["stats"]+[name for name in output_schema(p) if name != "stats"]
            for name in order:
                count, dtype = output_schema(p)[name]
                payload = np.zeros(pes*count, dtype=np.uint32 if dtype == "uint16" else np.dtype(dtype))
                invoke("d2h:"+name, runner.memcpy_d2h, payload, ids[name], 0, 0, p["C"], p["R"], count,
                       streaming=False, data_type=Types.MEMCPY_16BIT if dtype == "uint16" else Types.MEMCPY_32BIT,
                       order=Order.ROW_MAJOR, nonblock=False)
                if dtype == "uint16":
                    require(np.all(payload <= 65535), "u16 receive container upper bits")
                raw[name] = payload.astype(np.dtype(dtype), copy=False).reshape(pes, count).copy()
                if name == "stats":
                    require(np.all(raw[name][:, 0] == 0x600D), "completion-first status mismatch")
                    report["launch_to_completion_read_seconds"] = time.monotonic()-launch_start
        except Exception as error:
            report["error"] = type(error).__name__+": "+str(error)
        finally:
            if runner is not None:
                try:
                    invoke("stop", runner.stop)
                except Exception as error:
                    report["stop_error"] = type(error).__name__+": "+str(error)
            os.chdir(previous)
    with (output/"raw.npz").open("xb") as stream:
        np.savez(stream, **raw)
    expected = [{"op": operation, "event": event} for operation in api_names(p) for event in ("enter", "exit")]
    if "error" not in report and "stop_error" not in report and report["api"] == expected:
        try:
            post = validate_build(case)[4]
            report["post_runtime_inventory"] = post["current_scan"]
            report["status"] = "COMPLETED"
        except Exception as error:
            report["error"] = type(error).__name__+": "+str(error)
    report.update(wall_seconds=time.monotonic()-start, raw_sha256=digest(output/"raw.npz"),
                  api_sha256=digest(output/"api.jsonl"), request_sha256=digest(output/"request.json"))
    save_json(output/"runtime.json", report)
    require(report["status"] == "COMPLETED", "SDK capture incomplete; partial evidence retained")
    return report


def audit_run(case, output):
    manifest, inputs, node_ids, reference, inventory = validate_build(case)
    request, runtime = read_json(output/"request.json"), read_json(output/"runtime.json")
    p = manifest["params"]
    require(runtime.get("status") == "COMPLETED" and "error" not in runtime and "stop_error" not in runtime,
            "runtime did not complete")
    require(all(runtime.get(key) == value for key, value in request.items()) and
            request.get("case") == str(case) and request.get("output") == str(output) and
            request.get("case_sha256") == digest(case/"case.json") and
            request.get("build_sha256") == digest(case/"build.json") and
            request.get("source_sha256") == source_hashes() and request.get("runtime") == RUNTIME and
            runtime.get("request_sha256") == digest(output/"request.json") and
            runtime.get("raw_sha256") == digest(output/"raw.npz") and
            runtime.get("api_sha256") == digest(output/"api.jsonl"), "runtime/raw/source provenance mismatch")
    expected = [{"op": operation, "event": event} for operation in api_names(p) for event in ("enter", "exit")]
    require(runtime.get("api") == expected, "SDK API sequence incomplete")
    durations = runtime.get("api_seconds")
    require(isinstance(durations, dict) and set(durations) == set(api_names(p)) and
            all(type(value) in (int, float) and finite_nonnegative(value) for value in durations.values()),
            "SDK API duration inventory")
    with (output/"api.jsonl").open() as stream:
        receipt = [json.loads(line) for line in stream]
    require(receipt == expected, "persisted SDK API sequence differs")
    phases = ("setup_until_prelaunch_complete_seconds", "launch_to_completion_read_seconds")
    require(all(type(runtime.get(name)) in (int, float) and
                finite_nonnegative(runtime[name]) for name in phases), "diagnostic phase durations")
    raw = load_npz(output/"raw.npz", capture_schema(p))
    result = audit_arrays(p, inputs, node_ids, reference, raw)
    result.update(profile=PROFILE, case=str(case), output=str(output), params=p,
                  runtime_sha256=digest(output/"runtime.json"), case_sha256=digest(case/"case.json"),
                  source_sha256=source_hashes(), build_sha256=digest(case/"build.json"),
                  raw_sha256=digest(output/"raw.npz"), run_id=request.get("run_id"),
                  runtime_wall_seconds=runtime["wall_seconds"],
                  launch_seconds=durations["launch:f_run"], api_seconds=durations,
                  max_static_sram_end=inventory["max_static_sram_end"], artifact_bytes=inventory["artifact_bytes"])
    result.update({name: runtime[name] for name in phases})
    return result


def finite_nonnegative(value):
    # SDK-compatible scalar check without importing a native numerical module.
    return 0 <= value < float("inf")


def simulate_case(case, output):
    case, output = Path(case).resolve(strict=True), Path(output).resolve()
    validate_build(case)
    runs = (ROOT/"research/runs").resolve()
    require(runs in output.parents, "simulator output must be a dedicated research/runs descendant")
    require(case != output and case not in output.parents and output not in case.parents,
            "runtime output must be separate from immutable case")
    output.mkdir(parents=True, exist_ok=False)
    command = [str(ROOT/"cs_python"), str(HERE/"driver.py"), "_sdk-run", "--case-dir", str(case), "--output", str(output)]
    request = {"profile": PROFILE, "case": str(case), "output": str(output), "run_id": str(uuid.uuid4()),
               "case_sha256": digest(case/"case.json"), "build_sha256": digest(case/"build.json"),
               "source_sha256": source_hashes(), "runtime": RUNTIME, "command": command, "launcher_cwd": str(ROOT)}
    save_json(output/"request.json", request)
    result = dict(request, status="FAILED")
    try:
        completed = subprocess.run(command, cwd=str(ROOT), check=False)
        result["returncode"] = completed.returncode
        require(completed.returncode == 0, "SDK child failed; inspect runtime.json/api.jsonl")
        result.update(audit_run(case, output))
    except Exception as error:
        result["error"] = type(error).__name__+": "+str(error)
        raise
    finally:
        save_json(output/"result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compile", "simulate", "audit", "_sdk-run"))
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    case = args.case_dir.resolve(strict=True)
    if args.mode == "compile":
        result = compile_case(case)
    else:
        require(args.output is not None, "--output required")
        output = args.output.resolve()
        if args.mode == "simulate":
            result = simulate_case(case, output)
        elif args.mode == "audit":
            result = audit_run(case, output)
        else:
            result = execute_sdk(case, output)
    print(json.dumps({key: result[key] for key in
                      ("status", "errors", "max_static_sram_end", "local_interval_max", "wall_seconds")
                      if key in result}, sort_keys=True))


if __name__ == "__main__":
    main()
