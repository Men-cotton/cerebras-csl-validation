"""Portable audit of the finite E166 resident/staged/download ABI; no SDK imports."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
try:
    from . import portable as p
except ImportError:
    import portable as p

# Archive/ELF handling is shared source; the native H128 contract is not imported.
for _name in ("JSON_LIMIT", "INVENTORY_JSON_LIMIT", "ELF_TOTAL_LIMIT", "CASE_LIMIT", "CAPTURE_LIMIT",
              "HOST_FILES", "CASE_REQUIRED", "require", "bounded", "sha", "read_json", "write_json",
              "local_file", "load_npz", "compile_workers", "compiler_arguments", "compile_binding",
              "unpack_archive", "compiled_inventory", "admit_compiled_inventory"):
    globals()[_name] = getattr(p, _name)
ARMS = ("resident", "staged", "download")
KINDS = ("mean", "adversarial", "tiled6")
STEPS = 8
ATOL, RTOL = 2e-5, 2e-4
PARAMS = ("C", "R", "K", "L", "FW", "H", "O", "T", "MX", "MG", "STEPS")


def arm(value):
    require(value in ARMS, "finite resident/staged/download arm")
    return value


def fields(params):
    l, k, h, o, fw = (params[n] for n in ("L", "K", "H", "O", "FW"))
    return dict(pre=l*h, hidden=l*h, logits=l*o, D1=l*o, E1=l*o, E1_shared=k*o,
                dH=l*h, D0=l*h, E0=l*h, E0_shared=k*h, dW0_local=fw*h, dW0=fw*h,
                dW1_local=h*o, dW1=h*o, W0_old=fw*h, W1_old=h*o, W0=fw*h, W1=h*o,
                loss=1, count=1)


def schema(params, output=False):
    if output:
        return dict(history=[2*sum(fields(params).values()), "float32"], stats=[32, "uint32"],
                    release=[1, "uint32"], w0=[params["FW"]*params["H"], "float32"],
                    w1=[params["H"]*params["O"], "float32"], checkpoint_step=[2, "uint32"],
                    run_summary=[4, "uint32"], control_status=[4, "uint32"], step_times=[96, "uint16"])
    dims = dict(x_ptr=params["K"]+1, x_col=params["MX"], x_val=params["MX"],
                w0=params["FW"]*params["H"], w1=params["H"]*params["O"],
                g_ptr=params["R"]*params["L"]+1, g_src=params["MG"], g_val=params["MG"],
                slots=params["L"], targets=params["L"]*params["O"], mask=params["L"], rate=1)
    result = {n: [v, "float32" if n in ("x_val", "w0", "w1", "g_val", "targets", "rate") else "uint16"]
              for n, v in dims.items()}
    result["control"] = [2, "uint32"]
    return result


def inputs(case, manifest, selected):
    selected = arm(selected)
    arrays = load_npz(Path(case)/"inputs.npz")
    arrays["control"] = np.tile(np.array([int(selected != "resident"), 0], np.uint32),
                                (manifest["params"]["C"]*manifest["params"]["R"], 1))
    return arrays


def markers(params, index, selected):
    require(type(index) is int and 1 <= index <= STEPS, "marker index")
    n = STEPS if arm(selected) == "resident" else index
    tags = np.zeros((params["C"]*params["R"], 2), np.uint32)
    for step in range(n):
        tags[:, step % 2] = step+1
    return dict(summary=np.tile(np.array([n, 0, n, 0x600D if n == STEPS else 0], np.uint32),
                                (params["C"]*params["R"], 1)), tags=tags)


def load(case):
    case = Path(case).absolute()
    manifest = read_json(local_file(case, "case.json"))
    require(manifest.get("family") == "resident-control" and manifest.get("schema") == 1
            and manifest.get("kind") in KINDS and manifest.get("arms") == list(ARMS), "finite case identity")
    params = manifest.get("params")
    require(isinstance(params, dict) and set(params) == set(PARAMS)
            and all(type(v) is int and 1 <= v <= 32767 for v in params.values())
            and params["STEPS"] == STEPS, "eight-step parameters")
    require((params["C"], params["R"]) in ((2, 2), (3, 3), (6, 6)), "finite E166 topologies")
    require(manifest.get("inputs") == schema(params) and manifest.get("outputs") == schema(params, True), "fixed ABI schema")
    require(manifest.get("immutable_inputs") == sorted(set(schema(params)) - {"w0", "w1"}), "immutable input contract")
    require(manifest.get("input_features") == (2 if manifest["kind"] == "mean" else 7), "fixture input features")
    require(manifest.get("atol") == ATOL and manifest.get("rtol") == RTOL, "frozen FP64 tolerance")
    require(isinstance(manifest.get("scope"), str) and 1 <= len(manifest["scope"]) <= 8192, "explicit scope")
    files = manifest.get("files")
    require(isinstance(files, dict) and CASE_REQUIRED <= set(files) and len(files) <= 64, "frozen inventory")
    total = len(bounded(case/"case.json", JSON_LIMIT))
    for name, digest in files.items():
        path = local_file(case, name)
        require(sha(path) == digest, "frozen file identity: " + name)
        total += path.stat().st_size
        require(total <= CASE_LIMIT, "frozen case byte bound")
    provenance = read_json(case/"provenance.json")
    require(provenance.get("origin") == "E166" and provenance.get("arithmetic_origin") == "E159"
            and isinstance(provenance.get("source_sha256"), dict), "frozen source provenance")
    arrays = load_npz(case/"inputs.npz")
    pes = params["C"]*params["R"]
    require(set(arrays) == set(schema(params)), "input inventory")
    for name, (count, dtype) in schema(params).items():
        require(arrays[name].shape == (pes, count) and arrays[name].dtype == np.dtype(dtype)
                and np.isfinite(arrays[name]).all(), "input ABI: " + name)
    require(np.all(arrays["control"] == 0), "canonical resident compile inputs")
    expected = load_npz(case/"expected.npz")
    wanted = {"history"+str(n): [schema(params, True)["history"][0], "float64"] for n in range(1, STEPS+1)}
    wanted.update({name: [count, "float64" if dtype == "float32" else dtype]
                   for name, (count, dtype) in schema(params, True).items()
                   if name not in ("history", "control_status", "step_times")})
    require(set(expected) == set(wanted), "expected inventory")
    for name, (count, dtype) in wanted.items():
        require(expected[name].shape == (pes, count) and expected[name].dtype == np.dtype(dtype)
                and np.isfinite(expected[name]).all(), "finite oracle ABI: " + name)
    return manifest


def check_abi(build, manifest):
    require({k: int(v) for k, v in read_json(Path(build)/"out.json")["params"].items()} == manifest["params"], "compiled parameter ABI")
    symbols = read_json(Path(build)/"bin/out_rpc.json")["rpc_symbols"]
    expected = {name: ("Var", {"float32": "[*]f32", "uint32": "[*]u32", "uint16": "[*]u16"}[dtype], ())
                for name, (_, dtype) in dict(manifest["inputs"], **manifest["outputs"]).items()}
    expected.update(f_run=("Func", "void", ()), f_resume=("Func", "void", (("expected", "u32"),)))
    require(len(symbols) == len(expected) and len({s["name"] for s in symbols}) == len(symbols)
            and all(type(s["id"]) is int and s["id"] >= 0 for s in symbols)
            and len({s["id"] for s in symbols}) == len(symbols), "unique complete compiled symbols")
    require({s["name"]: (s["kind"], s["type"], tuple((a["name"], a["type"]) for a in s.get("inputs", [])))
             for s in symbols} == expected, "f_run/f_resume typed ABI")


def raw_schema(manifest, selected):
    arm(selected)
    result = dict(manifest["outputs"])
    for index in range(1, STEPS+1):
        result[f"marker{index}_summary"] = [4, "uint32"]
        result[f"marker{index}_tags"] = [2, "uint32"]
        if selected == "download" and index < STEPS:
            result[f"history{index}"] = manifest["outputs"]["history"]
    result.update({"after_"+name: manifest["inputs"][name] for name in manifest["immutable_inputs"]})
    return result


def api_sequence(manifest, selected):
    arm(selected)
    calls = (["construct"] + ["get_id:"+n for n in dict(manifest["inputs"], **manifest["outputs"])]
             + ["load", "run"] + ["h2d:"+n for n in manifest["inputs"]] + ["launch:f_run"])
    for index in range(1, STEPS+1):
        calls += [f"d2h:marker{index}_summary", f"d2h:marker{index}_tags"]
        if selected == "download" and index < STEPS:
            calls.append(f"d2h:history{index}")
        if selected != "resident" and index < STEPS:
            calls.append(f"launch:f_resume:{index}")
    return calls + ["d2h:"+n for n in manifest["outputs"]] + ["d2h:after_"+n for n in manifest["immutable_inputs"]] + ["stop"]


def execution_arrays(case, output):
    manifest = load(case)
    execution = read_json(Path(output)/"execution.json")
    selected = arm(execution.get("arm"))
    require(execution.get("status") == "COMPLETED" and not any(k in execution for k in ("error", "stop_error"))
            and execution.get("backend") in ("hardware", "simulator"), "completed execution")
    require(execution.get("case_sha256") == sha(Path(case)/"case.json")
            and execution.get("raw_sha256") == sha(Path(output)/"raw.npz", CAPTURE_LIMIT), "bound execution arrays")
    calls = api_sequence(manifest, selected)
    require(execution.get("api") == calls, "exact SDK call sequence")
    timing = execution.get("api_timings", [])
    require([row.get("api") for row in timing] == calls
            and all(type(row.get("wall_seconds")) in (int, float) and np.isfinite(row["wall_seconds"])
                    and row["wall_seconds"] >= 0 for row in timing), "completed API timing inventory")
    require(isinstance(execution.get("clock_domain"), str) and ":monotonic_ns:" in execution["clock_domain"], "host clock domain")
    previous = 0
    for row in timing:
        require(type(row.get("start_ns")) is int and type(row.get("end_ns")) is int
                and row["end_ns"] >= row["start_ns"] >= previous
                and row["wall_seconds"] == (row["end_ns"]-row["start_ns"])/1e9, "ordered host API endpoints")
        label = row["api"]
        method = {"construct": "SdkRuntime", "load": "load", "run": "run", "stop": "stop"}.get(label)
        for prefix, native in (("get_id:", "get_id"), ("h2d:", "memcpy_h2d"), ("d2h:", "memcpy_d2h"), ("launch:", "launch")):
            if label.startswith(prefix):
                method = native
        require(row.get("method") == method, "actual SDK callable")
        if label.startswith(("h2d:", "d2h:", "launch:")):
            require(row.get("nonblock") is False, "blocking dependent SDK calls")
        if label.startswith("launch:"):
            require(row.get("symbol") == label.split(":")[1], "actual exported launch symbol")
        previous = row["end_ns"]
    typed = [[]] if selected == "resident" else [[]]+[[dict(dtype="uint32", value=n)] for n in range(1, STEPS)]
    require(execution.get("launch_typed_arguments") == typed, "actual typed launch arguments")
    require(execution.get("launch_arguments") == ([[]] if selected == "resident" else [[]] + [[n] for n in range(1, STEPS)]), "typed resume generation schedule")
    raw = load_npz(Path(output)/"raw.npz", CAPTURE_LIMIT)
    require(set(raw) == set(raw_schema(manifest, selected)), "complete capture inventory")
    pes = manifest["params"]["C"]*manifest["params"]["R"]
    for name, (count, dtype) in raw_schema(manifest, selected).items():
        require(raw[name].shape == (pes, count) and raw[name].dtype == np.dtype("uint32" if dtype == "uint16" else dtype), "native carrier ABI: " + name)
    return manifest, execution, raw


def admit_execution(case, output):
    manifest, execution, raw = execution_arrays(case, output)
    return dict(status="PASS", execution_status="PASS", numerical_status="NOT_RUN", arm=execution["arm"],
                backend=execution["backend"], case_sha256=execution["case_sha256"], raw_sha256=execution["raw_sha256"],
                arrays=len(raw), params=manifest["params"], claim="bound SDK capture complete; numerical audit not yet run")


def history_parts(params, history, generation):
    offset = ((generation-1) % 2)*sum(fields(params).values())
    result = {}
    for name, size in fields(params).items():
        result[name] = history[:, offset:offset+size]
        offset += size
    return result


def equal_bits(a, b, message):
    require(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a.view(np.uint32), b.view(np.uint32)), message)


def history_relations(manifest, arrays, history, index):
    params = manifest["params"]
    width, pes = params["C"], params["C"]*params["R"]
    own = np.count_nonzero(arrays["slots"] != 65535, axis=1)
    group = own.reshape(params["R"], width).sum(axis=1)
    if index == 1:
        require(np.all(history[:, sum(fields(params).values()):].view(np.uint32) == 0), "unused first history slot")
    previous = None
    for generation in range(max(1, index-1), index+1):
        step = history_parts(params, history, generation)
        for pe in range(pes):
            for name in ("W0_old", "W0", "dW0"):
                equal_bits(step[name][pe], step[name][pe % width], "column replica: "+name)
            for name in ("W1_old", "W1", "dW1", "loss", "count"):
                equal_bits(step[name][pe], step[name][0], "global replica: "+name)
            for name in ("E0_shared", "E1_shared"):
                equal_bits(step[name][pe], step[name][pe//width*width], "row replica: "+name)
            for name in ("pre", "hidden", "logits", "D1", "E1", "dH", "D0", "E0"):
                feature = params["O"] if name in ("logits", "D1", "E1") else params["H"]
                require(np.all(step[name][pe, own[pe]*feature:] == 0), "owner padding: "+name)
            for name in ("E0_shared", "E1_shared"):
                feature = params["O"] if name == "E1_shared" else params["H"]
                require(np.all(step[name][pe, group[pe//width]*feature:] == 0), "shared padding: "+name)
            feature_count = len(range(pe % width, manifest["input_features"], width))
            for name in ("W0_old", "W0", "dW0", "dW0_local"):
                require(np.all(step[name][pe, feature_count*params["H"]:] == 0), "feature padding: "+name)
        for name in ("W0", "W1"):
            if previous is not None or generation == 1:
                before = previous[name] if previous is not None else arrays[name.lower()]
                equal_bits(step[name+"_old"], before, "old weight generation: "+name)
        previous = step
    return previous


def audit(case, output):
    manifest, execution, raw = execution_arrays(case, output)
    selected, params = execution["arm"], manifest["params"]
    arrays, expected = inputs(case, manifest, selected), load_npz(Path(case)/"expected.npz")
    for name, (_, dtype) in raw_schema(manifest, selected).items():
        require(np.isfinite(raw[name]).all(), "finite capture: "+name)
        if dtype == "uint16":
            require(np.all(raw[name] < 65536), "u16 native carrier upper half: "+name)
    for name in manifest["immutable_inputs"]:
        value = raw["after_"+name]
        if arrays[name].dtype == np.uint16:
            require(np.array_equal(value, arrays[name].astype(np.uint32)), "immutable input: "+name)
        else:
            equal_bits(value, arrays[name], "immutable input: "+name)
    for index in range(1, STEPS+1):
        for name, wanted in markers(params, index, selected).items():
            require(np.array_equal(raw[f"marker{index}_{name}"], wanted), "generation marker")
    status = np.tile(np.array([3, 0 if selected == "resident" else STEPS-1, 0, 1], np.uint32), (params["C"]*params["R"], 1))
    require(np.array_equal(raw["control_status"], status), "terminal continuation status")
    errors = {}
    for name in manifest["outputs"]:
        if name in ("control_status", "step_times"):
            continue
        want = expected["history8" if name == "history" else name]
        if want.dtype.kind == "f":
            delta = np.abs(raw[name].astype(np.float64)-want)
            require(np.all(delta <= ATOL+RTOL*np.abs(want)), "FP64 output: "+name)
            errors[name] = float(delta.max())
        else:
            require(np.array_equal(raw[name], want), "exact output: "+name)
    last = history_relations(manifest, arrays, raw["history"], STEPS)
    for name in ("w0", "w1"):
        equal_bits(raw[name], last[name.upper()], "final weight: "+name)
    if selected == "download":
        previous = None
        for index in range(1, STEPS+1):
            history = raw["history" if index == STEPS else "history"+str(index)]
            want = expected["history"+str(index)]
            delta = np.abs(history.astype(np.float64)-want)
            require(np.all(delta <= ATOL+RTOL*np.abs(want)), "prefix FP64 history")
            history_relations(manifest, arrays, history, index)
            errors["history"+str(index)] = float(delta.max())
            if previous is not None:
                slot = (index-2) % 2
                frame = sum(fields(params).values())
                equal_bits(history[:, slot*frame:(slot+1)*frame], previous[:, slot*frame:(slot+1)*frame], "retained history overlap")
            previous = history
    words = raw["step_times"].astype(np.uint64)
    require(np.all(words[:, STEPS*6:] == 0), "unused timestamps")
    triples = words[:, :STEPS*6].reshape(-1, STEPS, 2, 3)
    clocks = triples[..., 0] | (triples[..., 1] << np.uint64(16)) | (triples[..., 2] << np.uint64(32))
    require(np.all((clocks > 0) & (clocks < 1 << 47)), "positive bounded no-wrap timestamps")
    require(np.all(clocks[:, :, 1] > clocks[:, :, 0]) and np.all(clocks[:, 1:, 0] > clocks[:, :-1, 1]), "ordered local intervals")
    active = clocks[:, :, 1]-clocks[:, :, 0]
    gaps = clocks[:, 1:, 0]-clocks[:, :-1, 1]
    span = clocks[:, -1, 1]-clocks[:, 0, 0]
    require(np.array_equal(active.sum(axis=1)+gaps.sum(axis=1), span), "same-PE interval decomposition")
    timing = {row["api"]: row for row in execution["api_timings"]}
    host_timing = dict(clock_domain=execution["clock_domain"],
                       constructor_to_stop_ns=timing["stop"]["end_ns"]-timing["construct"]["start_ns"],
                       launch_to_last_marker_ns=timing["d2h:marker8_tags"]["end_ns"]-timing["launch:f_run"]["start_ns"],
                       scope="host monotonic process domain; inter-call journal and Python gaps included in windows")
    return dict(status="PASS", numerical_status="PASS", backend=execution["backend"], arm=selected, host_timing=host_timing,
                errors=errors, max_absolute_error=max(errors.values()), atol=ATOL, rtol=RTOL,
                active_cycles=active.tolist(), gap_cycles=gaps.tolist(), span_cycles=span.tolist(),
                api_timings=execution["api_timings"], scope=manifest["scope"],
                claim="frozen eight-step arithmetic and continuation agreement; same-PE clocks, no global makespan or speedup claim")


def admit_capture(folder):
    folder = Path(folder).absolute()
    record, build, request = (read_json(folder/n) for n in ("capture.json", "manifest.json", "request.json"))
    selected = arm(request.get("arm"))
    require(record.get("status") == "COMPLETED" and record.get("backend") == build.get("backend")
            and build.get("backend") in ("hardware", "simulator"), "completed capture backend")
    require(record.get("run_id") == request.get("run_id") and record.get("manifest_sha256") == request.get("manifest_sha256")
            == sha(folder/"manifest.json"), "capture manifest binding")
    require(record.get("request_sha256") == sha(folder/"request.json") and request.get("case_name") == build.get("case_name"), "capture request binding")
    compile_binding(build)
    require(set(build.get("host_sha256", {})) == set(HOST_FILES), "captured host inventory")
    for name, digest in build["host_sha256"].items():
        require(sha(local_file(folder/"host", name)) == digest, "captured host source")
    manifest = load(folder/"case")
    require(sha(folder/"case/case.json") == build.get("case_sha256") and manifest["params"] == build.get("params")
            and manifest["files"] == build.get("case_files"), "captured case identity")
    check_abi(folder/"build", manifest)
    metadata = record.get("compiled_metadata_sha256", {})
    require(set(metadata) == {"out.json", "bin/out_rpc.json", "inventory.json"}, "compiled metadata inventory")
    for name, digest in metadata.items():
        require(sha(folder/"build"/name) == digest, "compiled metadata bytes")
    inventory = read_json(folder/"build/inventory.json", limit=INVENTORY_JSON_LIMIT)
    admit_compiled_inventory(inventory)
    result = admit_execution(folder/"case", folder/"runtime")
    require(result["backend"] == build["backend"] and result["arm"] == selected, "runtime request identity")
    result.update(run_id=request["run_id"], manifest_sha256=sha(folder/"manifest.json"),
                  request_sha256=sha(folder/"request.json"), artifact_sha256=build["artifact_sha256"],
                  host_sha256=build["host_sha256"], compiled_inventory=inventory["max_static_sram_end"],
                  compiled_elf_files=len(inventory["files"]), compile_workers=build["compile_workers"])
    return result


def audit_report(folder):
    admission = admit_capture(folder)
    try:
        result = audit(Path(folder)/"case", Path(folder)/"runtime")
    except (AssertionError, ValueError) as error:
        return dict(admission, status="FAIL", numerical_status="FAIL", numerical_error=str(error)[:65536])
    return dict(admission, **result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists() and not args.out.is_symlink(), "fresh audit report")
    try:
        if args.capture.is_dir():
            report = audit_report(args.capture)
        else:
            with tempfile.TemporaryDirectory(prefix="resident-control-audit-") as temporary:
                folder = unpack_archive(args.capture, Path(temporary)/"capture", "capture", CAPTURE_LIMIT)
                report = audit_report(folder)
        write_json(args.out, report)
        return 0 if report["status"] == "PASS" else 2
    except Exception as error:
        write_json(args.out, dict(status="FAIL", numerical_status="NOT_RUN", error=str(error)))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
