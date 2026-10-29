"""Standalone original-Cora reconstruction and complete GCN capture audit."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import tempfile
import numpy as np
try:
    from . import portable as p
except ImportError:
    import portable as p

for _name in ("JSON_LIMIT", "INVENTORY_JSON_LIMIT", "ELF_TOTAL_LIMIT", "CASE_LIMIT", "CAPTURE_LIMIT",
              "HOST_FILES", "CASE_REQUIRED", "require", "bounded", "sha", "read_json", "write_json",
              "local_file", "load_npz", "compile_workers", "compiler_arguments", "compile_binding",
              "unpack_archive", "compiled_inventory", "admit_compiled_inventory"):
    globals()[_name] = getattr(p, _name)
PARAMS = ("C", "R", "K", "L", "FW", "H", "O", "T", "MX", "MG")
ATOL, RTOL = 2e-5, 2e-4
PINS = {
    "dataset/cora.content": "b8ebed0fd45e53995abafd1b70cf6a9fcc8f86abc5826d69b97fe6ca46e9c8ee",
    "dataset/cora.cites": "ec1a372391b7f0f60a6aff0084e8abd8f19f0faa7e1f2441a41c492042d5945e",
    "checkpoint.npz": "836c82b82871c76bb1822ace2014c9552690459a952de1b23d4399d5ecce4542",
}


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def schema(params, output=False):
    if output:
        return dict(pre=[params["L"]*16, "float32"], logits=[params["L"]*7, "float32"],
                    panel_sum=[params["K"]*params["T"], "float32"], stats=[8, "uint32"], clocks=[6, "uint16"])
    dims = dict(x_ptr=params["K"]+1, x_col=params["MX"], x_val=params["MX"],
                w0=params["FW"]*16, w1=16*7, g_ptr=params["R"]*params["L"]+1,
                g_src=params["MG"], g_val=params["MG"], slots=params["L"])
    return {name: [count, "float32" if name in ("x_val", "w0", "w1", "g_val") else "uint16"]
            for name, count in dims.items()}


def named_bytes(params):
    c, r, k, l, fw, h, o, t, mx, mg = (params[name] for name in PARAMS)
    integers = k+1+mx+r*l+1+mg+l
    floats = mx+fw*h+h*o+mg+l*h+l*o+3*k*t+l*t
    return 2*integers+4*floats+8*4+6*2+8


def validate_params(params):
    require(isinstance(params, dict) and set(params) == set(PARAMS)
            and all(type(v) is int and 1 <= v <= 65535 for v in params.values()), "u16 parameter contract")
    require(params["H"] == 16 and params["O"] == 7 and params["T"] in (4, 8, 16)
            and params["C"] >= 2 and params["R"] >= 2 and params["C"]*params["R"] <= 4096, "native H16/O7 rectangle")
    require(params["K"]*params["T"] <= 32767
            and all(count <= 65535 for count, _ in dict(schema(params), **schema(params, True)).values()), "descriptor/array bounds")


def native_inputs(case, manifest):
    """Reconstruct from raw corpus and retained trained weights, never packed tables."""
    case = Path(case)
    if manifest["kind"] == "tiny":
        raw = load_npz(case/"original.npz")
        require(set(raw) == {"x", "src", "dst", "coeff", "w0", "w1"}, "tiny raw inventory")
        require(raw["x"].shape == (7, 23) and raw["w0"].shape == (23, 16) and raw["w1"].shape == (16, 7), "beyond-prefix tiny dimensions")
        require(np.any(raw["x"][:, 17] != 0) and np.any(raw["x"][:, 22] != 0), "features beyond first sixteen are live")
        raw.update(labels=None, valid=None, full=False)
        return raw
    for name, digest in PINS.items():
        require(sha(case/name) == digest, "original corpus/checkpoint pin: "+name)
    rows = [line.split() for line in bounded(case/"dataset/cora.content", CASE_LIMIT).decode().splitlines()]
    ids = {row[0]: i for i, row in enumerate(rows)}
    classes = sorted({row[-1] for row in rows})
    labels = np.array([classes.index(row[-1]) for row in rows], np.int64)
    x = np.array([row[1:-1] for row in rows], np.float32)
    require(len(ids) == 2708 and x.shape == (2708, 1433) and len(classes) == 7, "native Cora dimensions")
    x /= np.maximum(x.sum(axis=1, keepdims=True), 1)
    pairs = set()
    for line in bounded(case/"dataset/cora.cites", CASE_LIMIT).decode().splitlines():
        a, b = (ids[key] for key in line.split())
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    edges = np.array(sorted(pairs), np.int64)
    require(edges.shape == (5278, 2), "complete original unpruned graph")
    rng = np.random.default_rng(20260916)
    train = np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:20] for c in range(7)])
    valid = rng.permutation(np.setdiff1d(np.arange(2708), train))[:500]
    record = read_json(case/"checkpoint.json")
    require(record.get("profile") == "original" and record.get("seed") == 11 and record.get("epochs") == 200
            and record.get("weights_sha256") == PINS["checkpoint.npz"]
            and record.get("retained_sha256") == array_sha(edges), "retained trained checkpoint identity")
    dataset = record["dataset"]
    for name, wanted in dict(content_sha256=PINS["dataset/cora.content"], cites_sha256=PINS["dataset/cora.cites"],
                             classes=classes, train_indices=train.tolist(), validation_indices=valid.tolist(),
                             split_seed=20260916, test_evaluated=False, features=1433, nodes=2708,
                             feature_normalization="row sum", retained_edges=5278, removed_edges=0).items():
        require(dataset.get(name) == wanted, "checkpoint dataset/split: "+name)
    weights = load_npz(case/"checkpoint.npz")
    require(set(weights) == {"w0", "w1"} and weights["w0"].shape == (1433, 16)
            and weights["w1"].shape == (16, 7) and all(v.dtype == np.float32 and np.isfinite(v).all() for v in weights.values()), "trained weight tensor contract")
    n = manifest["nodes"]
    if n != 2708:
        edges = edges[np.max(edges, axis=1) < n]
    src = np.concatenate((edges[:, 0], edges[:, 1], np.arange(n)))
    dst = np.concatenate((edges[:, 1], edges[:, 0], np.arange(n)))
    degree = np.ones(n, np.int64)
    np.add.at(degree, edges.ravel(), 1)
    coeff = (1/np.sqrt(degree[src]*degree[dst])).astype(np.float32)
    return dict(x=x[:n], src=src, dst=dst, coeff=coeff, **weights,
                labels=labels, valid=valid, full=n == 2708)


def decode_uploads(params, arrays, nodes, original):
    """Independent exact sparse-X, W0 shard and occurrence decoding."""
    c, r, k, l, fw = (params[name] for name in ("C", "R", "K", "L", "FW"))
    x, src, dst, coeff = (original[name] for name in ("x", "src", "dst", "coeff"))
    n, fin = x.shape
    require(nodes.shape == (c*r, l) and nodes.dtype == np.int64 and np.all((nodes >= -1) & (nodes < n)), "node ownership ABI")
    require(np.array_equal(np.sort(nodes[nodes >= 0]), np.arange(n)), "every node owned exactly once")
    require(fw == (fin+c-1)//c, "all native input features sharded")
    owner = np.empty(n, np.int64)
    for pe in range(c*r):
        owner[nodes[pe, nodes[pe] >= 0]] = pe
    groups = [np.flatnonzero(owner//c == row) for row in range(r)]
    require(max(map(len, groups)) == k, "source row capacity")
    decoded = np.zeros_like(x)
    occurrences = Counter()
    for pe in range(c*r):
        group = groups[pe//c]
        shard = np.zeros((fw, 16), np.float32)
        values = original["w0"][pe % c::c]
        shard[:len(values)] = values
        require(np.array_equal(arrays["w0"][pe].view(np.uint32), shard.ravel().view(np.uint32))
                and np.array_equal(arrays["w1"][pe].view(np.uint32), original["w1"].ravel().view(np.uint32)), "exact supplied weight shard/replica")
        wanted_slots = np.full(l, 65535, np.uint16)
        active = nodes[pe] >= 0
        wanted_slots[active] = np.searchsorted(group, nodes[pe, active])
        require(np.array_equal(arrays["slots"][pe], wanted_slots), "canonical local-to-group slots")
        for prefix, row_count, count in (("x", k, params["MX"]), ("g", r*l, params["MG"])):
            ptr = arrays[prefix+"_ptr"][pe].astype(np.int64)
            index = arrays["x_col" if prefix == "x" else "g_src"][pe]
            value = arrays[prefix+"_val"][pe]
            require(ptr[0] == 0 and np.all(ptr[1:] >= ptr[:-1]) and ptr[-1] <= count, "bounded monotone sparse pointers")
            require(np.all(index[ptr[-1]:] == 0) and np.all(value[ptr[-1]:] == 0), "sparse storage padding")
            for row in range(row_count):
                lo, hi = ptr[row:row+2]
                if prefix == "x":
                    require(row < len(group) or lo == hi, "sparse X padded source")
                    features = index[lo:hi].astype(np.int64)*c+pe % c
                    require(np.all(features < fin) and len(np.unique(features)) == len(features)
                            and np.all(value[lo:hi] != 0), "sparse X feature identity")
                    if hi > lo:
                        decoded[group[row], features] = value[lo:hi]
                else:
                    source_group, local = divmod(row, l)
                    require(nodes[pe, local] >= 0 or lo == hi, "graph padded destination")
                    for edge in range(lo, hi):
                        require(index[edge] < len(groups[source_group]), "graph source slot")
                        occurrences[(int(groups[source_group][index[edge]]), int(nodes[pe, local]), float(value[edge]))] += 1
    require(np.array_equal(decoded.view(np.uint32), x.view(np.uint32)), "complete decoded native sparse features")
    require(occurrences == Counter((int(s), int(d), float(v)) for s, d, v in zip(src, dst, coeff)), "graph occurrence/coefficient identity")
    tiles = (16+params["T"]-1)//params["T"]+(7+params["T"]-1)//params["T"]
    stats = np.zeros((c*r, 8), np.uint32)
    stats[:, :4], stats[:, 7] = (0x600D, tiles, tiles, tiles*r), tiles
    stats[:, 4] = arrays["x_ptr"][:, -1].astype(np.uint32)*16
    stats[:, 5] = np.bincount(owner[dst], minlength=c*r)*23
    stats[:, 6] = np.count_nonzero(nodes >= 0, axis=1)*112
    return groups, stats


def independent_expected(case, manifest, arrays=None):
    params = manifest["params"]
    arrays = load_npz(Path(case)/"inputs.npz") if arrays is None else arrays
    nodes = load_npz(Path(case)/"node_ids.npz")["nodes"]
    original = native_inputs(case, manifest)
    groups, stats = decode_uploads(params, arrays, nodes, original)
    x, w0, w1 = (original[name].astype(np.float64) for name in ("x", "w0", "w1"))
    src, dst, coeff = original["src"], original["dst"], original["coeff"].astype(np.float64)
    projected = x@w0
    pre = np.zeros((len(x), 16), np.float64)
    np.add.at(pre, dst, coeff[:, None]*projected[src])
    projected = np.maximum(pre, 0)@w1
    logits = np.zeros((len(x), 7), np.float64)
    np.add.at(logits, dst, coeff[:, None]*projected[src])
    active = nodes >= 0
    expected = dict(stats=stats)
    for name, tensor in (("pre", pre), ("logits", logits)):
        packed = np.zeros((*nodes.shape, tensor.shape[1]), np.float64)
        packed[active] = tensor[nodes[active]]
        expected[name] = packed.reshape(len(nodes), -1)
    panel = np.zeros((params["R"], params["C"], params["K"], params["T"]), np.float64)
    first = ((7-1)//params["T"])*params["T"]
    for row, group in enumerate(groups):
        panel[row, :, :len(group), :7-first] = projected[group, first:]
    expected["panel_sum"] = panel.reshape(len(nodes), -1)
    return expected, dict(original=original, nodes=nodes, groups=groups, logits=logits)


def load(case):
    case = Path(case).absolute()
    manifest = read_json(local_file(case, "case.json"))
    require(manifest.get("schema") == 1 and manifest.get("family") == "native-cora"
            and manifest.get("kind") in ("tiny", "cora"), "fixed native-Cora family")
    params = manifest.get("params")
    validate_params(params)
    require(manifest.get("inputs") == schema(params) and manifest.get("outputs") == schema(params, True)
            and manifest.get("immutable_inputs") == sorted(schema(params)), "fixed inference/immutable ABI")
    require(manifest.get("launch") == dict(entrypoint="f_run", arguments=[]), "one zero-argument f_run")
    require(manifest.get("atol") == ATOL and manifest.get("rtol") == RTOL, "fixed FP64 tolerances")
    require(type(manifest.get("nodes")) is int and 1 <= manifest["nodes"] <= 2708
            and manifest.get("features") == (23 if manifest["kind"] == "tiny" else 1433), "complete feature contract")
    require(manifest["kind"] != "tiny" or manifest["nodes"] == 7, "tiny node count")
    require(manifest.get("profile") == ("adversarial" if manifest["kind"] == "tiny" else "original"), "unpruned trained profile")
    files = manifest.get("files")
    required = CASE_REQUIRED | ({"original.npz"} if manifest["kind"] == "tiny" else set(PINS) | {"checkpoint.json"})
    require(isinstance(files, dict) and set(files) == required, "exact portable frozen inventory")
    total = len(bounded(case/"case.json", JSON_LIMIT))
    for name, digest in files.items():
        path = local_file(case, name)
        require(sha(path) == digest, "frozen file identity: "+name)
        total += path.stat().st_size
        require(total <= CASE_LIMIT, "complete case byte bound")
    arrays = load_npz(case/"inputs.npz")
    require(set(arrays) == set(schema(params)), "input inventory")
    pes = params["C"]*params["R"]
    for name, (count, dtype) in schema(params).items():
        require(arrays[name].shape == (pes, count) and arrays[name].dtype == np.dtype(dtype)
                and np.isfinite(arrays[name]).all(), "input ABI: "+name)
    expected = load_npz(case/"expected.npz")
    require(set(expected) == set(schema(params, True))-{"clocks"}, "frozen expected inventory")
    for name, value in expected.items():
        count, dtype = schema(params, True)[name]
        require(value.shape == (pes, count) and value.dtype == np.dtype("float64" if dtype == "float32" else dtype)
                and np.isfinite(value).all(), "finite expected ABI: "+name)
    admission = manifest.get("admission", {})
    reserve = admission.get("code_reserve_bytes")
    require(type(reserve) is int and 0 <= reserve <= 44*1024
            and admission.get("named_bytes") == named_bytes(params)
            and admission.get("forecast_bytes") == named_bytes(params)+reserve
            and admission.get("admitted") is (named_bytes(params)+reserve < 44*1024), "explicit static forecast")
    raw_bytes = sum(count*4*pes for count, _ in raw_schema(manifest).values())
    require(total+raw_bytes+(32 << 20) <= CAPTURE_LIMIT, "projected complete capture bound")
    return manifest


def check_abi(build, manifest):
    require({k: int(v) for k, v in read_json(Path(build)/"out.json")["params"].items()} == manifest["params"], "compiled parameter ABI")
    symbols = read_json(Path(build)/"bin/out_rpc.json")["rpc_symbols"]
    expected = {name: ("Var", {"float32": "[*]f32", "uint16": "[*]u16", "uint32": "[*]u32"}[dtype], ())
                for name, (_, dtype) in dict(manifest["inputs"], **manifest["outputs"]).items()}
    expected["f_run"] = ("Func", "void", ())
    require(len(symbols) == len(expected) and len({s["id"] for s in symbols}) == len(symbols)
            and len({s["name"] for s in symbols}) == len(symbols), "unique complete RPC symbols")
    require({s["name"]: (s["kind"], s["type"], tuple(a["type"] for a in s.get("inputs", []))) for s in symbols} == expected, "fixed f_run/pointer ABI")


def raw_schema(manifest):
    return dict(manifest["outputs"], initial_stats=[8, "uint32"],
                **{prefix+name: field for prefix in ("before_", "after_") for name, field in manifest["inputs"].items()})


def api_sequence(manifest):
    return (["construct"]+["get_id:"+n for n in dict(manifest["inputs"], **manifest["outputs"])]
            +["load", "run"]+["h2d:"+n for n in manifest["inputs"]]
            +["d2h:before_"+n for n in manifest["inputs"]]+["d2h:initial_stats", "launch:f_run"]
            +["d2h:"+n for n in ["stats"]+[k for k in manifest["outputs"] if k != "stats"]]
            +["d2h:after_"+n for n in manifest["inputs"]]+["stop"])


def execution_arrays(case, output):
    manifest = load(case)
    execution = read_json(Path(output)/"execution.json")
    require(execution.get("status") == "COMPLETED" and not any(n in execution for n in ("error", "stop_error"))
            and execution.get("backend") in ("simulator", "hardware"), "completed explicit execution backend")
    require(execution.get("case_sha256") == sha(Path(case)/"case.json")
            and execution.get("raw_sha256") == sha(Path(output)/"raw.npz", CAPTURE_LIMIT), "bound raw capture")
    calls = api_sequence(manifest)
    require(execution.get("api") == calls and execution.get("launch_arguments") == [[]]
            and execution.get("launch_typed_arguments") == [[]], "exact one-launch SDK sequence")
    timing = execution.get("api_timings", [])
    require([row.get("api") for row in timing] == calls
            and isinstance(execution.get("clock_domain"), str) and ":monotonic_ns:" in execution["clock_domain"], "complete host timing domain")
    previous = 0
    for row in timing:
        require(type(row.get("start_ns")) is int and type(row.get("end_ns")) is int
                and row["end_ns"] >= row["start_ns"] >= previous
                and row.get("wall_seconds") == (row["end_ns"]-row["start_ns"])/1e9, "ordered host API endpoints")
        label = row["api"]
        method = {"construct": "SdkRuntime", "load": "load", "run": "run", "stop": "stop"}.get(label)
        for prefix, native in (("get_id:", "get_id"), ("h2d:", "memcpy_h2d"), ("d2h:", "memcpy_d2h"), ("launch:", "launch")):
            if label.startswith(prefix):
                method = native
        require(row.get("method") == method, "actual SDK callable")
        if label.startswith(("h2d:", "d2h:", "launch:")):
            require(row.get("nonblock") is False, "blocking dependent SDK calls")
        if label.startswith("launch:"):
            require(row.get("symbol") == "f_run", "actual launch symbol")
        previous = row["end_ns"]
    raw = load_npz(Path(output)/"raw.npz", CAPTURE_LIMIT)
    require(set(raw) == set(raw_schema(manifest)), "complete raw array inventory")
    pes = manifest["params"]["C"]*manifest["params"]["R"]
    for name, (count, dtype) in raw_schema(manifest).items():
        require(raw[name].shape == (pes, count) and raw[name].dtype == np.dtype("uint32" if dtype == "uint16" else dtype), "native carrier ABI: "+name)
    return manifest, execution, raw


def admit_execution(case, output):
    manifest, execution, raw = execution_arrays(case, output)
    return dict(status="PASS", execution_status="PASS", numerical_status="NOT_RUN", backend=execution["backend"],
                case_sha256=execution["case_sha256"], raw_sha256=execution["raw_sha256"], arrays=len(raw), params=manifest["params"],
                claim="bound complete SDK capture; numerical correctness not yet checked")


def audit(case, output):
    manifest, execution, raw = execution_arrays(case, output)
    arrays = load_npz(Path(case)/"inputs.npz")
    expected, context = independent_expected(case, manifest, arrays)
    frozen = load_npz(Path(case)/"expected.npz")
    for name, wanted in expected.items():
        require(np.allclose(frozen[name], wanted, atol=2e-12, rtol=2e-12), "independent frozen reference: "+name)
    for name, (_, dtype) in raw_schema(manifest).items():
        require(np.isfinite(raw[name]).all(), "finite raw: "+name)
        if dtype == "uint16":
            require(np.all(raw[name] < 65536), "u16 native carrier upper half: "+name)
    for prefix in ("before_", "after_"):
        for name, wanted in arrays.items():
            carrier = wanted.astype(np.uint32) if wanted.dtype == np.uint16 else wanted
            require(np.array_equal(raw[prefix+name].view(np.uint32), carrier.view(np.uint32)), "immutable input: "+prefix+name)
    require(np.all(raw["initial_stats"] == 0), "prelaunch zero statistics")
    errors = {}
    for name, wanted in expected.items():
        if name == "stats":
            require(np.array_equal(raw[name], wanted), "exact counters/completion")
        else:
            delta = np.abs(raw[name].astype(np.float64)-wanted)
            require(np.all(delta <= ATOL+RTOL*np.abs(wanted)), "independent FP64 output: "+name)
            errors[name] = float(delta.max())
    params, nodes = manifest["params"], context["nodes"]
    for name, width in (("pre", 16), ("logits", 7)):
        require(np.all(raw[name].reshape(*nodes.shape, width)[nodes < 0] == 0), "structural owner padding: "+name)
    panel = raw["panel_sum"].reshape(params["R"], params["C"], params["K"], params["T"])
    active = 7-((7-1)//params["T"])*params["T"]
    for row, group in enumerate(context["groups"]):
        require(np.all(panel[row, :, len(group):] == 0) and np.all(panel[row, :, :, active:] == 0), "structural final panel padding")
        require(np.array_equal(panel[row].view(np.uint32), np.broadcast_to(panel[row, :1], panel[row].shape).copy().view(np.uint32)), "exact final panel replicas")
    device = np.empty((manifest["nodes"], 7), np.float32)
    device[nodes[nodes >= 0]] = raw["logits"].reshape(*nodes.shape, 7)[nodes >= 0]
    disagreements = int(np.count_nonzero(device.argmax(1) != context["logits"].argmax(1)))
    require(disagreements == 0, "complete prediction agreement")
    prediction = dict(all_nodes_disagree=disagreements)
    if context["original"]["full"]:
        original = context["original"]
        prediction.update(validation_nodes=500, device_validation_accuracy=float(np.mean(device[original["valid"]].argmax(1) == original["labels"][original["valid"]])),
                          reference_validation_accuracy=float(np.mean(context["logits"][original["valid"]].argmax(1) == original["labels"][original["valid"]])))
    words = raw["clocks"].astype(np.uint64)
    times = words[:, 0::3] | (words[:, 1::3] << np.uint64(16)) | (words[:, 2::3] << np.uint64(32))
    cycles = (times[:, 1]-times[:, 0]) & np.uint64((1 << 48)-1)
    require(np.all((cycles > 0) & (cycles < 1 << 47)), "same-PE clock intervals")
    timing = {row["api"]: row for row in execution["api_timings"]}
    return dict(status="PASS", numerical_status="PASS", backend=execution["backend"], nodes=manifest["nodes"], features=manifest["features"],
                profile=manifest["profile"], calibration_only=not context["original"]["full"], params=params,
                max_abs_error=errors, predictions=prediction, local_interval_min_max=[int(cycles.min()), int(cycles.max())],
                host_timing=dict(clock_domain=execution["clock_domain"], constructor_to_stop_ns=timing["stop"]["end_ns"]-timing["construct"]["start_ns"],
                                 launch_to_completion_read_ns=timing["d2h:stats"]["end_ns"]-timing["launch:f_run"]["start_ns"]),
                api_timings=execution["api_timings"], atol=ATOL, rtol=RTOL, scope=manifest["scope"],
                claim="independent complete GCN inference agreement; same-PE clocks, no global makespan or device-training claim")

def admit_capture(folder):
    folder = Path(folder).absolute()
    record, build, request = (read_json(folder/n) for n in ("capture.json", "manifest.json", "request.json"))
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
    require(result["backend"] == build["backend"], "runtime request identity")
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
            with tempfile.TemporaryDirectory(prefix="native-cora-audit-") as temporary:
                folder = unpack_archive(args.capture, Path(temporary)/"capture", "capture", CAPTURE_LIMIT)
                report = audit_report(folder)
        write_json(args.out, report)
        return 0 if report["status"] == "PASS" else 2
    except Exception as error:
        write_json(args.out, dict(status="FAIL", numerical_status="NOT_RUN", error=str(error)))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
