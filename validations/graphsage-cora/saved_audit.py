"""Portable original-occurrence GraphSAGE oracle and complete capture admission.

Transport admission derives from native_cora_appliance/saved_audit.py (E217).
The model oracle and packed-input decoder below do not import the device packer.
"""
import argparse
from collections import defaultdict
import io
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
WEIGHTS = ("ws0", "wn0", "b0", "ws1", "wn1", "b1")
INTEGERS = ("src", "dst", "labels", "train_idx", "val_idx", "test_idx")
MODEL_KEYS = {"x", *WEIGHTS, *INTEGERS}
ATOL, RTOL = 2e-5, 2e-4
MODEL_CONTRACT = "mean GraphSAGE classifier; hidden ReLU/L2; final raw logits; no implicit self loops"
CAPTURE_CONTRACT = "last slots upload witness plus complete outputs; no replicated weight/input D2H audit"


def exact(left, right):
    return left.shape == right.shape and left.dtype == right.dtype and np.array_equal(
        np.ascontiguousarray(left).view(np.uint8), np.ascontiguousarray(right).view(np.uint8))


def schema(params, output=False):
    h, o, l = (params[name] for name in ("H", "O", "L"))
    if output:
        return dict(stats=[10, "uint32"], clocks=[6, "uint16"], pre=[l*h, "float32"],
                    hidden=[l*h, "float32"], logits=[l*o, "float32"])
    # This order is the upload ABI: slots is the final upload, even after JSON
    # serialization sorts dictionary keys.
    return dict(x_ptr=[params["K"]+1, "uint16"], x_col=[params["MX"], "uint16"],
                x_val=[params["MX"], "float32"], ws0=[params["FW"]*h, "float32"],
                wn0=[params["FW"]*h, "float32"], ws1=[h*o, "float32"], wn1=[h*o, "float32"],
                b0=[h, "float32"], b1=[o, "float32"], g_ptr=[params["R"]*l+1, "uint16"],
                g_src=[params["MG"], "uint16"], g_val=[params["MG"], "float32"], slots=[l, "uint16"])


def named_bytes(params):
    arrays = sum(size*np.dtype(dtype).itemsize for size, dtype in
                 dict(schema(params), **schema(params, True)).values())
    return arrays+4*(3*params["K"]*params["T"]+2*params["L"]*params["T"])+10


def validate_params(params):
    require(isinstance(params, dict) and set(params) == set(PARAMS)
            and all(type(v) is int and 1 <= v <= 65535 for v in params.values()), "u16 parameter contract")
    require(params["T"] in (4, 8, 16) and params["C"] >= 2 and params["R"] >= 2
            and params["C"]*params["R"] <= 4096, "bounded collective rectangle")
    require(params["K"]*params["T"] <= 32767 and all(count <= 65535 for count, _ in
            dict(schema(params), **schema(params, True)).values()), "descriptor/array bounds")


def checkpoint(case, manifest):
    """Validate the original checkpoint before selecting an induced prefix."""
    case = Path(case)
    raw = load_npz(case/"input_model.npz", 64 << 20)
    require(set(raw) == MODEL_KEYS, "original checkpoint inventory")
    for name, array in raw.items():
        require(array.dtype == np.dtype("int64" if name in INTEGERS else "float32")
                and np.isfinite(array).all(), "original checkpoint dtype/finite: "+name)
    x = raw["x"]
    require(x.ndim == 2 and 0 < x.shape[0] <= 10000 and x.shape[1] > 0, "bounded original features")
    n, f = x.shape
    h, o = manifest["params"]["H"], manifest["params"]["O"]
    shapes = dict(ws0=(f, h), wn0=(f, h), b0=(h,), ws1=(h, o), wn1=(h, o), b1=(o,))
    require(all(raw[name].shape == shape for name, shape in shapes.items()), "original weight shapes")
    src, dst = raw["src"], raw["dst"]
    require(src.ndim == dst.ndim == 1 and src.shape == dst.shape and len(src) <= 1000000
            and np.all((src >= 0) & (src < n)) and np.all((dst >= 0) & (dst < n)), "original edge endpoints")
    require(raw["labels"].shape == (n,) and np.all((raw["labels"] >= 0) & (raw["labels"] < o)), "original labels")
    splits = []
    for name in ("train_idx", "val_idx", "test_idx"):
        ids = raw[name]
        require(ids.ndim == 1 and np.all((ids >= 0) & (ids < n)) and len(np.unique(ids)) == len(ids), "original split: "+name)
        splits.extend(ids.tolist())
    require(len(set(splits)) == len(splits), "disjoint original splits")
    require(n == manifest["original_nodes"] and f == manifest["features"], "original dimensions")
    if manifest["kind"] == "cora":
        require((n, f, h, o) == (2708, 1433, 16, 7) and len(src) == 10556
                and np.all(src != dst), "native full Cora without self loops")
        training = read_json(case/"training.json")
        require(training.get("model") == "full-neighbor mean GraphSAGE classifier, hidden ReLU+L2, raw final logits"
                and training.get("nodes") == n and training.get("features") == f and training.get("hidden") == h
                and training.get("classes") == o and training.get("directed_nonself_occurrences") == len(src), "training model provenance")
        for name, path in (("input.npz", "input_model.npz"), ("weights.npz", "weights.npz")):
            pin = training.get("files", {}).get(name, {})
            require(pin.get("sha256") == sha(case/path) and pin.get("bytes") == (case/path).stat().st_size, "training file binding: "+name)
        dataset = training.get("dataset", {})
        require(dataset.get("content_sha256") == "b8ebed0fd45e53995abafd1b70cf6a9fcc8f86abc5826d69b97fe6ca46e9c8ee"
                and dataset.get("cites_sha256") == "ec1a372391b7f0f60a6aff0084e8abd8f19f0faa7e1f2441a41c492042d5945e"
                and dataset.get("feature_normalization") == "row sum"
                and dataset.get("nodes") == n and dataset.get("features") == f
                and dataset.get("undirected_nonself_edges") == 5278, "recorded raw corpus provenance")
        require(dataset.get("train_indices") == raw["train_idx"].tolist()
                and dataset.get("validation_indices") == raw["val_idx"].tolist()
                and dataset.get("test_evaluated") is False and training.get("test_evaluated") is False
                and raw["test_idx"].size == 0, "training split binding; held-out test remains unused")
        weights = load_npz(case/"weights.npz")
        require(set(weights) == set(WEIGHTS) and all(exact(weights[name], raw[name]) for name in WEIGHTS), "exact retained weight tensors")
    else:
        require(n <= 32 and f <= 16 and max(h, o) <= 16, "tiny semantic fixture bounds")
    selected = manifest["nodes"]
    require(type(selected) is int and 0 < selected <= n, "selected prefix")
    result = {name: value.copy() for name, value in raw.items()}
    result["x"], result["labels"] = raw["x"][:selected].copy(), raw["labels"][:selected].copy()
    keep = (src < selected) & (dst < selected)
    result["src"], result["dst"] = src[keep].copy(), dst[keep].copy()
    for name in ("train_idx", "val_idx", "test_idx"):
        result[name] = raw[name][raw[name] < selected].copy()
    return result


def prepared_nodes(path):
    """The original NPY is retained byte-for-byte beside its bounded NPZ copy."""
    payload = bounded(path, 1 << 20)
    stream = io.BytesIO(payload)
    version = np.lib.format.read_magic(stream)
    require(version in ((1, 0), (2, 0)), "node-map NPY version")
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(stream)
    require(len(shape) == 2 and all(type(n) is int and 0 < n <= 65535 for n in shape)
            and not fortran and dtype == np.dtype("int64")
            and len(payload)-stream.tell() == shape[0]*shape[1]*8, "bounded original node map")
    return np.load(io.BytesIO(payload), allow_pickle=False)


def decode_uploads(params, arrays, nodes, original):
    """Check canonical feature shards and every ordered original edge occurrence."""
    c, r, k, l, fw, h = (params[name] for name in ("C", "R", "K", "L", "FW", "H"))
    x, src, dst = (original[name] for name in ("x", "src", "dst"))
    n, f = x.shape
    owner = np.arange(n, dtype=np.int64) % (c*r)
    groups = [np.flatnonzero(owner//c == row) for row in range(r)]
    vertices = [np.flatnonzero(owner == pe) for pe in range(c*r)]
    expected_nodes = np.full((c*r, l), -1, np.int64)
    require(max(1, max(map(len, groups))) == k and max(1, max(map(len, vertices))) == l
            and fw == (f+c-1)//c, "canonical node/feature capacities")
    for pe, ids in enumerate(vertices):
        expected_nodes[pe, :len(ids)] = ids
    require(exact(nodes, expected_nodes), "canonical original-ID ownership including empty partitions")
    degree = np.bincount(dst, minlength=n)
    occurrences = defaultdict(list)
    for source, dest in zip(src, dst):
        occurrences[(int(dest), int(owner[source]//c))].append(int(source))
    decoded = np.zeros_like(x)
    seen = np.zeros(x.shape, np.uint8)
    for pe in range(c*r):
        group = groups[pe//c]
        for name in ("ws0", "wn0"):
            shard = np.zeros((fw, h), np.float32)
            values = original[name][pe % c::c]
            shard[:len(values)] = values
            require(exact(arrays[name][pe], shard.ravel()), "exact first-layer weight shard: "+name)
        for name in ("ws1", "wn1", "b0", "b1"):
            require(exact(arrays[name][pe], original[name].ravel()), "exact replicated parameter: "+name)
        wanted_slots = np.full(l, 65535, np.uint16)
        wanted_slots[:len(vertices[pe])] = np.searchsorted(group, vertices[pe])
        require(exact(arrays["slots"][pe], wanted_slots), "canonical local-to-group slots")
        for prefix, row_count, capacity in (("x", k, params["MX"]), ("g", r*l, params["MG"])):
            ptr = arrays[prefix+"_ptr"][pe].astype(np.int64)
            index = arrays["x_col" if prefix == "x" else "g_src"][pe]
            value = arrays[prefix+"_val"][pe]
            require(ptr[0] == 0 and np.all(ptr[1:] >= ptr[:-1]) and ptr[-1] <= capacity, "bounded sparse pointers")
            require(np.all(index[ptr[-1]:] == 0) and np.all(value[ptr[-1]:].view(np.uint32) == 0), "exact sparse storage padding")
            for row in range(row_count):
                lo, hi = (int(v) for v in ptr[row:row+2])
                if prefix == "x":
                    if row >= len(group):
                        require(lo == hi, "empty source-group padding")
                        continue
                    features = index[lo:hi].astype(np.int64)*c+pe % c
                    require(np.all(features < f) and np.all(features[1:] > features[:-1])
                            and np.all(value[lo:hi] != 0), "canonical sparse feature columns")
                    decoded[group[row], features] = value[lo:hi]
                    seen[group[row], features] += 1
                else:
                    source_row, local = divmod(row, l)
                    dest = int(nodes[pe, local])
                    if dest < 0:
                        require(lo == hi, "empty destination padding")
                        continue
                    require(np.all(index[lo:hi] < len(groups[source_row])), "bounded source-group index")
                    sources = groups[source_row][index[lo:hi]].tolist()
                    require(sources == occurrences[(dest, source_row)], "ordered original edge occurrences including duplicates")
                    expected_value = np.full(hi-lo, np.float32(1.0/max(1, degree[dest])), np.float32)
                    require(exact(value[lo:hi], expected_value), "destination mean coefficient")
    require(np.all(seen <= 1) and exact(decoded, x), "complete exact original features")
    return groups


def reference(original):
    """Feature mean precedes projection; no uploaded coefficients or CSR used."""
    x = original["x"].astype(np.float64)
    src, dst = original["src"], original["dst"]
    degree = np.maximum(np.bincount(dst, minlength=len(x)), 1)
    def mean(values):
        result = np.zeros_like(values)
        np.add.at(result, dst, values[src])
        return result/degree[:, None]
    weights = {name: original[name].astype(np.float64) for name in WEIGHTS}
    pre = x@weights["ws0"]+mean(x)@weights["wn0"]+weights["b0"]
    positive = np.maximum(pre, 0)
    scale = positive.max(axis=1, keepdims=True)
    scaled = np.divide(positive, scale, out=np.zeros_like(positive), where=scale != 0)
    norm = np.sqrt(np.sum(scaled*scaled, axis=1, keepdims=True))
    hidden = np.divide(scaled, norm, out=np.zeros_like(scaled), where=norm != 0)
    logits = hidden@weights["ws1"]+mean(hidden)@weights["wn1"]+weights["b1"]
    result = dict(pre=pre, hidden=hidden, logits=logits)
    require(all(np.isfinite(value).all() for value in result.values()), "finite independent FP64 reference")
    return result


def independent_expected(case, manifest, arrays):
    original = checkpoint(case, manifest)
    nodes_file = load_npz(Path(case)/"node_ids.npz")
    require(set(nodes_file) == {"nodes"}, "node map inventory")
    nodes = nodes_file["nodes"]
    params = manifest["params"]
    groups = decode_uploads(params, arrays, nodes, original)
    refs = reference(original)
    count = params["C"]*params["R"]
    expected = {name: np.zeros((count, size), "uint32" if name == "stats" else "float64")
                for name, (size, _) in schema(params, True).items() if name != "clocks"}
    for name, values in refs.items():
        shaped = expected[name].reshape(*nodes.shape, values.shape[1])
        shaped[nodes >= 0] = values[nodes[nodes >= 0]]
    tiles = (params["H"]+params["T"]-1)//params["T"]+(params["O"]+params["T"]-1)//params["T"]
    counts = expected["stats"]
    counts[:, 0], counts[:, 1:3], counts[:, 3] = 0x600D, 2*tiles, params["R"]*tiles
    # These counts come from original features/occurrences and ownership, rather
    # than trusting the uploaded pointer endpoints as their own oracle.
    for pe in range(count):
        owned = nodes[pe][nodes[pe] >= 0]
        counts[pe, 4] = np.count_nonzero(original["x"][groups[pe//params["C"]]][:, pe % params["C"]::params["C"]])*2*params["H"]
        counts[pe, 5] = np.count_nonzero(np.isin(original["dst"], owned))*(params["H"]+params["O"])
        counts[pe, 6] = len(owned)*2*params["H"]*params["O"]
        counts[pe, 7:9] = tiles, len(owned)
        counts[pe, 9] = np.count_nonzero(np.any(refs["hidden"][owned] != 0, axis=1))
    return expected, dict(nodes=nodes, original=original, **refs)


def load(case):
    case = Path(case)
    manifest = read_json(case/"case.json")
    require(manifest.get("schema") == 1 and manifest.get("family") == "graphsage-cora"
            and manifest.get("kind") in ("tiny", "cora"), "GraphSAGE case family")
    params = manifest["params"]
    validate_params(params)
    require(manifest.get("inputs") == schema(params) and manifest.get("outputs") == schema(params, True)
            and manifest.get("launch") == dict(symbol="f_run", arguments=[]), "fixed model ABI")
    require(manifest.get("tolerance") == dict(atol=ATOL, rtol=RTOL)
            and manifest.get("model_contract") == MODEL_CONTRACT
            and manifest.get("capture_contract") == CAPTURE_CONTRACT, "frozen numerical/capture contract")
    required = CASE_REQUIRED | ({"training.json", "weights.npz"} if manifest["kind"] == "cora" else set())
    require(set(manifest.get("files", {})) == required, "complete case file inventory")
    total = 0
    for name, digest in manifest["files"].items():
        path = local_file(case, name)
        require(sha(path) == digest, "case file identity: "+name)
        total += path.stat().st_size
    require(total <= CASE_LIMIT, "bounded complete case")
    prepared = read_json(case/"prepared-case.json")
    provenance = read_json(case/"provenance.json")
    require(provenance.get("prepared_case_sha256") == sha(case/"prepared-case.json")
            and provenance.get("source_sha256") == prepared.get("source_sha256"), "prepared provenance binding")
    require(prepared.get("params") == params and prepared.get("selected_nodes") == manifest["nodes"]
            and prepared.get("original_nodes") == manifest["original_nodes"]
            and prepared.get("dataset_scope") == manifest["dataset_scope"]
            and prepared.get("model_contract") == MODEL_CONTRACT and prepared.get("capture_contract") == CAPTURE_CONTRACT,
            "prepared model/shape binding")
    require(prepared.get("files", {}).get("input_model.npz") == sha(case/"input_model.npz")
            == prepared.get("original_checkpoint_sha256") == manifest.get("original_checkpoint_sha256")
            and prepared.get("files", {}).get("arrays.npz") == sha(case/"inputs.npz"), "exact original input/upload bytes")
    require(prepared.get("files", {}).get("node_ids.npy") == sha(case/"node_ids.npy"), "exact original node-map bytes")
    node_copy = load_npz(case/"node_ids.npz")
    require(set(node_copy) == {"nodes"} and exact(node_copy["nodes"], prepared_nodes(case/"node_ids.npy")), "node-map archive copy")
    for name in ("layout.csl", "pe.csl"):
        require(prepared.get("source_sha256", {}).get("models/graphsage/cora_device/"+name)
                == sha(case/"source"/name), "frozen device source: "+name)
    admission = manifest.get("admission", {})
    require(admission.get("named_bytes") == named_bytes(params)
            and admission.get("limit_bytes") == 44*1024
            and admission.get("admitted") is (named_bytes(params) <= 44*1024), "declared-state forecast consistency")
    arrays = load_npz(case/"inputs.npz")
    require(set(arrays) == set(schema(params)), "packed upload inventory")
    pes = params["C"]*params["R"]
    for name, (count, dtype) in schema(params).items():
        require(arrays[name].shape == (pes, count) and arrays[name].dtype == dtype
                and np.isfinite(arrays[name]).all(), "packed upload ABI: "+name)
    frozen = load_npz(case/"expected.npz")
    require(set(frozen) == set(schema(params, True))-{ "clocks" }, "expected output inventory")
    for name, values in frozen.items():
        require(values.shape == (pes, schema(params, True)[name][0])
                and values.dtype == ("uint32" if name == "stats" else "float64")
                and np.isfinite(values).all(), "expected output ABI: "+name)
    scope = "synthetic-semantic" if manifest["kind"] == "tiny" else (
        "full-checkpoint" if manifest["nodes"] == 2708 else "induced-prefix-calibration")
    require(manifest["dataset_scope"] == scope, "dataset scope")
    require(total+sum(count*4*pes for count, _ in raw_schema(manifest).values())+(32 << 20) <= CAPTURE_LIMIT,
            "projected complete capture bound")
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
    return dict(schema(manifest["params"], True), initial_stats=[10, "uint32"], upload_tail=[manifest["params"]["L"], "uint16"])


def api_sequence(manifest):
    inputs, outputs = schema(manifest["params"]), schema(manifest["params"], True)
    return (["construct"]+["get_id:"+n for n in dict(inputs, **outputs)]+["load", "run"]
            +["h2d:"+n for n in inputs]+["d2h:upload_tail", "d2h:initial_stats", "launch:f_run"]
            +["d2h:"+n for n in outputs]+["stop"])


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
        require(np.array_equal(frozen[name], wanted) if name == "stats" else
                np.allclose(frozen[name], wanted, atol=2e-12, rtol=2e-12), "independent frozen reference: "+name)
    for name, (_, dtype) in raw_schema(manifest).items():
        require(np.isfinite(raw[name]).all(), "finite raw: "+name)
        if dtype == "uint16":
            require(np.all(raw[name] < 65536), "u16 native carrier upper half: "+name)
    require(exact(raw["upload_tail"], arrays["slots"].astype(np.uint32)), "final slots upload witness")
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
    for name, width in (("pre", params["H"]), ("hidden", params["H"]), ("logits", params["O"])):
        padding = raw[name].reshape(*nodes.shape, width)[nodes < 0]
        require(np.all(padding.view(np.uint32) == 0), "exact zero owner padding: "+name)
    device = np.empty((manifest["nodes"], params["O"]), np.float32)
    device[nodes[nodes >= 0]] = raw["logits"].reshape(*nodes.shape, params["O"])[nodes >= 0]
    disagreements = int(np.count_nonzero(device.argmax(1) != context["logits"].argmax(1)))
    require(disagreements == 0, "complete prediction agreement")
    prediction = dict(all_nodes_disagree=disagreements, all_nodes=manifest["nodes"])
    if manifest["dataset_scope"] == "full-checkpoint":
        original = context["original"]
        prediction["declared_splits"] = {}
        for name in ("train_idx", "val_idx", "test_idx"):
            ids = original[name]
            if len(ids):
                prediction["declared_splits"][name] = dict(total=len(ids),
                    device_correct=int(np.count_nonzero(device[ids].argmax(1) == original["labels"][ids])),
                    reference_correct=int(np.count_nonzero(context["logits"][ids].argmax(1) == original["labels"][ids])))
    words = raw["clocks"].astype(np.uint64)
    times = words[:, 0::3] | (words[:, 1::3] << np.uint64(16)) | (words[:, 2::3] << np.uint64(32))
    cycles = (times[:, 1]-times[:, 0]) & np.uint64((1 << 48)-1)
    require(np.all((cycles > 0) & (cycles < 1 << 47)), "same-PE clock intervals")
    timing = {row["api"]: row for row in execution["api_timings"]}
    return dict(status="PASS", numerical_status="PASS", backend=execution["backend"], nodes=manifest["nodes"], features=manifest["features"],
                dataset_scope=manifest["dataset_scope"], calibration_only=manifest["dataset_scope"] != "full-checkpoint", params=params,
                max_abs_error=errors, predictions=prediction, local_interval_min_max=[int(cycles.min()), int(cycles.max())],
                per_pe_cycles=cycles.tolist(),
                host_timing=dict(clock_domain=execution["clock_domain"], constructor_to_stop_ns=timing["stop"]["end_ns"]-timing["construct"]["start_ns"],
                                 launch_to_completion_read_ns=timing["d2h:stats"]["end_ns"]-timing["launch:f_run"]["start_ns"]),
                api_timings=execution["api_timings"], atol=ATOL, rtol=RTOL, scope=manifest["scope"],
                input_observation="exact pinned host upload buffers and final slots witness; other device inputs not read back",
                claim="independent complete GraphSAGE inference agreement; same-PE clocks, no global makespan or device-training claim")




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
            with tempfile.TemporaryDirectory(prefix="graphsage-cora-audit-") as temporary:
                folder = unpack_archive(args.capture, Path(temporary)/"capture", "capture", CAPTURE_LIMIT)
                report = audit_report(folder)
        write_json(args.out, report)
        return 0 if report["status"] == "PASS" else 2
    except Exception as error:
        write_json(args.out, dict(status="FAIL", numerical_status="NOT_RUN", error=str(error)))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
