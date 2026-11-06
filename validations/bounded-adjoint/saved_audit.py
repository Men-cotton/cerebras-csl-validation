"""Standalone original-occurrence audit for two resident AZ/A-transpose-D launches.

Archive/ELF helpers derive from native_cora_appliance/portable.py. Numerical
formulas follow the E211 original-edge oracle and E212 traversal accounting;
this module imports neither the preparer nor any repository model module.
"""
import argparse
from pathlib import Path
import tempfile
import numpy as np
try:
    from . import portable as p
except ImportError:
    import portable as p

for _name in ("JSON_LIMIT", "INVENTORY_JSON_LIMIT", "ELF_TOTAL_LIMIT", "CASE_LIMIT", "CAPTURE_LIMIT",
              "HOST_FILES", "require", "bounded", "sha", "read_json", "write_json", "local_file",
              "load_npz", "compile_workers", "compiler_arguments", "compile_binding",
              "unpack_archive", "compiled_inventory", "admit_compiled_inventory"):
    globals()[_name] = getattr(p, _name)
CASE_REQUIRED = {"source-manifest.json", "inputs.npz", "original.npz", "reference.npz",
                 "source/layout.csl", "source/pe.csl"}
PARAMS = {"C", "R", "B", "T", "F", "L", "G", "U", "E", "CURSOR"}
ATOL, RTOL = 2e-5, 2e-4
RAW_LIMIT = 64 << 20
CLOCK_SCOPE = ("same-PE first forward descriptor through final adjoint RELEASE/status; "
               "excludes initialization, output reset, export, host transfer and global makespan")


def schema(q, output=False):
    if output:
        return dict(output=[q["L"]*q["T"], "float32"], panel_local=[q["B"]*q["T"], "float32"],
                    panel_home=[q["B"]*q["T"], "float32"], panel_receive=[q["B"]*q["T"], "float32"],
                    stats=[12, "uint32"], gradient=[q["L"]*q["T"], "float32"],
                    clocks=[6, "uint32"], traversal=[10, "uint32"])
    return dict(counts=[4, "uint32"], group_length=[1, "uint32"], owner_id=[q["L"], "uint32"],
                owner_slot=[q["L"], "uint32"], source_values=[q["L"]*q["T"], "float32"],
                group_id=[q["G"], "uint32"], group_begin=[q["G"]+1, "uint16"],
                source_slot=[q["U"], "uint32"], source_id=[q["U"], "uint32"],
                edge_begin=[q["U"]+1, "uint16"], edge_dst=[q["E"], "uint16"],
                edge_value=[q["E"], "float32"], derivative=[q["L"]*q["T"], "float32"])


def validate_params(q):
    require(isinstance(q, dict) and set(q) == PARAMS
            and all(type(v) is int for v in q.values()), "fixed integer parameter ABI")
    require(2 <= q["C"] <= 755 and 2 <= q["R"] <= 1170 and q["CURSOR"] in (0, 1), "rectangle/mode bounds")
    require(1 <= q["F"] <= q["T"] and 1 <= q["B"] and 1 <= q["L"]
            and max(q["B"]*q["T"], q["L"]*q["T"]) <= 32767, "owner/panel signed DSD bounds")
    require(all(1 <= q[k] < 65534 for k in ("G", "U")) and 1 <= q["E"] < 65535,
            "native u16 directory bounds")


def named_bytes(q):
    # Exported arrays, descriptor/release 28 B, private clocks 12 B, scalars 26 B.
    return sum(n*np.dtype(t).itemsize for n, t in dict(schema(q), **schema(q, True)).values())+66


def raw_schema(manifest):
    result = dict(initial_stats=[12, "uint32"])
    for epoch in range(2):
        prefix = f"epoch{epoch}_"
        result.update({prefix+n: v for n, v in manifest["outputs"].items()})
        for stage in ("before_", "after_"):
            for name, field in manifest["inputs"].items():
                result[prefix+stage+name] = field
                if field[1] == "uint16":
                    result[prefix+stage+name+"_native_u32"] = [field[0], "uint32"]
    return result


def exact(actual, wanted, label):
    require(actual.shape == wanted.shape and actual.dtype == wanted.dtype
            and actual.tobytes() == wanted.tobytes(), label)


def original_arrays(case, q):
    raw = load_npz(Path(case)/"original.npz", RAW_LIMIT)
    required = {"ids", "owner", "values", "src", "dst", "coeff", "derivative0", "derivative1"}
    require(required <= set(raw) <= required | {"occurrence_id", "labels", "train", "valid"}, "original array inventory")
    values = raw["values"]
    require(values.dtype == np.float32 and values.ndim == 2 and values.shape[1] == q["F"], "original features")
    n = len(values)
    require(1 <= n <= 32767 and n*q["F"] <= 1 << 20, "original node/feature bound")
    require(raw["ids"].dtype == np.uint32 and raw["ids"].shape == (n,)
            and len(np.unique(raw["ids"])) == n, "unique external IDs")
    require(raw["owner"].dtype == np.int64 and raw["owner"].shape == (n,)
            and np.all((raw["owner"] >= 0) & (raw["owner"] < q["C"]*q["R"])), "original owners")
    require(raw["src"].dtype == raw["dst"].dtype == np.int64 and raw["src"].ndim == 1
            and raw["src"].shape == raw["dst"].shape == raw["coeff"].shape
            and raw["coeff"].dtype == np.float32 and len(raw["src"]) <= 1 << 20, "original occurrence ABI")
    require(np.all((raw["src"] >= 0) & (raw["src"] < n) & (raw["dst"] >= 0) & (raw["dst"] < n)), "original edge bounds")
    for name in ("values", "coeff", "derivative0", "derivative1"):
        require(np.isfinite(raw[name]).all(), "finite original "+name)
    for name in ("derivative0", "derivative1"):
        require(raw[name].dtype == np.float32 and raw[name].shape == values.shape, "original derivative ABI")
    if "occurrence_id" in raw:
        ids = raw["occurrence_id"]
        require(ids.dtype == np.uint32 and ids.shape == raw["src"].shape
                and (len(ids) < 2 or np.all(ids[1:] > ids[:-1])), "original occurrence identity/order")
    for name in ("labels", "train", "valid"):
        if name in raw:
            require(raw[name].shape == (n,) and raw[name].dtype == np.dtype("int32" if name == "labels" else "bool"),
                    "retained node metadata: "+name)
    return raw


def owner_array(raw, q, values):
    out = np.zeros((q["C"]*q["R"], q["L"]*q["T"]), values.dtype)
    for pe in range(len(out)):
        nodes = sorted(np.flatnonzero(raw["owner"] == pe), key=lambda node: int(raw["ids"][node]))
        require(len(nodes) <= q["L"], "owner capacity")
        for slot, node in enumerate(nodes):
            out[pe, slot*q["T"]:slot*q["T"]+q["F"]] = values[node]
    return out


def scatter(raw, values, transpose=False):
    result = np.zeros(values.shape, np.float64)
    for source, dest, coefficient in zip(raw["src"], raw["dst"], raw["coeff"]):
        target, origin = (source, dest) if transpose else (dest, source)
        result[target] += float(coefficient)*values[origin].astype(np.float64)
    return result


def independent_inputs(raw, q):
    """Decode the expected ordered directory directly from original occurrences.

    This is deliberately separate from the E212 packer. Duplicate/zero edges
    retain their original ordinal within each source, including padding bytes.
    """
    c, r, b, t = (q[k] for k in ("C", "R", "B", "T"))
    groups = [sorted(np.flatnonzero(raw["owner"]//c == row), key=lambda n: int(raw["ids"][n])) for row in range(r)]
    owners = [sorted(np.flatnonzero(raw["owner"] == pe), key=lambda n: int(raw["ids"][n])) for pe in range(c*r)]
    slots = {node: slot for group in groups for slot, node in enumerate(group)}
    local = {node: slot for group in owners for slot, node in enumerate(group)}
    arrays = {name: np.zeros((c*r, count), dtype) for name, (count, dtype) in schema(q).items()}
    directories = []
    for pe in range(c*r):
        ordinal = [i for i, dest in enumerate(raw["dst"]) if raw["owner"][dest] == pe]
        ordinal.sort(key=lambda i: (int(raw["owner"][raw["src"][i]]//c), int(raw["ids"][raw["src"][i]]), i))
        source_order = list(dict.fromkeys(int(raw["src"][i]) for i in ordinal))
        group_order = list(dict.fromkeys(int(raw["owner"][node]//c) for node in source_order))
        require(len(owners[pe]) <= q["L"] and len(group_order) <= q["G"] and len(source_order) <= q["U"]
                and len(ordinal) <= q["E"], "original graph fits directory")
        arrays["counts"][pe] = [len(owners[pe]), len(group_order), len(source_order), len(ordinal)]
        if pe % c == 0:
            arrays["group_length"][pe, 0] = len(groups[pe//c])
        for slot, node in enumerate(owners[pe]):
            arrays["owner_id"][pe, slot], arrays["owner_slot"][pe, slot] = raw["ids"][node], slots[node]
        edge_at = 0
        for si, node in enumerate(source_order):
            arrays["source_slot"][pe, si], arrays["source_id"][pe, si] = slots[node], raw["ids"][node]
            arrays["edge_begin"][pe, si] = edge_at
            for i in (i for i in ordinal if int(raw["src"][i]) == node):
                arrays["edge_dst"][pe, edge_at], arrays["edge_value"][pe, edge_at] = local[int(raw["dst"][i])], raw["coeff"][i]
                edge_at += 1
        arrays["edge_begin"][pe, len(source_order)] = edge_at
        begin = 0
        row = {}
        for gi, group in enumerate(group_order):
            members = [node for node in source_order if raw["owner"][node]//c == group]
            arrays["group_id"][pe, gi], arrays["group_begin"][pe, gi] = group, begin
            begin += len(members)
            row[group] = members
        arrays["group_begin"][pe, len(group_order)] = begin
        directories.append(row)
    arrays["source_values"] = owner_array(raw, q, raw["values"])
    arrays["derivative"] = owner_array(raw, q, raw["derivative0"])
    panels = [(len(group)+b-1)//b for group in groups]
    require(max(2*sum(panels), 2*len(raw["src"])*q["F"], 2*r, 0x600D)
            <= np.iinfo(np.uint32).max, "protocol/work u32 counter bound")
    traversal = np.zeros((c*r, 10), np.uint32)
    for pe, row in enumerate(directories):
        if q["CURSOR"] == 0:
            numbers = [panels[pe//c]*q["L"], sum(panels)*q["G"], sum(panels[g]*len(nodes) for g, nodes in row.items())]
        else:
            owned = [slots[node] for node in owners[pe]]
            numbers = [len(owned)+sum(any(slot >= end for slot in owned) for end in range(b, len(groups[pe//c]), b)),
                       sum(any(group >= current for group in row) for current in range(r)),
                       sum(len(nodes)+sum(any(slots[node] >= end for node in nodes) for end in range(b, len(groups[g]), b)) for g, nodes in row.items())]
        numbers += [len(owners[pe]), sum(map(len, row.values()))]
        require(max(numbers) <= np.iinfo(np.uint32).max, "traversal u32 bound")
        traversal[pe] = numbers*2
    stats = np.zeros((c*r, 12), np.uint32)
    stats[:, 0], stats[:, 1], stats[:, 3], stats[:, 4] = 0x600D, 2*sum(panels), 2*sum(panels), 2*r
    stats[:, 2], stats[:, 5] = 2*arrays["counts"][:, 3], 2*arrays["counts"][:, 3]*q["F"]
    stats[:, 6], stats[:, 7], stats[:, 10], stats[:, 11] = r, len(groups[-1]), q["F"], 1
    return arrays, groups, stats, traversal


def independent_expected(case, manifest, arrays=None):
    q = manifest["params"]
    raw = original_arrays(case, q)
    wanted, groups, stats, traversal = independent_inputs(raw, q)
    if arrays is None:
        arrays = load_npz(Path(case)/"inputs.npz", RAW_LIMIT)
    for name, value in wanted.items():
        exact(arrays[name], value, "original ordered upload: "+name)
    forward = scatter(raw, raw["values"])
    expected, panels = {}, []
    last = max(row for row, nodes in enumerate(groups) if nodes)
    selected = groups[last][((len(groups[last])-1)//q["B"])*q["B"]:]
    slots = {node: slot for slot, node in enumerate(selected)}
    for epoch in range(2):
        prefix, d = f"epoch{epoch}_", raw[f"derivative{epoch}"]
        expected[prefix+"derivative"] = owner_array(raw, q, d)
        expected[prefix+"output"] = owner_array(raw, q, forward)
        expected[prefix+"gradient"] = owner_array(raw, q, scatter(raw, d, True))
        epoch_stats = stats.copy()
        epoch_stats[:, 9] = epoch+1
        expected[prefix+"stats"], expected[prefix+"traversal"] = epoch_stats, traversal.copy()
        partial = np.zeros((q["C"]*q["R"], q["B"]*q["T"]), np.float64)
        for source, dest, coefficient in zip(raw["src"], raw["dst"], raw["coeff"]):
            if source in slots:
                at = slots[source]*q["T"]
                partial[raw["owner"][dest], at:at+q["F"]] += float(coefficient)*d[dest].astype(np.float64)
        columns = partial.reshape(q["R"], q["C"], -1).sum(axis=0)
        panels.append(dict(last_group=last, panel_local=partial, source_row_home=columns,
                           source_row_receive=np.broadcast_to(columns.sum(axis=0), columns.shape)))
    require(not len(raw["src"]) or not np.array_equal(expected["epoch0_gradient"], expected["epoch1_gradient"]), "changed-D witness")
    return expected, panels


def load(case):
    case = Path(case).absolute()
    manifest = read_json(local_file(case, "case.json"))
    require(manifest.get("schema") == 1 and manifest.get("family") == "bounded-adjoint", "dedicated family identity")
    q = manifest.get("params")
    validate_params(q)
    require(manifest.get("inputs") == schema(q) and manifest.get("outputs") == schema(q, True), "fixed pointer ABI")
    require(manifest.get("launches") == [dict(entrypoint="run", arguments=[], update=None),
                                         dict(entrypoint="run", arguments=[], update="derivative")], "exact resident launch contract")
    require(manifest.get("atol") == ATOL and manifest.get("rtol") == RTOL
            and manifest.get("clock_scope") == CLOCK_SCOPE, "fixed tolerance/clock scope")
    files = manifest.get("files")
    require(isinstance(files, dict) and set(files) == CASE_REQUIRED, "frozen portable inventory")
    total = len(bounded(case/"case.json", JSON_LIMIT))
    for name, digest in files.items():
        path = local_file(case, name)
        require(sha(path) == digest, "frozen file identity: "+name)
        total += path.stat().st_size
        require(total <= CASE_LIMIT, "complete case bound")
    source = read_json(case/"source-manifest.json")
    require(source.get("experiment") == "E214" and source.get("params") == q, "original prepared E214 identity")
    require(manifest.get("metadata") == source.get("metadata"), "parent metadata binding")
    require(source.get("files") == {n: files[n] for n in CASE_REQUIRED-{"source-manifest.json"}}, "original file binding")
    for name in ("layout.csl", "pe.csl"):
        origin = "models/hierarchical_gnn/grouped_csr/bounded_panels/adjoint_capacity/"+name
        require(source.get("sources", {}).get(origin) == files["source/"+name], "unchanged prepared CSL origin")
    require(manifest.get("source_case_sha256") == files["source-manifest.json"], "source case binding")
    admission = manifest.get("admission", {})
    reserve = admission.get("code_reserve_bytes")
    require(type(reserve) is int and 0 <= reserve <= 48*1024 and admission.get("named_bytes") == named_bytes(q)
            and admission.get("forecast_bytes") == named_bytes(q)+reserve
            and admission.get("admitted") is (named_bytes(q)+reserve <= 44*1024), "source storage forecast")
    require(source.get("metadata", {}).get("forecast_static_bytes") == admission["forecast_bytes"], "parent admission binding")
    arrays = load_npz(case/"inputs.npz", RAW_LIMIT)
    require(set(arrays) == set(schema(q)), "input inventory")
    for name, (count, dtype) in schema(q).items():
        value = arrays[name]
        require(value.dtype == np.dtype(dtype) and value.shape == (q["C"]*q["R"], count)
                and np.isfinite(value).all(), "finite input ABI: "+name)
    raw_bytes = q["C"]*q["R"]*sum(count*np.dtype(dtype).itemsize for count, dtype in raw_schema(manifest).values())
    require(raw_bytes+4096*96 <= RAW_LIMIT and total+raw_bytes+(32 << 20) <= CAPTURE_LIMIT, "complete capture forecast")
    frozen = load_npz(case/"reference.npz", RAW_LIMIT)
    expected_keys = {f"epoch{i}_{n}" for i in range(2) for n in ("derivative", "output", "gradient", "stats", "traversal")}
    require(set(frozen) == expected_keys, "two-launch reference inventory")
    for name, value in frozen.items():
        field = name.split("_", 1)[1]
        count, dtype = (schema(q) if field == "derivative" else schema(q, True))[field]
        if field in ("output", "gradient"):
            dtype = "float64"
        require(value.shape == (q["C"]*q["R"], count) and value.dtype == np.dtype(dtype)
                and np.isfinite(value).all(), "frozen reference ABI: "+name)
    return manifest


def check_abi(build, manifest):
    require({k: int(v) for k, v in read_json(Path(build)/"out.json")["params"].items()} == manifest["params"], "compiled parameter ABI")
    symbols = read_json(Path(build)/"bin/out_rpc.json")["rpc_symbols"]
    expected = {name: ("Var", {"float32": "[*]f32", "uint16": "[*]u16", "uint32": "[*]u32"}[dtype], ())
                for name, (_, dtype) in dict(manifest["inputs"], **manifest["outputs"]).items()}
    expected["run"] = ("Func", "void", ())
    require(len(symbols) == len(expected) and len({s["id"] for s in symbols}) == len(symbols)
            and len({s["name"] for s in symbols}) == len(symbols), "unique complete RPC ABI")
    require({s["name"]: (s["kind"], s["type"], tuple(a["type"] for a in s.get("inputs", []))) for s in symbols} == expected, "run/pointer ABI")


def api_sequence(manifest):
    inputs, outputs = manifest["inputs"], manifest["outputs"]
    calls = ["construct"]+["get_id:"+n for n in dict(inputs, **outputs)]+["load", "run"]
    calls += ["h2d:"+n for n in inputs]+["d2h:initial_stats"]
    for epoch in range(2):
        prefix = f"epoch{epoch}_"
        if epoch:
            calls += ["h2d:"+prefix+"derivative"]
        calls += ["d2h:"+prefix+"before_"+n for n in inputs]+["launch:"+prefix+"run"]
        calls += ["d2h:"+prefix+n for n in ["stats"]+[n for n in outputs if n != "stats"]]
        calls += ["d2h:"+prefix+"after_"+n for n in inputs]
    return calls+["stop"]


def execution_arrays(case, output):
    manifest = load(case)
    execution = read_json(Path(output)/"execution.json")
    require(execution.get("status") == "COMPLETED" and not any(n in execution for n in ("error", "stop_error", "pending_api"))
            and execution.get("backend") in ("simulator", "hardware"), "completed explicit execution")
    require(execution.get("case_sha256") == sha(Path(case)/"case.json")
            and execution.get("raw_sha256") == sha(Path(output)/"raw.npz", RAW_LIMIT), "bound raw capture")
    calls = api_sequence(manifest)
    require(execution.get("api") == calls and execution.get("launch_arguments") == [[], []]
            and execution.get("launch_typed_arguments") == [[], []], "exact two-launch SDK sequence")
    timing = execution.get("api_timings", [])
    require([row.get("api") for row in timing] == calls and isinstance(execution.get("clock_domain"), str)
            and ":monotonic_ns:" in execution["clock_domain"], "complete host timing domain")
    previous = 0
    for row in timing:
        require(type(row.get("start_ns")) is int and type(row.get("end_ns")) is int
                and row["end_ns"] >= row["start_ns"] >= previous
                and row.get("wall_seconds") == (row["end_ns"]-row["start_ns"])/1e9, "ordered host endpoints")
        label = row["api"]
        method = {"construct": "SdkRuntime", "load": "load", "run": "run", "stop": "stop"}.get(label)
        for prefix, native in (("get_id:", "get_id"), ("h2d:", "memcpy_h2d"), ("d2h:", "memcpy_d2h"), ("launch:", "launch")):
            if label.startswith(prefix):
                method = native
        require(row.get("method") == method, "actual SDK callable")
        if label.startswith(("h2d:", "d2h:", "launch:")):
            require(row.get("nonblock") is False, "blocking dependent calls")
        if label.startswith("launch:"):
            require(row.get("symbol") == "run", "actual launch symbol")
        previous = row["end_ns"]
    raw = load_npz(Path(output)/"raw.npz", RAW_LIMIT)
    require(set(raw) == set(raw_schema(manifest)), "complete 81-array capture")
    pes = manifest["params"]["C"]*manifest["params"]["R"]
    for name, (count, dtype) in raw_schema(manifest).items():
        require(raw[name].shape == (pes, count) and raw[name].dtype == np.dtype(dtype), "raw ABI: "+name)
    return manifest, execution, raw


def admit_execution(case, output):
    manifest, execution, raw = execution_arrays(case, output)
    return dict(status="PASS", execution_status="PASS", numerical_status="NOT_RUN", backend=execution["backend"],
                case_sha256=execution["case_sha256"], raw_sha256=execution["raw_sha256"], arrays=len(raw), params=manifest["params"],
                claim="bound complete SDK capture; numerical correctness not yet checked")


def audit(case, output):
    manifest, execution, raw = execution_arrays(case, output)
    q = manifest["params"]
    arrays = load_npz(Path(case)/"inputs.npz", RAW_LIMIT)
    expected, panels = independent_expected(case, manifest, arrays)
    frozen = load_npz(Path(case)/"reference.npz", RAW_LIMIT)
    for name, wanted in expected.items():
        exact(frozen[name], wanted, "independent frozen reference: "+name)
    require(not raw["initial_stats"].any(), "initial zero statistics")
    errors, products, intervals = {}, [], []
    for epoch in range(2):
        prefix = f"epoch{epoch}_"
        inputs = dict(arrays, derivative=expected[prefix+"derivative"])
        for name, value in inputs.items():
            for stage in ("before_", "after_"):
                label = prefix+stage+name
                exact(raw[label], value, "resident input: "+label)
                if value.dtype == np.uint16:
                    exact(raw[label+"_native_u32"], value.astype(np.uint32), "native carrier: "+label)
        for name in ("stats", "traversal"):
            exact(raw[prefix+name], expected[prefix+name], "operator/reset counters: "+prefix+name)
        for name in ("output", "gradient"):
            value, wanted = raw[prefix+name], expected[prefix+name]
            require(np.isfinite(value).all() and np.allclose(value, wanted, atol=ATOL, rtol=RTOL), "original-edge FP64 "+prefix+name)
            errors[prefix+name] = float(np.max(np.abs(value-wanted)))
            view = value.reshape(-1, q["L"], q["T"])
            require(not view[:, :, q["F"]:].view(np.uint32).any(), "owner feature padding")
            for pe in range(len(view)):
                require(not view[pe, int(arrays["counts"][pe, 0]):].view(np.uint32).any(), "owner slot padding")
        lhs = float(np.sum(raw[prefix+"output"].astype(np.float64)*expected[prefix+"derivative"].astype(np.float64)))
        rhs = float(np.sum(arrays["source_values"].astype(np.float64)*raw[prefix+"gradient"].astype(np.float64)))
        require(np.isclose(lhs, rhs, atol=ATOL, rtol=RTOL), "captured adjoint inner product")
        products.append(dict(forward_dot_derivative=lhs, source_dot_gradient=rhs))
        for name in ("panel_local", "panel_home", "panel_receive"):
            value = raw[prefix+name]
            require(np.isfinite(value).all() and not value.reshape(-1, q["B"], q["T"])[:, :, q["F"]:].view(np.uint32).any(), "panel finiteness/padding")
        panel = panels[epoch]
        row = slice(panel["last_group"]*q["C"], (panel["last_group"]+1)*q["C"])
        for observed, wanted in ((raw[prefix+"panel_local"], panel["panel_local"]),
                                 (raw[prefix+"panel_home"][row], panel["source_row_home"]),
                                 (raw[prefix+"panel_receive"][row], panel["source_row_receive"])):
            require(np.allclose(observed, wanted, atol=ATOL, rtol=RTOL), "original-edge defined return panel")
        words = raw[prefix+"clocks"].astype(np.uint64)
        require(np.all(words <= 65535), "u32 clock words have zero high halves")
        start = words[:, 0]+(words[:, 1] << 16)+(words[:, 2] << 32)
        end = words[:, 3]+(words[:, 4] << 16)+(words[:, 5] << 32)
        cycles = (end-start) & ((1 << 48)-1)
        require(np.all((cycles > 0) & (cycles < 1 << 47)), "positive same-PE intervals")
        intervals.append(dict(min=int(cycles.min()), max=int(cycles.max())))
    exact(raw["epoch0_output"], raw["epoch1_output"], "forward bits repeat after D-only update")
    timings = {row["api"]: row for row in execution["api_timings"]}
    windows = [timings[f"d2h:epoch{i}_stats"]["end_ns"]-timings[f"launch:epoch{i}_run"]["start_ns"] for i in range(2)]
    return dict(status="PASS", numerical_status="PASS", backend=execution["backend"], params=q, arrays=len(raw),
                max_abs_error=errors, inner_products=products, local_cycles_by_launch=intervals,
                forward_repeat_bits_equal=True, clock_scope=CLOCK_SCOPE,
                host_timing=dict(clock_domain=execution["clock_domain"], launch_to_completion_read_ns=windows,
                                 constructor_to_stop_ns=timings["stop"]["end_ns"]-timings["construct"]["start_ns"]),
                scope=manifest["scope"], metadata=manifest["metadata"],
                panel_scope="all local partials and final source-row reductions; other reduction scratch checked for finiteness/padding",
                claim="independent linear AZ/ATD agreement; no classifier, training, global makespan or E2E speed claim")


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
    require(result["backend"] == build["backend"], "runtime backend binding")
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
    if args.capture.is_dir():
        result = audit_report(args.capture)
    else:
        with tempfile.TemporaryDirectory(prefix="bounded-adjoint-audit-") as temporary:
            folder = unpack_archive(args.capture, Path(temporary)/"capture", "capture", CAPTURE_LIMIT)
            result = audit_report(folder)
    write_json(args.out, result)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
