"""Frozen Cora inputs and resident Shah router/worker/accumulator host contract.

The FP64 reference reads original COO occurrences, not packed SELL words. The
FP32 model separately executes packed sparse X and source-shard graph streams.
Arithmetic association differs from the grouped-CSR baseline; cross-family
acceptance uses the frozen tolerance rather than requiring equal FP32 bits.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
END_ROW = 0xfffffffe
EMPTY = 0xffffffff
ATOL, RTOL = 2e-5, 2e-4
PINS = {
    "models/gcn/data/cora/cora.content": "b8ebed0fd45e53995abafd1b70cf6a9fcc8f86abc5826d69b97fe6ca46e9c8ee",
    "models/gcn/data/cora/cora.cites": "ec1a372391b7f0f60a6aff0084e8abd8f19f0faa7e1f2441a41c492042d5945e",
    "research/experiments/data/E087/captures/checkpoint-original.npz": "836c82b82871c76bb1822ace2014c9552690459a952de1b23d4399d5ecce4542",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def array_identity(value):
    value = np.ascontiguousarray(value)
    return dict(dtype=value.dtype.str, shape=list(value.shape),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def _bounded(path, limit=32 << 20):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
            "bounded regular input: " + str(path))
    return path.read_bytes()


def load_original(kind="tiny", nodes=2708, root=ROOT):
    """Read the E217 raw-corpus/checkpoint contract, or deterministic adversary.

    The corpus reconstruction follows native_cora_appliance/saved_audit.py:
    original corpus order, row-sum X normalization, sorted undirected pairs,
    forward/reverse/self occurrence order, and FP32 symmetric normalization.
    This function imports no runtime SDK or model implementation.
    """
    if kind == "tiny":
        return tiny_original()
    require(kind == "cora" and type(nodes) is int and 1 <= nodes <= 2708,
            "kind and Cora node bound")
    root = Path(root)
    payloads = {name: _bounded(root / name) for name in PINS}
    require(all(hashlib.sha256(payloads[k]).hexdigest() == v for k, v in PINS.items()),
            "frozen Cora corpus/checkpoint identity")
    content = "models/gcn/data/cora/cora.content"
    cites = "models/gcn/data/cora/cora.cites"
    rows = [line.split() for line in payloads[content].decode().splitlines()]
    ids = {row[0]: i for i, row in enumerate(rows)}
    classes = sorted({row[-1] for row in rows})
    labels = np.array([classes.index(row[-1]) for row in rows], np.int64)
    x = np.array([row[1:-1] for row in rows], np.float32)
    require(len(ids) == 2708 and x.shape == (2708, 1433) and len(classes) == 7,
            "complete native Cora dimensions")
    x /= np.maximum(x.sum(axis=1, keepdims=True), 1)
    pairs = set()
    for line in payloads[cites].decode().splitlines():
        a, b = (ids[key] for key in line.split())
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    edges = np.array(sorted(pairs), np.int64)
    require(edges.shape == (5278, 2), "unpruned Cora graph")
    rng = np.random.default_rng(20260916)
    train = np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:20] for c in range(7)])
    valid = rng.permutation(np.setdiff1d(np.arange(2708), train))[:500]
    checkpoint = root / "research/experiments/data/E087/captures/checkpoint-original"
    record_bytes = _bounded(checkpoint.with_suffix(".json"))
    record = json.loads(record_bytes)
    require(record.get("profile") == "original" and record.get("seed") == 11
            and record.get("epochs") == 200
            and record.get("weights_sha256") == PINS[str(checkpoint.with_suffix(".npz").relative_to(root))]
            and record.get("retained_sha256") == array_identity(edges)["sha256"],
            "trained checkpoint identity")
    expected = dict(content_sha256=PINS[content], cites_sha256=PINS[cites], classes=classes,
                    train_indices=train.tolist(), validation_indices=valid.tolist(), split_seed=20260916,
                    test_evaluated=False, features=1433, nodes=2708, feature_normalization="row sum",
                    retained_edges=5278, removed_edges=0)
    require(all(record["dataset"].get(k) == v for k, v in expected.items()),
            "checkpoint dataset/split contract")
    with np.load(checkpoint.with_suffix(".npz"), allow_pickle=False) as saved:
        require(set(saved.files) == {"w0", "w1"}, "checkpoint tensor inventory")
        weights = {name: saved[name].copy() for name in saved.files}
    require(weights["w0"].shape == (1433, 16) and weights["w1"].shape == (16, 7)
            and all(v.dtype == np.float32 and np.isfinite(v).all() for v in weights.values()),
            "FP32 weight contract")
    edges = edges[np.max(edges, axis=1) < nodes]
    src = np.concatenate((edges[:, 0], edges[:, 1], np.arange(nodes)))
    dst = np.concatenate((edges[:, 1], edges[:, 0], np.arange(nodes)))
    degree = np.ones(nodes, np.int64)
    np.add.at(degree, edges.ravel(), 1)
    coeff = (1 / np.sqrt(degree[src] * degree[dst])).astype(np.float32)
    return dict(x=x[:nodes].copy(), src=src, dst=dst, coeff=coeff, **weights,
                labels=labels[:nodes].copy(), valid=valid.copy() if nodes == 2708 else None,
                full=nodes == 2708, kind="cora", provenance=dict(source_sha256=PINS,
                checkpoint_json_sha256=hashlib.sha256(record_bytes).hexdigest(), seed=11, epochs=200,
                graph="original undirected pairs plus self loops; induced subsets recompute degrees",
                feature_normalization="row sum", validation_scope="custom500, not Planetoid test"))


def tiny_original(empty=False):
    """Seven nodes, native-feature tails, signed duplicates and empty source rows."""
    rng = np.random.default_rng(712)
    x = rng.normal(size=(7, 23)).astype(np.float32)
    x[rng.random(x.shape) < .6] = 0
    x[:, 17], x[:, 22] = np.float32(.75), np.float32(-.5)
    src = np.array([0, 0, 0, 1, 1, 6, 6, 6, 6], np.int64)
    dst = np.array([0, 0, 6, 1, 3, 0, 1, 6, 6], np.int64)
    coeff = np.array([.5, -.25, 1, .7, 0, -.3, .4, 1, .25], np.float32)
    if empty:
        src, dst, coeff = src[:0], dst[:0], coeff[:0]
    return dict(x=x, src=src, dst=dst, coeff=coeff,
                w0=(rng.normal(size=(23, 16)) * .1).astype(np.float32),
                w1=(rng.normal(size=(16, 7)) * .1).astype(np.float32),
                labels=None, valid=None, full=False, kind="tiny", provenance=dict(seed=712))


def validate_original(original):
    result = {}
    for name in ("x", "w0", "w1", "coeff"):
        a = np.asarray(original[name])
        require(a.dtype == np.float32 and np.isfinite(a).all(), "finite FP32 " + name)
        result[name] = a
    x, w0, w1, coeff = (result[k] for k in ("x", "w0", "w1", "coeff"))
    require(x.ndim == w0.ndim == w1.ndim == 2 and all(x.shape) and all(w0.shape)
            and all(w1.shape) and w0.shape[0] == x.shape[1] and w1.shape[0] == w0.shape[1]
            and 1 <= w1.shape[1] <= w0.shape[1] <= 256 and coeff.ndim == 1,
            "GCN dimensions and O <= F")
    for name in ("src", "dst"):
        a = np.asarray(original[name])
        require(a.ndim == 1 and a.dtype.kind in "iu" and len(a) == len(coeff)
                and (not len(a) or (a.min() >= 0 and a.max() < len(x))), "original edge " + name)
        result[name] = a.astype(np.int64, copy=False)
    return result


def ah_reference(original, h):
    """FP64 scatter of original occurrences; never inspect packed graph words."""
    o = validate_original(original)
    h = np.asarray(h)
    require(h.ndim == 2 and h.shape[0] == len(o["x"]) and h.dtype in (np.float32, np.float64)
            and np.isfinite(h).all(), "AH input matrix")
    result = np.zeros(h.shape, np.float64)
    np.add.at(result, o["dst"], o["coeff"].astype(np.float64)[:, None] * h[o["src"]].astype(np.float64))
    return result


def reference(original):
    o = validate_original(original)
    xw = o["x"].astype(np.float64) @ o["w0"].astype(np.float64)
    pre = ah_reference(o, xw)
    hidden = np.maximum(pre, 0)
    hw = hidden @ o["w1"].astype(np.float64)
    return dict(xw=xw, projected0=xw, pre=pre, hidden=hidden, hw=hw, projected1=hw, logits=ah_reference(o, hw))


def _router_words(original, R, V):
    grouped = [[[] for _ in range(R * V)] for _ in range(R)]
    for src, dst, val in zip(original["src"], original["dst"], original["coeff"]):
        grouped[int(src) // V][int(dst)].append((int(src), int(val.view(np.uint32))))
    packed = []
    for shard in grouped:
        words = []
        for chunk in range(R):
            records = []
            for destination in range(chunk * V, (chunk + 1) * V):
                records.extend(shard[destination])
                if records and records[-1][0] == END_ROW:
                    records[-1] = (END_ROW, records[-1][1] + 1)
                else:
                    records.append((END_ROW, 1))
            words.extend(records)
        packed.append(np.array(words, np.uint32).reshape(-1))
    return packed


def graph_occurrences(case):
    """Decode resident router records and validate framing/ownership/padding."""
    p, arrays = case["params"], case["arrays"]
    R, V = p["R"], p["V"]
    records = []
    for shard in range(R):
        length = int(arrays["graph_len"][shard, 0, 0])
        words = arrays["graph_words"][shard, 0]
        require(length % 2 == 0 and 0 < length <= p["GP"], "paired graph word count")
        require(np.all(words[length:] == 0), "graph array tail is zero")
        row = 0
        for index, bits in words[:length].reshape(-1, 2):
            index, bits = int(index), int(bits)
            if index == END_ROW:
                require(1 <= bits <= V - row % V and row < R * V,
                        "END_ROW must end within its declared chunk")
                row += bits
            elif index == EMPTY:
                raise ValueError("resident graph has no EMPTY transfer padding")
            else:
                require(row < case["nodes"] and shard * V <= index < min((shard + 1) * V, case["nodes"]),
                        "source shard and destination bounds")
                val = np.uint32(bits).view(np.float32)
                require(np.isfinite(val), "finite graph coefficient")
                records.append((shard, row, index - shard * V, val))
        require(row == R * V, "all destination rows terminated")
    return records


def build_case(original, R=2):
    o = validate_original(original)
    require(type(R) is int and 1 <= R <= 256, "source-row count")
    N, FIN = o["x"].shape
    F, O = o["w1"].shape
    V = (N + R - 1) // R
    require(V > 0 and R * V < 65535 and FIN < 65535, "u16 geometry")
    words = _router_words(o, R, V)
    GP = max(map(len, words))
    nnz = [int(np.count_nonzero(o["x"][r * V:min((r + 1) * V, N)])) for r in range(R)]
    MX = max(1, max(nnz))
    require(GP < 32768 and MX < 65535, "router descriptor and sparse-X capacity")
    p = dict(R=R, V=V, FIN=FIN, F=F, O=O, MX=MX, GP=GP)
    dims = dict(graph_words=(GP, np.uint32), graph_len=(1, np.uint32),
                x_ptr=(V + 1, np.uint16), x_col=(MX, np.uint16), x_val=(MX, np.float32),
                w0=(FIN, np.float32), h=(V, np.float32), w1=(F, np.float32))
    arrays = {name: np.zeros((2 * R, F + 1, count), dtype) for name, (count, dtype) in dims.items()}
    for r in range(R):
        arrays["graph_words"][r, 0, :len(words[r])] = words[r]
        arrays["graph_len"][r, 0, 0] = len(words[r])
        ptr, cols, vals = [0], [], []
        for node in range(r * V, (r + 1) * V):
            features = np.flatnonzero(o["x"][node]) if node < N else []
            cols.extend(features)
            vals.extend(o["x"][node, features] if node < N else [])
            ptr.append(len(cols))
        for f in range(F):
            arrays["x_ptr"][r, f + 1] = ptr
            arrays["x_col"][r, f + 1, :len(cols)] = cols
            arrays["x_val"][r, f + 1, :len(vals)] = vals
            arrays["w0"][r, f + 1] = o["w0"][:, f]
            if f < O:
                arrays["w1"][R + r, f + 1] = o["w1"][:, f]
    case = dict(params=p, nodes=N, arrays=arrays, original=dict(original), reference=reference(o))
    audit_packing(case)
    per_shard = np.bincount(o["src"] // V, minlength=R)
    # Named numerical/index arrays only. The device implementation adds scratch,
    # counters, code, SDK and runtime stack; actual ELF admission is separate.
    case["metadata"] = dict(input_identity={k: array_identity(v) for k, v in o.items()},
        source_provenance=original.get("provenance", {}), nodes=N, features=FIN,
        edge_occurrences=len(o["src"]), source_node_counts=[max(0, min(V, N - r * V)) for r in range(R)],
        sparse_x_nnz_per_shard=nnz, graph_words_per_router=list(map(len, words)),
        graph_nnz_per_source_shard=per_shard.tolist(), source_work_max_over_mean=float(per_shard.max() / per_shard.mean()) if per_shard.sum() else 0,
        logical_projection_fmas=int(np.count_nonzero(o["x"])) * F + N * F * O,
        logical_graph_fmas=len(o["src"]) * (F + O),
        executed_graph_fmas=2 * len(o["src"]) * F,
        executed_projection_fmas=int(np.count_nonzero(o["x"])) * F + R * V * F * O,
        executed_work_scope="two graph phases execute all F lanes, including zero lanes O..F-1; second projection includes padded nodes",
        application_rectangle_pes=2 * R * (F + 1),
        worker_pes=R * F, accumulator_pes=R * F,
        named_payload_bytes_per_role=dict(router=4 * GP + 4, worker=2 * (V + 1 + MX) + 4 * (MX + FIN + V),
                                         accumulator=4 * (F + 3 * V)),
        storage_scope="named immutable inputs and visible h/pre/projected/logits only; excludes protocol scratch, code, SDK and stack",
        sparse_semantics="original edge occurrences preserved including duplicates, zero and signed coefficients; source-shard reduction order",
        graph_order="each router: destination chunks then rows, original occurrence order within each row; compressed END_ROW runs",
        upload_roles=dict(router=["graph_words", "graph_len"], worker=["x_ptr", "x_col", "x_val", "w0", "h"], accumulator=["w1"]))
    return case


def audit_packing(case):
    p, a, o = case["params"], case["arrays"], validate_original(case["original"])
    R, V, F, O = (p[k] for k in ("R", "V", "F", "O"))
    expected = Counter((int(s), int(d), int(v.view(np.uint32))) for s, d, v in zip(o["src"], o["dst"], o["coeff"]))
    actual = Counter((r * V + local, dst, int(v.view(np.uint32))) for r, dst, local, v in graph_occurrences(case))
    require(actual == expected, "original graph occurrence identity")
    wanted_order = sorted(range(len(o["src"])), key=lambda i: (int(o["src"][i]) // V, int(o["dst"][i])))
    wanted_records = [(int(o["src"][i]), int(o["dst"][i]), int(o["coeff"][i].view(np.uint32))) for i in wanted_order]
    actual_records = [(r * V + local, dst, int(v.view(np.uint32))) for r, dst, local, v in graph_occurrences(case)]
    require(actual_records == wanted_records, "original occurrence order within destination/source shard")
    for r in range(R):
        for f in range(F):
            ptr, cols, vals = (a[k][r, f + 1] for k in ("x_ptr", "x_col", "x_val"))
            require(ptr[0] == 0 and np.all(ptr[1:] >= ptr[:-1]) and ptr[-1] <= p["MX"], "sparse X pointer bounds")
            require(np.all(cols[ptr[-1]:] == 0) and np.all(vals[ptr[-1]:] == 0), "sparse X padding")
            for local in range(V):
                node = r * V + local
                begin, end = map(int, ptr[local:local + 2])
                features = np.flatnonzero(o["x"][node]) if node < case["nodes"] else np.array([], np.int64)
                require(np.array_equal(cols[begin:end], features), "sparse X feature identity")
                require(np.array_equal(vals[begin:end].view(np.uint32), o["x"][node, features].view(np.uint32))
                        if node < case["nodes"] else begin == end, "sparse X value identity")
            require(np.array_equal(a["w0"][r, f + 1].view(np.uint32), o["w0"][:, f].view(np.uint32)), "W0 column identity")
            expected_w1 = o["w1"][:, f] if f < O else np.zeros(F, np.float32)
            require(np.array_equal(a["w1"][R + r, f + 1].view(np.uint32), expected_w1.view(np.uint32)), "W1 column identity")
    return True


def pack_h(case, h):
    p = case["params"]
    h = np.asarray(h)
    require(h.dtype == np.float32 and h.ndim == 2 and h.shape[0] == case["nodes"]
            and 1 <= h.shape[1] <= p["F"] and np.isfinite(h).all(), "finite FP32 AH input")
    padded = np.zeros((p["R"] * p["V"], p["F"]), np.float32)
    padded[:case["nodes"], :h.shape[1]] = h
    return padded.reshape(p["R"], p["V"], p["F"]).transpose(0, 2, 1).copy()


def unpack_output(case, values, width=None):
    p = case["params"]
    values = np.asarray(values)
    require(values.dtype == np.float32 and values.size == p["R"] * p["F"] * p["V"], "accumulator output shape")
    matrix = values.reshape(p["R"], p["F"], p["V"]).transpose(0, 2, 1).reshape(p["R"] * p["V"], p["F"])
    width = p["F"] if width is None else width
    require(type(width) is int and 1 <= width <= p["F"] and np.isfinite(matrix).all(), "finite output and width")
    require(np.all(matrix[case["nodes"]:] == 0) and np.all(matrix[:, width:] == 0),
            "structural output padding must remain zero")
    return matrix[:case["nodes"], :width].copy()


def emulate_ah(case, h):
    """FP32 packed-router arithmetic, followed by north-to-south shard sum."""
    p = case["params"]
    tiled = pack_h(case, h)
    partials = np.zeros((p["R"], p["R"] * p["V"], p["F"]), np.float32)
    for shard, dst, local, value in graph_occurrences(case):
        partials[shard, dst] += value * tiled[shard, :, local]
    result = np.zeros(partials.shape[1:], np.float32)
    for shard in range(p["R"]):
        result += partials[shard]
    return result[:case["nodes"], :h.shape[1]].copy()


def emulate(case):
    """Execute packed projection and graph tables; no reference tensor is read."""
    p, a = case["params"], case["arrays"]
    R, V, F, O = (p[k] for k in ("R", "V", "F", "O"))
    xw = np.zeros((R * V, F), np.float32)
    for r in range(R):
        for f in range(F):
            ptr, cols, vals = (a[k][r, f + 1] for k in ("x_ptr", "x_col", "x_val"))
            for local in range(V):
                for index in range(int(ptr[local]), int(ptr[local + 1])):
                    xw[r * V + local, f] += vals[index] * a["w0"][r, f + 1, cols[index]]
    xw = xw[:case["nodes"]]
    pre = emulate_ah(case, xw)
    hidden = np.maximum(pre, np.float32(0))
    hw = np.zeros((case["nodes"], O), np.float32)
    for r in range(R):
        begin, end = r * V, min((r + 1) * V, case["nodes"])
        for f in range(O):
            for source_feature in range(F):
                hw[begin:end, f] += hidden[begin:end, source_feature] * a["w1"][R + r, f + 1, source_feature]
    return dict(xw=xw, pre=pre, hidden=hidden, hw=hw, logits=emulate_ah(case, hw))


def ah_cases(case):
    """Fixed FP32 projection inputs for both standalone AH comparisons.

    AH7 uses the staged packed FP32 first-layer output. Both implementations
    must consume these exact matrix bits; whole-GCN validation separately uses
    the original complete-model FP64 reference.
    """
    staged = emulate(case)
    return {name: dict(h=staged[key].copy(), reference=ah_reference(case["original"], staged[key]),
                      input_identity=array_identity(staged[key]), width=staged[key].shape[1])
            for name, key in (("ah16", "xw"), ("ah7", "hw"))}


def streaming_fixture(case, h, Q=64):
    """Expose identical AH inputs in E231's existing power-of-two chunk ABI."""
    require(type(Q) is int and Q > 0 and Q & (Q - 1) == 0, "streaming Q power of two")
    pack_h(case, h)
    p = case["params"]
    require((p["R"] - 1) * p["V"] < case["nodes"], "E231 requires a nonempty final source shard")
    params = dict(R=p["R"], F=h.shape[1], V=p["V"], Q=Q,
                  B=(case["nodes"] + Q - 1) // Q, K=case["nodes"])
    params["N"] = params["B"] * Q
    o = case["original"]
    return dict(params=params, h=h.copy(), edges=[(int(d), int(s), np.float32(v))
                for s, d, v in zip(o["src"], o["dst"], o["coeff"])],
                reference=ah_reference(o, h), input_identity=array_identity(h))


def independent_reference(original):
    """Complete-model FP64 alias for the portable driver/auditor."""
    return reference(original)


def project_inputs(original, R=2):
    """Return fixed FP32 AH16/AH7 inputs and their separate FP64 AH references."""
    return ah_cases(build_case(original, R))


def expected_stats(case, mode="full", epoch=1):
    """Exact role counters; epoch is the launch count within one loaded runtime."""
    require(mode in ("full", "ah") and type(epoch) is int and 1 <= epoch < 2**32, "mode and epoch")
    p, a, m = case["params"], case["arrays"], case["metadata"]
    R, V, F, O = (p[k] for k in ("R", "V", "F", "O"))
    phases = 2 if mode == "full" else 1
    stats = np.zeros((2 * R, F + 1, 8), np.uint32)
    stats[:, :, 0] = 0x600d
    stats[:, :, 7] = epoch
    for r in range(R):
        stats[r, 0, 1] = phases * int(a["graph_len"][r, 0, 0]) // 2
        stats[r, 1:, 1] = phases * m["graph_nnz_per_source_shard"][r]
        stats[r, 1:, 2] = phases * R * V
        stats[r, 1:, 3] = phases * R
        stats[R + r, 1:, 4] = phases * (R - r) * V
        stats[R + r, 1:, 5] = phases * (R - r - 1) * V
        if mode == "full":
            stats[r, 1:, 6] = m["sparse_x_nnz_per_shard"][r]
            stats[R + r, 1:O + 1, 6] = F * V
    return stats
