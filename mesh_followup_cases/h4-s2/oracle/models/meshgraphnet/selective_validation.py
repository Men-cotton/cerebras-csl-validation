"""Independent graph-coverage and capture checks for selective sender delivery.

The graph determines required deliveries; producer schedules never define the
reference graph. A delivery may feed several incoming edges, including repeated
edges. Communication counters describe completed work, not hardware latency.
"""
from collections.abc import Mapping
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
if __package__:
    from . import colored_validation
else:
    import colored_validation


U32_MAX = (1 << 32) - 1
SCHEDULE_FIELDS = ("tx_begin", "tx_slots", "rx_count", "rx_slot")
FINAL_FIELDS = ("velocity", "h", "e", "aggregate", "cached_e0", "sender_cache")
COMM_FIELDS = ("tx_packets", "rx_packets", "tx_feature_words", "rx_feature_words",
               "tx_wire_words", "rx_wire_words", "local_edge_copies",
               "remote_edge_copies", "completed_rounds", "last_generation",
               "last_source", "error")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value, name):
    require(type(value) is int and 0 < value <= U32_MAX, "positive integer " + name)
    return value


def same_array(actual, expected, name):
    require(isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray) and
            actual.shape == expected.shape and actual.dtype == expected.dtype and
            actual.tobytes(order="C") == expected.tobytes(order="C"),
            "array differs: " + name)


def uint_array(arrays, name, shape):
    require(name in arrays and isinstance(arrays[name], np.ndarray) and
            arrays[name].dtype == np.dtype("uint32") and arrays[name].shape == shape,
            "schedule shape/dtype: " + name)
    return arrays[name]


def dimensions(p):
    for name in ("C", "R", "L", "E", "H", "S", "STEPS", "T", "K"):
        positive(p[name], name)
    require(type(p.get("MODE")) is int and p["MODE"] in (1,2), "selective MODE required")
    require(p["H"] <= 27 and p["K"] * p["H"] <= 27,
            "packet capacity: four application headers plus at most 27 feature words")
    return p["C"] * p["R"], p["L"], p["E"], p["H"], p["K"]


def validate_schedule(graph, maps, arrays, p):
    """Validate exact remote pair coverage against original edges and ownership.

    tx_begin indexes destination groups in a source owner's tx_slots. rx_count
    gives the corresponding receive lengths. Each remote edge's rx_slot indexes
    its sender within that pair's sorted unique slots. Local features require no
    network delivery and their rx_slot is zero.
    All schedule padding is zero; edge multiplicity and packed order survive.
    """
    pes, local, edges, hidden, batch = dimensions(p)
    require(isinstance(graph, Mapping) and isinstance(maps, Mapping), "graph/mapping objects")
    nodes = len(graph["velocity"])
    senders, receivers = graph["senders"], graph["receivers"]
    require(isinstance(senders, np.ndarray) and isinstance(receivers, np.ndarray) and
            senders.ndim == receivers.ndim == 1 and senders.shape == receivers.shape and
            senders.dtype.kind in "iu" and receivers.dtype.kind in "iu",
            "directed edge arrays")
    require(nodes > 0 and bool(((senders >= 0) & (senders < nodes)).all()) and
            bool(((receivers >= 0) & (receivers < nodes)).all()), "graph endpoint range")
    ni, ei = maps["node_ids"], maps["edge_ids"]
    for name, ids, shape, total in (("node_ids", ni, (pes, local), nodes),
                                    ("edge_ids", ei, (pes, edges), len(senders))):
        require(isinstance(ids, np.ndarray) and ids.shape == shape and ids.dtype.kind == "i",
                "mapping shape/dtype: " + name)
        require(bool(((ids >= -1) & (ids < total)).all()), "mapping index range: " + name)
        require(np.array_equal(np.sort(ids[ids >= 0]), np.arange(total)),
                "mapping coverage: " + name)
        for row in ids:
            count = int(np.count_nonzero(row >= 0))
            require(bool((row[:count] >= 0).all()) and bool((row[count:] == -1).all()),
                    "mapping packed prefix: " + name)
    owner = np.empty(nodes, dtype=np.int64)
    slot = np.empty(nodes, dtype=np.uint32)
    for pe, row in enumerate(ni):
        present = row >= 0
        owner[row[present]] = pe
        slot[row[present]] = np.arange(int(present.sum()), dtype=np.uint32)
    counts = np.column_stack(((ni >= 0).sum(axis=1), (ei >= 0).sum(axis=1))).astype(np.uint32)
    same_array(uint_array(arrays, "counts", (pes, 2)), counts, "counts")
    expected_edge = {name: np.zeros((pes, edges), np.uint32)
                     for name in ("src_owner", "src_slot", "dst_slot")}
    src_begin = np.zeros((pes, pes + 1), np.uint32)
    for pe, row in enumerate(ei):
        edge_ids = row[row >= 0]
        require(bool((owner[receivers[edge_ids]] == pe).all()), "edge receiver ownership")
        # The existing receiver-owned packing stably groups by source owner.
        original_ids = np.flatnonzero(owner[receivers] == pe)
        expected_ids = sorted(original_ids.tolist(), key=lambda edge: int(owner[senders[edge]]))
        require(np.array_equal(edge_ids, expected_ids), "stable packed edge order")
        source_owners = owner[senders[edge_ids]]
        n = len(edge_ids)
        expected_edge["src_owner"][pe, :n] = source_owners
        expected_edge["src_slot"][pe, :n] = slot[senders[edge_ids]]
        expected_edge["dst_slot"][pe, :n] = slot[receivers[edge_ids]]
        src_begin[pe, 1:] = np.cumsum(np.bincount(source_owners, minlength=pes), dtype=np.uint32)
    for name, expected in expected_edge.items():
        same_array(uint_array(arrays, name, expected.shape), expected, name)
    same_array(uint_array(arrays, "src_begin", src_begin.shape), src_begin, "src_begin")

    pairs = [[set() for _ in range(pes)] for _ in range(pes)]
    for source, target in zip(senders, receivers):
        source_owner, destination = int(owner[source]), int(owner[target])
        if source_owner != destination:
            pairs[source_owner][destination].add(int(slot[source]))
    tx_begin = np.zeros((pes, pes + 1), np.uint32)
    tx_slots = np.zeros((pes, p["T"]), np.uint32)
    rx_count = np.zeros((pes, pes), np.uint32)
    rx_slot = np.zeros((pes, edges), np.uint32)
    for source in range(pes):
        packed = []
        for destination in range(pes):
            slots = sorted(pairs[source][destination])
            packed.extend(slots)
            tx_begin[source, destination + 1] = len(packed)
            rx_count[destination, source] = len(slots)
        require(len(packed) <= p["T"], "transmit schedule exceeds T")
        tx_slots[source, :len(packed)] = packed
    for destination, row in enumerate(ei):
        for local_edge, edge in enumerate(row[row >= 0]):
            source = int(owner[senders[edge]])
            if source != destination:
                group = sorted(pairs[source][destination])
                rx_slot[destination, local_edge] = group.index(int(slot[senders[edge]]))
    for name, expected in (("tx_begin", tx_begin), ("tx_slots", tx_slots),
                           ("rx_count", rx_count), ("rx_slot", rx_slot)):
        same_array(uint_array(arrays, name, expected.shape), expected, name)
    packets = sum((len(group) + batch - 1) // batch for row in pairs for group in row)
    deliveries = sum(len(group) for row in pairs for group in row)
    return dict(distinct_remote_deliveries=deliveries, packets_per_block=packets,
                feature_words_per_block=deliveries*hidden,
                wire_words_per_block=deliveries*hidden + 5*packets,
                scope="endpoint word counts include one SDK and four application header words per packet; no link-hop or latency claim")


def validate_comm(p, arrays, raw, steps, cumulative_steps):
    """Check per-launch counters after validate_schedule established graph coverage."""
    pes, _, edges, hidden, batch = dimensions(p)
    positive(steps, "steps")
    positive(cumulative_steps, "cumulative_steps")
    require(steps <= cumulative_steps <= p["STEPS"], "communication step range")
    begin = uint_array(arrays, "tx_begin", (pes, pes + 1))
    receive = uint_array(arrays, "rx_count", (pes, pes))
    counts = uint_array(arrays, "counts", (pes, 2))
    owners = uint_array(arrays, "src_owner", (pes, edges))
    expected = np.zeros((pes, len(COMM_FIELDS)), np.uint32)
    if p["MODE"] == 2:
        colored_validation.validate_rounds(arrays, p)
        rounds = p["Q"]
        last_sources = arrays["round_sources"][:, -1]
    else:
        rounds = pes
        last_sources = np.full(pes, pes-1, np.uint32)
    repeats = p["S"] * steps
    for pe in range(pes):
        tx = np.diff(begin[pe].astype(np.int64))
        rx = receive[pe].astype(np.int64)
        require(bool((tx >= 0).all()) and begin[pe, 0] == 0 and begin[pe, -1] <= p["T"],
                "transmit prefix bounds")
        require(tx[pe] == 0 and rx[pe] == 0 and counts[pe, 1] <= edges, "local delivery/count bounds")
        tx_packets = sum((int(n) + batch - 1) // batch for n in tx)
        rx_packets = sum((int(n) + batch - 1) // batch for n in rx)
        tx_features, rx_features = int(tx.sum()) * hidden, int(rx.sum()) * hidden
        local_edges = int(np.count_nonzero(owners[pe, :int(counts[pe, 1])] == pe))
        remote_edges = int(counts[pe, 1]) - local_edges
        row = [tx_packets*repeats, rx_packets*repeats, tx_features*repeats, rx_features*repeats,
               (tx_features + 5*tx_packets)*repeats, (rx_features + 5*rx_packets)*repeats,
               local_edges*repeats, remote_edges*repeats, rounds*repeats,
               cumulative_steps*p["S"], int(last_sources[pe]), 0]
        require(all(0 <= value <= U32_MAX for value in row), "communication counter overflow")
        expected[pe] = row
    same_array(uint_array(raw, "comm_stats", expected.shape), expected, "comm_stats")
    return dict(totals=dict(zip(COMM_FIELDS[:8], expected[:, :8].astype(np.uint64).sum(axis=0).tolist())),
                completed_rounds_per_pe=rounds*repeats,
                last_generation=cumulative_steps*p["S"], last_source=(last_sources.tolist() if p["MODE"] == 2 else pes-1))


def _read_bytes(case, name, limit):
    relative = Path(name)
    require(not relative.is_absolute() and bool(relative.parts) and
            all(part not in (".", "..") for part in relative.parts), "relative capture path")
    path = case / name
    require(path.resolve(strict=True) == path and path.is_file(), "regular nonsymlink capture file")
    require(path.stat().st_size <= limit, "capture file byte bound")
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    require(len(payload) <= limit, "capture file grew beyond bound")
    return payload


def _read_json(case, name):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, "duplicate JSON key")
            obj[key] = value
        return obj
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    payload = _read_bytes(case, name, 2 << 20)
    value = json.loads(payload, object_pairs_hook=unique, parse_constant=constant)
    require(isinstance(value, dict), "capture JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _read_npz(case, name):
    payload = _read_bytes(case, name, 64 << 20)
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        require(0 < len(entries) <= 512 and len({item.filename for item in entries}) == len(entries),
                "capture NPZ inventory")
        require(sum(item.file_size for item in entries) <= 64 << 20, "expanded capture NPZ byte bound")
        for item in entries:
            require(not item.is_dir() and item.filename.endswith(".npy") and
                    Path(item.filename).name == item.filename and not (item.flag_bits & 1),
                    "capture NPZ member")
            content = archive.read(item)
            stream = io.BytesIO(content)
            version = np.lib.format.read_magic(stream)
            require(version in ((1, 0), (2, 0)), "capture NPY format version")
            read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                           else np.lib.format.read_array_header_2_0)
            shape, fortran, dtype = read_header(stream, max_header_size=16384)
            require(not dtype.hasobject and dtype.kind in "fiu" and dtype.itemsize in (1, 2, 4, 8),
                    "capture numeric dtype")
            require(len(shape) <= 4 and all(type(size) is int and 0 <= size <= 64 << 20 for size in shape),
                    "capture shape bound")
            count = math.prod(shape)
            require(count*dtype.itemsize == len(content)-stream.tell(), "capture NPY byte length")
            value = np.frombuffer(content, dtype=dtype, count=count, offset=stream.tell()).reshape(
                shape, order="F" if fortran else "C")
            require(bool(np.isfinite(value).all()), "capture finite arrays")
            arrays[item.filename[:-4]] = value
    return arrays, hashlib.sha256(payload).hexdigest()


def _bound_case(path):
    case = Path(path).resolve(strict=True)
    m, case_hash = _read_json(case, "case.json")
    build, _ = _read_json(case, "build.json")
    execution, _ = _read_json(case, "execution.json")
    result, _ = _read_json(case, "result.json")
    require(build.get("status") == "PASS" and execution.get("status") == "COMPLETED" and
            result.get("status") == "PASS", "individually audited completed capture required")
    require(all(record.get("case_sha256") == case_hash for record in (build, execution, result)),
            "case identity binding")
    require(isinstance(m.get("files"), dict) and 0 < len(m["files"]) <= 32,
            "bounded prepared file inventory")
    for name, digest in m["files"].items():
        content = _read_bytes(case, name, (64 << 20) if name.endswith(".npz") else (2 << 20))
        require(hashlib.sha256(content).hexdigest() == digest, "prepared bytes differ: " + name)
    saved = {}
    for name in ("original", "mapping", "inputs", "raw"):
        saved[name], digest = _read_npz(case, name + ".npz")
        expected = execution["raw_sha256"] if name == "raw" else m["files"][name + ".npz"]
        require(digest == expected, "prepared/captured bytes differ: " + name)
        if name == "raw":
            require(digest == result["raw_sha256"], "result raw identity binding")
    require(result["params"] == m["params"] and result["residency"] == m["residency"],
            "result model/session binding")
    return m, saved, dict(case_sha256=case_hash, raw_sha256=result["raw_sha256"])


def compare_cases(first, second):
    """Compare two individually audited captures, permitting only transport changes.

    This binds saved bytes and compares all numerical checkpoints. It does not
    repeat SDK execution, numerical oracle evaluation, or compiled ELF checks.
    The prior individual audit is a prerequisite, not inferred from equality.
    """
    left, a, left_identity = _bound_case(first)
    right, b, right_identity = _bound_case(second)
    require(left["params"]["MODE"] != right["params"]["MODE"] and
            {left["params"]["MODE"], right["params"]["MODE"]} <= {0,1,2},
            "two different transport modes required")
    for field in ("sources", "residency", "partition", "model", "edge_encoder_probe",
                  "graph_scope", "graph_limit", "original_graph_counts", "data_provenance"):
        require(left[field] == right[field], "matched case metadata: " + field)
    omit = {"MODE", "T", "K", "Q"}
    require({k: v for k, v in left["params"].items() if k not in omit} ==
            {k: v for k, v in right["params"].items() if k not in omit}, "matched model parameters")
    for archive in ("original", "mapping"):
        require(set(a[archive]) == set(b[archive]), "matched archive inventory: " + archive)
        for name in a[archive]:
            same_array(a[archive][name], b[archive][name], archive + ":" + name)
    common_a = set(a["inputs"]) - (set(SCHEDULE_FIELDS)|{"round_sources"})
    common_b = set(b["inputs"]) - (set(SCHEDULE_FIELDS)|{"round_sources"})
    require(common_a == common_b, "matched arithmetic input inventory")
    for name in common_a:
        same_array(a["inputs"][name], b["inputs"][name], "inputs:" + name)
    for m, arrays in ((left, a), (right, b)):
        actual_schedule = set(arrays["inputs"]) & (set(SCHEDULE_FIELDS)|{"round_sources"})
        expected_schedule = set(SCHEDULE_FIELDS) if m["params"]["MODE"] else set()
        if m["params"]["MODE"] == 2:
            expected_schedule.add("round_sources")
        require(actual_schedule == expected_schedule,
                "transport schedule inventory")
        if m["params"]["MODE"]:
            validate_schedule(arrays["original"], arrays["mapping"], arrays["inputs"], m["params"])
            if m["params"]["MODE"] == 2:
                colored_validation.validate_rounds(arrays["inputs"],m["params"])
    chunks = left["residency"]["chunks"]
    for index in range(len(chunks)):
        for name in FINAL_FIELDS:
            field = f"chunk{index}_" + name
            same_array(a["raw"][field], b["raw"][field], field)
    return dict(status="PASS", first=left_identity, second=right_identity,
                checkpoints=len(chunks), bit_exact_fields=list(FINAL_FIELDS),
                scope="matched captured arithmetic; prior individual oracle/build audits required; no new SDK or hardware execution")
