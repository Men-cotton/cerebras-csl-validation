"""Fixed Cora final aggregation, resident projection banks, and common physical reply."""

import hashlib
import json
from pathlib import Path

import numpy as np

COLS, LANES, ROWS, FEATURES = 4, 2, 256, 7
NODES, EDGES, HEAD = 2708, 13264, 20
SCALES = (1.0, -1.0, 1.0)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def word_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def load_graph(path):
    with np.load(path, allow_pickle=False) as data:
        graph = {k: data[k] for k in data.files}
    src, dst, weights, panel = (graph[k] for k in ("src", "dst", "weights", "features"))
    require(
        src.shape == dst.shape == weights.shape == (EDGES,) and panel.shape == (NODES, FEATURES),
        "fixed Cora",
    )
    require(np.all((src >= 0) & (src < NODES) & (dst >= 0) & (dst < NODES)), "valid original node IDs")
    ref = np.zeros_like(panel, dtype=np.float64)
    np.add.at(ref, dst, weights[:, None].astype(np.float64) * panel[src].astype(np.float64))
    np.testing.assert_array_equal(ref, graph["reference"])
    np.testing.assert_array_equal(ref.argmax(1), graph["original"].argmax(1))
    np.testing.assert_array_equal(graph["bounds"], np.minimum(np.arange(12) * ROWS, NODES))
    return graph


def sizes(shards):
    require(shards in (4, 16), "4 or 16 fixed source shards")
    return (NODES + shards - 1) // shards, {4: 957, 16: 404}[shards]


def checksum(words):
    # 32-bit rotate/xor over the exact little-endian words in device traversal order.
    h = 0x811C9DC5
    for value in np.ascontiguousarray(words, dtype=np.uint32).ravel():
        h = ((h << 5) & 0xFFFFFFFF | h >> 27) ^ int(value)
    return h


def banks(graph, shards):
    source_capacity, _ = sizes(shards)
    result = []
    for sign in (1.0, -1.0):
        rows = np.zeros((shards * source_capacity, COLS * LANES), np.float32)
        rows[:NODES, :FEATURES] = graph["features"] * np.float32(sign)
        result.append(
            rows.reshape(shards, source_capacity, LANES, COLS)
            .transpose(0, 3, 1, 2)
            .copy()
            .reshape(shards, COLS, -1)
        )
    return result


def window(graph, shards, index):
    start, end = graph["bounds"][index : index + 2]
    source_cap, edge_cap = sizes(shards)
    src, dst, weights = (graph[k] for k in ("src", "dst", "weights"))
    pairs = np.zeros((shards, edge_cap), np.uint32)
    values = np.zeros_like(pairs, dtype=np.float32)
    counts = np.empty(shards, np.uint32)
    hashes = np.empty(shards, np.uint32)
    encoded_counts = np.empty(shards, np.uint32)
    wire = [np.array([index], np.uint32)]
    for y in range(shards):
        which = np.flatnonzero((src // source_cap == y) & (dst >= start) & (dst < end))
        idx = which[np.argsort(dst[which], kind="stable")]
        local_destinations = dst[idx]
        last = np.ones(len(idx), dtype=bool)
        last[:-1] = local_destinations[:-1] != local_destinations[1:]
        eligible = np.flatnonzero(last & (src[idx] == local_destinations))
        singleton = np.zeros(len(idx), dtype=bool)
        if eligible.size:
            breaks = np.flatnonzero(np.diff(local_destinations[eligible]) != 1) + 1
            runs = np.split(eligible, breaks)
            singleton[max(runs, key=len)] = True
        residual = int(np.count_nonzero(~singleton))
        idx = np.concatenate((idx[~singleton],idx[singleton]))
        length = len(idx)
        require(length <= edge_cap, "per-shard edge capacity")
        counts[y] = length
        encoded_counts[y] = (residual << 16) | length
        soff = (src[idx] - y * source_cap).astype(np.uint32) * LANES
        doff = (dst[idx] - start).astype(np.uint32) * LANES
        pairs[y, :length] = ((soff*4) << 16) | (doff*4)
        values[y, :length] = weights[idx]
        packed = np.stack([pairs[y, :length], values[y, :length].view(np.uint32)], axis=1).ravel()
        hashes[y] = checksum(packed)
        wire.extend([encoded_counts[y : y + 1], packed])
    return dict(
        start=int(start),
        end=int(end),
        pairs=pairs,
        values=values,
        counts=counts,
        encoded_counts=encoded_counts,
        hashes=hashes,
        wire=np.concatenate(wire),
    )


def ticks(words):
    words = words.astype(np.int64)
    require(np.all(words <= 65535), "48-bit TSC word")
    a, b = np.split(words, 2, axis=-1)
    ticks = np.array([1, 65536, 1 << 32], np.int64)
    result = ((b @ ticks) - (a @ ticks)) % (1 << 48)
    require(np.all((result > 0) & (result < 100_000_000)), "valid PE interval")
    return result


def check_frame(graph, sl, index, trial, shards, reply, bank_hashes):
    column_words = shards * HEAD + ROWS * LANES
    words = np.asarray(reply, np.uint32).reshape(COLS, column_words)
    header = words[:, : shards * HEAD].reshape(COLS, shards, HEAD).transpose(1, 0, 2)
    xy = np.arange(shards * COLS).reshape(shards, COLS)
    for k, expected in enumerate(
        (
            index,
            trial,
            xy,
            sl["counts"][:, None],
            sl["hashes"][:, None],
            bank_hashes[0],
            bank_hashes[1],
            bank_hashes[trial == 2],
        )
    ):
        np.testing.assert_array_equal(header[:, :, k], np.broadcast_to(expected, xy.shape))
    setup, compute = ticks(header[:, :, 8:14]), ticks(header[:, :, 14:20])
    owner = (
        words[:, shards * HEAD :]
        .view(np.float32)
        .reshape(COLS, ROWS, LANES)
        .transpose(1, 2, 0)
        .reshape(ROWS, COLS * LANES)
    )
    rows = sl["end"] - sl["start"]
    require(
        np.all(owner[rows:] == 0.0) and np.all(owner[:, FEATURES:] == 0.0), "padded rows and features"
    )
    logits = owner[:rows, :FEATURES]
    expected = graph["reference"][sl["start"] : sl["end"]] * SCALES[trial - 1]
    np.testing.assert_allclose(logits, expected, rtol=1e-5, atol=2e-6)
    delta = logits.astype(np.float64) - expected
    return logits.copy(), setup, compute, float(np.abs(delta).max())


def audit(graph, record, raw):
    role, shards = record["language"], record["shards"]
    require(
        record["stage"] == "PASS" and record["stopped"] and len(record["windows"]) == 11,
        "completed eleven-window run",
    )
    a, b = banks(graph, shards)
    hashes = [[[] for _ in range(shards)] for _ in range(2)]
    for i, bank in enumerate((a, b)):
        hashes[i] = np.array(
            [[checksum(bank[y, x].view(np.uint32)) for x in range(COLS)] for y in range(shards)]
        )
    bank_wire = np.concatenate((a, b), axis=2)
    require(record["bank_sha256"] == word_sha(bank_wire), "resident feature payload")
    logits = np.empty((3, NODES, FEATURES), np.float32)
    compute_sum = np.zeros(3, np.int64)
    prep_sum = 0
    owner_sum = np.zeros(3, np.int64)
    phase_sum = np.zeros(3, np.int64)
    report = []
    for index, actual in enumerate(record["windows"]):
        sl = window(graph, shards, index)
        require(
            actual["index"] == index
            and actual["packed_graph_sha256"] == word_sha(sl["wire"])
            and len(actual["frames"]) == 3,
            "window identity",
        )
        prep = None
        times = []
        for t, frame in enumerate(actual["frames"], 1):
            words = raw[f"w{index}_t{t}"]
            require(
                frame["trial"] == t
                and word_sha(words) == frame["reply_sha256"]
                and frame["host_interval_ns"] > 0,
                "captured reply and host clock",
            )
            incoming = (
                frame_packet(bank_wire, sl, index, t) if role == "tungsten" else np.array([t], np.uint32)
            )
            require(frame["control_sha256"] == word_sha(incoming), "runtime control packet")
            out, setup, cycles, error = check_frame(graph, sl, index, t, shards, words, hashes)
            if prep is None:
                prep = setup
            np.testing.assert_array_equal(setup, prep)
            logits[t - 1, sl["start"] : sl["end"]] = out
            times.append(cycles)
        timing = np.stack(times)
        maxima = timing.max(axis=(1, 2))
        cum = (np.cumsum(timing, axis=0) + prep).max(axis=(1, 2))
        phases = np.cumsum(maxima) + prep.max()
        compute_sum += maxima
        prep_sum += int(prep.max())
        owner_sum += cum
        phase_sum += phases
        report.append(
            dict(
                start=sl["start"],
                end=sl["end"],
                edges=int(sl["counts"].sum()),
                setup_max=int(prep.max()),
                compute_max=maxima.tolist(),
                max_owner_cumulative=cum.tolist(),
                sum_phase_max=phases.tolist(),
            )
        )
    require(
        set(raw) == {f"w{i}_t{t}" for i in range(11) for t in range(1, 4)}, "exact capture inventory"
    )
    np.testing.assert_array_equal(logits[0].view(np.uint32), logits[2].view(np.uint32))
    errors = []
    classes = []
    for t, scale in enumerate(SCALES):
        ref = graph["reference"] * scale
        diff = logits[t].astype(np.float64) - ref
        errors.append(
            dict(
                max_abs=float(np.abs(diff).max()),
                relative_l2=float(np.linalg.norm(diff) / np.linalg.norm(ref)),
            )
        )
        classes.append(
            int(np.count_nonzero(logits[t].argmax(1) != (graph["original"] * scale).argmax(1)))
        )
    require(classes == [0, 0, 0], "all trained original classifications under panel sign")
    valid = graph["validation"]
    correct = int(np.count_nonzero(logits[0].argmax(1)[valid] == graph["labels"][valid]))
    require(correct == 402 and len(valid) == 500, "trained validation")
    return dict(
        windows=report,
        edges=sum(w["edges"] for w in report),
        nodes=NODES,
        shards=shards,
        pe=4 * shards,
        words=int(logits.size),
        owner_sha256=[word_sha(v) for v in logits],
        fp64_error=errors,
        original_class_mismatches=classes,
        validation=[correct, len(valid)],
        setup_sum=prep_sum,
        compute_sum=compute_sum.tolist(),
        max_owner_cumulative_sum=owner_sum.tolist(),
        phase_max_cumulative_sum=phase_sum.tolist(),
    )


def frame_packet(bank_wire, sl, index, trial):
    pieces = []
    if index == 0 and trial == 1:
        pieces.append(bank_wire.ravel().view(np.uint32))
    if trial == 1:
        pieces.append(sl["wire"])
    pieces.append(np.array([trial], np.uint32))
    return np.concatenate(pieces)


"""Check recovered outgoing records and their order independently of the selected host packing logic."""

import numpy as np


def expected_plan(graph, sl, index, shards, method):
    src,dst,w=(graph[field] for field in ("src","dst","weights"))
    lo,hi=map(int,graph["bounds"][index:index+2])
    capacity=(len(graph["features"])+shards-1)//shards
    result=[]
    for y in range(shards):
        indices=np.flatnonzero((dst>=lo)&(dst<hi)&(src//capacity==y))
        by_row={}
        for e in indices:
            by_row.setdefault(int(dst[e]),[]).append(int(e))
        eligible=[]
        for row,edges in sorted(by_row.items()):
            last=edges[-1]
            if int(src[last])==row and (method=="tail" or len(edges)==1):
                eligible.append(row)
        if method=="tail":
            runs=[]
            for row in eligible:
                if not runs or runs[-1][-1]+1!=row:
                    runs.append([])
                runs[-1].append(row)
            eligible=max(runs,key=len) if runs else []
        selected={by_row[row][-1] for row in eligible}
        ordered=[e for row in sorted(by_row) for e in by_row[row]]
        wire_order=[e for e in ordered if e not in selected]+[e for e in ordered if e in selected]
        n=len(ordered);k=n-len(selected)
        assert int(sl["encoded_counts"][y])==k<<16|n
        pair=sl["pairs"][y,:n]
        assert np.all(((pair>>16)%8==0)&((pair&65535)%8==0))
        np.testing.assert_array_equal((pair>>16)//8+y*capacity,src[wire_order])
        np.testing.assert_array_equal((pair&65535)//8+lo,dst[wire_order])
        np.testing.assert_array_equal(sl["values"][y,:n].view(np.uint32),w[wire_order].view(np.uint32))
        # A stable merge by destination restores all original per-row FMA inputs, even duplicate edges.
        merged=sorted(wire_order,key=lambda e:int(dst[e]))
        assert merged==ordered
        width=2*(eligible[-1]-eligible[0]+1) if eligible else 0
        assert method!="tail" or width==2*(n-k)
        result.append([k,n-k,width])
    return np.asarray(result,np.uint32)


def check_original_plan(graph, sl, index, shards, actual, method):
    wanted=expected_plan(graph,sl,index,shards,method)
    assert actual.shape==(shards,4,3)
    np.testing.assert_array_equal(actual,np.broadcast_to(wanted[:,None,:],actual.shape))
