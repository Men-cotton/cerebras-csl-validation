"""Split the original owner assignment over two physical rows without changing the ordered reductions."""

from pathlib import Path
import json
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.extend([str(HERE.parent.parent), str(HERE.parent)])
from chain import TRIALS, audit, device, prepare as prepare, source_tiles, upload

# Frame zero loads both copies of the first-layer weights. Each warmed path has a serial and concurrent send.
SCHEDULE = (
    (0, True),
    (1, False),
    (1, True),
    (2, True),
    (2, False),
    (3, False),
    (3, True),
    (4, True),
    (4, False),
)
CHECK = 0xC510A11


def parts(data):
    a, _, cfg, groups, owner, slot, _, _, _ = data
    if cfg["P"] % 2 or cfg["P"] < 2:
        raise ValueError("two input rows require an even owner count")
    width = cfg["P"] // 2
    rows = []
    for lane in range(2):
        first_pe = lane * width
        selected = np.flatnonzero((owner >= first_pe) & (owner < first_pe + width))
        local = [np.searchsorted(selected, g) for g in groups[first_pe : first_pe + width]]
        for previous, ids in zip(groups[first_pe : first_pe + width], local):
            np.testing.assert_array_equal(selected[ids], previous)
        shard = dict(a)
        for key in ("seed", "root", "labels", "fp64", "gpu"):
            shard[key] = a[key][selected]
        shard["src"] = np.concatenate([a["src"][a["ptr"][n] : a["ptr"][n + 1]] for n in selected])
        shard["ptr"] = np.r_[0, np.diff(a["ptr"])[selected].cumsum()]
        shard["shape"] = np.array([len(selected), len(a["node"]), len(shard["src"]), *a["shape"][-2:]])
        params = dict(cfg, P=width, FRAMES=len(SCHEDULE))
        tiles = source_tiles(shard, params, owner[selected] - first_pe, slot[selected], "dictionary")
        rows.append((shard, params, local, tiles))
    return rows


def trial_inputs(build, data, order=None):
    original = parts(data)
    if order is None:
        return [(trial, concurrent, data, original) for trial, concurrent in SCHEDULE]
    p = data[2]["P"]
    if not isinstance(order, list) or sorted(order) != list(range(p)):
        raise ValueError("owner order must be a permutation of all original compute owners")
    groups = [data[3][i] for i in order]
    owner = np.argsort(order)[data[4]]
    changed = (*data[:3], groups, owner, *data[5:])
    placed = parts(changed)
    sources = json.loads((build / "build.json").read_text())["source_sha256"]
    for lane, (_, _, _, blocks) in enumerate(placed):
        source = build / "src" / f"owner{lane}.w"
        if not source.is_file():
            source = build / source.name
        if device.sha(source) != sources[source.name]:
            raise ValueError("selector capacity source differs from the compiled image")
        sizes = re.findall(r"u32 (?:selectors|ignored)\[n=(\d+)\];", source.read_text())
        needed = max((len(ref) for _, owners in blocks for ref in owners), default=0)
        if len(sizes) != 2 or min(map(int, sizes)) < needed:
            raise ValueError("placement exceeds the compiled selector buffer")
    # Both allocations use concurrent sends. The original serial controls become
    # the paired alternative at each warm input and echo setting.
    return [
        (trial, True, data, original) if baseline else (trial, True, changed, placed)
        for trial, baseline in SCHEDULE
    ]


def packets(rows, position):
    logical, _ = SCHEDULE[position]
    echo, scale = TRIALS[logical]
    index = 0 if position == 0 else 1 if scale == 1 else 2
    result = []
    for lane, (z, cfg, groups, tiles) in enumerate(rows):
        stream = upload(z, cfg, groups, tiles, index)
        padding = (len(stream) + 1) % 2
        header = [position + 1, int(echo and lane == 0) + 2 * padding] + ([0] if padding else [])
        words = np.r_[np.array(header, np.uint32), stream[1:]]
        if len(words) % 2:
            raise ValueError("a Direct Link message needs an even word count")
        result.append(words)
    return result


def verify(data, position, raw, lower_ack, gate_ack, hidden, expected_hidden=None):
    logical, _ = SCHEDULE[position]
    tag = position + 1
    np.testing.assert_array_equal(gate_ack, [tag, CHECK])
    width = data[2]["P"] // 2
    np.testing.assert_array_equal(lower_ack, [tag, width])
    # The single-row oracle uses its logical trial ID. Check every physical tag before normalizing.
    np.testing.assert_array_equal(raw[-4], tag)
    triplet = raw[-3:].astype(np.int64)
    if np.any(triplet > 65535):
        raise ValueError("invalid collector data-start clock")
    weights = np.array([1, 1 << 16, 1 << 32], dtype=np.int64)
    start = int(triplet @ weights)
    stop = int(raw[14:17].astype(np.int64) @ weights)
    raw = raw[:-4]
    fields = [0, *range(17, 17 + 16 * data[2]["P"], 16)]
    last = data[7]
    if (17 + 16 * data[2]["P"] + last["Z"] * (1 + last["Q"])) % 2:
        fields.append(len(raw) - 1)
    np.testing.assert_array_equal(raw[fields], tag)
    copy = raw.copy()
    copy[fields] = logical + 1
    logits, metrics = audit(data, logical, copy, hidden, expected_hidden)
    interval = (stop - start) % (1 << 48)
    if not 0 < interval < metrics["collector_first_to_last_cycles"]:
        raise ValueError("the data interval must fit in the enclosing collector gate")
    metrics["collector_from_data_tag_cycles"] = interval
    return logits, metrics
