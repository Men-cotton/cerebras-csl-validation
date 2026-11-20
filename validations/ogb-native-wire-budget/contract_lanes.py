"""Split original destinations over independent source tables and West ports."""

from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from source_table import TRIALS, assign, audit, device, source_tiles, upload

# Paired original/parallel deliveries, swapped ordering at the next condition.
# Index zero of the original protocol loads the weights exactly once.
TRIAL_PLAN = [
    (0, True),
    (5, False),
    (1, False),
    (1, True),
    (2, True),
    (2, False),
    (3, False),
    (3, True),
    (4, True),
    (4, False),
    (5, False),
    (5, True),
]


def lane_inputs(z, lanes, owners, capacity, assignment=None, row_caps=None):
    if lanes not in (1, 2, 4, 8) or owners < 1 or len(z["seed"]) < lanes:
        raise ValueError("insufficient destinations or invalid PE array")
    output = []
    if assignment is None:
        assignment = np.array_split(np.arange(len(z["seed"])), lanes)
    if len(assignment) != lanes or min(map(len, assignment)) == 0:
        raise ValueError("invalid destination group")
    np.testing.assert_array_equal(np.sort(np.concatenate(assignment)), np.arange(len(z["seed"])))
    for lane, rows in enumerate(assignment):
        src = np.concatenate([z["src"][z["ptr"][r] : z["ptr"][r + 1]] for r in rows])
        roots = z["root"][rows]
        active = np.unique(np.concatenate([roots, src]))
        shard = dict(z)
        for name in ("seed", "labels", "fp64", "gpu"):
            shard[name] = z[name][rows]
        shard.update(
            node=z["node"][active],
            root=np.searchsorted(active, roots),
            ptr=np.r_[0, np.diff(z["ptr"])[rows].cumsum()],
            src=np.searchsorted(active, src),
            x=z["x"][active],
            shape=np.array([len(rows), len(active), len(src), *z["shape"][-2:]]),
        )
        cfg, groups, owner, slot = assign(shard, owners, capacity)
        if row_caps is not None:
            if not cfg["R"] <= row_caps[lane] <= capacity:
                raise ValueError("shared row bank has incompatible capacity")
            cfg["R"] = row_caps[lane]
        tiles = {mode: source_tiles(shard, cfg, owner, slot, mode) for mode, _ in TRIALS[:2]}
        output.append((shard, cfg, groups, owner, slot, tiles))
    return output


def configured_parts(z, desc, policy):
    policies = desc.get("partitions", {"contiguous": None})
    if policy not in policies:
        raise ValueError(f"partition not covered by this image: {policy}")
    return lane_inputs(
        z,
        desc["lanes"],
        desc["owners_per_lane"],
        desc["entries"],
        policies[policy],
        desc.get("row_caps"),
    )


def wire(part, logical):
    z, cfg, groups, _, _, tiles = part
    mode, scale = TRIALS[logical]
    payload = upload(z, cfg, groups, tiles[mode], logical)
    padding = (len(payload) + 1) % 2
    return np.concatenate([np.array([padding] + [0] * padding, np.uint32), payload])


def reply_shape(part):
    cfg = part[1]
    return cfg["P"], 16 + cfg["R"] * (1 + cfg["Q"])


def evaluate(z, parts, packets, logical):
    if len(parts) != len(packets):
        raise ValueError("missing port reply")
    results = []
    metrics = []
    for part, p in zip(parts, packets):
        shard, cfg, groups, *_ = part
        if p.shape != reply_shape(part):
            raise ValueError("wrong lane packet size")
        y, m = audit(p, shard, cfg, groups, logical)
        results.append(y)
        metrics.append(m)
    y = np.concatenate(results)
    lane_ids = np.concatenate([s[0]["seed"] for s in parts])
    if not np.array_equal(lane_ids, z["seed"]):
        original_rows = {int(v): i for i, v in enumerate(z["seed"])}
        restored = np.empty_like(y)
        restored[[original_rows[int(v)] for v in lane_ids]] = y
        y = restored
    truth = device.oracle(z, TRIALS[logical][1])
    np.testing.assert_allclose(y, truth, rtol=3e-4, atol=2e-4)
    np.testing.assert_array_equal(y.argmax(1), truth.argmax(1))
    if TRIALS[logical][1] == 1:
        np.testing.assert_array_equal(y.argmax(1), z["gpu"].argmax(1))
    return y, metrics


def decode_clock(reply, tag):
    if reply.shape != (8,) or tuple(reply[[0, 1]]) != (tag, tag):
        raise ValueError("invalid timing fence")
    limbs = reply[2:].reshape(2, 3)
    if np.any(limbs > 65535):
        raise ValueError("invalid 48-bit timestamp")
    start, end = limbs.astype(np.int64) @ np.array([1, 65536, 1 << 32])
    delta = int(end - start) % (1 << 48)
    if not 0 < delta < 1 << 41:
        raise ValueError("unexpected port fence interval")
    return delta
