"""Tile original OGB sources without changing a destination's CSR summation order."""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEVICE = HERE.parent / "device"
contract_path = HERE / "ogb_contract.py"
if not contract_path.is_file():
    contract_path = DEVICE / "contract.py"
spec = importlib.util.spec_from_file_location("ogb_device_contract", contract_path)
device = importlib.util.module_from_spec(spec)
spec.loader.exec_module(device)
TRIALS = (("edges", 1.), ("dictionary", 1.), ("dictionary", .5),
          ("edges", .5), ("dictionary", 1.), ("edges", 1.))


def assign(z, p, capacity):
    groups = device.place(z, p, "balanced")
    rows = max(map(len, groups))
    if not 1 <= rows <= capacity <= 65535:
        raise ValueError("the table also stages root ingress for other owners")
    n, _, _, f, c = map(int, z["shape"])
    params = dict(P=p, R=rows, F=f, Q=(c+3)//4*4, E=capacity, RELU=int(z["relu"]), FRAMES=len(TRIALS))
    owner = np.empty(n, np.uint16)
    slot = np.empty(n, np.uint16)
    for pe, group in enumerate(groups):
        owner[group] = pe
        slot[group] = np.arange(len(group))
    return params, groups, owner, slot


def source_tiles(z, params, owner, slot, method):
    """Return the common table and the ordered (owner, row, source slot) reads per block."""
    src = z["src"]
    row = np.repeat(np.arange(len(z["seed"])), np.diff(z["ptr"]))
    order = np.argsort(src, kind="stable")
    ordered, rows = src[order], row[order]
    if method == "dictionary":
        table, indices = np.unique(ordered, return_inverse=True)
    elif method == "edges":
        table, indices = ordered, np.arange(len(ordered))
    else:
        raise ValueError(method)
    result = []
    seen = [[] for _ in range(len(z["seed"]))]
    for start in range(0, len(table), params["E"]):
        stop = min(start + params["E"], len(table))
        lo, hi = np.searchsorted(indices, [start, stop])
        dest = rows[lo:hi]
        pe = owner[dest]
        local = (indices[lo:hi] - start).astype(np.uint32)
        refs = (slot[dest].astype(np.uint32) << 16) | local
        words = [refs[pe == who] for who in range(params["P"])]
        entries = table[start:stop]
        for r, v in zip(dest, entries[local]):
            seen[r].append(v)
        result.append((entries, words))
    for i, observed in enumerate(seen):
        np.testing.assert_array_equal(observed, src[z["ptr"][i]:z["ptr"][i+1]])
    return result


def upload(z, cfg, groups, tiles, trial):
    method, scale = TRIALS[trial]
    q, f = cfg["Q"], cfg["F"]
    classes = z["wr"].shape[1]
    words = [np.array([trial+1], np.uint32)]
    if trial == 0:
        matrix = np.zeros((2*f+1, q), np.float32)
        matrix[:f,:classes] = z["wr"]
        matrix[f:2*f,:classes] = z["wn"]
        matrix[-1,:classes] = z["b"]
        words.append(matrix.ravel().view(np.uint32))
    for group in groups:
        degree = np.diff(z["ptr"])[group]
        inv = (1 / np.maximum(1, degree)).astype(np.float32)
        words += [np.array([len(group)], np.uint32), z["seed"][group].astype(np.uint32),
                  inv.view(np.uint32), (z["x"][z["root"][group]] * scale).ravel().view(np.uint32)]
    words.append(np.array([len(tiles)], np.uint32))
    for source, references in tiles:
        words.extend((np.array([len(source)], np.uint32),
                      (z["x"][source] * scale).ravel().view(np.uint32)))
        for pe_refs in references:
            words += [np.array([len(pe_refs)], np.uint32), pe_refs]
    return np.concatenate(words)


def audit(reply, z, cfg, groups, trial):
    r, q = cfg["R"], cfg["Q"]
    expected_edges = [int(np.diff(z["ptr"])[group].sum()) for group in groups]
    for i, ref in enumerate(([trial+1]*cfg["P"], range(cfg["P"]), list(map(len, groups)), expected_edges)):
        np.testing.assert_array_equal(reply[:, i], ref)
    y = np.empty((len(z["seed"]), z["wr"].shape[1]), np.float32)
    for pe, group in enumerate(groups):
        np.testing.assert_array_equal(reply[pe,16:16+len(group)], z["seed"][group])
        out = reply[pe,16+r:].view(np.float32).reshape(r, q)
        y[group] = out[:len(group), :y.shape[1]]
        np.testing.assert_array_equal(out[:len(group), y.shape[1]:], 0)
    limbs = reply[:,4:16].reshape(-1, 4, 3)
    if np.any(limbs > 65535):
        raise ValueError("illegal clock limb")
    t = limbs.astype(np.int64) @ np.array([1,65536,1<<32])
    ticks = np.diff(t, axis=1) % (1<<48)
    if np.any(ticks <= 0) or np.any(ticks > 1<<36):
        raise ValueError("inconsistent PE clock interval")
    expected = device.oracle(z, TRIALS[trial][1])
    np.testing.assert_allclose(y, expected, rtol=3e-4, atol=2e-4)
    diff = y.astype(np.float64)-expected
    return y, dict(peak_first_to_last=int(ticks.sum(1).max()), peak_edge_interval=int(ticks[:,1].max()),
                   peak_roots_clear=int(ticks[:,0].max()), peak_projection=int(ticks[:,2].max()),
                   fp64_max_abs=float(np.abs(diff).max()), fp64_relative_l2=float(np.linalg.norm(diff)/np.linalg.norm(expected)),
                   argmax_differences=int(np.count_nonzero(y.argmax(1)!=expected.argmax(1))))
