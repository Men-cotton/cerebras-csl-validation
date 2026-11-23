"""Construct original source transfers and audit all 384 final predictions and PE clock regions."""

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
HOT = HERE.parent
sys.path.insert(0, str(HOT))
spec = importlib.util.spec_from_file_location("source_window_worker", HOT / "worker.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from rectangles import rectangles


def materialize(z, meta, use, scale):
    hot, table, wire, payload, hits = base.pack(z, scale, use, meta["slots"])
    counts = z["counts"]
    f = z["bank"].shape[-1]
    kept = z["indexed_banks"].tolist() if use else []
    if use:
        np.testing.assert_array_equal(hot, z["hot"])
        plain = sorted(set(range(len(counts))) - set(kept))
        hits[plain] = 0
        wire[plain] = counts[plain] * f
        dense = np.ascontiguousarray(z["bank"][plain] * np.float32(scale)).view(
            np.uint32
        )
        payload[plain] = (
            dense.reshape(len(plain), counts.shape[1], -1) if plain else payload[plain]
        )
    words = int(table.size)
    calls = int(len(table) > 0)
    for row in wire:
        for lo, hi, size in rectangles(row, 1):
            words += (hi - lo) * size
            calls += 1
    base.require(
        {"words": words, "calls": calls} == meta["selected" if use else "dense"],
        "all measured source DMA rectangles",
    )
    return hot, table, wire, payload, hits, kept


def raw_output(raw, z, meta, tag, scale):
    nodes = z["original_destinations"]
    p = meta["params"]["P"]
    groups = [np.searchsorted(nodes, rows[:n]) for rows, n in zip(z["ids"], z["n"])]
    step = dict(
        mode=2, parts=meta["parts"], tag=tag, rows=np.arange(len(nodes)), groups=groups
    )
    y, ticks, elapsed = base.decode(
        raw, {"b1": np.empty(meta["classes"])}, step, z, meta["params"]
    )
    truth = z["fp64_a" if scale == 1 else "fp64_half"]
    np.testing.assert_allclose(y, truth, rtol=4e-4, atol=3e-4)
    np.testing.assert_array_equal(y.argmax(axis=1), truth.argmax(axis=1))
    if scale == 1:
        np.testing.assert_allclose(y, z["gpu_logits"], rtol=4e-4, atol=3e-4)
        np.testing.assert_array_equal(y.argmax(axis=1), z["gpu_logits"].argmax(axis=1))
    base.require(
        y.shape == (384, 47) and ticks.shape == (p, meta["parts"] + 2),
        "every original class and PE region",
    )
    return y, dict(
        peak_active_cycles=int(ticks.sum(axis=1).max()),
        peak_edge_cycles=int(ticks[:, 1:-1].sum(axis=1).max()),
        peak_elapsed_cycles=int(elapsed.max()),
        fp64_max_abs=float(np.max(np.abs(y - truth))),
        ordered_logits_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
    )


def source_hashes():
    return {
        p.name: base.sha(p)
        for p in [
            HERE / "contract.py",
            HERE / "worker_final.py",
            HOT / "worker.py",
            HOT.parent / "window_cache.py",
            *(Path(sys.modules[m].__file__) for m in ("graph", "run", "rectangles")),
        ]
    }
