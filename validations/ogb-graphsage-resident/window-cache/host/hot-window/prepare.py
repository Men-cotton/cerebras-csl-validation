"""Freeze the actual original-A device projections and the complete hidden aggregation window."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from codec import captured_edge_banks
from graph import (
    decode,
    load_case,
    oracle,
    packed,
    read,
    require,
    schedule,
    sha,
    weights,
)


def prepare(case, original, capture, output):
    identity, owners, cfg, windows, sources, banks = captured_edge_banks(
        case, original, capture, None
    )
    _, g = load_case(case)
    p, r, e, f = (cfg[k] for k in ("P", "R", "E", "F"))
    steps = [s for s in schedule(g, cfg) if s["trial"] == 1]
    hidden = [s for s in steps if s["mode"] == 1]
    require(len(hidden) == 1, "one complete first hidden window")
    step = hidden[0]
    projected = np.empty((len(g["raw_ids"]), 2 * f), np.float32)
    for s in steps:
        if s["mode"]:
            continue
        with np.load(
            capture / f"t1-s0-w{s['index']:04d}.npz", allow_pickle=False
        ) as cap:
            value, _, _ = decode(cap["reply"], g, s, packed(g, s, cfg, None, 1.0), cfg)
            projected[s["rows"]] = value
    batch = packed(g, step, cfg, projected, 1.0)
    with np.load(
        capture / f"t1-s1-w{step['index']:04d}.npz", allow_pickle=False
    ) as cap:
        y, _, _ = decode(cap["reply"], g, step, batch, cfg)
    chosen = np.full((step["parts"], p, e), -1, np.int32)
    for part, per_pe in enumerate(sources[1]):
        for pe, ids in enumerate(per_pe):
            chosen[part, pe, : len(ids)] = ids
            np.testing.assert_array_equal(
                batch["bank"][part, pe, : len(ids)].view(np.uint32), banks[1][part][pe]
            )
    reference_raw = g["x"].astype(np.float64) @ np.c_[g["wr0"], g["wn0"]].astype(
        np.float64
    )
    ref_a = oracle(g, step, reference_raw, 1.0)
    ref_half = oracle(g, step, reference_raw * 0.5, 0.5)
    np.testing.assert_allclose(y, ref_a, rtol=4e-4, atol=3e-4)
    output.mkdir(parents=True, exist_ok=True)
    features = projected[:, f:]
    fixture = output / "fixture.npz"
    np.savez_compressed(
        fixture,
        sources=chosen,
        bank=batch["bank"],
        n=batch["n"],
        ids=batch["ids"],
        inv=batch["inv"],
        root=batch["root"],
        ends=batch["ends"],
        counts=batch["counts"],
        original_nodes=g["raw_ids"],
        feature_table=features,
        weights=weights(g, cfg),
        measured_hidden=y,
        fp64_a=ref_a,
        fp64_half=ref_half,
        hidden_nodes=g["a_ids"],
    )
    record = dict(
        dataset=identity["dataset"],
        case_sha256=sha(case / "case.json"),
        original_build_sha256=sha(original / "build.json"),
        original_graph_sha256=sha(case / "graph.npz"),
        original_trial_A_windows=windows,
        params=cfg,
        parts=step["parts"],
        original_arcs=int(batch["counts"].sum()),
        original_hidden_words=y.size,
        original_order_sha256=hashlib.sha256(g["a_ids"].tobytes()).hexdigest(),
        files={fixture.name: sha(fixture)},
    )
    (output / "fixture.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    for n in ("case", "original", "capture", "out"):
        a.add_argument(n, type=Path)
    args = a.parse_args()
    prepare(args.case, args.original, args.capture, args.out)
