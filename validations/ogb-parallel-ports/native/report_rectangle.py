"""Recover native GraphSAGE rectangle traffic, clocks and original scores from saved frames."""

import argparse
import json
from pathlib import Path
import numpy as np

from contract_lanes import TRIAL_PLAN, TRIALS, decode_clock, device, evaluate
from worker_rectangle import packets


def replay(fixture, build, capture, expected=None):
    meta = json.loads((build / "build.json").read_text())
    doc = json.loads((capture / "result.json").read_text())
    if (
        not doc["stopped"]
        or len(doc["frames"]) != len(TRIAL_PLAN)
        or doc["role"] != meta["role"]
        or doc["build_sha256"] != device.sha(build / "build.json")
        or doc["fixture_sha256"] != device.sha(fixture)
        or doc["fixture_sha256"] != meta["fixture_sha256"]
    ):
        raise ValueError("incomplete native rectangle")
    for name, h in meta["source_sha256"].items():
        if device.sha(build / "src" / name) != h:
            raise ValueError("native source differs")
    if {p.name: device.sha(p) for p in (build / "out/bin").glob("*.elf")} != meta["elf_sha256"]:
        raise ValueError("different native PE binaries")
    z = device.load(fixture)
    pr = meta["params"]
    parts, upload = packets(z, pr)
    result = dict(
        role=doc["role"],
        fixture_sha256=doc["fixture_sha256"],
        build_sha256=doc["build_sha256"],
        lanes=pr["L"],
        owners_per_lane=pr["P"],
        destinations=len(z["seed"]),
        arcs=len(z["src"]),
        frames=[],
    )
    for key, off in [("empty_before", 100), ("empty_after", 200)]:
        if len(doc[key]) != 5:
            raise ValueError("missing empty samples")
        values = []
        for k, s in enumerate(doc[key]):
            tag = off + k
            t = decode_clock(
                np.r_[np.array([tag, tag], np.uint32), np.array(s["clock"], np.uint32)], tag
            )
            if t != s["cycles"] or tag != s["tag"] or s["synchronized_host_ns"] <= 0:
                raise ValueError("invalid empty fence")
            values.append(t)
        result[key + "_cycles"] = values
    states = {}
    for i, ((logical, combined), u, row) in enumerate(zip(TRIAL_PLAN, upload, doc["frames"]), 1):
        mode, scale = TRIALS[logical]
        f = capture / f"frame-{i}.npz"
        if (i, mode, scale, combined, device.sha(f)) != (
            row["frame"],
            row["mode"],
            row["scale"],
            row["combined_rectangle"],
            row["raw_sha256"],
        ):
            raise ValueError("native capture schedule differs")
        with np.load(f, allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw["nodes"], z["seed"])
            y, metrics = evaluate(z, parts, raw["reply"], logical)
            np.testing.assert_array_equal(y.view("u4"), raw["logits"].view("u4"))
            if raw["fence"].shape != (pr["L"], pr["P"], 6) or (raw["fence"] > 65535).any():
                raise ValueError("invalid fence shape/limbs")
            ticks = decode_clock(np.r_[np.array([i, i], np.uint32), raw["fence"][0, 0]], i)
            peak = int(raw["active_edge_cycles"].max())
        if scale in states:
            np.testing.assert_array_equal(y.view("u4"), states[scale].view("u4"))
        else:
            states[scale] = y.copy()
        if expected is not None:
            np.testing.assert_array_equal(y.view("u4"), expected[scale].view("u4"))
        weights, ctrl, idx, inv, roots, blocks = u
        tables = sum(a.size for a, _, _, _ in blocks)
        useful = sum(s for _, _, _, s in blocks)
        words = (
            tables
            + sum(v.size for v in (ctrl, idx, inv, roots))
            + sum(b.size + c.size for _, b, c, _ in blocks)
        )
        if weights is not None:
            words += weights.size
        measured = dict(
            common_clock_cycles=ticks,
            peak_local_cycles=max(v["peak_first_to_last"] for v in metrics),
            peak_active_edge_cycles=peak,
            h2d_words=words,
            table_words=tables,
            useful_table_words=useful,
            table_calls=len(blocks) * (1 if combined else pr["L"]),
        )
        if not all(row[k] == v for k, v in measured.items()) or row["synchronized_host_ns"] <= 0:
            raise ValueError("native counters or traffic differ")
        result["frames"].append(
            dict(
                frame=i,
                mode=mode,
                scale=scale,
                combined=combined,
                **measured,
                fp64_max_abs=max(v["fp64_max_abs"] for v in metrics),
                raw_sha256=device.sha(f),
            )
        )
    return result, states


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("fixture", "build", "capture"):
        p.add_argument(n, type=Path)
    p.add_argument("--tungsten", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    other = None
    if a.tungsten:
        from report_lanes import replay as tungsten

        _, other = tungsten(a.fixture, a.tungsten, a.tungsten / "run")
    report, _ = replay(a.fixture, a.build, a.capture, other)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if a.output:
        a.output.write_text(text)
    else:
        print(text, end="")
