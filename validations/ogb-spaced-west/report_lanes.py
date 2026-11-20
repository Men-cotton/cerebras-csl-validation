"""Verify original graph scores, per-lane packets and the common 48-bit port fence."""

import argparse
import json
from pathlib import Path
import numpy as np

from contract_lanes import TRIAL_PLAN, TRIALS, configured_parts, decode_clock, device, evaluate, wire


def replay(fixture, build, run, expected=None):
    desc = json.loads((build / "build.json").read_text())
    capture = json.loads((run / "result.json").read_text())
    role = capture["role"]
    image = build / role / "out"
    if (
        not capture["stopped"]
        or len(capture["frames"]) != len(TRIAL_PLAN)
        or role not in ("simulator", "hardware")
        or capture["build_sha256"] != device.sha(build / "build.json")
        or capture["fixture_sha256"] != device.sha(fixture)
        or desc["fixture_sha256"] != capture["fixture_sha256"]
        or device.sha(image / "program.elf") != desc["roles"][role]["elf_sha256"]
        or device.sha(image / "out_port_map.json") != desc["port_sha256"]
        or device.sha(build / "lanes.map") != desc["map_sha256"]
    ):
        raise ValueError("incorrect firmware or unfinished capture")
    for name, h in desc["source_sha256"].items():
        if device.sha(build / "src" / name) != h:
            raise ValueError("changed Tungsten source")
    z = device.load(fixture)
    n = desc["lanes"]
    policy = capture.get("partition", "contiguous")
    parts = configured_parts(z, desc, policy)
    if [p[1] for p in parts] != desc["lane_params"]:
        raise ValueError("incorrect partition")
    result = dict(
        role=role,
        build_sha256=device.sha(build / "build.json"),
        fixture_sha256=device.sha(fixture),
        port_sha256=desc["port_sha256"],
        original_destinations=len(z["seed"]),
        arcs=len(z["src"]),
        lanes=n,
        owners_per_lane=desc["owners_per_lane"],
        lane_destinations=[s[0]["seed"].tolist() for s in parts],
        lane_arcs=[len(s[0]["src"]) for s in parts],
        unique_sources_per_lane=[len(np.unique(s[0]["src"])) for s in parts],
        frames=[],
    )
    if "partitions" in desc:
        result["partition"] = policy
    for group, start in [("empty_before", 100), ("empty_after", 200)]:
        samples = capture[group]
        cycles = []
        if len(samples) != 5:
            raise ValueError("missing empty fence")
        for k, s in enumerate(samples):
            tag = start + k
            np.testing.assert_array_equal(s["ack"], [tag, 0xC510A11])
            ticks = decode_clock(np.array(s["timer"], np.uint32), tag)
            if ticks != s["cycles"] or s["tag"] != tag or s["host_ns"] <= 0:
                raise ValueError("wrong empty timing measure")
            cycles.append(ticks)
        result[group + "_cycles"] = cycles
    states = {}
    for i, ((logical, concurrent), frame) in enumerate(zip(TRIAL_PLAN, capture["frames"])):
        mode, scale = TRIALS[logical]
        tag = i + 1
        raw = run / f"frame-{tag}.npz"
        if (tag, mode, scale, concurrent, device.sha(raw)) != (
            frame["frame"],
            frame["method"],
            frame["scale"],
            frame["concurrent"],
            frame["raw_sha256"],
        ):
            raise ValueError("inconsistent recorded frame")
        with np.load(raw, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["ack"], [tag, 0xC510A11])
            ticks = decode_clock(data["timer"], tag)
            y, m = evaluate(z, parts, [data[f"lane{k}"] for k in range(n)], logical)
            np.testing.assert_array_equal(data["nodes"], z["seed"])
            np.testing.assert_array_equal(y.view("u4"), data["logits"].view("u4"))
        if scale in states:
            np.testing.assert_array_equal(y.view("u4"), states[scale].view("u4"))
        else:
            states[scale] = y.copy()
        if expected is not None:
            np.testing.assert_array_equal(y.view("u4"), expected[scale].view("u4"))
        words = [len(wire(s, logical)) for s in parts]
        peaks = [v["peak_first_to_last"] for v in m]
        if (
            frame["common_clock_cycles"] != ticks
            or peaks != frame["lane_compute_peaks"]
            or words != frame["port_words"]
            or frame["fence_words_each_direction"] != [4, 10]
            or not 0 < frame["synchronized_data_host_ns"] < frame["synchronized_fenced_host_ns"]
        ):
            raise ValueError("word counts, clock or host boundary mismatch")
        result["frames"].append(
            dict(
                frame=tag,
                mode=mode,
                scale=scale,
                concurrent=concurrent,
                common_clock_cycles=ticks,
                per_lane_peak_cycles=peaks,
                port_data_words=words,
                fence_input_words=4,
                fence_output_words=10,
                synchronized_data_host_ns=frame["synchronized_data_host_ns"],
                synchronized_fenced_host_ns=frame["synchronized_fenced_host_ns"],
                fp64_max_abs=max(v["fp64_max_abs"] for v in m),
                raw_sha256=device.sha(raw),
            )
        )
    if set(states) != {0.5, 1.0} or np.array_equal(states[0.5], states[1.0]):
        raise ValueError("inactive scale control")
    return result, states


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "build", "run"):
        p.add_argument(name, type=Path)
    p.add_argument(
        "--peer", type=Path, help="compare each logit with another parallel-port build and run/"
    )
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    other = replay(a.fixture, a.peer, a.peer / "run")[1] if a.peer else None
    data = replay(a.fixture, a.build, a.run, other)[0]
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    if a.output:
        a.output.write_text(text)
    else:
        print(text, end="")
