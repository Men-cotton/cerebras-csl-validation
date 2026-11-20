"""Replay all two-port hidden states, predictions and PE clocks against the original one-port device."""

import argparse
import json
from pathlib import Path
import numpy as np

from two_rows import SCHEDULE, TRIALS, device, packets, prepare, trial_inputs, verify
from report_chain import replay as replay_original


def replay(first, last, build, capture, original, original_capture):
    desc = json.loads((build / "build.json").read_text())
    raw_report = json.loads((capture / "result.json").read_text())
    if not raw_report["stopped"] or len(raw_report["frames"]) != len(SCHEDULE):
        raise ValueError("incomplete two-port capture")
    peer, expected = replay_original(first, last, original, original_capture)
    if desc["original_build_sha256"] != peer["build_sha256"]:
        raise ValueError("different one-port control")
    role = raw_report["role"]
    if role not in ("simulator", "hardware"):
        raise ValueError("unknown platform")
    for path, digest in [
        (first, desc["first_sha256"]),
        (last, desc["last_sha256"]),
        (build / "build.json", raw_report["build_sha256"]),
        (build / "chain.map", desc["map_sha256"]),
        (build / role / "out/program.elf", desc["roles"][role]["elf_sha256"]),
        (build / role / "out/out_port_map.json", desc["port_sha256"]),
        *((build / "src" / n, h) for n, h in desc["source_sha256"].items()),
    ]:
        if device.sha(path) != digest:
            raise ValueError(f"changed firmware, raw input or reference: {path.name}")
    if (raw_report["first_sha256"], raw_report["last_sha256"]) != (device.sha(first), device.sha(last)):
        raise ValueError("capture identifies different weights or graph")
    cfg = desc["first"]
    data = prepare(first, last, cfg["P"], cfg["E"])
    if (
        cfg != data[2]
        or data[7] != desc["final"]
        or raw_report["setup_words"] != len(data[8]) + (len(data[8]) % 2)
    ):
        raise ValueError("different shape or setup")
    ordering = raw_report.get("compared_owner_order")
    trials = trial_inputs(build, data, ordering)
    frames = []
    for pos, ((logical, concurrent, data, rows), row) in enumerate(zip(trials, raw_report["frames"])):
        echo, scale = TRIALS[logical]
        tag = pos + 1
        path = capture / f"frame-{tag}.npz"
        if (
            row["tag"],
            row["logical"],
            row["concurrent"],
            row["echo"],
            row["scale"],
            row["raw_sha256"],
        ) != (tag, logical, concurrent, echo, scale, device.sha(path)):
            raise ValueError("inconsistent trial or raw response")
        with np.load(path, allow_pickle=False) as raw:
            required = {"gate_ack", "lower_ack", "reply", "logits", "nodes"} | (
                {"echo"} if echo else set()
            )
            if set(raw.files) != required:
                raise ValueError("unexpected transfer of intermediate states")
            hid = raw["echo"] if echo else None
            logits, metrics = verify(
                data,
                pos,
                raw["reply"],
                raw["lower_ack"],
                raw["gate_ack"],
                hid,
                expected["hidden"][scale],
            )
            np.testing.assert_array_equal(raw["nodes"], data[1]["seed"])
            np.testing.assert_array_equal(raw["logits"].view(np.uint32), logits.view(np.uint32))
            np.testing.assert_array_equal(
                logits.view(np.uint32), expected["logits"][scale].view(np.uint32)
            )
            if echo:
                np.testing.assert_array_equal(hid, expected["hidden"][scale])
        for k, v in metrics.items():
            if k.endswith("max_abs") and v is not None:
                np.testing.assert_allclose(v, row[k], rtol=1e-8, atol=1e-13)
            elif v != row[k]:
                raise ValueError(f"wrong measurement: {k}")
        inputs = list(map(len, packets(rows, pos)))
        num_hidden = data[7]["H"] * data[7]["N"] if echo else 0
        if (
            row["port_input_words"],
            row["gate_input_words"],
            row["gate_output_words"],
            row["lower_ack_words"],
            row["hidden_h2d_words"],
            row["hidden_d2h_words"],
        ) != (inputs, 2, 2, 2, num_hidden, num_hidden):
            raise ValueError("wire counts disagree with the original graph")
        if row["synchronized_host_ns"] <= 0:
            raise ValueError("missing synchronized host time")
        frames.append(
            dict(
                tag=tag,
                logical=logical,
                concurrent=concurrent,
                echo=echo,
                scale=scale,
                input_words_by_port=inputs,
                one_port_data_interval=peer["frames"][logical]["collector_first_to_last_cycles"],
                hidden_round_trip_words=2 * num_hidden,
                **metrics,
            )
        )
    return dict(
        role=role,
        **({"compared_owner_order": ordering} if ordering is not None else {}),
        first_sha256=device.sha(first),
        last_sha256=device.sha(last),
        build_sha256=device.sha(build / "build.json"),
        original_build_sha256=peer["build_sha256"],
        original_role=peer["role"],
        first_owners=cfg["P"],
        hidden_vertices=data[7]["N"],
        final_vertices=data[7]["Z"],
        arcs=[len(data[0]["src"]), len(data[1]["src"])],
        one_port_geometry=[cfg["P"], 2],
        two_port_geometry=[cfg["P"] // 2, 4],
        frames=frames,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("first", "last", "build", "capture", "original", "original_capture"):
        parser.add_argument(n, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = (
        json.dumps(
            replay(
                args.first, args.last, args.build, args.capture, args.original, args.original_capture
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
