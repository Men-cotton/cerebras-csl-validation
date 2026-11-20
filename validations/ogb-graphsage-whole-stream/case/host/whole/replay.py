"""Audit original node coverage, classifications and all per-PE clocks of a full final layer."""
import argparse
from pathlib import Path
import sys

import numpy as np

from dispatch import load, pack, read, save, sha, slice_graph, windows

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "device"))
from contract import check_frame


def require(test, message):
    if not test:
        raise ValueError(message)


def read_clocks(reply, config):
    count = config["K"] + 2
    clocks = reply[:, 4:4 + 6 * count].astype(np.int64).reshape(config["P"], count, 6)
    base = np.array([1, 1 << 16, 1 << 32], dtype=np.int64)
    start = clocks[:, 0, :3] @ base
    end = clocks[:, -1, 3:] @ base
    return (end - start) % (1 << 48)


def audit(build, fixture, capture, selected=None, role="simulator", peer=None):
    model = load(fixture)
    descriptor = read(build / "build.json")
    state = read(capture / "result.json")
    require(state["transfer_sha256"] == sha(HERE / "rectangles.py"), "compact transfer code identity")
    config = descriptor["params"]
    expected_cfg, plan, guards = windows(model, config["P"], config["R"])
    require(expected_cfg == config == state["params"], "compiled parameters differ")
    require(read(build / "windows.json") == plan, "frozen windows differ from original graph")
    require(sha(build / "windows.json") == descriptor["windows_sha256"] == state["windows_sha256"], "window identity")
    require(sha(fixture) == descriptor["input_sha256"] == state["input_sha256"], "fixture identity")
    require(sha(build / "build.json") == state["build_sha256"], "executed build identity")
    require(state["language"] == descriptor["language"] and state["role"] == role, "wrong language or target")
    require(state["platform"] == {"system": role == "hardware", "simulation": role == "simulator"}, "SDK target")
    expected_image = descriptor["roles"][role]["elf"] if state["language"] == "tungsten" else descriptor["elf"]
    require(state["elf_sha256"] == expected_image, "loaded image identity")
    require(state["source"] == {p.name: sha(p) for p in (HERE / "run.py", HERE / "dispatch.py",
                                                              HERE.parent / "device/contract.py")}, "runtime source identity")
    complete = {entry["window"]: entry for entry in state["completed"]}
    require(len(complete) == len(state["completed"]), "repeated window")
    if selected is None:
        require(state["stage"] == "PASS" and state["stopped"] and set(complete) == set(range(len(plan))),
                "whole graph was not collected")
        selected = list(range(len(plan)))
    else:
        selected = list(selected)
        require(set(selected) <= set(complete) and len(set(selected)) == len(selected), "unavailable or repeated sample")
    if peer:
        other = read(peer / "result.json")
        require(other["input_sha256"] == state["input_sha256"] and other["params"] == config, "peer input or parameters")
        peers = {e["window"]: e for e in other["completed"]}
        require(set(selected) <= set(peers), "missing peer windows")
    n, q = len(model["x"]), len(model["b"])
    seen = np.zeros(n, bool)
    correct, total = np.zeros(3, np.int64), np.zeros(3, np.int64)
    actual, predicted = np.zeros(q, np.int64), np.zeros(q, np.int64)
    summary = {"role": role, "language": state["language"], "params": config, "build_sha256": state["build_sha256"],
               "input_sha256": state["input_sha256"], "expected_nodes": n, "expected_arcs": len(model["src"]),
               "expected_windows": len(plan), "windows": [], "selected_controls": [], "edges": 0,
               "sum_of_window_peak_active_cycles": 0, "sum_of_window_phase_maxima_cycles": 0,
               "sum_of_window_peak_elapsed_cycles": 0, "A_input_words_across_windows": 0,
               "A_reply_words_across_windows": 0, "max_abs": 0., "fp64_argmax_mismatches": 0,
               "whole_model_argmax_mismatches": 0, "peer_argmax_mismatches": 0 if peer else None,
               "peer_different_words": 0 if peer else None, "peer_max_abs": 0. if peer else None}
    error_sq, norm_sq = 0., 0.
    for index in selected:
        window = plan[index]
        entry = complete[index]
        z = slice_graph(model, window)
        packed = pack(model, z, config, window)
        local = dict(config, K=window["parts"])
        require(all(entry[k] == window[k] for k in ("first", "last", "edges", "parts")), "journal window identity")
        nodes = z["seed"]
        require(not seen[nodes].any(), "repeated original destination")
        seen[nodes] = True
        source = capture / f"w{index:04d}.npz"
        require(sha(source) == entry["raw_sha256"], "captured raw words changed")
        expected_trials = [1, 2, 3] if index in guards else [1]
        require([f["trial"] for f in entry["frames"]] == expected_trials, "missing input frame")
        if index in guards:
            summary["selected_controls"].append(index)
        detail = {"index": index, "raw_sha256": entry["raw_sha256"], "edges": window["edges"], "parts": local["K"], "frames": []}
        with np.load(source, allow_pickle=False) as raw:
            require(set(raw.files) == {"t" + str(t) for t in expected_trials}, "unexpected capture members")
            for frame in entry["frames"]:
                trial = frame["trial"]
                reply = raw[f"t{trial}"]
                values, ticks, err = check_frame(reply, z, packed, local, trial)
                maxsum = int(ticks.sum(1).max())
                phasemax = int(ticks.max(0).sum())
                elapsed = int(read_clocks(reply, local).max())
                require(maxsum == frame["peak_sum_pe_cycles"] and phasemax == frame["sum_phase_max_cycles"] and
                        int(ticks[:, 1:-1].sum(1).max()) == frame["max_aggregation_cycles"] and
                        int(ticks[:, -1].max()) == frame["max_projection_cycles"], "archived clocks differ")
                require(frame["argmax_vs_final_fp64"] == err["argmax_mismatch"], "FP64 class mismatch count differs")
                # Summaries produced by two BLAS implementations can round differently at ~1e-15.
                np.testing.assert_allclose([frame["max_abs"], frame["relative_l2"]], [err["max_abs"], err["relative_l2"]],
                                           rtol=1.e-9, atol=2.e-13)
                detail["frames"].append({"trial": trial, "peak_active_cycles": maxsum, "sum_phase_max_cycles": phasemax,
                                         "peak_elapsed_cycles": elapsed, "max_abs": err["max_abs"]})
                if trial == 3:
                    np.testing.assert_array_equal(base_values.view(np.uint32), values.view(np.uint32))
                if peer:
                    peer_file = peer / source.name
                    require(sha(peer_file) == peers[index]["raw_sha256"], "peer raw words changed")
                    with np.load(peer_file, allow_pickle=False) as other_raw:
                        other_reply = other_raw[f"t{trial}"]
                        other_values, _, _ = check_frame(other_reply, z, packed, local, trial)
                    require(np.allclose(values, other_values, rtol=4e-4, atol=3e-4), "matched outputs differ")
                    if trial == 1:
                        summary["peer_argmax_mismatches"] += int(np.sum(values.argmax(1) != other_values.argmax(1)))
                        summary["peer_different_words"] += int(np.sum(values.view(np.uint32) != other_values.view(np.uint32)))
                        summary["peer_max_abs"] = max(summary["peer_max_abs"], float(np.abs(values - other_values).max()))
                if trial != 1:
                    continue
                base_values = values.copy()
                labels = model["labels"][nodes]
                ids = values.argmax(1)
                split = model["split"][nodes]
                for k in range(3):
                    mask = split == k
                    total[k] += int(mask.sum())
                    correct[k] += int(np.sum(ids[mask] == labels[mask]))
                actual += np.bincount(labels, minlength=q)
                predicted += np.bincount(ids, minlength=q)
                delta = values.astype(np.float64) - z["fp64"]
                error_sq += float(np.square(delta).sum())
                norm_sq += float(np.square(z["fp64"]).sum())
                summary["edges"] += window["edges"]
                summary["sum_of_window_peak_active_cycles"] += maxsum
                summary["sum_of_window_phase_maxima_cycles"] += phasemax
                summary["sum_of_window_peak_elapsed_cycles"] += elapsed
                summary["A_input_words_across_windows"] += frame["input_words"]
                summary["A_reply_words_across_windows"] += frame["response_words"]
                summary["max_abs"] = max(summary["max_abs"], float(np.abs(delta).max()))
                compared = int(np.sum(ids != model["original_prediction"][nodes]))
                require(compared == frame["argmax_vs_original_model"], "original-model class mismatch count differs")
                summary["whole_model_argmax_mismatches"] += compared
                summary["fp64_argmax_mismatches"] += int(err["argmax_mismatch"])
        summary["windows"].append(detail)
    summary["nodes"] = int(seen.sum())
    summary["graph_complete"] = bool(seen.all() and summary["edges"] == len(model["src"]) and
                                     set(selected) == set(range(len(plan))))
    summary["relative_l2"] = float(np.sqrt(error_sq / norm_sq))
    summary["splits"] = {name: {"correct": int(correct[k]), "count": int(total[k])}
                         for k, name in enumerate(("train", "valid", "test"))}
    summary["true_class_counts"] = actual.tolist()
    summary["predicted_class_counts"] = predicted.tolist()
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("build", type=Path)
    p.add_argument("fixture", type=Path)
    p.add_argument("capture", type=Path)
    p.add_argument("--windows", type=int, nargs="+")
    p.add_argument("--role", choices=("simulator", "hardware"), default="simulator")
    p.add_argument("--peer", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.build, args.fixture, args.capture, args.windows, args.role, args.peer)
    save(args.out, result)
    print(result["language"], result["nodes"], result["edges"], result["max_abs"], result["graph_complete"])
