"""Bind the 32 selected roots and their two complete CSR layers to the trained full graph."""

import argparse
from pathlib import Path
import sys

import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAMILY))
from originals import digest, raw_inputs, read, require, sha


def verify_window(case):
    plan = read(case / "plan.json")
    batch = read(case / "batch.json")
    model = read(case / "one/model.json")
    require(
        plan["dataset"] == model["dataset"]
        and plan["graph"] == model["graph"]
        and plan["checkpoint_sha256"]
        == sha(case / "one/weights.npz")
        == model["weights_sha256"]
        and plan["gpu_logits_sha256"] == model["gpu_logits_sha256"],
        "whole model and trained checkpoint",
    )
    require(
        batch == plan["batches"][batch["index"]]
        and len(batch["original_targets"]) == 32,
        "selected batch and all 32 roots",
    )
    require(
        batch["fixtures"]
        == {n: sha(case / "one" / n) for n in ("first.npz", "last.npz")},
        "both immutable two-hop inputs",
    )
    require(
        sha(case / "window.npz") == plan["window_sha256"],
        "original RCM median 128-root window",
    )
    siblings = [row["original_targets"] for row in plan["batches"]]
    offset = sum(map(len, siblings[: batch["index"]]))
    with (
        np.load(case / "window.npz", allow_pickle=False) as parent,
        np.load(case / "one/first.npz", allow_pickle=False) as first,
        np.load(case / "one/last.npz", allow_pickle=False) as last,
    ):
        np.testing.assert_array_equal(parent["seed"], np.concatenate(siblings))
        require(
            len(parent["seed"]) == 128 and len(np.unique(parent["seed"])) == 128,
            "128 unique original targets",
        )
        np.testing.assert_array_equal(last["seed"], batch["original_targets"])
        for field in ("seed", "gpu", "fp64"):
            np.testing.assert_array_equal(
                last[field], parent[field][offset : offset + 32]
            )
        np.testing.assert_array_equal(first["seed"], last["node"])
        for i in range(32):
            original = offset + i
            sources = parent["node"][
                parent["src"][parent["ptr"][original] : parent["ptr"][original + 1]]
            ]
            actual = last["node"][last["src"][last["ptr"][i] : last["ptr"][i + 1]]]
            np.testing.assert_array_equal(actual, sources)
        require(
            [
                len(first["node"]),
                len(first["seed"]),
                len(first["src"]),
                len(last["src"]),
            ]
            == [
                batch[n] for n in ("raw_rows", "hidden_rows", "raw_arcs", "final_arcs")
            ],
            "whole batch counts",
        )
    return batch


def source_receipt(case, cache):
    batch = verify_window(case)
    model = read(case / "one/model.json")
    dataset = model["dataset"]
    resident = cache / f"resident-{dataset}-whole"
    info = read(resident / "case.json")
    original = cache / dataset
    require(
        info["original_checkpoint_sha256"] == model["weights_sha256"]
        and info["original_graph"] == model["graph"]
        and info["raw_nodes"] == model["nodes"],
        "whole normalized CSR and original feature corpus",
    )
    paths = {
        "ptr": ("ptr.npy", model["graph"]["ptr_sha256"]),
        "index": ("index.npy", model["graph"]["index_sha256"]),
        "hidden": ("cpu_input_gpu_hidden.npy", model["gpu_hidden_sha256"]),
        "logits": ("gpu_logits.npy", model["gpu_logits_sha256"]),
    }
    for name, pin in paths.values():
        require(sha(original / name) == pin, "original source " + name)
    full = {
        k: np.load(original / name, mmap_mode="r", allow_pickle=False)
        for k, (name, _) in paths.items()
    }
    full64 = np.load(original / "fp64-hidden.npy", mmap_mode="r", allow_pickle=False)
    ptr, idx = full["ptr"], full["index"]
    layers = []
    for stage, name in enumerate(("first.npz", "last.npz")):
        with np.load(case / "one" / name, allow_pickle=False) as z:
            seed, node = z["seed"], z["node"]
            neighbors = node[z["src"]]
            np.testing.assert_array_equal(
                z["ptr"], np.r_[0, np.cumsum(ptr[seed + 1] - ptr[seed])]
            )
            np.testing.assert_array_equal(
                neighbors, np.concatenate([idx[ptr[s] : ptr[s + 1]] for s in seed])
            )
            np.testing.assert_array_equal(
                np.sort(node), np.unique(np.r_[seed, neighbors])
            )
            np.testing.assert_array_equal(node[z["root"]], seed)
            np.testing.assert_array_equal(
                z["gpu"], full[("hidden", "logits")[stage]][seed]
            )
            if stage == 0:
                raw_inputs(node, z["x"], resident, info)
                np.testing.assert_allclose(
                    z["fp64"], full64[seed], rtol=1e-12, atol=1e-12
                )
            else:
                np.testing.assert_array_equal(z["x"], full["hidden"][node])
                hd = full64[node]
                sums = np.zeros((len(seed), hd.shape[1]), dtype=np.float64)
                for i in range(len(seed)):
                    sources = z["src"][z["ptr"][i] : z["ptr"][i + 1]]
                    if len(sources):
                        sums[i] = hd[sources].mean(axis=0)
                expected = (
                    hd[z["root"]] @ z["wr"].astype(np.float64)
                    + sums @ z["wn"].astype(np.float64)
                    + z["b"]
                )
                np.testing.assert_allclose(z["fp64"], expected, rtol=1e-12, atol=1e-12)
            layers.append(
                dict(
                    rows_sha256=digest(seed),
                    original_arc_sources_sha256=digest(neighbors),
                    row_pointers_sha256=digest(z["ptr"]),
                    input_feature_sha256=digest(z["x"]),
                    fp64_output_sha256=digest(z["fp64"]),
                    gpu_output_sha256=digest(z["gpu"]),
                )
            )
    return dict(
        normalized_csr=model["graph"],
        weights_sha256=model["weights_sha256"],
        layers=layers,
        complete_raw_input_case_sha256=sha(resident / "case.json"),
        source_graph_sha256=model["graph"]["processed_sha256"],
        original_batch=batch["index"],
        original_128_sha256=sha(case / "window.npz"),
        complete_fp64_hidden_sha256=sha(original / "fp64-hidden.npy"),
        complete_gpu_hidden_sha256=model["gpu_hidden_sha256"],
        complete_gpu_logits_sha256=model["gpu_logits_sha256"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    require(
        source_receipt(args.case, args.cache) == read(args.case / "one/original.json"),
        "foreign source audit",
    )
