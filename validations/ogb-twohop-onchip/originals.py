"""Rebuild the frozen two-hop provenance record from the trained whole-graph cache."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    with path.open("rb") as infile:
        return hashlib.file_digest(infile, "sha256").hexdigest()


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def raw_inputs(node_ids, features, case, info):
    mapping = case / "graph.npz"
    require(sha(mapping) == info["graph_sha256"], "changed source-node mapping")
    with np.load(mapping, allow_pickle=False) as graph:
        all_ids = graph["raw_ids"]
    positions = np.searchsorted(all_ids, node_ids)
    np.testing.assert_array_equal(all_ids[positions], node_ids)
    end = 0
    covered = 0
    for chunk in info["chunks"]:
        begin, end = end, end + chunk["rows"]
        selected = np.flatnonzero((positions >= begin) & (positions < end))
        if len(selected):
            path = case / chunk["file"]
            require(path.parent == case and sha(path) == chunk["sha256"], "changed raw feature chunk")
            with np.load(path, allow_pickle=False) as archive:
                actual = archive["x"][positions[selected]-begin]
                np.testing.assert_array_equal(actual.view(np.uint32), features[selected].view(np.uint32))
            covered += len(selected)
    require(covered == len(node_ids), "incomplete original feature coverage")


def source_record(case, cache):
    model = read(case / "model.json")
    dataset = model["dataset"]
    scope = "whole" if case.name == "small-arxiv-p2" else "four"
    resident = cache / f"resident-{dataset}-{scope}"
    info = read(resident / "case.json")
    original = cache / dataset
    require(info["original_checkpoint_sha256"] == sha(case / "weights.npz") == model["weights_sha256"]
            and info["original_graph"] == model["graph"], "original training dataset or checkpoint")
    full = {}
    for filename, expected in (("ptr.npy", model["graph"]["ptr_sha256"]),
                                ("index.npy", model["graph"]["index_sha256"]),
                                ("cpu_input_gpu_hidden.npy", model["gpu_hidden_sha256"]),
                                ("gpu_logits.npy", model["gpu_logits_sha256"])):
        file = original / filename
        require(sha(file) == expected, "changed original graph or GPU reference "+filename)
        full[filename] = np.load(file, mmap_mode="r", allow_pickle=False)
    fp64_hidden = np.load(original / "fp64-hidden.npy", mmap_mode="r", allow_pickle=False)
    ptr, index = full["ptr.npy"], full["index.npy"]
    hidden, logits = full["cpu_input_gpu_hidden.npy"], full["gpu_logits.npy"]
    layers = []
    first_ids = None
    with np.load(case / "weights.npz", allow_pickle=False) as weights:
        for level, name in enumerate(("first.npz", "last.npz")):
            with np.load(case / name, allow_pickle=False) as layer:
                nodes, rows = layer["node"], layer["seed"]
                require(np.all(rows >= 0) and np.all(rows < model["nodes"]), "original seed range")
                flattened = nodes[layer["src"]]
                np.testing.assert_array_equal(layer["ptr"], np.r_[0, np.cumsum(ptr[rows+1]-ptr[rows])])
                expected = np.concatenate([index[ptr[v]:ptr[v+1]] for v in rows])
                np.testing.assert_array_equal(flattened, expected)
                np.testing.assert_array_equal(np.sort(nodes), np.unique(np.r_[rows, expected]))
                np.testing.assert_array_equal(nodes[layer["root"]], rows)
                expected_gpu = (hidden if level == 0 else logits)[rows]
                np.testing.assert_array_equal(layer["gpu"].view(np.uint32), expected_gpu.view(np.uint32))
                for field, key in (("wn", f"neigh{level}.weight"), ("wr", f"root{level}.weight"), ("b", f"neigh{level}.bias")):
                    np.testing.assert_array_equal(layer[field].view(np.uint32), weights[key].T.view(np.uint32))
                if level == 0:
                    first_ids = rows.copy()
                    raw_inputs(nodes, layer["x"], resident, info)
                    np.testing.assert_allclose(layer["fp64"], fp64_hidden[rows], rtol=1e-13, atol=1e-13)
                else:
                    np.testing.assert_array_equal(first_ids, nodes)
                    np.testing.assert_array_equal(layer["x"].view(np.uint32), hidden[nodes].view(np.uint32))
                    # The full-graph FP64 final reference uses FP64 hidden values through both layers.
                    hd = fp64_hidden[nodes]
                    agg = np.zeros((len(rows), hd.shape[1]), dtype=np.float64)
                    for i in range(len(rows)):
                        neighbors = layer["src"][layer["ptr"][i]:layer["ptr"][i+1]]
                        if len(neighbors):
                            agg[i] = hd[neighbors].mean(axis=0, dtype=np.float64)
                    both_layers = hd[layer["root"]] @ np.asarray(layer["wr"],np.float64)
                    both_layers += agg @ np.asarray(layer["wn"],np.float64)
                    both_layers += layer["b"]
                    np.testing.assert_allclose(layer["fp64"], both_layers, rtol=1e-12, atol=1e-12)
                layers.append(dict(rows_sha256=digest(rows), row_pointers_sha256=digest(layer["ptr"]),
                                   original_arc_sources_sha256=digest(flattened), input_feature_sha256=digest(layer["x"]),
                                   fp64_output_sha256=digest(layer["fp64"]), gpu_output_sha256=digest(layer["gpu"])))
    return dict(normalized_csr=model["graph"], weights_sha256=model["weights_sha256"], layers=layers,
                complete_raw_input_case_sha256=sha(resident/"case.json"),
                original_data_sha256=model["graph"]["processed_sha256"],
                complete_gpu_hidden_sha256=model["gpu_hidden_sha256"], complete_gpu_logits_sha256=model["gpu_logits_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    actual = source_record(args.case.resolve(), args.cache.resolve())
    require(actual == read(args.case / "original.json"), "frozen source provenance differs")
    print("verified", args.case.name, sha(args.case / "original.json"))
