"""Compare returned Cora captures only after independently auditing both appliance runs."""

import argparse
import json
from pathlib import Path

import numpy as np

def split_effects(control, split, roles, nodes, columns):
    owner, = np.flatnonzero(roles == 1)
    helper, = np.flatnonzero(roles == 2)
    peers = np.setdiff1d(np.arange(len(nodes)), [owner, helper])
    before = control["stats"].astype(np.int64)
    after = split["stats"].astype(np.int64)
    np.testing.assert_array_equal(before[:, [0, 1, 2, 4, 6, 7]], after[:, [0, 1, 2, 4, 6, 7]])
    shift = after[:, 5] - before[:, 5]
    assert shift[owner] == -shift[helper] < 0 and not shift[peers].any()
    elapsed = []
    for raw in (control, split):
        times = raw["clocks"].astype(np.int64).reshape(-1, 2, 3) @ np.array([1, 65536, 1 << 32])
        elapsed.append((times[:, 1] - times[:, 0]) % (1 << 48))
    delta = elapsed[1] - elapsed[0]
    layers = {}
    valid = nodes >= 0
    for field in ("pre", "logits"):
        ordered = []
        for raw in (control, split):
            tile_values = raw[field].reshape(*nodes.shape, -1)
            original = np.empty((int(valid.sum()), tile_values.shape[-1]), dtype=np.float32)
            original[nodes[valid]] = tile_values[valid]
            ordered.append(original)
        normal, divided = ordered
        np.testing.assert_allclose(normal, divided, atol=3e-5, rtol=2e-4)
        layers[field] = dict(fp32_different=int(np.count_nonzero(normal.view("<u4") != divided.view("<u4"))),
                             max_abs=float(np.max(np.abs(normal.astype(np.float64) - divided))))
        if field == "logits":
            np.testing.assert_array_equal(normal.argmax(1), divided.argmax(1))
    return dict(moved_edge_macs=int(shift[helper]), owner_xy=[int(owner % columns), int(owner // columns)],
                helper_xy=[int(helper % columns), int(helper // columns)],
                owner_cycles=[int(v[owner]) for v in elapsed], helper_cycles=[int(v[helper]) for v in elapsed],
                other_pes=len(peers), other_pes_faster=int(np.count_nonzero(delta[peers] < 0)),
                other_pe_median_cycle_change=float(np.median(delta[peers])),
                critical_pe_ids=[[int(i) for i in np.flatnonzero(v == v.max())] for v in elapsed],
                max_pe_cycles=[int(v.max()) for v in elapsed], original_id_outputs=layers, all_node_class_mismatches=0)


def compare(control, split):
    import saved_audit as native
    captures = (control, split)
    for path in captures:
        native.require(native.sha(Path(__file__)) == native.sha(path / "host/compare_pair.py"), "physical comparison program identity")
    traces = [native.audit_report(path) for path in captures]
    native.require(all(t["backend"] == "hardware" and t["status"] == "PASS" for t in traces), "audited physical controls")
    native.require(traces[0]["params"] == traces[1]["params"] and traces[0]["nodes"] == traces[1]["nodes"] == 2708,
                   "same full Cora and compiled dimensions")
    inventories = [native.read_json(path / "build/inventory.json", limit=native.INVENTORY_JSON_LIMIT) for path in captures]
    digests = [{name: field["sha256"] for name, field in item["files"].items()} for item in inventories]
    rpc = [native.read_json(path / "build/bin/out_rpc.json") for path in captures]
    native.require(rpc[0] == rpc[1], "physical RPC identity")
    for filename in ("source/layout.csl", "source/pe.csl", "dataset/cora.cites", "dataset/cora.content",
                     "checkpoint.npz", "node_ids.npz"):
        native.require(native.sha(control / "case" / filename) == native.sha(split / "case" / filename),
                       "matching frozen "+filename)
    # Split roles are independently reconstructed from the original COO by audit_report.
    roles = native.load_npz(split / "case/inputs.npz")["split_control"][:, 0]
    hops = 23 * int(np.count_nonzero((roles == 2) | (roles == 3)))
    arrays = [native.load_npz(path / "runtime/raw.npz") for path in captures]
    for raw, wanted_hops in zip(arrays, (0, hops)):
        np.testing.assert_array_equal(raw["split_words"].sum(axis=0), [wanted_hops, wanted_hops])
        native.require(int(raw["stats"][:, 4:7].sum()) == 1395824, "complete logical work")
    cycles = [t["local_interval_min_max"][1] for t in traces]
    same_images = digests[0] == digests[1]
    nodes = native.load_npz(control / "case/node_ids.npz")["nodes"]
    effects = split_effects(*arrays, roles, nodes, traces[0]["params"]["C"])
    native.require(effects["max_pe_cycles"] == cycles, "signed interval reconstruction")
    return dict(status="PASS", same_physical_elf_bytes=same_images, identical_rpc=True,
                cycles={"control": cycles[0], "split": cycles[1]}, speedup=cycles[0] / cycles[1],
                scope="same physical images" if same_images else "matched parameters and source; physical ELF hashes differ", effects=effects,
                numerical_status=[t["numerical_status"] for t in traces], compiled_elfs=[len(d) for d in digests])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = compare(args.control, args.split)
    args.out.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(value))


if __name__ == "__main__":
    main()
