"""Validate grouped GCN uploads with disjoint horizontal partial reductions on original Cora."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def audit_uploads(grouped, params, arrays, nodes, original, plans):
    c, r, k, l = (params[key] for key in ("C", "R", "K", "L"))
    active_pe, slot = np.nonzero(nodes >= 0)
    n = len(original["x"])
    np.testing.assert_array_equal(np.sort(nodes[active_pe, slot]), np.arange(n))
    owner = np.empty(n, np.int64)
    owner[nodes[active_pe, slot]] = active_pe
    row_sizes = np.bincount(owner // c, minlength=r)
    if row_sizes.max() != k:
        raise ValueError("incorrect original source-row capacity")
    extended_nodes = np.pad(nodes, ((0, 0), (0, 1)), constant_values=-1)
    extended_slots = np.pad(arrays["slots"], ((0, 0), (0, 1)), constant_values=65535)
    destinations = original["dst"].astype(np.int64).copy()
    roles = np.zeros((c*r, 2), np.uint16)
    splits_per_row = np.zeros(r, np.int64)
    helpers = []
    for i, plan in enumerate(plans):
        hub, home, helper, moved = (int(plan[key]) for key in ("hub", "home", "helper", "moved"))
        if not (0 <= hub < n and 0 <= home < helper < c*r and home//c == helper//c and owner[hub] == home):
            raise ValueError("incorrect original hub or horizontal helper")
        if np.any(roles[home:helper+1, 0]):
            raise ValueError("overlapping horizontal partial reductions")
        occurrences = np.flatnonzero(original["dst"] == hub)
        if not 0 < moved < len(occurrences):
            raise ValueError("empty owner prefix or suffix")
        stable_rows = np.argsort(owner[original["src"][occurrences]] // c, kind="stable")
        virtual = n + i
        destinations[occurrences[stable_rows[-moved:]]] = virtual
        roles[home] = [1, int(np.flatnonzero(nodes[home] == hub)[0])]
        roles[home+1:helper, 0] = 3
        roles[helper, 0] = 2
        row = home // c
        extended_nodes[helper, -1] = virtual
        extended_slots[helper, -1] = row_sizes[row] + splits_per_row[row]
        splits_per_row[row] += 1
        helpers.append(helper)
    np.testing.assert_array_equal(arrays["split_control"], roles)
    # Virtual leaves have zero source features and are never referenced by an edge.
    # Move only the local/remote index boundary for the ordinary grouped audit.
    new_k = int((row_sizes + splits_per_row).max())
    checked = dict(arrays, slots=extended_slots)
    if new_k > k:
        checked["x_ptr"] = np.pad(arrays["x_ptr"], ((0, 0), (0, new_k-k)), mode="edge")
        checked["edge_src"] = arrays["edge_src"].copy()
        remote = checked["edge_src"] >= k
        checked["edge_src"][remote] += new_k - k
    features = np.pad(original["x"], ((0, len(plans)), (0, 0)))
    groups, stats, traffic = grouped.audit_uploads(
        dict(params, K=new_k, L=l+1), checked, extended_nodes, dict(original, x=features, dst=destinations))
    for helper in helpers:
        stats[helper, 6] -= params["H"] * params["O"]
    return [group[group < n] for group in groups], stats, traffic


def certify(reference, candidate):
    sys.path.insert(0, str(reference.parent.parent))
    import grouped_contract
    import hub_contract
    import saved_audit as host
    manifest = host.load(reference)
    params = manifest["params"]
    original = host.native_inputs(reference, manifest)
    nodes = host.load_npz(reference / "node_ids.npz")["nodes"]
    spec = json.loads((candidate / "case.json").read_text())
    plans = spec["split_plans"]
    if (host.sha(reference / "inputs.npz") != spec["parent_inputs_sha256"]
            or host.sha(candidate / "inputs.npz") != spec["packed_inputs_sha256"]):
        raise ValueError("different immutable input archives")
    np.testing.assert_array_equal(nodes, np.load(candidate / "nodes.npy"))
    one = host.load_npz(reference / "inputs.npz")
    old = hub_contract.audit_uploads(params, one, nodes, original)
    checked = audit_uploads(grouped_contract, params, one, nodes, original, plans[:1])
    for a, b in zip(old[0], checked[0], strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(old[1], checked[1])
    np.testing.assert_array_equal(old[2], checked[2])
    two = host.load_npz(candidate / "inputs.npz")
    groups, stats, traffic = audit_uploads(grouped_contract, params, two, nodes, original, plans)
    lanes = params["H"] + params["O"]
    np.testing.assert_array_equal(stats[:, 5], two["edge_ptr"][:, -1].astype(np.uint32) * lanes)
    helper = plans[-1]["helper"]
    first_partial = two["edge_ptr"][helper, params["L"]]
    two["edge_weight"][helper, first_partial] += np.float32(0.001)
    try:
        audit_uploads(grouped_contract, params, two, nodes, original, plans)
    except ValueError as exc:
        if str(exc) != "ordered edge identity":
            raise
    else:
        raise ValueError("altered helper coefficient passed the independent upload audit")
    return dict(original_nodes=sum(map(len, groups)), ordered_occurrences=int(stats[:, 5].sum() // lanes),
                grouped_auditor_sha256=host.sha(Path(grouped_contract.__file__)),
                reference_case_sha256=host.sha(reference / "case.json"), candidate_case_sha256=host.sha(candidate / "case.json"),
                parent_input_sha256=spec["parent_inputs_sha256"], candidate_input_sha256=spec["packed_inputs_sha256"],
                stats_sha256=hashlib.sha256(stats.tobytes()).hexdigest(), traffic_sha256=hashlib.sha256(traffic.tobytes()).hexdigest(),
                corrupted_helper_edge_rejected=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path, help="frozen single-hub case in the h16-hub-cora watcher family")
    p.add_argument("candidate", type=Path, help="multiple-hub input from multi_split.py")
    p.add_argument("out", type=Path)
    args = p.parse_args()
    certificate = certify(args.reference.resolve(), args.candidate.resolve())
    args.out.write_text(json.dumps(certificate, indent=2) + "\n")
    print(certificate["original_nodes"], certificate["ordered_occurrences"])
