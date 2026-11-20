"""Audit an ordered split of Cora vertex 163 across two PEs in one row."""

import numpy as np

import grouped_contract as grouped


def schema(params, output=False):
    fields = grouped.schema(params, output)
    if output:
        fields["split_words"] = [2, "uint32"]
    else:
        fields["edge_ptr"] = [params["L"] + 2, "uint16"]
        fields["split_control"] = [2, "uint16"]
    return fields


def named_bytes(params):
    # One extra row pointer, two control halfwords, two counters and one accumulator tile.
    return grouped.named_bytes(params) + 2 + 4 + 8 + 4 * params["T"] + 4 * params["K"] * (params["H"] - params["T"])


def split_words(params, arrays):
    roles = arrays["split_control"][:, 0]
    outgoing = (roles == 2) | (roles == 3)
    incoming = (roles == 1) | (roles == 3)
    return np.column_stack((outgoing, incoming)).astype(np.uint32) * (params["H"] + params["O"])


def audit_uploads(params, arrays, nodes, raw):
    columns = params["C"]
    hub = 163
    owners, slots = np.nonzero(nodes >= 0)
    owner = np.empty(len(raw["x"]), np.int64)
    np.testing.assert_array_equal(np.sort(nodes[owners, slots]), np.arange(len(owner)))
    owner[nodes[owners, slots]] = owners
    home = int(owner[hub])
    row = home // columns
    degree = np.bincount(owner[raw["dst"]], minlength=len(nodes))
    occurrences = np.flatnonzero(raw["dst"] == hub)
    source_rows = owner[raw["src"][occurrences]] // columns
    ordered = occurrences[np.argsort(source_rows, kind="stable")]
    # Minimize the maximum PE edge count, then distance, then imbalance of the pair.
    plans = []
    all_pes = np.arange(len(nodes))
    for helper in range(home + 1, (row + 1) * columns):
        other_peak = int(degree[(all_pes != home) & (all_pes != helper)].max())
        for moved in range(1, len(ordered)):
            pair_peak = max(int(degree[home]) - moved, int(degree[helper]) + moved)
            plans.append((max(other_peak, pair_peak), helper - home, pair_peak, moved))
    grouped.require(plans, "same-row split target")
    _, distance, _, moved = min(plans)
    helper = home + distance
    enabled = bool(arrays["split_control"].any())
    expected_roles = np.zeros((len(nodes), 2), np.uint16)
    destinations = raw["dst"].astype(np.int64).copy()
    virtual = len(owner)
    if enabled:
        expected_roles[home] = [1, int(np.flatnonzero(nodes[home] == hub)[0])]
        expected_roles[home + 1:helper, 0] = 3
        expected_roles[helper, 0] = 2
        destinations[ordered[-moved:]] = virtual
    np.testing.assert_array_equal(arrays["split_control"], expected_roles)

    # The ordinary grouped audit can check every original occurrence and transport
    # slot if the helper's extra aggregate is represented as a zero-feature vertex.
    # It is introduced only inside this check, never in the FP64 model or device inputs.
    extended_nodes = np.pad(nodes, ((0, 0), (0, 1)), constant_values=-1)
    extended_nodes[helper, -1] = virtual
    row_sizes = (nodes >= 0).sum(axis=1).reshape(params["R"], columns).sum(axis=1)
    grouped.require(int(row_sizes.max()) == params["K"], "original row extent")
    extended_slots = np.pad(arrays["slots"], ((0, 0), (0, 1)), constant_values=65535)
    extended_slots[helper, -1] = row_sizes[row]
    new_k = max(params["K"], int(row_sizes[row]) + 1)
    index_shift = new_k - params["K"]
    features = np.pad(raw["x"], ((0, 1), (0, 0)))
    checked_arrays = dict(arrays, slots=extended_slots)
    if index_shift:
        checked_arrays["x_ptr"] = np.pad(arrays["x_ptr"], ((0, 0), (0, index_shift)), mode="edge")
        src = arrays["edge_src"].copy()
        src[src >= params["K"]] += index_shift
        checked_arrays["edge_src"] = src
    groups, stats, traffic = grouped.audit_uploads(
        dict(params, K=new_k, L=params["L"] + 1), checked_arrays, extended_nodes,
        dict(raw, x=features, dst=destinations))
    stats[helper, 6] -= params["H"] * params["O"]
    groups[row] = groups[row][groups[row] != virtual]
    return groups, stats, traffic
