"""Validate source-star rounds independently of the scheduling algorithm.

The caller must first validate pair counts against the original graph. This
module checks only their round assignment: every source occurs exactly once,
active closed stars have disjoint endpoints, and every PE has its exact role.
Idle PEs still participate in caller completion barriers; idle is encoded by P.
"""
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def uint_array(arrays, name, shape):
    require(name in arrays and isinstance(arrays[name], np.ndarray) and
            arrays[name].shape == shape and arrays[name].dtype == np.dtype("uint32"),
            "round array shape/dtype: " + name)
    return arrays[name]


def validate_rounds(arrays, p):
    """Return Q and each PE's source in the last round, including idle P.

    Source s has the closed star {s} union its nonempty destination owners.
    Distinct stars within a round cannot share a source or receiver endpoint.
    Every owner must act as source once, even if it has no remote destinations;
    its local-only edge copies remain part of its source turn. No greedy order
    or minimal coloring is required by this correctness check.
    """
    for name in ("C", "R", "Q"):
        require(type(p.get(name)) is int and p[name] > 0, "positive integer " + name)
    require(type(p.get("MODE")) is int and p["MODE"] == 2, "colored MODE2 required")
    owners, rounds = p["C"] * p["R"], p["Q"]
    require(owners <= 32767 and rounds <= owners, "round count bound 1<=Q<=P")
    begin = uint_array(arrays, "tx_begin", (owners, owners + 1))
    receive = uint_array(arrays, "rx_count", (owners, owners))
    assignments = uint_array(arrays, "round_sources", (owners, rounds))
    require(bool((assignments <= owners).all()), "round source/idle range")
    transmit = np.diff(begin.astype(np.int64), axis=1)
    require(bool((begin[:, 0] == 0).all()) and bool((transmit >= 0).all()), "transmit prefix bounds")
    require(np.array_equal(transmit, receive.T), "transmit/receive pair counts disagree")
    require(bool((np.diag(transmit) == 0).all()), "local copies must not be network destinations")

    source_roles = assignments == np.arange(owners, dtype=np.uint32)[:, None]
    require(bool((source_roles.sum(axis=1) == 1).all()), "each source owner must occur exactly once")
    source_round = np.argmax(source_roles, axis=1)
    counts = []
    for round_index in range(rounds):
        active = np.flatnonzero(source_roles[:, round_index])
        require(len(active) > 0, "empty source-star round")
        expected = np.full(owners, owners, np.uint32)
        occupied = np.zeros(owners, bool)
        for source in active:
            destinations = np.flatnonzero(receive[:, source] > 0)
            endpoints = np.concatenate((np.array([source]), destinations))
            require(not bool(occupied[endpoints].any()), "overlapping closed source stars")
            occupied[endpoints] = True
            expected[endpoints] = source
        require(np.array_equal(assignments[:, round_index], expected),
                "round source/receiver/idle assignment differs")
        counts.append(int(len(active)))
    return dict(round_count=rounds, last_source=assignments[:, -1].tolist(),
                source_round=source_round.tolist(), active_sources_per_round=counts,
                scope="Graph-validated pairs scheduled exactly once with disjoint source/receiver endpoints; no minimal-coloring or timing claim")
