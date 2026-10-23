"""Color source stars into endpoint-disjoint sparse-delivery rounds.

The existing directed delivery schedule determines each source's closed star:
the source PE and all destination PEs that consume its features. Disjoint stars
may send during one round. Payloads, per-edge sender caches, and the order of
GNN edge arithmetic are unchanged. Color count is an analytical schedule size,
not a prediction of cycles, network contention, or simulator time.
"""

import numpy as np


MAX_OWNERS = 128


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build(arrays, columns):
    """Return ``round_sources[P,Q]`` and a deterministic coloring census.

    An active source and each of its remote destinations store that source's ID
    in the round column. Other PEs store the idle sentinel P. Each source,
    including sources without remote work, occurs in exactly one round. The
    source's local sender-cache copies can therefore retain their current
    ownership and occur once per processor block.
    """
    require(type(columns) is int and columns > 0, "positive integer columns")
    require(isinstance(arrays, dict) and "tx_begin" in arrays and "rx_count" in arrays,
            "transmit and receive schedules required")
    begin, receive = arrays["tx_begin"], arrays["rx_count"]
    require(isinstance(begin, np.ndarray) and begin.dtype == np.dtype("uint32")
            and begin.ndim == 2, "uint32 transmit prefix matrix")
    pes = begin.shape[0]
    require(0 < pes <= MAX_OWNERS and pes % columns == 0 and begin.shape == (pes, pes + 1),
            "bounded rectangular owner grid")
    require(isinstance(receive, np.ndarray) and receive.dtype == np.dtype("uint32")
            and receive.shape == (pes, pes), "uint32 receive count matrix")
    pair_counts = np.diff(begin.astype(np.int64), axis=1)
    require(bool((begin[:, 0] == 0).all()) and bool((pair_counts >= 0).all())
            and bool((begin[:, -1] <= 32767).all()), "bounded monotone transmit prefixes")
    require(bool((np.diag(pair_counts) == 0).all()), "local features cannot be remote deliveries")
    require(np.array_equal(pair_counts.T, receive), "transmit/receive pair counts disagree")

    closed = [set(np.flatnonzero(pair_counts[source] > 0).tolist()) | {source}
              for source in range(pes)]
    conflicts = [set() for _ in range(pes)]
    for first in range(pes):
        for second in range(first + 1, pes):
            if not closed[first].isdisjoint(closed[second]):
                conflicts[first].add(second)
                conflicts[second].add(first)
    work = pair_counts.sum(axis=1).tolist()
    order = sorted(range(pes), key=lambda source: (-len(conflicts[source]), -work[source], source))
    colors = [-1] * pes
    for source in order:
        forbidden = {colors[neighbor] for neighbor in conflicts[source] if colors[neighbor] >= 0}
        color = 0
        while color in forbidden:
            color += 1
        colors[source] = color
    rounds = max(colors) + 1
    packed = np.full((pes, rounds), pes, np.uint32)
    sources_per_round = [[] for _ in range(rounds)]
    for source, color in enumerate(colors):
        sources_per_round[color].append(source)
        for endpoint in closed[source]:
            require(packed[endpoint, color] == pes, "internal coloring endpoint collision")
            packed[endpoint, color] = source
    require(1 <= rounds <= pes and all(sources_per_round), "nonempty bounded round coverage")
    require(all(np.count_nonzero(packed[source] == source) == 1 for source in range(pes)),
            "each source owns exactly one round")
    last_active = [int(row[np.flatnonzero(row != pes)[-1]]) for row in packed]
    return dict(round_sources=packed), dict(
        schema=1, Q=rounds, owners=pes, columns=columns, rows=pes // columns,
        idle=pes, colors=colors, greedy_order=order,
        closed_endpoints=[sorted(star) for star in closed],
        conflict_counts=[len(group) for group in conflicts],
        source_unique_vertices=work,
        round_sources=sources_per_round,
        round_active_endpoint_counts=np.count_nonzero(packed != pes, axis=0).tolist(),
        final_round_sources=packed[:, -1].tolist(),
        last_active_sources=last_active,
        sources_covered=pes, source_round_assignments=sum(map(len, sources_per_round)),
        remote_pairs_covered=int(np.count_nonzero(pair_counts)), endpoint_collisions=0,
        schedule_bytes_per_pe=rounds * np.dtype("uint32").itemsize,
        scope="analytical deterministic closed-source-star coloring; "
              "endpoint exclusion and source coverage checked; no routing-link exclusion or cycle prediction",
    )


def crossing_graph():
    """ID4x2: 0->3 and 1->2 share link1->2 with disjoint endpoints.

    Each PE owns eight nodes. The lower row has no edges. Reciprocal pairs
    change endpoint roles in the second color. Pair lengths7/8 and5/3 test
    multiple K6 packets, unequal lengths, and short tails; one edge repeats.
    """
    n = 64
    positions = np.array([[i % 8, i // 8] for i in range(n)], np.float32)
    senders = np.array(list(range(7))+list(range(8,16))+list(range(24,29))+list(range(16,19))+[0],np.int32)
    receivers = np.array([24]*7+[16]*8+[0]*5+[8]*3+[24],np.int32)
    delta = positions[senders]-positions[receivers]
    features = np.concatenate((delta,np.sqrt(np.sum(delta*delta,axis=1))[:,None]),axis=1)
    velocity = (np.arange(n*2,dtype=np.float32).reshape(n,2)%13-6)/32
    node_type = np.zeros((n,1),np.int32)
    node_type[32:]=6
    return dict(mesh_pos=positions,velocity=velocity,node_type=node_type,
                senders=senders,receivers=receivers,edge_features=features)
