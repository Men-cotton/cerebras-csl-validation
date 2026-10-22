"""Pack exact sparse feature delivery while preserving the owned edge order.

Every remote (source vertex, destination PE) appears once on the wire. An edge
keeps its own sender-cache entry, so duplicate edges retain their multiplicity.
The traffic metadata models Y-then-X routes, four application-header words and
one SDK routing-header word per packet. It does not measure routing, contention,
completion traffic, or simulator time.
"""

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive(value, name):
    _require(type(value) is int and value > 0, "positive integer " + name)
    return value


def _ids(value, name):
    _require(isinstance(value, np.ndarray) and value.ndim == 2
             and value.dtype.kind == "i" and min(value.shape) > 0,
             "signed two-dimensional " + name)
    # Each owner has one valid prefix; a hole would invalidate the device count.
    for row in value:
        count = int(np.count_nonzero(row >= 0))
        _require(bool((row[:count] >= 0).all()) and bool((row[count:] == -1).all()),
                 "valid prefix and -1 padding in " + name)
    return value


def _route(source, target, columns):
    """Yield directed links on the configured analytical Y-then-X path."""
    x, y = source % columns, source // columns
    tx, ty = target % columns, target // columns
    while y != ty:
        ny = y + (1 if ty > y else -1)
        yield (x, y, x, ny)
        y = ny
    while x != tx:
        nx = x + (1 if tx > x else -1)
        yield (x, y, nx, y)
        x = nx


def build(graph, node_ids, edge_ids, columns, hidden, batch):
    """Return immutable-input schedule arrays and analytical traffic metadata.

    ``tx_begin[source, target:target+2]`` indexes ``tx_slots[source]``.
    The slots in each pair are sorted and unique. ``rx_count[target, source]``
    is that pair's length; ``rx_slot[target, edge]`` selects the vertex within
    the pair before chunking. Local edges and padding have zero rx_slot.

    A packet contains at most ``batch * hidden`` payload words, with a 27-word
    payload limit. All arrays use uint32 host carriers. No input is changed.
    """
    _positive(columns, "columns")
    _positive(hidden, "hidden")
    _positive(batch, "batch")
    _require(hidden <= 27 and batch * hidden <= 27, "packet payload exceeds 27 words")
    node_ids, edge_ids = _ids(node_ids, "node_ids"), _ids(edge_ids, "edge_ids")
    pes, local_nodes = node_ids.shape
    _require(pes == edge_ids.shape[0] and pes % columns == 0,
             "owner grid and node/edge mappings disagree")
    _require(max(pes, local_nodes, edge_ids.shape[1]) <= 32767,
             "mapping exceeds signed device index bound")
    positions = np.asarray(graph["mesh_pos"])
    _require(positions.ndim == 2 and positions.shape[1] == 2 and len(positions) > 0,
             "nonempty two-dimensional mesh positions")
    nodes = len(positions)
    senders, receivers = np.asarray(graph["senders"]), np.asarray(graph["receivers"])
    _require(senders.ndim == 1 and receivers.shape == senders.shape
             and senders.dtype.kind in "iu" and receivers.dtype.kind in "iu",
             "integer directed edge arrays")
    _require(bool(((senders >= 0) & (senders < nodes)).all())
             and bool(((receivers >= 0) & (receivers < nodes)).all()),
             "edge endpoint outside graph")
    _require(np.array_equal(np.sort(node_ids[node_ids >= 0]), np.arange(nodes)),
             "node mapping must contain every node exactly once")
    _require(np.array_equal(np.sort(edge_ids[edge_ids >= 0]), np.arange(len(senders))),
             "edge mapping must contain every occurrence exactly once")

    owner = np.empty(nodes, np.int32)
    slot = np.empty(nodes, np.int32)
    for pe, row in enumerate(node_ids):
        valid = row[row >= 0]
        owner[valid] = pe
        slot[valid] = np.arange(len(valid))

    pairs = [[set() for _ in range(pes)] for _ in range(pes)]
    local_edges = np.zeros(pes, np.uint32)
    remote_edges = np.zeros(pes, np.uint32)
    for target, row in enumerate(edge_ids):
        for edge in row[row >= 0]:
            _require(int(owner[receivers[edge]]) == target,
                     "edge mapping violates receiver ownership")
            source = int(owner[senders[edge]])
            if source == target:
                local_edges[target] += 1
            else:
                pairs[source][target].add(int(slot[senders[edge]]))
                remote_edges[target] += 1

    pairs = [[sorted(values) for values in row] for row in pairs]
    totals = [sum(map(len, row)) for row in pairs]
    capacity = max(1, max(totals))
    _require(capacity <= 32767, "send schedule exceeds signed device index bound")
    tx_begin = np.zeros((pes, pes + 1), np.uint32)
    tx_slots = np.zeros((pes, capacity), np.uint32)
    rx_count = np.zeros((pes, pes), np.uint32)
    rx_slot = np.zeros(edge_ids.shape, np.uint32)
    for source in range(pes):
        offset = 0
        for target in range(pes):
            values = pairs[source][target]
            tx_slots[source, offset:offset + len(values)] = values
            offset += len(values)
            tx_begin[source, target + 1] = offset
            rx_count[target, source] = len(values)
    for target, row in enumerate(edge_ids):
        indices = [{value: index for index, value in enumerate(pairs[source][target])}
                   for source in range(pes)]
        for edge_slot, edge in enumerate(row):
            if edge < 0:
                break
            source = int(owner[senders[edge]])
            if source != target:
                rx_slot[target, edge_slot] = indices[source][int(slot[senders[edge]])]

    packets_out = [0] * pes
    packets_in = [0] * pes
    payload_out = [0] * pes
    payload_in = [0] * pes
    payload_links, header_links = {}, {}
    max_pair = 0
    for source in range(pes):
        for target in range(pes):
            count = len(pairs[source][target])
            max_pair = max(max_pair, count)
            packets = (count + batch - 1) // batch
            words = count * hidden
            packets_out[source] += packets
            packets_in[target] += packets
            payload_out[source] += words
            payload_in[target] += words
            if count:
                for link in _route(source, target, columns):
                    payload_links[link] = payload_links.get(link, 0) + words
                    header_links[link] = header_links.get(link, 0) + 5 * packets

    arrays = dict(tx_begin=tx_begin, tx_slots=tx_slots, rx_count=rx_count, rx_slot=rx_slot)
    metadata = dict(
        schema=1, T=capacity, K=batch, batch=batch, hidden=hidden,
        owners=pes, columns=columns, rows=pes // columns,
        max_pair_vertices=max_pair, remote_unique=sum(totals),
        local_edge_counts=local_edges.tolist(), remote_edge_counts=remote_edges.tolist(),
        packet_counts_source=packets_out, packet_counts_sink=packets_in,
        source_payload_words=payload_out, sink_payload_words=payload_in,
        packet_count=sum(packets_out), payload_words=sum(payload_out),
        application_header_words=4 * sum(packets_out),
        routing_header_words=sum(packets_out),
        header_words=5 * sum(packets_out),
        yx_payload_word_hops=sum(payload_links.values()),
        yx_application_header_word_hops=4 * sum(header_links.values()) // 5,
        yx_routing_header_word_hops=sum(header_links.values()) // 5,
        yx_header_word_hops=sum(header_links.values()),
        yx_max_link_payload_words=max(payload_links.values(), default=0),
        yx_max_link_header_words=max(header_links.values(), default=0),
        yx_max_link_total_words=max((value + header_links[link]
                                     for link, value in payload_links.items()), default=0),
        schedule_array_bytes_per_pe=sum(value.shape[1] * value.dtype.itemsize
                                        for value in arrays.values()),
        traffic_scope="analytical Y-then-X payload, four-word application headers "
                      "and one-word SDK routing headers; excludes completion/control "
                      "traffic, contention and timing",
    )
    return arrays, metadata


def adversarial_graph():
    """Directed fixture with duplicates, fanout, self edge and an isolated node.

    With ID ownership on a 4x2 grid, each node gets one owner. Edge 0->7 takes
    four Y-then-X hops, including owners that do not consume source 0. With a 2x2
    grid, edges 0->6 and 0->7 share a source/destination packet entry. Owner 5
    is isolated on 4x2, while larger grids also contain empty owners.
    """
    positions = np.array([[0, 0], [1, 0], [2, 0], [3, 0],
                          [0, 1], [1, 1], [2, 1], [3, 1]], np.float32)
    senders = np.array([0, 0, 0, 0, 1, 7, 2, 3, 4], np.int32)
    receivers = np.array([7, 7, 6, 1, 0, 0, 2, 2, 3], np.int32)
    delta = positions[senders] - positions[receivers]
    features = np.concatenate((delta, np.sqrt(np.sum(delta * delta, axis=1))[:, None]), axis=1)
    velocity = np.array([[0.125, -0.25], [0.5, 0.125], [1, 0], [0, 0],
                         [-0.25, 0.125], [0.0, 0.0], [0.75, -0.125], [0.1, 0.2]], np.float32)
    return dict(mesh_pos=positions, velocity=velocity,
                node_type=np.array([[0], [5], [4], [6], [0], [6], [0], [5]], np.int32),
                senders=senders, receivers=receivers, edge_features=features)
