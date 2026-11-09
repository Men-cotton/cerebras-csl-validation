"""Paired host streams and an independent reference for optimized Shah SpMM."""

import numpy as np


END_ROW = 0xfffffffe
EMPTY = 0xffffffff
DONE = 0xffff


def _validate_params(params):
    for name in ("R", "F", "V", "Q", "B", "N", "K"):
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if params["Q"] & (params["Q"] - 1):
        raise ValueError("Q must be a power of two")
    if params["N"] != params["B"] * params["Q"]:
        raise ValueError("N must contain exactly B padded chunks")
    if not (params["R"] - 1) * params["V"] < params["K"] <= params["R"] * params["V"]:
        raise ValueError("K must occupy the declared source shards")
    if params["V"] > DONE or params["K"] >= END_ROW:
        raise ValueError("source indices collide with control markers")


def _edges(params, edges):
    result = []
    for row, col, value in edges:
        if (isinstance(row, bool) or isinstance(col, bool) or
                not isinstance(row, (int, np.integer)) or
                not isinstance(col, (int, np.integer)) or
                not 0 <= row < params["N"] or not 0 <= col < params["K"]):
            raise ValueError("invalid edge index")
        value = np.float32(value)
        if not np.isfinite(value):
            raise ValueError("edge values must be finite f32")
        result.append((int(row), int(col), value))
    return result


def pack(params, edges):
    """Emit one rectangular paired-u32 stream per chunk; retain duplicate edges."""
    _validate_params(params)
    edges = _edges(params, edges)
    grouped = [[[] for _ in range(params["N"])] for _ in range(params["R"])]
    for row, col, value in edges:
        grouped[col // params["V"]][row].append((col, int(value.view(np.uint32))))
    chunks = []
    for chunk in range(params["B"]):
        shards = []
        for shard in grouped:
            records = []
            for row in range(chunk * params["Q"], (chunk + 1) * params["Q"]):
                records.extend(shard[row])
                if records and records[-1][0] == END_ROW:
                    records[-1] = (END_ROW, records[-1][1] + 1)
                else:
                    records.append((END_ROW, 1))
            shards.append(records)
        capacity = max(map(len, shards))
        array = np.zeros((params["R"], capacity, 2), np.uint32)
        array[:, :, 0] = EMPTY
        for shard, records in enumerate(shards):
            array[shard, :len(records)] = records
        chunks.append(array.reshape(params["R"], 2 * capacity))
    return chunks


def router_events(params, shard, words):
    """Drain every pair and expand each valid row run to individual DONE events."""
    _validate_params(params)
    if not 0 <= shard < params["R"]:
        raise ValueError("invalid router shard")
    if not isinstance(words, np.ndarray) or words.dtype != np.uint32 or words.ndim != 1 or len(words) % 2:
        raise ValueError("stream must contain paired u32 words")
    row, padded, events = 0, False, []
    for raw_index, raw_bits in words.reshape(-1, 2):
        index, bits = int(raw_index), int(raw_bits)
        if index == EMPTY:
            if row != params["Q"] or bits:
                raise ValueError("invalid terminal padding")
            padded = True
        elif padded:
            raise ValueError("data after padding")
        elif index == END_ROW:
            if not 1 <= bits <= params["Q"] - row:
                raise ValueError("invalid row run")
            events.extend((DONE, None) for _ in range(bits))
            row += bits
        else:
            if (row >= params["Q"] or not shard * params["V"] <= index <
                    min((shard + 1) * params["V"], params["K"])):
                raise ValueError("invalid source ownership or row")
            value = np.uint32(bits).view(np.float32)
            if not np.isfinite(value):
                raise ValueError("invalid value bits")
            events.append((index - shard * params["V"], value))
    if row != params["Q"]:
        raise ValueError("incomplete chunk")
    return events


def h_tiles(params, h):
    """Return worker H as (source shard, feature, local source), with zero tails."""
    _validate_params(params)
    if h.shape != (params["K"], params["F"]) or h.dtype != np.float32 or not np.all(np.isfinite(h)):
        raise ValueError("invalid H")
    padded = np.zeros((params["R"] * params["V"], params["F"]), np.float32)
    padded[:params["K"]] = h
    return padded.reshape(params["R"], params["V"], params["F"]).transpose(0, 2, 1).copy()


def decode(params, streams, h):
    """Model worker arithmetic and ordered north-to-south partial reduction."""
    tiles = h_tiles(params, h)
    if len(streams) != params["B"]:
        raise ValueError("incorrect number of chunks")
    output = np.zeros((params["N"], params["F"]), np.float32)
    for chunk, words in enumerate(streams):
        if not isinstance(words, np.ndarray) or words.ndim != 2 or words.shape[0] != params["R"]:
            raise ValueError("invalid router rectangle")
        partial = np.zeros((params["R"], params["Q"], params["F"]), np.float32)
        for shard in range(params["R"]):
            row = 0
            for index, value in router_events(params, shard, words[shard]):
                if index == DONE:
                    row += 1
                else:
                    partial[shard, row] += value * tiles[shard, :, index]
        result = np.zeros((params["Q"], params["F"]), np.float32)
        for shard in range(params["R"]):
            result += partial[shard]
        output[chunk * params["Q"]:(chunk + 1) * params["Q"]] = result
    return output


def sparse_reference(params, edges, h):
    """Accumulate original edge occurrences in f64 without reading the streams."""
    _validate_params(params)
    h_tiles(params, h)
    result = np.zeros((params["N"], params["F"]), np.float64)
    for row, col, value in _edges(params, edges):
        result[row] += float(value) * h[col].astype(np.float64)
    return result


def metrics(params, edges, streams):
    """Count host payloads and logical work; exclude SDK headers and code storage."""
    _validate_params(params)
    edges = _edges(params, edges)
    per_shard = np.zeros(params["R"], np.int64)
    for _, col, _ in edges:
        per_shard[col // params["V"]] += 1
    markers = padding = 0
    for words in streams:
        for shard in range(params["R"]):
            router_events(params, shard, words[shard])
        markers += int(np.count_nonzero(words[:, ::2] == END_ROW))
        padding += int(np.count_nonzero(words[:, ::2] == EMPTY))
    nnz = len(edges)
    return {
        "nnz_occurrences": nnz,
        "nnz_per_worker_row": per_shard.tolist(),
        "fmacs_per_worker_in_each_row": per_shard.tolist(),
        "max_to_mean_worker_work": float(per_shard.max() / per_shard.mean()) if nnz else 0.0,
        "h2d_stream_calls": params["B"],
        "d2h_output_calls": 1,
        "h2d_stream_payload_bytes": sum(words.nbytes for words in streams),
        "h2d_edge_payload_bytes": 8 * nnz,
        "h2d_row_marker_pairs": markers,
        "h2d_padding_pairs": padding,
        "h2d_padding_bytes": 8 * padding,
        "h_upload_padded_payload_bytes": 4 * params["R"] * params["V"] * params["F"],
        "d2h_output_payload_bytes": 4 * params["N"] * params["F"],
        "row_broadcast_edge_deliveries": nnz * params["F"],
        "worker_done_deliveries": params["N"] * params["R"] * params["F"],
        "worker_reduction_fadds": params["N"] * params["F"] * (params["R"] - 1),
        "worker_reduction_word_hops": params["N"] * params["F"] * (params["R"] - 1),
        "worker_h_plus_one_chunk_bytes": 4 * (params["V"] + params["Q"]),
        "accumulator_bytes_per_pe": 4 * params["Q"],
        "accumulator_rows": params["B"],
        "active_role_pes": params["R"] * (params["F"] + 1) + params["B"] * params["F"],
        "scope": "host payload and arithmetic model; buffer lower bounds exclude SDK, code, stack and routing state",
    }


def fixture(R=2, F=2, V=3, Q=2, B=3, empty=False):
    params = dict(R=R, F=F, V=V, Q=Q, B=B, N=B * Q, K=R * V - 1)
    _validate_params(params)
    # Binary fractions keep tiny protocol checks insensitive to FMA contraction.
    h = ((np.arange(params["K"] * F).reshape(params["K"], F) % 13 - 6) / 8).astype(np.float32)
    edges = []
    if not empty:
        edges = [(0, 0, 1.0), (0, 0, -0.25),
                 (0, min(V - 1, params["K"] - 1), 0.5),
                 (params["N"] - 1, params["K"] - 1, -2.0)]
        # Concentrate extra work on the first source shard; intermediate shards
        # remain empty when R > 2, and whole output rows remain empty when N > 2.
        if params["N"] > 2:
            edges.extend((min(Q, params["N"] - 1), col, (col % 3 + 1) / 4)
                         for col in range(min(V, params["K"])))
    streams = pack(params, edges)
    reference = sparse_reference(params, edges, h)
    return dict(params=params, edges=edges, h=h, streams=streams,
                reference=reference, metrics=metrics(params, edges, streams))


def random_fixture(R, F, V, Q, B, seed=7, degree=8):
    """Generate bounded random f32 inputs with duplicate edges and uneven work.

    Degree counts occurrences, so it may exceed K. Row zero has 4*degree
    occurrences; the last row has 2*degree; other nonempty rows have degree.
    Every seventh row starting at row three is empty, except the last row.
    Source shard one is empty when R >= 4. Reject, rather than clamp, inputs
    whose conservative bounds exceed one million edges or eight million array
    elements (each H/output array and the total padded stream words).
    """
    params = dict(R=R, F=F, V=V, Q=Q, B=B, N=B * Q, K=R * V - 1)
    _validate_params(params)
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError("degree must be a positive occurrence count")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    edge_bound = (params["N"] + 4) * int(degree)
    if (edge_bound > 1_000_000 or params["K"] * F > 8_000_000 or
            params["N"] * F > 8_000_000 or
            2 * R * (edge_bound + params["N"]) > 8_000_000):
        raise ValueError("random fixture exceeds host allocation bounds")
    rng = np.random.default_rng(seed)
    h = rng.uniform(-1.0, 1.0, (params["K"], F)).astype(np.float32)
    sources = np.arange(params["K"])
    if R >= 4:
        sources = sources[(sources < V) | (sources >= 2 * V)]
    edges = []
    for row in range(params["N"]):
        if row % 7 == 3 and row != params["N"] - 1:
            continue
        count = degree * (4 if row == 0 else 2 if row == params["N"] - 1 else 1)
        columns = rng.choice(sources, count)
        first_shard = rng.random(count) < 0.75
        columns[first_shard] = rng.integers(0, min(V, params["K"]), int(first_shard.sum()))
        if row == 0:
            columns[:2] = 0
        if row == params["N"] - 1:
            columns[-1] = params["K"] - 1
        values = rng.uniform(-1.0, 1.0, count).astype(np.float32)
        edges.extend((row, int(col), float(value)) for col, value in zip(columns, values))
    streams = pack(params, edges)
    return dict(params=params, edges=edges, h=h, streams=streams,
                reference=sparse_reference(params, edges, h),
                metrics=metrics(params, edges, streams))
