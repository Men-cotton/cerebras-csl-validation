"""Original host contract for declared-order FP32 scoring and exact Top-K.

This module does not import the challenge packet, a grader, an SDK, or BLAS.
P denotes a number of source shards, not a P-by-P mesh.
"""
from dataclasses import dataclass
import numpy as np


INVALID_ID = 0xffffffff
MAX_ID = INVALID_ID - 1
PAIR_BYTES = 8


class ScoreOverflow(ValueError):
    def __init__(self, phase, row, dimension):
        self.phase, self.row, self.dimension = phase, row, dimension
        super().__init__(f"nonfinite {phase} at row {row}, dimension {dimension}")


def integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(name + " outside admitted range")
    return value


def admit_inputs(database, query):
    database, query = np.asarray(database), np.asarray(query)
    if (database.dtype != np.float32 or query.dtype != np.float32 or
            database.ndim != 2 or query.ndim != 1 or query.size == 0 or
            database.shape[1] != query.size):
        raise ValueError("database/query must be matching nonzero-width float32 arrays")
    if not np.isfinite(database).all() or not np.isfinite(query).all():
        raise ValueError("database/query must be finite")
    return database, query


def admit_ids(ids, count):
    raw = list(range(count)) if ids is None else list(ids)
    if len(raw) != count:
        raise ValueError("ID count does not match rows")
    values = [integer(value, "ID", 0, MAX_ID) for value in raw]
    if len(set(values)) != count:
        raise ValueError("original IDs must be unique")
    return np.asarray(values, dtype=np.uint32)


def checked(value, phase, row, dimension):
    value = np.float32(value)
    if not np.isfinite(value):
        raise ScoreOverflow(phase, row, dimension)
    return value


def direct_scores(database, query):
    """Round subtraction, square, and sum separately, dimensions 0..d-1.

    Reject overflow at every operation. Underflow follows NumPy float32;
    subnormal preservation/flush behavior on any future device is unproven.
    """
    database, query = admit_inputs(database, query)
    scores = np.empty(len(database), np.float32)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for row in range(len(database)):
            total = np.float32(0)
            for dim in range(query.size):
                delta = checked(database[row, dim]-query[dim], "difference", row, dim)
                square = checked(delta*delta, "square", row, dim)
                total = checked(total+square, "sum", row, dim)
            scores[row] = total
    return scores


def norm_dot_scores(database, query):
    """Separate experimental formulation; no clamp or implicit admission."""
    database, query = admit_inputs(database, query)
    scores = np.empty(len(database), np.float32)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        qnorm = np.float32(0)
        for dim in range(query.size):
            term = checked(query[dim]*query[dim], "query-square", -1, dim)
            qnorm = checked(qnorm+term, "query-norm-sum", -1, dim)
        for row in range(len(database)):
            norm, dot = np.float32(0), np.float32(0)
            for dim in range(query.size):
                term = checked(database[row, dim]*database[row, dim], "row-square", row, dim)
                norm = checked(norm+term, "row-norm-sum", row, dim)
                term = checked(database[row, dim]*query[dim], "dot-product", row, dim)
                dot = checked(dot+term, "dot-sum", row, dim)
            norms = checked(norm+qnorm, "norm-add", row, query.size)
            doubled = checked(np.float32(2)*dot, "double-dot", row, query.size)
            score = checked(norms-doubled, "norm-minus-dot", row, query.size)
            if score < 0:
                raise ValueError(f"negative norm-dot score at row {row}")
            scores[row] = score
    return scores


@dataclass(frozen=True)
class Candidate:
    distance: np.float32
    original_id: int

    @property
    def key(self):
        return (float(self.distance), self.original_id)


PADDING = Candidate(np.float32(np.inf), INVALID_ID)


def validate_pairs(pairs, k=None, sorted_required=False):
    if k is not None and len(pairs) != k:
        raise ValueError("candidate capacity mismatch")
    seen = set()
    previous = None
    for pair in pairs:
        identity = integer(pair.original_id, "candidate ID", 0, INVALID_ID)
        if not isinstance(pair.distance, np.float32):
            raise ValueError("candidate distance must be float32")
        if identity == INVALID_ID:
            if not np.isposinf(pair.distance):
                raise ValueError("padding must be (+inf, UINT32_MAX)")
        else:
            if not np.isfinite(pair.distance) or pair.distance < 0 or identity in seen:
                raise ValueError("invalid distance or duplicate candidate ID")
            seen.add(identity)
        if sorted_required and previous is not None and pair.key < previous:
            raise ValueError("candidate list is not ordered")
        previous = pair.key
    return seen


def local_topk(scores, ids, k):
    k = integer(k, "K", 1, MAX_ID)
    scores = np.asarray(scores)
    if scores.dtype != np.float32 or scores.ndim != 1:
        raise ValueError("scores must be a float32 vector")
    ids = admit_ids(ids, len(scores))
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("scores must be finite and nonnegative")
    best, comparisons = [PADDING]*k, 0
    for score, identity in zip(scores, ids):
        candidate = Candidate(score, int(identity))
        for index in range(k):
            comparisons += 1
            if candidate.key < best[index].key:
                best.insert(index, candidate)
                best.pop()
                break
    return best, comparisons


def merge_topk(left, right, k):
    k = integer(k, "K", 1, MAX_ID)
    left_ids = validate_pairs(left, k, True)
    right_ids = validate_pairs(right, k, True)
    if left_ids & right_ids:
        raise ValueError("candidate ownership overlaps")
    result, i, j, comparisons = [], 0, 0, 0
    while len(result) < k:
        comparisons += 1
        if left[i].key <= right[j].key:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    return result, comparisons


def encode_pairs(pairs):
    validate_pairs(pairs)
    words = np.empty((len(pairs), 2), np.uint32)
    for index, pair in enumerate(pairs):
        words[index, 0] = np.asarray(pair.distance).view(np.uint32)
        words[index, 1] = pair.original_id
    return words


def decode_pairs(words):
    words = np.asarray(words)
    if words.dtype != np.uint32 or words.ndim != 2 or words.shape[1] != 2:
        raise ValueError("pair wire format is [K,2] uint32")
    pairs = [Candidate(np.array(row[0], np.uint32).view(np.float32)[()], int(row[1]))
             for row in words]
    validate_pairs(pairs)
    return pairs


def shard_rows(count, partitions):
    partitions = integer(partitions, "partitions", 1, MAX_ID)
    width = (count+partitions-1)//partitions
    return [list(range(min(i*width, count), min((i+1)*width, count))) for i in range(partitions)]


def distributed_topk(scores, ids, k, partitions, arrival=None, ownership=None):
    scores = np.asarray(scores)
    ids = admit_ids(ids, len(scores))
    k = integer(k, "K", 1, len(scores))
    partitions = integer(partitions, "partitions", 1, MAX_ID)
    ownership = shard_rows(len(scores), partitions) if ownership is None else ownership
    if len(ownership) != partitions:
        raise ValueError("wrong shard count")
    flattened = [integer(row, "owned row", 0, len(scores)-1) for shard in ownership for row in shard]
    if sorted(flattened) != list(range(len(scores))):
        raise ValueError("shards must partition every row exactly once")
    arrival = list(range(partitions)) if arrival is None else list(arrival)
    arrival = [integer(shard, "arrival shard", 0, partitions-1) for shard in arrival]
    if sorted(arrival) != list(range(partitions)):
        raise ValueError("arrival must be a shard permutation")
    local, local_counts = [], []
    for rows in ownership:
        selected, count = local_topk(scores[rows], ids[rows], k)
        local.append(selected)
        local_counts.append(count)
    merged, merge_counts = local[arrival[0]], []
    for shard in arrival[1:]:
        merged, comparisons = merge_topk(merged, local[shard], k)
        merge_counts.append(comparisons)
    ledger = dict(partitions=partitions, rows_per_shard=list(map(len, ownership)),
                  local_comparisons_per_shard=local_counts, local_comparisons=sum(local_counts),
                  merge_comparisons=sum(merge_counts), total_comparisons=sum(local_counts)+sum(merge_counts),
                  candidate_storage_bytes_per_shard=PAIR_BYTES*k,
                  candidate_storage_bytes_all_shards=PAIR_BYTES*k*partitions,
                  candidate_payload_bytes_per_nonroot_shard=PAIR_BYTES*k,
                  candidate_payload_bytes_to_root=PAIR_BYTES*k*(partitions-1),
                  maximum_local_selection_live_pair_bytes=PAIR_BYTES*(k+1),
                  maximum_merge_live_pair_bytes=3*PAIR_BYTES*k if partitions > 1 else PAIR_BYTES*k,
                  additional_merge_output_scratch_bytes=PAIR_BYTES*k if partitions > 1 else 0,
                  final_output_bytes=PAIR_BYTES*k,
                  scope="modeled fixed-length candidate buffers/direct-to-root payload; excludes Python objects, inputs, query, scalar state, routes and SDK; comparisons exclude validation")
    return merged, ledger
