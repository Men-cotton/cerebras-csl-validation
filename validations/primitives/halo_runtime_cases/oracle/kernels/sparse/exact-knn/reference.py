"""Independent host references, authored locally without upstream imports."""
import math
from functools import cmp_to_key
import numpy as np


def global_sort(scores, ids, k):
    # This reference deliberately does not call insertion selection, merge,
    # Candidate.key, or the wire encoder.
    order = sorted(range(len(scores)), key=lambda i: (float(scores[i]), int(ids[i])))[:k]
    return np.asarray([ids[i] for i in order], np.uint32), np.asarray([scores[i] for i in order])


def fp64_scores(database, query):
    return np.array([math.fsum((float(value)-float(query[dim]))**2
                              for dim, value in enumerate(row)) for row in database], np.float64)


def fp64_topk(database, query, ids, k):
    return global_sort(fp64_scores(database, query), ids, k)


def global_sort_counted(scores, ids, k):
    comparisons = 0
    def compare(left, right):
        nonlocal comparisons
        comparisons += 1
        a, b = (float(scores[left]), int(ids[left])), (float(scores[right]), int(ids[right]))
        return (a > b) - (a < b)
    order = sorted(range(len(scores)), key=cmp_to_key(compare))[:k]
    return [int(ids[i]) for i in order], comparisons


def scalar_fp32_scores(database, query):
    """Separate arithmetic witness using Python doubles and explicit casts.

    It avoids calling the implementation scorer, its validation, or reduction.
    """
    answer = []
    for row in database:
        running = 0.0
        for left, right in zip(row.tolist(), query.tolist()):
            delta = float(np.float32(left-right))
            product = float(np.float32(delta*delta))
            running = float(np.float32(running+product))
        answer.append(running)
    return np.asarray(answer, np.float32)
