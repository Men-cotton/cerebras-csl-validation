"""Assign destination rows to West ports using only the graph and the transfer format."""

from collections import Counter
import numpy as np


def assignments(z, lanes, owners, entries):
    degrees = np.diff(np.asarray(z["ptr"], dtype=np.int64))
    sources = [set(map(int, z["src"][lo:hi])) for lo, hi in zip(z["ptr"][:-1], z["ptr"][1:])]
    contiguous = [rows.tolist() for rows in np.array_split(np.arange(len(degrees)), lanes)]
    capacity = list(map(len, contiguous))
    balanced = [[] for _ in range(lanes)]
    edges = [0] * lanes
    for row in np.argsort(-degrees, kind="stable"):
        available = [i for i in range(lanes) if len(balanced[i]) < capacity[i]]
        index = min(available, key=lambda i: (edges[i], len(balanced[i]), i))
        balanced[index].append(int(row))
        edges[index] += int(degrees[row])

    def counts(groups):
        refs = [Counter(s for row in group for s in sources[row]) for group in groups]
        arcs = [int(degrees[group].sum()) for group in groups]
        return refs, [len(v) for v in refs], arcs

    def score(unique, arcs):
        # The remaining start/end controls and cached weights are independent of row assignment.
        f = z["x"].shape[1]
        payload = [
            u * f + a + ((u + entries - 1) // entries) * (1 + owners) + rows * (f + 2)
            for u, a, rows in zip(unique, arcs, capacity)
        ]
        return max(payload), sum(payload), max(arcs), sum(n * n for n in payload)

    def cost(groups):
        _, unique, arcs = counts(groups)
        return score(unique, arcs)

    groups = [rows.copy() for rows in min([contiguous, balanced], key=cost)]
    for _ in range(24):
        refs, unique, arcs = counts(groups)
        best = score(unique, arcs)
        best_swap = None
        keys = list(map(set, refs))
        lost = {
            row: {s for s in sources[row] if refs[lane][s] == 1}
            for lane, rows in enumerate(groups)
            for row in rows
        }
        added = [[len(s - keys[lane]) for s in sources] for lane in range(lanes)]
        for left in range(lanes):
            for right in range(left):
                for a in groups[left]:
                    for b in groups[right]:
                        # Sources shared by a and b remain present after swapping them.
                        u = unique.copy()
                        u[left] += added[left][b] - len(lost[a]) + len(lost[a] & sources[b])
                        u[right] += added[right][a] - len(lost[b]) + len(lost[b] & sources[a])
                        e = arcs.copy()
                        edge_delta = int(degrees[b] - degrees[a])
                        e[left] += edge_delta
                        e[right] -= edge_delta
                        candidate = score(u, e)
                        if candidate < best:
                            best = candidate
                            best_swap = (left, right, a, b)
        if best_swap is None:
            break
        left, right, a, b = best_swap
        groups[left].remove(a)
        groups[right].remove(b)
        groups[left].append(b)
        groups[right].append(a)
        if cost(groups) != best:
            raise ValueError("incremental partition objective changed")
    return dict(
        contiguous=contiguous,
        balanced=[sorted(rows) for rows in balanced],
        dictionary=[sorted(rows) for rows in groups],
    )
