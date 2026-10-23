"""Receiver ownership and explicit traffic census; no graph normalization."""
import numpy as np


def owners(positions, columns, rows, method):
    n, parts = len(positions), columns * rows
    if n == 0 or min(columns, rows) < 2:
        raise ValueError("nonempty graph and at least 2x2 owners required")
    result = np.empty(n, dtype=np.int32)
    if method == "id":
        for p, ids in enumerate(np.array_split(np.arange(n), parts)):
            result[ids] = p
    elif method == "spatial":
        # Recursive rectangular bisection maps neighboring coordinate regions
        # to neighboring PE rectangles. Original IDs break coordinate ties.
        def split(ids, x, y, c, r):
            if c == r == 1:
                result[ids] = y * columns + x
                return
            axis = 0 if c > 1 and (r == 1 or c >= r) else 1
            extent = c if axis == 0 else r
            left = extent // 2
            ordered = ids[np.lexsort((ids, positions[ids, axis]))]
            cut = len(ids) * left // extent
            if axis == 0:
                split(ordered[:cut], x, y, left, r)
                split(ordered[cut:], x+left, y, c-left, r)
            else:
                split(ordered[:cut], x, y, c, left)
                split(ordered[cut:], x, y+left, c, r-left)
        split(np.arange(n), 0, 0, columns, rows)
    else:
        raise ValueError("unknown partition")
    return result


def census(graph, owner, columns, rows, h):
    n, p = len(owner), columns*rows
    src, dst = graph["senders"], graph["receivers"]
    edges = len(src)
    nodes_per = np.bincount(owner, minlength=p)
    edges_per = np.bincount(owner[dst], minlength=p)
    # A remote source can serve several receiver edges at one owner.
    deliveries = sorted({(int(s), int(owner[d])) for s, d in zip(src, dst)
                         if owner[s] != owner[d]})
    links = {}
    hops = 0
    for s, target in deliveries:
        x, y = int(owner[s]) % columns, int(owner[s]) // columns
        tx, ty = target % columns, target // columns
        while x != tx:
            nx = x + (1 if tx > x else -1)
            key = (x, y, nx, y)
            links[key] = links.get(key, 0) + h
            x = nx
            hops += h
        while y != ty:
            ny = y + (1 if ty > y else -1)
            key = (x, y, x, ny)
            links[key] = links.get(key, 0) + h
            y = ny
            hops += h
    l = int(nodes_per.max())
    return dict(nodes=n, directed_edges=edges, owners=p,
                node_counts=nodes_per.tolist(), edge_counts=edges_per.tolist(),
                max_nodes=l, max_edges=int(edges_per.max()),
                edge_imbalance=float(edges_per.max()/(edges/p)) if edges else 0.,
                cut_edges=int(np.sum(owner[src] != owner[dst])),
                distinct_remote_source_deliveries=len(deliveries),
                selective_xy_word_hops=hops,
                selective_xy_max_link_words=max(links.values(), default=0),
                selective_scope="modeled source deduplication and XY unicast, excludes headers/contention; not implemented",
                broadcast_panel_words=p*l*h,
                broadcast_received_words_per_pe=(p-1)*l*h,
                broadcast_processed_panel_words_per_pe=p*l*h,
                broadcast_tree_word_hops=p*l*h*(p-1),
                broadcast_scope="implemented all-owner padded panels per processor block, excludes DONE/RELEASE",
                degree_min=int(np.bincount(dst, minlength=n).min()),
                degree_max=int(np.bincount(dst, minlength=n).max()))


def pack(graph, columns, rows, h, weights, method):
    owner = owners(graph["mesh_pos"], columns, rows, method)
    m = census(graph, owner, columns, rows, h)
    p, l, e = columns*rows, max(1, m["max_nodes"]), max(1, m["max_edges"])
    node_ids = np.full((p, l), -1, np.int32)
    edge_ids = np.full((p, e), -1, np.int32)
    slots = np.empty(len(owner), np.uint32)
    for pe in range(p):
        ids = np.flatnonzero(owner == pe)
        node_ids[pe, :len(ids)] = ids
        slots[ids] = np.arange(len(ids))
        ids = np.flatnonzero(owner[graph["receivers"]] == pe)
        ids = ids[np.argsort(owner[graph["senders"][ids]],kind="stable")]
        edge_ids[pe, :len(ids)] = ids
    arrays = dict(counts=np.array([m["node_counts"], m["edge_counts"]], np.uint32).T.copy(),
                  velocity=np.zeros((p,l*2), np.float32), node_type=np.zeros((p,l), np.uint32),
                  edge_features=np.zeros((p,e*3), np.float32),
                  src_owner=np.zeros((p,e),np.uint32), src_slot=np.zeros((p,e),np.uint32),
                  dst_slot=np.zeros((p,e),np.uint32),
                  src_begin=np.zeros((p,p+1),np.uint32),
                  weights=np.tile(weights.reshape(1,-1), (p,1)))
    for pe in range(p):
        ni, ei = node_ids[pe], edge_ids[pe]
        ns, es = ni[ni >= 0], ei[ei >= 0]
        arrays["velocity"][pe,:len(ns)*2] = graph["velocity"][ns].ravel()
        arrays["node_type"][pe,:len(ns)] = graph["node_type"].ravel()[ns]
        arrays["edge_features"][pe,:len(es)*3] = graph["edge_features"][es].ravel()
        arrays["src_owner"][pe,:len(es)] = owner[graph["senders"][es]]
        arrays["src_slot"][pe,:len(es)] = slots[graph["senders"][es]]
        arrays["dst_slot"][pe,:len(es)] = slots[graph["receivers"][es]]
        counts=np.bincount(owner[graph["senders"][es]],minlength=p)
        arrays["src_begin"][pe,1:]=np.cumsum(counts,dtype=np.uint32)
    return arrays, node_ids, edge_ids, m


def induced(graph, limit):
    """Prefix induced subgraph for calibration only; full cases use all vertices."""
    n = len(graph["mesh_pos"])
    if limit is None or limit >= n:
        return graph
    if limit <= 0:
        raise ValueError("positive vertex limit")
    keep = (graph["senders"] < limit) & (graph["receivers"] < limit)
    return dict(mesh_pos=graph["mesh_pos"][:limit], velocity=graph["velocity"][:limit],
                node_type=graph["node_type"][:limit], senders=graph["senders"][keep],
                receivers=graph["receivers"][keep], edge_features=graph["edge_features"][keep])
