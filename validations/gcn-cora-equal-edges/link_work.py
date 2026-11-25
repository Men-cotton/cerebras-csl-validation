"""Count actual vertical packet paths and both endpoints of the full two-layer GCN."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(vector):
    return hashlib.sha256(np.asarray(vector, dtype="<i8").tobytes()).hexdigest()


def packet_counts(arrays, columns, rows):
    # H=16 is transported as two identical T8 tiles, followed by one O7 tile.
    pes = columns * rows
    sent = np.zeros((2, pes, 2), np.int64)
    received = np.zeros_like(sent)
    length_histograms = np.zeros((2, 2, pes, 33), np.int64)
    links = np.zeros((2, 2, columns, rows - 1), np.int64)
    headers = arrays["packet_header"]
    stride = headers.shape[1] // 2
    for pe, packets in enumerate(headers):
        source, column = divmod(pe, columns)
        for phase in range(2):
            start = phase * stride
            for packet in packets[start:start + arrays["counts"][pe, phase]]:
                header = int(packet)
                dest = ((header >> 8) & 4095) - 1
                if (header >> 20) - 4 != column or dest == source or not 0 <= dest < rows:
                    raise ValueError("unexpected vertical packet address")
                words = (header & 31) + 1
                sent[phase, pe] += [1, words]
                received[phase, dest * columns + column] += [1, words]
                length_histograms[0, phase, pe, words] += 1
                length_histograms[1, phase, dest * columns + column, words] += 1
                links[phase, int(dest > source), column, min(source, dest):max(source, dest)] += words
    return sent, received, links, length_histograms


def report(baseline, candidate, sdk_baseline, sdk_nodes_path=None, require_sent=False):
    p = json.loads((baseline / "case.json").read_text())["params"]
    c, r = p["C"], p["R"]
    previous_nodes = np.load(baseline / "nodes.npy")
    nodes = np.load(candidate / "nodes.npy")
    sdk_nodes = previous_nodes if sdk_nodes_path is None else np.load(sdk_nodes_path)
    with np.load(sdk_baseline / "raw.npz") as raw:
        # The native second projection predicates its FMA on pre > 0.
        pre = raw["pre"].reshape(*sdk_nodes.shape, p["H"])
        # Padded owner slots carry -1 and select the trailing zero.
        positives = np.zeros(np.count_nonzero(previous_nodes >= 0) + 1, dtype=np.int64)
        positives[sdk_nodes[sdk_nodes >= 0]] = (pre[sdk_nodes >= 0] > 0).sum(axis=1)
    positive_by_owner = positives[previous_nodes].sum(axis=1)
    np.testing.assert_array_equal(positive_by_owner, positives[nodes].sum(axis=1))
    occupied = (previous_nodes >= 0).sum(axis=1)
    np.testing.assert_array_equal(occupied, (nodes >= 0).sum(axis=1))
    with np.load(baseline / "inputs.npz") as before, np.load(candidate / "inputs.npz") as after:
        unchanged = ["x_ptr", "x_col", "x_val", "w0", "w1", "split_control"]
        for key in unchanged:
            np.testing.assert_array_equal(before[key].view("u1"), after[key].view("u1"), err_msg=key)
        edges = before["edge_ptr"][:, -1]
        np.testing.assert_array_equal(edges, after["edge_ptr"][:, -1])
        tx0, rx0, ld0, hist0 = packet_counts(before, c, r)
        tx1, rx1, ld1, hist1 = packet_counts(after, c, r)
        valid = np.arange(before["edge_src"].shape[1])[None, :] < edges[:, None]
        local_edges = ((before["edge_src"] < p["K"]) & valid).sum(axis=1)
        np.testing.assert_array_equal(local_edges, ((after["edge_src"] < p["K"]) & valid).sum(axis=1))
        if require_sent:
            np.testing.assert_array_equal(before["counts"],after["counts"])
    np.testing.assert_array_equal(rx0, rx1)
    if require_sent:
        np.testing.assert_array_equal(tx0, tx1)
        np.testing.assert_array_equal(hist0, hist1)
    np.testing.assert_array_equal(tx0.sum(axis=1), tx1.sum(axis=1))
    np.testing.assert_array_equal(ld0.sum(axis=(1, 2, 3)), ld1.sum(axis=(1, 2, 3)))
    def full(a):
        return 2 * a[0] + a[1]
    def network(ld):
        loads = full(ld)
        return dict(peak_words=int(loads.max()), hops=int(loads.sum()),
                    peak_directed_links=[dict(column=int(x), from_row=int(y + 1 - direction),
                                              to_row=int(y + direction))
                                         for direction, x, y in np.argwhere(loads == loads.max())])
    before_peak = network(ld0)
    after_peak = network(ld1)
    if before_peak["peak_words"] == after_peak["peak_words"]:
        raise ValueError("unchanged maximum directed-link load")
    changed = np.flatnonzero(np.any(tx0 != tx1, axis=(0, 2)))
    sent0, sent1 = full(tx0), full(tx1)
    loads0, loads1 = full(ld0), full(ld1)
    changes = []
    for direction, col, y in np.argwhere(loads0 != loads1):
        edge = (direction, col, y)
        changes.append(dict(column=int(col),from_row=int(y+1-direction),to_row=int(y+direction),
                            baseline_words=int(loads0[edge]),candidate_words=int(loads1[edge])))
    return dict(
        schema=1, dimensions=[c, r], baseline_inputs_sha256=sha(baseline / "inputs.npz"),
        candidate_inputs_sha256=sha(candidate / "inputs.npz"), baseline_sdk_raw_sha256=sha(sdk_baseline / "raw.npz"),
        original_edge_occurrences=int(edges.sum()), equal_by_pe=dict(
            destination_edge_counts=edges.tolist(), output_vertices_i64_sha256=digest(occupied),
            same_row_edges_i64_sha256=digest(local_edges),
            positive_hidden_count_i64_sha256=digest(positive_by_owner),
            packet_receives_by_phase_i64_sha256=digest(rx0), byte_identical_uploads=unchanged),
        packet_sends_by_phase_i64_sha256=[digest(tx0), digest(tx1)],
        packet_length_histograms_by_endpoint_phase_and_pe_i64_sha256=[digest(hist0),digest(hist1)],
        packet_totals_by_tile=dict(hidden_T8=tx0[0].sum(axis=0).tolist(), output_O7=tx0[1].sum(axis=0).tolist(),
                                   unit_order=["packets", "words"], tile_multiplicity=[2,1]),
        full_sent_packets_words=sent0.sum(axis=0).tolist(), baseline_network=before_peak, candidate_network=after_peak,
        sent_changes=[dict(x=int(pe % c), y=int(pe // c), baseline_packets_words=sent0[pe].tolist(),
                           candidate_packets_words=sent1[pe].tolist()) for pe in changed],
        changed_directed_links=len(changes), link_changes=changes,
        scope="Decoded addressed vertical packet traffic from the uploads. Row collectives and split transfers use the same geometry and schedule.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "sdk_baseline", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--sdk-nodes",type=Path,help="original ID layout of the SDK capture")
    parser.add_argument("--require-same-sends", action="store_true")
    args = parser.parse_args()
    result = report(args.baseline, args.candidate, args.sdk_baseline,args.sdk_nodes,args.require_same_sends)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
