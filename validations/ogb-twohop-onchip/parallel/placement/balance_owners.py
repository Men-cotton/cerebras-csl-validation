"""Find the smallest legal peak input across both rows of a compiled 16-owner image."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np

from two_rows import device, packets, prepare, trial_inputs


def search(first, last, build):
    desc = json.loads((build / "build.json").read_text())
    cfg = desc["first"]
    if cfg["P"] != 16 or (device.sha(first), device.sha(last)) != (
        desc["first_sha256"],
        desc["last_sha256"],
    ):
        raise ValueError(
            "expected the matching original graph and a 16-owner two-row build"
        )
    data = prepare(first, last, cfg["P"], cfg["E"])
    z, owner = data[0], data[4]
    width, entries, channels = 8, cfg["E"], cfg["F"]
    sites = np.arange(16)
    degree = np.diff(z["ptr"])
    edge_owner = np.repeat(owner, degree)
    src = z["src"]
    arc_count = np.bincount(edge_owner, minlength=16)
    row_count = np.bincount(owner, minlength=16)
    row_mask = np.left_shift(1, sites)
    source_mask = np.zeros(len(z["node"]), np.int32)
    np.bitwise_or.at(source_mask, src, row_mask[edge_owner])
    used = source_mask[source_mask != 0]
    # Subset sums count sources whose owners all lie in the complementary row.
    subset_counts = np.bincount(used, minlength=1 << 16)
    for i in sites:
        groups = subset_counts.reshape(-1, 2, 1 << i)
        groups[:, 1, :] += groups[:, 0, :]
    selector_caps = []
    for i in range(2):
        text = (build / "src" / f"owner{i}.w").read_text()
        matches = re.findall(r"u32 (?:selectors|ignored)\[n=(\d+)\];", text)
        if len(matches) != 2:
            raise ValueError("unknown compiled selector bank")
        selector_caps.append(min(map(int, matches)))

    def volume(mask, arcs, rows):
        n = len(used) - int(subset_counts[65535 ^ mask])
        blocks = (n + entries - 1) // entries
        count = (
            3
            + width
            + arcs
            + rows * (2 + channels)
            + n * channels
            + blocks * (1 + width)
        )
        return count + count % 2

    def maxima(mask):
        sizes = []
        for lane_mask in (mask, 65535 ^ mask):
            selected = (row_mask[edge_owner] & lane_mask) != 0
            owners = edge_owner[selected]
            ids, inv = np.unique(src[selected], return_inverse=True)
            blocks = max(1, (len(ids) + entries - 1) // entries)
            slots = owners.astype(np.int64) * blocks + inv // entries
            sizes.append(int(np.bincount(slots).max(initial=0)))
        return sizes

    splits = []
    total_arcs, total_rows = len(src), len(owner)
    for left in combinations(range(16), width):
        idx = list(left)
        mask = int(row_mask[idx].sum())
        arcs, rows = int(arc_count[idx].sum()), int(row_count[idx].sum())
        words = (
            volume(mask, arcs, rows),
            volume(65535 ^ mask, total_arcs - arcs, total_rows - rows),
        )
        splits.append((max(words), sum(words), left, words, mask))
    splits.sort()
    for rank, (_, _, left, words, mask) in enumerate(splits, 1):
        banks = maxima(mask)
        if all(need <= cap for need, cap in zip(banks, selector_caps)):
            break
    else:
        raise ValueError("original allocation must fit its own firmware")
    order = [*left, *[int(i) for i in sites if i not in left]]
    spec = trial_inputs(build, data, order)
    old = list(map(len, packets(spec[7][-1], 7)))
    actual = list(map(len, packets(spec[8][-1], 8)))
    if list(words) != actual or max(actual) > max(old):
        raise ValueError("graph cost disagrees with the actual warmed streams")
    return dict(
        first_sha256=device.sha(first),
        last_sha256=device.sha(last),
        build_sha256=device.sha(build / "build.json"),
        legal_rank=rank,
        possible_assignments=len(splits),
        original_words=old,
        balanced_words=actual,
        compiled_selector_caps=selector_caps,
        balanced_selector_maxima=banks,
        order=order,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "last", "build"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(search(args.first, args.last, args.build), indent=2, allow_nan=False)
    )
