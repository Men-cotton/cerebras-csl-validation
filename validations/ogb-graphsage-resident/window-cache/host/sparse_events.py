"""Count exactly preplanned cache hits and only fills that a later edge consumes."""

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

from traffic import (
    HERE,
    OGB,
    assign,
    full_csr,
    load_case,
    read,
    regions,
    region_words,
    require,
    sha,
)

P32 = ctypes.POINTER(ctypes.c_int32)
P16 = ctypes.POINTER(ctypes.c_uint16)
P64 = ctypes.POINTER(ctypes.c_int64)
INT = ctypes.c_int


def ptr(a, t):
    return a.ctypes.data_as(t)


def build_lib(directory):
    out = directory / "events.so"
    subprocess.run(
        [
            "cc",
            "-O2",
            "-std=c99",
            "-fPIC",
            "-shared",
            str(HERE / "sparse_events.c"),
            "-o",
            str(out),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(out))
    lib.visit.argtypes = [P32, INT, INT, INT, INT, INT, INT, P32, P32, P16, P16]
    lib.visit.restype = None
    lib.dma.argtypes = [P16, P16, INT, INT, INT, INT, P64, P64]
    lib.dma.restype = None
    return lib


def dma(lib, data, lanes, *, payload=None, penalty=256):
    require(
        data.dtype == np.uint16 and data.flags.c_contiguous and 0 < data.shape[1] <= 32,
        "DMA geometry",
    )
    require(penalty >= 0, "nonnegative DMA selection cost")
    if payload is not None:
        require(
            payload.shape == data.shape
            and payload.dtype == np.uint16
            and payload.flags.c_contiguous
            and not np.any((data == 0) & (payload != 0)),
            "alternate packet region geometry",
        )
    volume = ctypes.c_int64()
    calls = ctypes.c_int64()
    lib.dma(
        ptr(data, P16),
        ptr(payload, P16) if payload is not None else None,
        len(data),
        data.shape[1],
        lanes,
        penalty,
        ctypes.byref(volume),
        ctypes.byref(calls),
    )
    return dict(words=volume.value, calls=calls.value)


def update(lib, seq, e, pe, p, block, cap, tag, pending, reads, writes):
    seq = np.asarray(seq, dtype=np.int32, order="C")
    lib.visit(
        ptr(seq, P32),
        len(seq),
        e,
        pe,
        p,
        block,
        cap,
        ptr(tag, P32),
        ptr(pending, P32),
        ptr(reads, P16),
        ptr(writes, P16),
    )


def selftest(lib):
    rng = np.random.default_rng(7701)
    a = rng.integers(0, 270, (400, 32), dtype=np.uint16)
    a[a % 5 == 0] = 0
    for f in (1, 9, 16):
        expected = [regions(tuple(map(int, row)), f) for row in a]
        require(
            dma(lib, a, f)
            == dict(
                words=sum(region_words(x, f) for x in expected),
                calls=sum(map(len, expected)),
            ),
            "compiled DMA calculation differs from actual resident grouping",
        )
    replacement = np.minimum(a, rng.integers(0, 173, a.shape, dtype=np.uint16))
    old = [regions(tuple(int(v) for v in row), 16) for row in a]
    actual = sum(
        (end - begin) * int(new[begin:end].max())
        for cuts, new in zip(old, replacement)
        for begin, end, _ in cuts
    )
    require(
        dma(lib, a, 16, payload=replacement)
        == dict(words=actual, calls=sum(map(len, old))),
        "padding variable packets on the original native region boundaries",
    )
    # A hit crosses a window. An overwritten candidate and an unreused final miss trigger no fill.
    tags = np.full((2, 4), -1, np.int32)
    fresh = tags.copy()
    reads = np.zeros((3, 2), np.uint16)
    writes = reads.copy()
    update(lib, [3, 5, 7], 2, 0, 2, 0, 4, tags, fresh, reads, writes)
    update(lib, [5, 1], 2, 0, 2, 2, 4, tags, fresh, reads, writes)
    update(lib, [5, 1], 2, 1, 2, 0, 4, tags, fresh, reads, writes)
    require(
        reads[2, 0] == 1
        and reads.sum() == 1
        and writes[0, 0] == 1
        and writes.sum() == 1,
        "sparse fill/read ordering",
    )


def run(a, lib):
    dataset, scope, cache = a.dataset, a.scope, a.cache
    baseline = read(a.baseline)
    cfg = baseline["params"]
    require(
        (baseline["dataset"], baseline["scope"]) == (dataset, scope),
        "original census alignment",
    )
    p, r, e, f, q = (cfg[k] for k in ("P", "R", "E", "F", "Q"))
    require(0 < p <= 32 and e < 256 and f > 1, "one explicit event word")
    record = read(OGB / "evidence" / dataset / "model.json")
    if scope == "whole":
        n = record["nodes"]
        ptr0, idx = full_csr(cache / dataset, record["graph"], n)
        views = [(n, ptr0, idx, None)] * 2
    else:
        _, g = load_case(cache / f"resident-{dataset}-{scope}")
        views = [
            (len(g["a_ids"]), g["a_ptr"], g["a_src"], g["raw_ids"]),
            (len(g["b_ids"]), g["b_ptr"], g["b_src"], g["a_ids"]),
        ]
    caps = (16, 32, 64, 128, 256)
    result = dict(
        dataset=dataset,
        scope=scope,
        baseline_sha256=sha(a.baseline),
        params=cfg,
        phases={},
    )
    for mode, (size, rowptr, source, mapping) in enumerate(views, 1):
        old = baseline["phases"][("aggregate_hidden", "aggregate_logits")[mode - 1]]
        parts = old["parts"]
        require(
            parts * p < (1 << 31) and source.dtype.kind in "iu",
            "CSR and block index bounds",
        )
        counts = np.zeros((parts, p), np.uint16)
        states = {}
        for cap in caps:
            tag = np.full((p, cap), -1, np.int32)
            pending = tag.copy()
            reads = np.zeros_like(counts)
            writes = reads.copy()
            states[cap] = (tag, pending, reads, writes)
        degree = np.diff(rowptr).astype(np.int64, copy=False)
        block = 0
        for i, start in enumerate(range(0, size, p * r)):
            rows = np.arange(start, min(start + p * r, size), dtype=np.int32)
            groups = assign(rows, degree, p, r, mode, f, q)
            seqs = [
                np.concatenate([source[rowptr[j] : rowptr[j + 1]] for j in group])
                if len(group)
                else np.empty(0, np.int32)
                for group in groups
            ]
            lengths = np.array(list(map(len, seqs)))
            nk = (int(lengths.max()) + e - 1) // e
            require(block + nk <= parts, "expected edge banks")
            counts[block : block + nk] = np.clip(
                lengths[None, :] - np.arange(nk)[:, None] * e, 0, e
            )
            for pe, seq in enumerate(seqs):
                if mapping is not None:
                    seq = mapping[seq]
                for cap, args in states.items():
                    update(lib, seq, e, pe, p, block, cap, *args)
            block += nk
            if i and i % 2000 == 0:
                print(dataset, mode, i, flush=True)
        base_dma = dma(lib, counts, f)
        occ = int(counts.sum())
        require(
            block == parts
            and occ == old["occurrences"]
            and base_dma["words"] == old["csl_edge_words"],
            "byte-identical original native owner and DMA trace",
        )
        comparisons = []
        for cap, (_, __, reads, writes) in states.items():
            hits = int(reads.sum())
            fills = int(writes.sum())
            events = reads + writes
            require(
                fills <= hits and np.all(events <= counts),
                "at most one event per real edge",
            )
            words = (counts - reads) * f + events
            actual = dma(lib, words, 1)
            logical = int(words.sum())
            require(
                logical == occ * f - hits * f + hits + fills,
                "serial activation payload",
            )
            comparisons.append(
                dict(
                    slots=cap,
                    hits=hits,
                    useful_fills=fills,
                    event_words=hits + fills,
                    logical_saved_bytes=(occ * f - logical) * 4,
                    padded_native_saved_bytes=(base_dma["words"] - actual["words"]) * 4,
                    planned_native_calls=actual["calls"],
                    current_native_calls=base_dma["calls"],
                    max_bank_words=int(words.max(initial=0)),
                    minimum_extra_data_bytes=4
                    * (cap * f + max(0, int(words.max(initial=0)) - e * f)),
                )
            )
        result["phases"][str(mode)] = dict(
            occurrences=occ, baseline_edge_dma=base_dma, comparisons=comparisons
        )
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", choices=("arxiv", "products"))
    ap.add_argument("scope", choices=("whole", "four", "128"))
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        lib = build_lib(Path(td))
        selftest(lib)
        run(a, lib)
