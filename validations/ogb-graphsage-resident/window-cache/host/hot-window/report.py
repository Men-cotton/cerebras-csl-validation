"""Recompute actual source regions, all PE clocks, and full original-ID hidden outputs from returns."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from worker import FRAMES, HERE, inputs, inspect, pack
from graph import read, require, sha
from rectangles import rectangles


def costs(counts, lanes):
    regions = rectangles(counts, lanes)
    return int(sum((b - a) * n * lanes for a, b, n in regions)), len(regions)


def replay(image, fixture, capture, completed_prefix=False):
    meta, z = inputs(fixture)
    cfg = meta["params"]
    p, f = cfg["P"], cfg["F"]
    desc = read(image / "build.json")
    journal = (capture / "result.json").read_bytes()
    log = json.loads(journal)
    observed_count = len(log["frames"])
    if completed_prefix:
        require(1 <= observed_count <= len(FRAMES), "one or more completed original SDK frames")
    else:
        require(log["stopped"] and observed_count == len(FRAMES), "all six frames and normal SDK stop")
    require(
        log["build_sha256"] == sha(image / "build.json")
        and log["fixture_sha256"] == sha(fixture / "fixture.json")
        and log["worker_sha256"] == sha(HERE / "worker.py")
        and log["wire_codec_sha256"] == sha(HERE.parent / "window_cache.py"),
        "same six-frame host, source-table codec and inputs",
    )
    require(
        log["loaded_image"]
        == desc["elf"]
        == {file.name: sha(file) for file in (image / "out/bin").glob("*.elf")}
        and log["platform"]
        == {
            "system": desc["role"] == "hardware",
            "simulation": desc["role"] == "simulator",
        },
        "source-matched platform and loaded executable",
    )
    require(
        desc["original_build_sha256"] == meta["original_build_sha256"]
        and desc["params"] == cfg,
        "original calibrated image and fixture hashes",
    )
    groups = {}
    records = []
    for i, ((use, scale), entry) in enumerate(
        zip(FRAMES[:observed_count], log["frames"], strict=True), 1
    ):
        file = capture / f"frame-{i}.npz"
        require(
            (entry["use_table"], entry["scale"], entry["frame"], entry["raw_sha256"])
            == (use, scale, i, sha(file)),
            "registered frames and captured words",
        )
        hot, table, wire, _, hits = pack(z, scale, use, 32)
        with np.load(file, allow_pickle=False) as raw:
            require(
                set(raw.files)
                == {"reply", "original_ids", "hidden", "hot", "hits", "wire"},
                "complete capture fields",
            )
            for k, v in (
                ("hot", hot),
                ("hits", hits),
                ("wire", wire),
                ("original_ids", z["hidden_nodes"]),
            ):
                np.testing.assert_array_equal(raw[k], v)
            y, measured = inspect(raw["reply"], z, cfg, i, scale)
            np.testing.assert_array_equal(
                y.view(np.uint32), raw["hidden"].view(np.uint32)
            )
            if scale in groups:
                np.testing.assert_array_equal(y.view(np.uint32), groups[scale])
            else:
                groups[scale] = y.view(np.uint32).copy()
        require(
            all(entry[k] == v for k, v in measured.items())
            and entry["source_table_words"] == table.size
            and entry["hits"] == int(hits.sum())
            and entry["source_words"] == int(wire.sum()),
            "device measurements and unpadded wire",
        )
        n = z["n"]
        parts = len(z["counts"])
        rc, rcalls = costs(n, 1)
        feature_cost = [costs(row, 1) for row in wire]
        symbols = dict(
            control=(5 * p * (1 + parts), 1 + parts),
            indices=(rc, rcalls),
            inverse=(rc, rcalls),
            roots=costs(n, f),
            targets=(rc * parts, rcalls * parts),
            features=tuple(map(sum, zip(*feature_cost))),
        )
        if use and len(hot):
            symbols["rawx"] = (table.size, 1)
        if i == 1:
            symbols["weights"] = (z["weights"].size, 1)
        observed = entry["by_symbol"]
        require(
            symbols.keys() == observed.keys()
            and all(
                tuple(observed[k][:2]) == v and observed[k][2] > 0
                for k, v in symbols.items()
            ),
            "exact actual blocking rectangles, rows and calls",
        )
        records.append(
            dict(
                frame=i,
                cached=use,
                scale=scale,
                active=measured["peak_active_cycles"],
                edge=measured["peak_edge_cycles"],
                elapsed=measured["peak_elapsed_cycles"],
                fp64_max_abs=measured["fp64_maximum_absolute_error"],
                output_sha256=measured["ordered_output_sha256"],
                source_words=entry["source_words"],
                table_words=table.size,
                source_wire_words=int(
                    sum(symbols.get(k, (0, 0))[0] for k in ("features", "rawx"))
                ),
                source_dma_calls=int(
                    sum(symbols.get(k, (0, 0))[1] for k in ("features", "rawx"))
                ),
                host_all_words=int(sum(t[0] for t in symbols.values())),
                host_all_calls=int(sum(t[1] for t in symbols.values())),
                host_source_dma_ns=sum(
                    observed.get(k, (0, 0, 0))[2] for k in ("features", "rawx")
                ),
                raw_sha256=sha(file),
            )
        )
    return dict(
        role=desc["role"],
        case_sha256=meta["case_sha256"],
        fixture_sha256=sha(fixture / "fixture.json"),
        image_sha256=sha(image / "build.json"),
        capture_sha256=hashlib.sha256(journal).hexdigest(),
        observed_frames=observed_count,
        required_frames=len(FRAMES),
        stopped=log["stopped"],
        original_A_windows=len(meta["original_trial_A_windows"]),
        original_arcs=meta["original_arcs"],
        original_hidden_words=meta["original_hidden_words"],
        frames=records,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("image", "fixture", "capture", "out"):
        parser.add_argument(n, type=Path)
    parser.add_argument("--completed-prefix", action="store_true", help="check available original SDK returns and record exact incomplete coverage")
    a = parser.parse_args()
    a.out.write_text(
        json.dumps(
            replay(a.image, a.fixture, a.capture, a.completed_prefix),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
