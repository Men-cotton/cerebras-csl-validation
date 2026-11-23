"""Recompute source bytes, array identities, final FP64 predictions and each PE's observed intervals."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from contract import base, raw_output, source_hashes
from rectangles import rectangles


def regions(counts, lanes):
    cuts = rectangles(counts, lanes)
    return [sum((end - start) * width * lanes for start, end, width in cuts), len(cuts)]


def audit(image, fixture, run, partial=False):
    desc = base.read(image / "build.json")
    meta, z = base.inputs(fixture)
    raw_state = (run / "result.json").read_bytes()
    state = json.loads(raw_state)
    role = state["role"]
    count = len(state["frames"])
    required = len(base.FRAMES)
    complete = count == required and state["stopped"]
    base.require(
        complete or (partial and 0 < count <= required and not state["stopped"]),
        "resident frame count and normal stop",
    )
    base.require(
        role in ("hardware", "simulator")
        and desc["role"] == role
        and state["build_sha256"] == base.sha(image / "build.json")
        and state["fixture_sha256"] == base.sha(fixture / "fixture.json")
        and state["platform"]
        == {"system": role == "hardware", "simulation": role == "simulator"}
        and state["loaded_image"] == desc["elf"]
        and state["host"] == source_hashes(),
        "actual native images, trained graph, workers and platform",
    )
    cfg = meta["params"]
    n = z["n"]
    p, f = cfg["P"], cfg["F"]
    counts = z["counts"]
    saved = {}
    frames = []
    for tag, ((use, scale), rec) in enumerate(
        zip(base.FRAMES[:count], state["frames"], strict=True), 1
    ):
        file = run / f"frame-{tag}.npz"
        base.require(
            (rec["tag"], rec["encoded"], rec["scale"], rec["raw_sha256"])
            == (tag, use, scale, base.sha(file)),
            "journal and original NPZ",
        )
        kept = z["indexed_banks"] if use else np.empty(0, np.int32)
        hot = z["hot"] if use else np.empty(0, np.int32)
        hits = np.zeros_like(counts)
        for k in kept:
            hits[k] = np.sum(np.isin(z["sources"][k], hot), axis=-1, dtype=np.uint32)
        wire = (counts - hits) * f + (hits + 1) // 2
        with np.load(file, allow_pickle=False) as raw:
            base.require(
                set(raw.files)
                == {"reply", "nodes", "hot", "selected_banks", "hits", "wire"},
                "complete physical reply inventory",
            )
            for name, value in [
                ("nodes", z["original_destinations"]),
                ("hot", hot),
                ("selected_banks", kept),
                ("hits", hits),
                ("wire", wire),
            ]:
                np.testing.assert_array_equal(raw[name], value)
            logits, metrics = raw_output(raw["reply"], z, meta, tag, scale)
        if scale in saved:
            np.testing.assert_array_equal(saved[scale], logits.view(np.uint32))
        else:
            saved[scale] = logits.view(np.uint32).copy()
        parts = len(counts)
        small = regions(n, 1)
        expected = {
            "control": [5 * p * (parts + 1), parts + 1],
            "indices": small,
            "inverse": small,
            "roots": regions(n, f),
            "features": np.sum([regions(w, 1) for w in wire], axis=0).tolist(),
            "targets": [parts * x for x in small],
        }
        if tag == 1:
            expected["weights"] = [z["weights"].size, 1]
        if len(hot):
            expected["rawx"] = [len(hot) * f, 1]
        base.require(
            {k: v[:2] for k, v in rec["by_symbol"].items()} == expected
            and all(x[2] >= 0 for x in rec["by_symbol"].values()),
            "every measured upload and rectangle",
        )
        base.require(
            all(rec[n] == value for n, value in metrics.items()),
            "independent numerical and raw clock audit",
        )
        source = np.array(expected["features"]) + expected.get("rawx", [0, 0])
        ref = meta["selected" if use else "dense"]
        base.require(
            source.tolist() == [ref["words"], ref["calls"]],
            "exact whole-model selection point",
        )
        frames.append(
            dict(
                tag=tag,
                encoded=use,
                scale=scale,
                source_words=int(source[0]),
                source_calls=int(source[1]),
                source_dma_ns=sum(
                    rec["by_symbol"].get(n, [0, 0, 0])[2] for n in ["features", "rawx"]
                ),
                raw_sha256=rec["raw_sha256"],
                **metrics,
            )
        )
    return dict(
        role=role,
        window=meta["window"],
        destinations=meta["destinations"],
        classes=meta["classes"],
        arcs=meta["arcs"],
        source_vertices=meta["source_vertices"],
        observed=count,
        required=required,
        stopped=state["stopped"],
        build_sha256=state["build_sha256"],
        fixture_sha256=state["fixture_sha256"],
        journal_sha256=hashlib.sha256(raw_state).hexdigest(),
        frames=frames,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ["image", "fixture", "run", "output"]:
        p.add_argument(n, type=Path)
    p.add_argument("--partial", action="store_true")
    a = p.parse_args()
    a.output.write_text(
        json.dumps(
            audit(a.image, a.fixture, a.run, a.partial), indent=2, sort_keys=True
        )
        + "\n"
    )
