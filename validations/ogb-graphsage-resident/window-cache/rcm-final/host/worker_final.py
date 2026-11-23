"""Run dense and shared-table final GraphSAGE on the same native P32 firmware and original RCM destinations."""

import argparse
import os
from pathlib import Path

import numpy as np

from contract import base, materialize, raw_output, source_hashes


def execute(image, fixture, out, cmaddr=None, artifact=None):
    desc = base.read(image / "build.json")
    meta, z = base.inputs(fixture)
    cfg = desc["params"]
    p, r, q = (cfg[k] for k in ("P", "R", "Q"))
    role = "hardware" if cmaddr else "simulator"
    expected = meta["physical_size_sha256" if cmaddr else "simulator_sha256"]
    base.require(
        desc["role"] == role
        and base.sha(image / "build.json") == expected
        and meta["params"] == cfg,
        "same measured resident ELF, 32 PE geometry and graph",
    )
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    target = base.Device(image, desc, role, cmaddr, artifact)
    journal = out / "result.json"
    peers = {}
    state = dict(
        role=role,
        stopped=False,
        loaded_image=target.image,
        platform=target.platform,
        host=source_hashes(),
        build_sha256=expected,
        fixture_sha256=base.sha(fixture / "fixture.json"),
        frames=[],
    )
    base.save(journal, state)
    try:
        for tag, (use, scale) in enumerate(base.FRAMES, 1):
            hot, table, wire, payload, hits, kept = materialize(z, meta, use, scale)
            n = z["n"]
            parts = meta["parts"]
            ctl = np.zeros((p, 5), np.uint32)
            ctl[:, 0] = tag
            ctl[:, 1] = 2
            ctl[:, 2] = parts
            ctl[:, 3] = n
            tx = base.Transfers(target)
            if tag == 1:
                tx.copy("weights", z["weights"], 0, p, broadcast=True)
            tx.copy("control", ctl, 0, p)
            tx.prefixes("indices", z["ids"], n)
            tx.prefixes("inverse", z["inv"], n)
            tx.prefixes("roots", z["root"] * np.float32(scale), n)
            if len(hot):
                tx.copy("rawx", table, 0, p, broadcast=True)
            target.dev.launch("begin_frame", nonblock=False)
            for k in range(parts):
                ctl[:, 4] = z["counts"][k] | (hits[k] << 16)
                tx.copy("control", ctl, 0, p)
                tx.prefixes("targets", z["ends"][k], n)
                tx.prefixes("features", payload[k], wire[k])
                target.dev.launch(
                    "accumulate_cached" if k in kept else "accumulate", nonblock=False
                )
            target.dev.launch("finish", nonblock=False)
            raw = np.empty((p, 4 + (parts + 2) * 6 + r * (1 + q)), np.uint32)
            target.dev.memcpy_d2h(
                raw.ravel(),
                target.dev.get_id("reply"),
                0,
                0,
                p,
                1,
                raw.shape[1],
                streaming=False,
                nonblock=False,
                data_type=target.dtype,
                order=target.order,
            )
            name = f"frame-{tag}.npz"
            np.savez_compressed(
                out / name,
                reply=raw,
                nodes=z["original_destinations"],
                hot=hot,
                selected_banks=np.array(kept, np.int32),
                hits=hits,
                wire=wire,
            )
            y, metrics = raw_output(raw, z, meta, tag, scale)
            if scale in peers:
                np.testing.assert_array_equal(y.view(np.uint32), peers[scale])
            else:
                peers[scale] = y.view(np.uint32).copy()
            sources = [
                sum(tx.amount.get(n, (0, 0, 0))[i] for n in ("features", "rawx"))
                for i in (0, 1)
            ]
            reference = meta["selected" if use else "dense"]
            base.require(
                sources == [reference["words"], reference["calls"]],
                "actual source transfers",
            )
            state["frames"].append(
                dict(
                    tag=tag,
                    encoded=use,
                    scale=scale,
                    raw_sha256=base.sha(out / name),
                    by_symbol=dict(tx.amount),
                    **metrics,
                )
            )
            base.save(journal, state)
            print(tag, use, sources, metrics, flush=True)
    finally:
        target.stop()
        state["stopped"] = len(state["frames"]) == len(base.FRAMES)
        base.save(journal, state)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path)
    p.add_argument("fixture", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--cmaddr")
    p.add_argument("--artifact", type=Path)
    a = p.parse_args()
    execute(
        a.image.resolve(),
        a.fixture.resolve(),
        a.out.resolve(),
        a.cmaddr,
        a.artifact.resolve() if a.artifact else None,
    )
