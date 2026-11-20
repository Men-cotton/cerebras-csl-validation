"""Recover the exact edge features of an original SDK capture using sparse cache events."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from uniform import RESIDENT, decode, load_case, packed, read, require, schedule, sha
from graph import rowsources


def event_plan(blocks, capacity):
    """Return one sorted event list per PE bank. Fills are scheduled at their original miss."""
    events = [[] for _ in blocks]
    if not blocks:
        return events
    owners = len(blocks[0])
    for pe in range(owners):
        tags = np.full(capacity, -1, np.int64)
        pending = [None] * capacity
        for i, block in enumerate(blocks):
            for position, source in enumerate(block[pe]):
                slot = int(source) % capacity
                if tags[slot] != source:
                    tags[slot] = source
                    pending[slot] = (i, position)
                    continue
                events[i].append((pe, position, slot, True))
                if pending[slot] is not None:
                    first_block, first_position = pending[slot]
                    events[first_block].append((pe, first_position, slot, False))
                    pending[slot] = None
    return [sorted(block) for block in events]


def encode(values, events):
    """Each event packs the edge position, cache slot and read flag into one u32."""
    commands = np.array(
        [
            position | (slot << 8) | (int(hit) << 16)
            for _, position, slot, hit in events
        ],
        dtype=np.uint32,
    )
    hits = {position for _, position, _, hit in events if hit}
    payload = values[[i for i in range(len(values)) if i not in hits]].ravel()
    return ((len(commands) << 16) | len(values)), np.r_[commands, payload]


def unpack(control, words, slots, initialized):
    count, commands = control & 65535, control >> 16
    width = slots.shape[1]
    require(commands <= count and len(words) >= commands, "event count and packet size")
    decoded = np.empty((count, width), np.uint32)
    event, payload = 0, commands
    for i in range(count):
        cmd = int(words[event]) if event < commands else None
        due = cmd is not None and (cmd & 255) == i
        if due:
            slot = (cmd >> 8) & 255
            require(cmd < (1 << 17) and slot < len(slots), "valid cache command")
            event += 1
            if cmd & (1 << 16):
                require(initialized[slot], "read-before-fill")
                decoded[i] = slots[slot]
                continue
        decoded[i] = words[payload : payload + width]
        payload += width
        if due:
            slots[slot] = decoded[i]
            initialized[slot] = True
    require(
        event == commands and payload == len(words), "complete ordered packet decode"
    )
    return decoded


def captured_edge_banks(case, build, capture, owners):
    identity, graph = load_case(case)
    desc = read(build / "build.json")
    state = read(capture / "result.json")
    require(
        state["params"] == desc["params"]
        and state["case_sha256"] == sha(case / "case.json")
        and state["build_sha256"] == sha(build / "build.json")
        and state["loaded_image"] == desc["elf"]
        and state["platform"] == {"simulation": True, "system": False},
        "original native SDK image and data",
    )
    paths = (
        RESIDENT / "run.py",
        RESIDENT / "graph.py",
        RESIDENT.parent / "transfer/rectangles.py",
    )
    require(
        state["host_sha256"] == {p.name: sha(p) for p in paths},
        "original resident transfer interface",
    )
    rows = [
        json.loads(line)
        for line in (capture / "windows.jsonl").read_text().splitlines()
    ]
    records = {r["name"]: r for r in rows if r["trial"] == 1}
    cfg = desc["params"]
    f, e = cfg["F"], cfg["E"]
    require(e < 256, "u8 event position")
    output = [
        np.full((len(graph[name]), width), np.nan, np.float32)
        for name, width in (
            ("raw_ids", 2 * f),
            ("a_ids", f),
            ("b_ids", len(graph["b1"])),
        )
    ]
    raw_hashes = {}
    for step in schedule(graph, cfg):
        if step["trial"] != 1:
            continue
        mode = step["mode"]
        filename = f"t1-s{mode}-w{step['index']:04d}.npz"
        raw_hash = sha(capture / filename)
        require(
            raw_hash == records[filename]["raw_sha256"], "actual returned SDK words"
        )
        raw_hashes[filename] = raw_hash
        previous = output[mode - 1] if mode else None
        batch = packed(graph, step, cfg, previous, 1.0)
        with np.load(capture / filename, allow_pickle=False) as z:
            returned, _, _ = decode(z["reply"], graph, step, batch, cfg)
        output[mode][step["rows"]] = returned
    require(
        len(raw_hashes) == len(records) and all(np.isfinite(x).all() for x in output),
        "complete original A two-layer SDK rollout",
    )

    measured_owners = cfg["P"]
    if owners:
        require(0 < owners <= 32, "configured owner count")
        cfg = dict(cfg, P=owners)
    sources = {1: [], 2: []}
    banks = {1: [], 2: []}
    for step in schedule(graph, cfg):
        mode = step["mode"]
        if step["trial"] != 1 or mode == 0:
            continue
        batch = packed(graph, step, cfg, output[mode - 1], 1.0)
        prefix, original = ("a", "raw_ids") if mode == 1 else ("b", "a_ids")
        ids = [
            graph[original][rowsources(graph, prefix, group)]
            for group in step["groups"]
        ]
        for part in range(step["parts"]):
            sources[mode].append([s[part * e : (part + 1) * e] for s in ids])
            banks[mode].append(
                [
                    batch["bank"][part, pe, :n].view(np.uint32)
                    for pe, n in enumerate(batch["counts"][part])
                ]
            )
    return identity, measured_owners, cfg, raw_hashes, sources, banks


def compare(sources, banks, cfg, capacity):
    planned = event_plan(sources, capacity)
    p, f = cfg["P"], cfg["F"]
    slots = np.empty((p, capacity, f), np.uint32)
    initialized = np.zeros((p, capacity), bool)
    original = hashlib.sha256()
    decoded = hashlib.sha256()
    wire = hashlib.sha256()
    edges = words = reads = writes = 0
    for i, (source, bank, events) in enumerate(zip(sources, banks, planned)):
        for pe, values in enumerate(bank):
            require(
                values.shape == (len(source[pe]), f),
                "original arc source and feature width",
            )
            commands = [event for event in events if event[0] == pe]
            ctl, packet = encode(values, commands)
            recovered = unpack(ctl, packet, slots[pe], initialized[pe])
            np.testing.assert_array_equal(recovered, values)
            original.update(values.tobytes())
            decoded.update(recovered.tobytes())
            wire.update(np.array([i, pe, ctl], dtype="<u4").tobytes())
            wire.update(packet.tobytes())
            edges += len(values)
            words += len(packet)
            reads += sum(int(x[3]) for x in commands)
            writes += sum(int(not x[3]) for x in commands)
    require(
        writes <= reads and words == f * (edges - reads) + reads + writes,
        "exact live wire counts",
    )
    return dict(
        slots=capacity,
        original_edges=edges,
        original_words=edges * f,
        packet_words=words,
        read_events=reads,
        fill_events=writes,
        original_words_sha256=original.hexdigest(),
        decoded_words_sha256=decoded.hexdigest(),
        planned_packets_sha256=wire.hexdigest(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("arxiv", "products"))
    parser.add_argument("scope", choices=("four", "128"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--owners", type=int, help="replan ownership on the captured FP32 tensors"
    )
    opt = parser.parse_args()
    case = opt.cache / f"resident-{opt.dataset}-{opt.scope}"
    build = opt.cache / f"resident-{opt.dataset}-csl-p32"
    capture = opt.cache / f"resident-{opt.dataset}-{opt.scope}-csl-p32-capture"
    identity, measured_owners, cfg, raw, sources, banks = captured_edge_banks(
        case, build, capture, opt.owners
    )
    result = dict(
        dataset=opt.dataset,
        scope=opt.scope,
        params=cfg,
        measured_sdk_owners=measured_owners,
        case_sha256=sha(case / "case.json"),
        build_sha256=sha(build / "build.json"),
        raw_sha256=raw,
        checkpoint=identity["original_checkpoint_sha256"],
        phases={
            mode: [
                compare(sources[mode], banks[mode], cfg, c)
                for c in (16, 32, 64, 128, 256)
            ]
            for mode in (1, 2)
        },
    )
    opt.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
