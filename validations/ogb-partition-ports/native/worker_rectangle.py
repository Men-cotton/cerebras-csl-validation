"""Time native row broadcasts for identical GraphSAGE tables, whole rectangle or separate rows."""

import argparse
import json
import os
from pathlib import Path
import time
import numpy as np

from contract_lanes import TRIALS, TRIAL_PLAN, decode_clock, device, evaluate, lane_inputs


def packets(z, params, assignment=None, plan=TRIAL_PLAN, load_first=False):
    lanes, p, r, f, q, cap = (params[k] for k in ("L", "P", "R", "F", "Q", "M"))
    parts = lane_inputs(z, lanes, p, params["E"], assignment)
    padded = [(a, dict(cfg, R=r), *other) for a, cfg, *other in parts]
    classes = z["wr"].shape[1]
    weights = np.zeros((2 * f + 1, q), np.float32)
    weights[:f, :classes], weights[f : 2 * f, :classes], weights[-1, :classes] = z["wr"], z["wn"], z["b"]
    weights = np.ascontiguousarray(np.broadcast_to(weights, (lanes, *weights.shape))).reshape(lanes, -1)
    control = np.zeros((lanes, p, 3), np.uint32)
    ids = np.full((lanes, p, r), 0xFFFFFFFF, np.uint32)
    inv = np.zeros((lanes, p, r), np.float32)
    roots = np.zeros((lanes, p, r, f), np.float32)
    for j, (a, cfg, groups, *_) in enumerate(parts):
        for i, group in enumerate(groups):
            n = len(group)
            control[j, i, 1] = n
            ids[j, i, :n] = a["seed"][group]
            inv[j, i, :n] = 1 / np.maximum(np.diff(a["ptr"])[group], 1)
            roots[j, i, :n] = a["x"][a["root"][group]]
    uploads = []
    for logical, combined in plan:
        mode, scale = TRIALS[logical]
        head = control.copy()
        head[:, :, 0] = logical + 1
        blocks = []
        sched = [part[-1][mode] for part in parts]
        for k in range(max(map(len, sched))):
            items = [
                row[k] if k < len(row) else (np.array([], np.int32), [[] for _ in range(p)])
                for row in sched
            ]
            length = max(len(sources) for sources, _ in items)
            extent = max(len(words) for _, refs in items for words in refs)
            if not 0 < extent <= cap:
                raise ValueError("selector bank too small")
            tables = np.zeros((lanes, length, f), np.float32)
            refs = np.zeros((lanes, p, extent), np.uint32)
            ctrl = head.copy()
            useful = 0
            for j, (sources, words) in enumerate(items):
                tables[j, : len(sources)] = parts[j][0]["x"][sources] * scale
                useful += len(sources) * f
                for x, entries in enumerate(words):
                    refs[j, x, : len(entries)] = entries
                    ctrl[j, x, 2] = len(entries)
            blocks.append((tables.reshape(lanes, -1), refs, ctrl, useful))
        uploads.append(
            (
                weights if logical == 0 or load_first and not uploads else None,
                head,
                ids,
                inv,
                roots * scale,
                blocks,
            )
        )
    return padded, uploads


def frame_plan(desc):
    if "partitions" not in desc:
        return [(i, combined, "contiguous") for i, combined in TRIAL_PLAN]
    order = ("contiguous", "balanced", "dictionary")
    return [(i, True, p) for i, policies in [(1, order), (2, order[::-1]), (4, order)] for p in policies]


def prepared_frames(z, desc):
    plan = frame_plan(desc)
    by_policy = {}
    for policy in dict.fromkeys(row[2] for row in plan):
        selected = [(i, j) for i, j, p in plan if p == policy]
        assignment = desc.get("partitions", {}).get(policy)
        first = "partitions" in desc and policy == plan[0][2]
        group, payload = packets(z, desc["params"], assignment, selected, first)
        by_policy[policy] = group, iter(payload)
    return [
        (i, combined, policy, by_policy[policy][0], next(by_policy[policy][1]))
        for i, combined, policy in plan
    ]


def execute(build, fixture, output, cmaddr=None, artifact=None):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

    desc = json.loads((build / "build.json").read_text())
    role = "hardware" if cmaddr else "simulator"
    artifact = artifact or build / "out"
    if (
        desc["role"] != role
        or device.sha(fixture) != desc["fixture_sha256"]
        or {p.name: device.sha(p) for p in (artifact / "bin").glob("*.elf")} != desc["elf_sha256"]
    ):
        raise ValueError("different original inputs or native CSL binary")
    z = device.load(fixture)
    pr = desc["params"]
    lanes, p, r, q = (pr[k] for k in ("L", "P", "R", "Q"))
    frames = prepared_frames(z, desc)
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    rt = SdkRuntime(
        str(artifact),
        cmaddr=cmaddr,
        **({} if cmaddr else dict(simfab_numthreads=1, suppress_simfab_trace=True)),
    )
    opts = dict(
        streaming=False,
        nonblock=False,
        order=MemcpyOrder.ROW_MAJOR,
        data_type=MemcpyDataType.MEMCPY_32BIT,
    )

    def unique(name, data):
        a = np.ascontiguousarray(data).reshape(lanes, p, -1)
        rt.memcpy_h2d(rt.get_id(name), a.ravel(), 0, 0, p, lanes, a.shape[2], **opts)
        return a.size

    def broadcast(name, tables, combined):
        count = tables.shape[1]
        if combined:
            rt.memcpy_h2d_rowbcast(rt.get_id(name), tables.ravel(), 0, 0, p, lanes, count, **opts)
        else:
            for row in range(lanes):
                rt.memcpy_h2d_rowbcast(rt.get_id(name), tables[row], 0, row, p, 1, count, **opts)
        return tables.size

    def receive(name, count):
        a = np.empty((lanes, p, count), np.uint32)
        rt.memcpy_d2h(a.ravel(), rt.get_id(name), 0, 0, p, lanes, count, **opts)
        return a

    def calibrate(start):
        samples = []
        for tag in range(start, start + 5):
            before = time.perf_counter_ns()
            rt.launch("open_fence", nonblock=False)
            rt.launch("close_fence", nonblock=False)
            wall = time.perf_counter_ns() - before
            raw = receive("fence_words", 6)[0, 0]
            ticks = decode_clock(np.r_[np.array([tag, tag], np.uint32), raw], tag)
            samples.append(dict(tag=tag, clock=raw.tolist(), cycles=ticks, synchronized_host_ns=wall))
        return samples

    report = dict(
        role=role,
        build_sha256=device.sha(build / "build.json"),
        fixture_sha256=device.sha(fixture),
        frames=[],
        stopped=False,
    )
    device.write(output / "result.json", report)
    rt.load()
    rt.run()
    previous = {}
    try:
        report["empty_before"] = calibrate(100)
        for frame, (logical, combined, policy, parts, session) in enumerate(frames, 1):
            weights, ctrl, ids, inv, roots, blocks = session
            words = table_words = useful_words = calls = 0
            started = time.perf_counter_ns()
            rt.launch("open_fence", nonblock=False)
            if weights is not None:
                words += broadcast("weights", weights, True)
            for name, a in [("control", ctrl), ("indices", ids), ("inverse", inv), ("roots", roots)]:
                words += unique(name, a)
            rt.launch("begin_frame", nonblock=False)
            for tables, refs, ctrl, useful in blocks:
                table_words += broadcast("features", tables, combined)
                useful_words += useful
                calls += 1 if combined else lanes
                words += unique("targets", refs) + unique("control", ctrl)
                rt.launch("accumulate", nonblock=False)
            rt.launch("finish", nonblock=False)
            reply = receive("reply", 16 + r * (1 + q))
            edge = receive("edge_ticks", 1)
            rt.launch("close_fence", nonblock=False)
            elapsed = time.perf_counter_ns() - started
            stamps = receive("fence_words", 6)
            ticks = decode_clock(np.r_[np.array([frame, frame], np.uint32), stamps[0, 0]], frame)
            y, metrics = evaluate(z, parts, reply, logical)
            mode, scale = TRIALS[logical]
            if scale in previous:
                np.testing.assert_array_equal(y.view("u4"), previous[scale])
            else:
                previous[scale] = y.view("u4").copy()
            fpath = output / f"frame-{frame}.npz"
            np.savez_compressed(
                fpath, reply=reply, nodes=z["seed"], logits=y, active_edge_cycles=edge, fence=stamps
            )
            report["frames"].append(
                dict(
                    frame=frame,
                    **({"partition": policy} if "partitions" in desc else {}),
                    scale=scale,
                    mode=mode,
                    combined_rectangle=combined,
                    common_clock_cycles=ticks,
                    peak_local_cycles=max(x["peak_first_to_last"] for x in metrics),
                    peak_active_edge_cycles=int(edge.max()),
                    table_calls=calls,
                    table_words=table_words,
                    useful_table_words=useful_words,
                    h2d_words=words + table_words,
                    synchronized_host_ns=elapsed,
                    raw_sha256=device.sha(fpath),
                )
            )
            device.write(output / "result.json", report)
        report["empty_after"] = calibrate(200)
    finally:
        rt.stop()
    if set(previous) != {0.5, 1.0} or np.array_equal(previous[0.5], previous[1.0]):
        raise ValueError("inactive scale control")
    report["stopped"] = True
    device.write(output / "result.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("build", "fixture", "output"):
        p.add_argument(k, type=Path)
    p.add_argument("--artifact", type=Path)
    p.add_argument("--cmaddr")
    a = p.parse_args()
    execute(
        a.build.resolve(),
        a.fixture.resolve(),
        a.output.resolve(),
        a.cmaddr,
        a.artifact.resolve() if a.artifact else None,
    )
