"""Measure serial and concurrent West port pairs on one image and one PE clock."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from contract_lanes import (
    TRIALS,
    TRIAL_PLAN,
    decode_clock,
    device,
    evaluate,
    lane_inputs,
    reply_shape,
    wire,
)


def execute(build, fixture, output, artifact=None, cmaddr=None):
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkCompileArtifacts,
        SdkRuntime,
        SdkTarget,
        SimfabConfig,
        get_platform,
    )

    desc = json.loads((build / "build.json").read_text())
    role = "hardware" if cmaddr else "simulator"
    firmware = artifact or build / role / "out"
    if (
        device.sha(fixture) != desc["fixture_sha256"]
        or device.sha(firmware / "program.elf") != desc["roles"][role]["elf_sha256"]
        or device.sha(firmware / "out_port_map.json") != desc["port_sha256"]
    ):
        raise ValueError("different input, firmware or port mapping")
    z = device.load(fixture)
    n = desc["lanes"]
    parts = lane_inputs(z, n, desc["owners_per_lane"], desc["entries"])
    if [p[1] for p in parts] != desc["lane_params"]:
        raise ValueError("changed original lane assignments")
    uploads = [[wire(s, k) for s in parts] for k, _ in TRIAL_PLAN]
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    plat = get_platform(cmaddr, SimfabConfig(num_threads=1, suppress_trace=True), SdkTarget.WSE3)
    if bool(plat.is_system()) != bool(cmaddr) or bool(plat.is_simulation()) == bool(cmaddr):
        raise ValueError("unexpected platform")
    art = SdkCompileArtifacts(str(firmware)).add_port_mapping(str(firmware / "out_port_map.json"))
    rt = SdkRuntime(art, plat, memcpy_required=False, msg_level="INFO")
    for x in range(desc["owners_per_lane"]):
        for y in range(2 * n):
            if tuple(rt.coord_logical_to_physical((x, y))) != (x, y):
                raise ValueError("unexpected physical rectangle")
    report = dict(
        role=role,
        fixture_sha256=device.sha(fixture),
        build_sha256=device.sha(build / "build.json"),
        frames=[],
        stopped=False,
    )
    device.write(output / "result.json", report)
    rt.load()
    rt.run()
    ports = [(rt.get_port_id(f"rx{i}_port"), rt.get_port_id(f"tx{i}_port")) for i in range(n)]

    def submit(i, source, target):
        rx, tx = ports[i]
        read = rt.receive(tx, target, target.size, nonblock=True)
        write = rt.send(rx, source, source.size, nonblock=True)
        return write, read

    def wait(tasks):
        for write, read in tasks:
            rt.task_wait(write)
            rt.task_wait(read)

    def calibrate(offset):
        samples = []
        for k in range(5):
            tag = offset + k
            gate = np.array([tag, 0xC510A11], np.uint32)
            finish = np.array([tag, 0xC510B22], np.uint32)
            ack = np.empty(2, np.uint32)
            marks = np.empty(8, np.uint32)
            before = time.perf_counter_ns()
            wait([submit(0, gate, ack)])
            wait([submit(0, finish, marks)])
            wall = time.perf_counter_ns() - before
            np.testing.assert_array_equal(ack, gate)
            samples.append(
                dict(
                    tag=tag,
                    ack=ack.tolist(),
                    timer=marks.tolist(),
                    cycles=decode_clock(marks, tag),
                    host_ns=wall,
                )
            )
        return samples

    states = {}
    try:
        report["empty_before"] = calibrate(100)
        for i, (logical, concurrent) in enumerate(TRIAL_PLAN):
            gate = np.array([i + 1, 0xC510A11], np.uint32)
            finish = np.array([i + 1, 0xC510B22], np.uint32)
            ack = np.full(2, 0xDEADBEEF, np.uint32)
            stamps = np.full(8, 0xDEADBEEF, np.uint32)
            guarded = [np.full(np.prod(reply_shape(s)) + 4, 0xDEADBEEF, np.uint32) for s in parts]
            before = time.perf_counter_ns()
            wait([submit(0, gate, ack)])
            inner = time.perf_counter_ns()
            if concurrent:
                wait([submit(j, source, guarded[j][2:-2]) for j, source in enumerate(uploads[i])])
            else:
                for j, source in enumerate(uploads[i]):
                    wait([submit(j, source, guarded[j][2:-2])])
            received = time.perf_counter_ns()
            wait([submit(0, finish, stamps)])
            stopped = time.perf_counter_ns()
            np.testing.assert_array_equal(ack, gate)
            delta = decode_clock(stamps, i + 1)
            packets = []
            for s, g in zip(parts, guarded):
                np.testing.assert_array_equal(g[[0, 1, -2, -1]], 0xDEADBEEF)
                packets.append(g[2:-2].reshape(reply_shape(s)))
            y, m = evaluate(z, parts, packets, logical)
            method, scale = TRIALS[logical]
            if scale in states:
                np.testing.assert_array_equal(states[scale], y.view(np.uint32))
            else:
                states[scale] = y.view(np.uint32).copy()
            f = output / f"frame-{i + 1}.npz"
            np.savez_compressed(
                f,
                ack=ack,
                timer=stamps,
                nodes=z["seed"],
                logits=y,
                **{f"lane{j}": v for j, v in enumerate(packets)},
            )
            report["frames"].append(
                dict(
                    frame=i + 1,
                    method=method,
                    scale=scale,
                    concurrent=concurrent,
                    common_clock_cycles=delta,
                    lane_compute_peaks=[v["peak_first_to_last"] for v in m],
                    port_words=[u.size for u in uploads[i]],
                    fence_words_each_direction=[4, 10],
                    synchronized_data_host_ns=received - inner,
                    synchronized_fenced_host_ns=stopped - before,
                    raw_sha256=device.sha(f),
                )
            )
            device.write(output / "result.json", report)
        report["empty_after"] = calibrate(200)
    finally:
        rt.stop()
    if set(states) != {0.5, 1.0} or np.array_equal(states[0.5], states[1.0]):
        raise ValueError("input scale change was inactive")
    report["stopped"] = True
    device.write(output / "result.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("build", "fixture", "output"):
        p.add_argument(n, type=Path)
    p.add_argument("--artifact", type=Path)
    p.add_argument("--cmaddr")
    a = p.parse_args()
    execute(
        a.build.resolve(),
        a.fixture.resolve(),
        a.output.resolve(),
        a.artifact.resolve() if a.artifact else None,
        a.cmaddr,
    )
