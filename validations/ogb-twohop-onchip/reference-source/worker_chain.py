"""Execute both layers with and without an intermediate host round trip on the same ELF."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from chain import TRIALS, audit, device, digest, first_upload, prepare


def execute(build, first, last, out, artifact, cmaddr):
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkCompileArtifacts,
        SdkRuntime,
        SimfabConfig,
        SdkTarget,
        get_platform,
    )

    desc = json.loads((build / "build.json").read_text())
    role = "hardware" if cmaddr else "simulator"
    image = artifact or build / role / "out"
    if (
        device.sha(image / "program.elf") != desc["roles"][role]["elf_sha256"]
        or device.sha(image / "out_port_map.json") != desc["port_sha256"]
        or [device.sha(first), device.sha(last)] != [desc["first_sha256"], desc["last_sha256"]]
    ):
        raise ValueError("images or original graph inputs differ")
    cfg = desc["first"]
    data = prepare(first, last, cfg["P"], cfg["E"])
    final = data[7]
    if data[2] != cfg or final != desc["final"]:
        raise ValueError("unexpected graph dimensions")
    size = 17 + 16 * cfg["P"] + final["Z"] * (1 + final["Q"])
    size += size % 2
    hidden_size = final["N"] * final["H"]
    inputs = [first_upload(data, i) for i in range(len(TRIALS))]
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    plat = get_platform(cmaddr, SimfabConfig(num_threads=1, suppress_trace=True), SdkTarget.WSE3)
    if bool(plat.is_system()) != bool(cmaddr) or bool(plat.is_simulation()) == bool(cmaddr):
        raise ValueError("wrong platform")
    firmware = SdkCompileArtifacts(str(image)).add_port_mapping(str(image / "out_port_map.json"))
    rt = SdkRuntime(firmware, plat, memcpy_required=False, msg_level="INFO")
    for x in range(cfg["P"]):
        for y in (0, 1):
            if tuple(rt.coord_logical_to_physical((x, y))) != (x, y):
                raise ValueError("unexpected PE coordinates")
    report = dict(
        role=role,
        build_sha256=device.sha(build / "build.json"),
        first_sha256=device.sha(first),
        last_sha256=device.sha(last),
        frames=[],
        stopped=False,
    )
    device.write(out / "result.json", report)
    rt.load()
    rt.run()
    rx, tx = rt.get_port_id("rx_port"), rt.get_port_id("tx_port")
    hidden = {}
    scores = {}
    try:
        for i, (echo, scale) in enumerate(TRIALS):
            answer = np.full(size + 4, 0xDEADBEEF, np.uint32)
            inter = np.full(hidden_size + 4, 0xDEADBEEF, np.uint32) if echo else None
            begin = time.perf_counter_ns()
            recv = rt.receive(
                tx, inter[2:-2] if echo else answer[2:-2], hidden_size if echo else size, nonblock=True
            )
            send = rt.send(rx, inputs[i], inputs[i].size, nonblock=True)
            rt.task_wait(send)
            rt.task_wait(recv)
            if echo:
                recv = rt.receive(tx, answer[2:-2], size, nonblock=True)
                send = rt.send(rx, inter[2:-2], hidden_size, nonblock=True)
                rt.task_wait(send)
                rt.task_wait(recv)
            elapsed = time.perf_counter_ns() - begin
            for array in [answer, inter] if echo else [answer]:
                np.testing.assert_array_equal(array[[0, 1, -2, -1]], 0xDEADBEEF)
            raw = answer[2:-2]
            states = inter[2:-2] if echo else None
            logits, metrics = audit(data, i, raw, states, hidden.get(scale))
            if scale in scores:
                np.testing.assert_array_equal(logits.view(np.uint32), scores[scale])
            else:
                scores[scale] = logits.view(np.uint32)
            if echo:
                hidden[scale] = states.copy()
            file = out / f"frame-{i + 1}.npz"
            arr = dict(reply=raw, logits=logits, node=data[1]["seed"])
            if echo:
                arr["echo"] = states
            np.savez_compressed(file, **arr)
            metrics.update(
                trial=i + 1,
                echo=echo,
                scale=scale,
                input_words=int(inputs[i].size) + (hidden_size if echo else 0),
                output_words=size + (hidden_size if echo else 0),
                hidden_to_host_words=hidden_size if echo else 0,
                hidden_from_host_words=hidden_size if echo else 0,
                synchronized_host_ns=elapsed,
                raw_sha256=device.sha(file),
            )
            report["frames"].append(metrics)
            device.write(out / "result.json", report)
    finally:
        rt.stop()
    if set(hidden) != {1.0, 0.5} or np.array_equal(scores[1.0], scores[0.5]):
        raise ValueError("changed inputs were not observed")
    for f in report["frames"]:
        if f["hidden_digest"] != digest(hidden[f["scale"]]):
            raise ValueError("fused/echo hidden digest differs")
    report["stopped"] = True
    device.write(out / "result.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("build", "first", "last", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--artifact", type=Path)
    p.add_argument("--cmaddr")
    a = p.parse_args()
    execute(
        a.build.resolve(),
        a.first.resolve(),
        a.last.resolve(),
        a.output.resolve(),
        a.artifact.resolve() if a.artifact else None,
        a.cmaddr,
    )
