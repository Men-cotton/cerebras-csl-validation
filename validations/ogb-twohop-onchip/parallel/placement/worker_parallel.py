"""Measure independent first-layer inputs and join the hidden states on the WSE fabric."""

import argparse
import json
import os
import hashlib
import two_rows
import chain
import source_table
from pathlib import Path
import time
import numpy as np

from two_rows import CHECK, TRIALS, device, packets, prepare, trial_inputs, verify


def execute(build, first, last, out, artifact=None, cmaddr=None, owner_order=None):
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkCompileArtifacts,
        SdkRuntime,
        SdkTarget,
        SimfabConfig,
        get_platform,
    )

    desc = json.loads((build / "build.json").read_text())
    role = "hardware" if cmaddr else "simulator"
    image = artifact or build / role / "out"
    for path, digest in [
        (first, desc["first_sha256"]),
        (last, desc["last_sha256"]),
        (image / "program.elf", desc["roles"][role]["elf_sha256"]),
        (image / "out_port_map.json", desc["port_sha256"]),
    ]:
        if device.sha(path) != digest:
            raise ValueError("input or installed image changed")
    cfg = desc["first"]
    data = prepare(first, last, cfg["P"], cfg["E"])
    if data[2] != cfg or desc["final"] != data[7]:
        raise ValueError("unexpected graph or PE shape")
    trials = trial_inputs(build, data, owner_order)
    final = data[7]
    size = 17 + 16 * cfg["P"] + final["Z"] * (1 + final["Q"])
    size += size % 2 + 4
    num_hidden = final["N"] * final["H"]
    setup = np.pad(data[8], (0, len(data[8]) % 2))
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    plat = get_platform(cmaddr, SimfabConfig(num_threads=1, suppress_trace=True), SdkTarget.WSE3)
    if bool(plat.is_system()) != bool(cmaddr) or bool(plat.is_simulation()) == bool(cmaddr):
        raise ValueError("wrong execution platform")
    fw = SdkCompileArtifacts(str(image)).add_port_mapping(str(image / "out_port_map.json"))
    rt = SdkRuntime(fw, plat, memcpy_required=False, msg_level="INFO")
    for x in range(cfg["P"] // 2):
        for y in range(4):
            if tuple(rt.coord_logical_to_physical((x, y))) != (x, y):
                raise ValueError("unexpected PE placement")
    result = dict(
        role=role,
        build_sha256=device.sha(build / "build.json"),
        first_sha256=device.sha(first),
        last_sha256=device.sha(last),
        setup_words=len(setup),
        setup_sha256=hashlib.sha256(setup.tobytes()).hexdigest(),
        actual_elf_sha256=device.sha(image / "program.elf"),
        actual_port_sha256=device.sha(image / "out_port_map.json"),
        platform={"system": bool(plat.is_system()), "simulation": bool(plat.is_simulation())},
        host_sha256={Path(x).name:device.sha(Path(x)) for x in (
            __file__, two_rows.__file__, chain.__file__, source_table.__file__, source_table.contract_path)},
        frames=[],
        stopped=False,
    )
    if owner_order is not None:
        result["compared_owner_order"] = owner_order
    device.write(out / "result.json", result)
    rt.load()
    rt.run()
    rx = [rt.get_port_id(f"rx{n}_port") for n in range(2)]
    tx = [rt.get_port_id(f"tx{n}_port") for n in range(2)]

    def send(lane, array):
        return rt.send(rx[lane], array, array.size, nonblock=True)

    def receive(lane, array):
        return rt.receive(tx[lane], array, array.size, nonblock=True)

    scores, states = {}, {}
    sentinel = 0xDEADBEEF
    try:
        rt.task_wait(send(0, setup))
        for pos, (logical, concurrent, data, rows) in enumerate(trials):
            tag = pos + 1
            echo, scale = TRIALS[logical]
            inputs = packets(rows, pos)
            guard = np.full(size + 4, sentinel, np.uint32)
            output = guard[2:-2]
            intermediate = np.full(num_hidden + 4, sentinel, np.uint32) if echo else None
            echo_words = intermediate[2:-2] if echo else None
            start = np.array([tag, CHECK], np.uint32)
            gate = np.full(2, sentinel, np.uint32)
            lower = np.full(2, sentinel, np.uint32)
            before = time.perf_counter_ns()
            latch = receive(0, gate)
            rt.task_wait(send(0, start))
            rt.task_wait(latch)
            np.testing.assert_array_equal(gate, start)
            response = receive(0, echo_words if echo else output)
            done = receive(1, lower)
            if concurrent:
                writes = [send(lane, array) for lane, array in enumerate(inputs)]
                for token in writes:
                    rt.task_wait(token)
            else:
                for lane, array in enumerate(inputs):
                    rt.task_wait(send(lane, array))
            rt.task_wait(response)
            if echo:
                response = receive(0, output)
                rt.task_wait(send(0, echo_words))
                rt.task_wait(response)
            rt.task_wait(done)
            elapsed = time.perf_counter_ns() - before
            for buf in (guard, intermediate) if echo else (guard,):
                np.testing.assert_array_equal(buf[[0, 1, -2, -1]], sentinel)
            logits, stats = verify(data, pos, output, lower, gate, echo_words, states.get(scale))
            if scale in scores:
                np.testing.assert_array_equal(scores[scale], logits.view(np.uint32))
            scores[scale] = logits.view(np.uint32).copy()
            raw = dict(
                reply=output, lower_ack=lower, gate_ack=gate, logits=logits, nodes=data[1]["seed"]
            )
            if echo:
                if scale in states:
                    np.testing.assert_array_equal(states[scale], echo_words)
                states[scale] = echo_words.copy()
                raw["echo"] = echo_words
            path = out / f"frame-{tag}.npz"
            np.savez_compressed(path, **raw)
            result["frames"].append(
                dict(
                    tag=tag,
                    logical=logical,
                    concurrent=concurrent,
                    echo=echo,
                    scale=scale,
                    port_input_words=[len(v) for v in inputs],
                    port_input_sha256=[hashlib.sha256(v.tobytes()).hexdigest() for v in inputs],
                    echo_h2d_sha256=hashlib.sha256(echo_words.tobytes()).hexdigest() if echo else None,
                    gate_input_words=2,
                    gate_output_words=2,
                    lower_ack_words=2,
                    hidden_h2d_words=num_hidden if echo else 0,
                    hidden_d2h_words=num_hidden if echo else 0,
                    synchronized_host_ns=elapsed,
                    raw_sha256=device.sha(path),
                    **stats,
                )
            )
            device.write(out / "result.json", result)
    finally:
        rt.stop()
    if set(states) != {0.5, 1.0} or np.array_equal(scores[0.5], scores[1.0]):
        raise ValueError("both scales and their complete intermediate controls are required")
    result["stopped"] = True
    device.write(out / "result.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("build", "first", "last", "output"):
        p.add_argument(n, type=Path)
    p.add_argument("--artifact", type=Path)
    p.add_argument("--cmaddr")
    p.add_argument("--compare-owner-order", type=int, nargs="+")
    a = p.parse_args()
    execute(
        a.build.resolve(),
        a.first.resolve(),
        a.last.resolve(),
        a.output.resolve(),
        a.artifact.resolve() if a.artifact else None,
        a.cmaddr,
        a.compare_owner_order,
    )
