"""Measure both source schedules in one resident ELF on the same OGB features."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from source_table import TRIALS, assign, audit, device, source_tiles, upload


def run(build, fixture, output, cmaddr, artifact):
    from cerebras.sdk.runtime.sdkruntimepybind import (SdkCompileArtifacts, SdkRuntime,
                                                       SdkTarget, SimfabConfig, get_platform)
    role = "hardware" if cmaddr else "simulator"
    desc = json.loads((build/"build.json").read_text())
    if device.sha(fixture) != desc["fixture_sha256"]:
        raise ValueError("fixture differs from the paired build")
    image_path = artifact or build/role/"out"
    if (device.sha(image_path/"program.elf") != desc["roles"][role]["elf_sha256"]
            or device.sha(image_path/"out_port_map.json") != desc["port_sha256"]):
        raise ValueError("ELF or physical port mapping changed")
    z = device.load(fixture)
    cfg, groups, owner, slot = assign(z, desc["params"]["P"], desc["params"]["E"])
    if cfg != desc["params"]:
        raise ValueError("geometry mismatch")
    schedules = {mode: source_tiles(z, cfg, owner, slot, mode) for mode, _ in TRIALS[:2]}
    uploads = [upload(z, cfg, groups, schedules[mode], i) for i, (mode, _) in enumerate(TRIALS)]
    stream = np.concatenate(uploads)
    size = cfg["P"]*(16+cfg["R"]*(1+cfg["Q"]))
    guarded = np.full(4+size*len(TRIALS), 0xdeadbeef, np.uint32)
    reply = guarded[2:-2]
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    plat = get_platform(cmaddr, SimfabConfig(num_threads=1, suppress_trace=True), SdkTarget.WSE3)
    if bool(plat.is_system()) != bool(cmaddr) or bool(plat.is_simulation()) == bool(cmaddr):
        raise ValueError("unexpected target platform")
    image = SdkCompileArtifacts(str(image_path)).add_port_mapping(str(image_path/"out_port_map.json"))
    runtime = SdkRuntime(image, plat, memcpy_required=False, msg_level="INFO")
    for y in range(2):
        for x in range(cfg["P"]):
            if tuple(runtime.coord_logical_to_physical((x,y))) != (x,y):
                raise ValueError("unexpected port or PE placement")
    result = dict(role=role, params=cfg, build_sha256=device.sha(build/"build.json"),
                  fixture_sha256=device.sha(fixture), frames=[], stopped=False)
    device.write(output/"result.json", result)
    runtime.load()
    runtime.run()
    try:
        rec = runtime.receive(runtime.get_port_id("tx_port"), reply, reply.size, nonblock=True)
        send = runtime.send(runtime.get_port_id("rx_port"), stream, stream.size, nonblock=True)
        runtime.task_wait(send)
        runtime.task_wait(rec)
    finally:
        runtime.stop()
    np.testing.assert_array_equal(guarded[[0,1,-2,-1]], 0xdeadbeef)
    previous = {}
    for i, (mode, scale) in enumerate(TRIALS):
        packet = reply[i*size:(i+1)*size].reshape(cfg["P"],-1)
        logits, measured = audit(packet, z, cfg, groups, i)
        if scale in previous:
            np.testing.assert_array_equal(logits.view(np.uint32), previous[scale])
        else:
            previous[scale] = logits.view(np.uint32)
        tiles = schedules[mode]
        entries = sum(len(sources) for sources, _ in tiles)
        path = output / f"frame-{i+1}.npz"
        np.savez_compressed(path, reply=packet, logits=logits, node=z["seed"])
        measured.update(trial=i+1, method=mode, scale=scale, input_words=len(uploads[i]),
                        blocks=len(tiles), source_vectors=entries, source_words=entries*cfg["F"],
                        edge_count=len(z["src"]), raw_sha256=device.sha(path))
        result["frames"].append(measured)
    if len(previous) != 2 or np.array_equal(previous[1.], previous[.5]):
        raise ValueError("the changed feature frame was inactive")
    result["stopped"] = True
    device.write(output/"result.json", result)
    print([(f["method"],f["input_words"],f["peak_first_to_last"]) for f in result["frames"]])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("build", type=Path)
    p.add_argument("fixture", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--cmaddr")
    p.add_argument("--artifact", type=Path)
    a = p.parse_args()
    run(a.build.resolve(), a.fixture.resolve(), a.output.resolve(), a.cmaddr, a.artifact.resolve() if a.artifact else None)
