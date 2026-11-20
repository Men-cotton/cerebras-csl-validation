"""Send full fresh stencil grids through the two physical Direct Link endpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from native_worker import reference, tile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(records, params, frame, prev, nxt, fp64):
    c, r, tx, ty, epochs = (params[k] for k in ("C", "R", "TX", "TY", "E"))
    y, x = np.indices((r,c)).reshape(2,-1)
    num = len(x)
    stamps = 3 + 6 * epochs
    size = (tx + 2) * (ty + 2)
    rows = records.reshape(num, 12 + stamps + 2 * size)
    for i, values in enumerate((frame, x, y, epochs)):
        np.testing.assert_array_equal(rows[:,i], np.broadcast_to(values, (num,)))
    east, west, north, south = (x+1<c)*ty, (x>0)*ty, (y>0)*tx, (y+1<r)*tx
    traffic = epochs * np.stack((east,west,north,south,west,east,south,north), axis=1)
    np.testing.assert_array_equal(rows[:,4:12], traffic)
    limbs = rows[:,12:12+stamps].reshape(num, -1, 3)
    if np.any(limbs > 65535):
        raise ValueError("invalid timestamp limbs")
    times = limbs.astype(np.int64) @ np.array([1,65536,1<<32])
    deltas = np.diff(times, axis=1) % (1<<48)
    if np.any(deltas == 0) or np.any(deltas >= 1<<32):
        raise ValueError("invalid local timestamp intervals")
    data = rows[:,12+stamps:].reshape(num,2,size)
    np.testing.assert_array_equal(data[:,0], prev.view(np.uint32).reshape(num,size))
    np.testing.assert_array_equal(data[:,1], nxt.view(np.uint32).reshape(num,size))
    actual = data[:,0].view(np.float32).reshape(r,c,ty+2,tx+2)[:,:,1:-1,1:-1]
    double = tile(fp64,c,r,tx,ty)
    np.testing.assert_allclose(actual,double,rtol=2e-5,atol=2e-6)
    return dict(frame=frame, max_cycles=int(deltas.sum(axis=1).max()),
                halo_max_cycles=int(deltas[:,::2].sum(axis=1).max()),
                compute_max_cycles=int(deltas[:,1::2].sum(axis=1).max()),
                sent_halo_words=int(traffic[:,:4].sum()),
                max_fp64_abs=float(np.abs(actual-double).max()))


def run(build, artifact, out, cmaddr):
    from cerebras.sdk.runtime.sdkruntimepybind import (SdkCompileArtifacts, SdkRuntime,
                                                      SdkTarget, SimfabConfig, get_platform)
    desc = json.loads(build.read_text())
    role = "hardware" if cmaddr else "simulator"
    if sha(artifact/"program.elf") != desc["roles"][role]["elf_sha256"]:
        raise ValueError("ELF differs from the paired build")
    if sha(artifact/"out_port_map.json") != desc["ports_sha256"]:
        raise ValueError("Direct Link port map digest mismatch")
    p = desc["params"]
    c,r,tx,ty,e = (p[k] for k in ("C","R","TX","TY","E"))
    gains = (1., -.5, 1.)
    fields = {gain: reference(c,r,tx,ty,e,gain) for gain in set(gains)}
    uploads = np.stack([fields[gain][0].view(np.uint32).ravel() for gain in gains])
    n = c*r*(15+6*e+2*(tx+2)*(ty+2))
    guard = np.full(len(gains)*n+4, 0xdeadc0de, np.uint32)
    reply = guard[2:-2]
    out.mkdir(parents=True,exist_ok=False)
    os.chdir(out)
    platform = get_platform(cmaddr, SimfabConfig(num_threads=1, suppress_trace=True), SdkTarget.WSE3)
    if bool(platform.is_system()) != bool(cmaddr) or bool(platform.is_simulation()) == bool(cmaddr):
        raise ValueError("incorrect runtime target")
    image = SdkCompileArtifacts(str(artifact)).add_port_mapping(str(artifact/"out_port_map.json"))
    runtime = SdkRuntime(image,platform,memcpy_required=False,msg_level="INFO")
    if any(tuple(runtime.coord_logical_to_physical((x,y))) != (x,y) for x in range(c) for y in range(r)):
        raise ValueError("physical allocation does not match the stencil placement")
    result = dict(backend=role, params=p, build_sha256=sha(build), trials=[], stopped=False)
    def stage(name):
        result["stage"] = name
        tmp = out/"capture.tmp"
        tmp.write_text(json.dumps(result,indent=2)+"\n")
        tmp.replace(out/"capture.json")
    stage("load")
    runtime.load()
    runtime.run()
    try:
        runtime.report_port_infos()
        rx, tx_port = runtime.get_port_id("rx_port"), runtime.get_port_id("tx_port")
        stage("streaming three frames")
        read = runtime.receive(tx_port,reply,reply.size,nonblock=True)
        write = runtime.send(rx,uploads.ravel(),uploads.size,nonblock=True)
        runtime.task_wait(write)
        runtime.task_wait(read)
    finally:
        runtime.stop()
    result["stopped"] = True
    np.testing.assert_array_equal(guard[[0,1,-2,-1]], np.full(4,0xdeadc0de,np.uint32))
    for frame, (gain, upload, output) in enumerate(zip(gains, uploads, reply.reshape(-1,n)),1):
        _,prev,nxt,fp64 = fields[gain]
        name = out/f"frame-{frame}.npz"
        np.savez_compressed(name, upload=upload, reply=output)
        trial = audit(output,p,frame,prev,nxt,fp64)
        trial.update(raw_sha256=sha(name), gain=gain, input_words=len(upload), output_words=n)
        result["trials"].append(trial)
    stage("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build",type=Path)
    parser.add_argument("artifact",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--cmaddr")
    a = parser.parse_args()
    run(a.build.resolve(),a.artifact.resolve(),a.output.resolve(),a.cmaddr)
