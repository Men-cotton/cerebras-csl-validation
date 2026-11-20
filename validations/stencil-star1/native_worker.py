"""Measure resident CSL launches and verify both padded physical grids independently."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def tile(grid, c, r, tx, ty):
    return grid.reshape(r,ty,c,tx).transpose(0,2,1,3)


def check_transport(data, c, r, tx, ty, epochs):
    y, x = np.indices((r, c))
    horizontal = (x > 0).astype(int) + (x + 1 < c)
    vertical = (y > 0).astype(int) + (y + 1 < r)
    faces = epochs * (horizontal + vertical)
    words = epochs * (ty * horizontal + tx * vertical)
    status = data["transport_status"]
    expected = (epochs, faces, faces, faces, faces, words, words, epochs, epochs,
                epochs, epochs, 4, 4, 0, 0, 0x44534431)
    meta = (0x4353544e, tx, ty, 1, 1, 4, tx * ty, 10, 6, 24,
            y * c + x, x, y, c, r, epochs)
    for values, recorded in ((expected, status), (meta, data["stencil_meta"])):
        for field, value in enumerate(values):
            np.testing.assert_array_equal(recorded[:, :, field], np.broadcast_to(value, (r, c)))
    np.testing.assert_array_equal(data["done_flag"], np.ones((r, c, 1), np.uint32))
    np.testing.assert_array_equal(data["diagnostic_delay"], np.zeros((r, c, 4), np.uint32))
    return int(words.sum())


def reference(c,r,tx,ty,epochs,gain):
    y,x = np.indices((r*ty,c*tx))
    start = (((17*x+29*y+3*x*y)%101-50)/16).astype(np.float32)*np.float32(gain)
    current, fp64 = start, start.astype(np.float64)
    for _ in range(epochs):
        old = np.pad(current,1)
        neighbor = (old[:-2,1:-1],old[2:,1:-1],old[1:-1,:-2],old[1:-1,2:])
        current = np.float32(.5)*current
        for face in neighbor:
            current += np.float32(.125)*face
        wide = np.pad(fp64,1)
        fp64 = .5*fp64 + .125*(wide[:-2,1:-1]+wide[2:,1:-1]+wide[1:-1,:-2]+wide[1:-1,2:])
    shape = (r,c,ty+2,tx+2)
    initial = np.zeros(shape,np.float32)
    initial[:,:,1:-1,1:-1] = tile(start,c,r,tx,ty)
    nxt = np.zeros_like(initial)
    nxt[:,:,1:-1,1:-1] = tile(current,c,r,tx,ty)
    prev = nxt.copy()
    for y in range(r):
        for x in range(c):
            grid = prev[y,x]
            halo = old[y*ty:y*ty+ty+2,x*tx:x*tx+tx+2]
            grid[0,1:-1],grid[-1,1:-1] = halo[0,1:-1],halo[-1,1:-1]
            grid[1:-1,0],grid[1:-1,-1] = halo[1:-1,0],halo[1:-1,-1]
    return initial,prev,nxt,fp64


def execute(build,out,cmaddr=None):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    params = json.loads((build/"out.json").read_text())["params"]
    c,r,tx,ty,e = (int(params[k]) for k in ("KERNEL_X_DIM","KERNEL_Y_DIM","TILE_X","TILE_Y","ITERATIONS"))
    out.mkdir(parents=True,exist_ok=False)
    opts = dict(cmaddr=cmaddr) if cmaddr else dict(simfab_numthreads=1,suppress_simfab_trace=True)
    runtime = SdkRuntime(str(build),**opts)
    w32 = dict(streaming=False,nonblock=False,order=MemcpyOrder.ROW_MAJOR,data_type=MemcpyDataType.MEMCPY_32BIT)
    w16 = dict(w32,data_type=MemcpyDataType.MEMCPY_16BIT)
    N = (tx+2)*(ty+2)
    trials = []
    runtime.load()
    runtime.run()
    try:
        for num,gain in enumerate((1.,-.5,1.),1):
            initial,expected_prev,expected_next,fp64 = reference(c,r,tx,ty,e,gain)
            runtime.memcpy_h2d(runtime.get_id("grid_prev"),initial.ravel(),0,0,c,r,N,**w32)
            runtime.launch("run_kernel",nonblock=False)
            data = {}
            for name,n in (("grid_prev",N),("grid_next",N),("stencil_meta",16),("done_flag",1),
                           ("transport_status",16),("diagnostic_delay",4),("tsc_start",3),("tsc_end",3)):
                words = np.empty((r,c,n),np.float32 if name.startswith("grid_") else np.uint32)
                runtime.memcpy_d2h(words.ravel(),runtime.get_id(name),0,0,c,r,n,**(w16 if name.startswith("tsc_") else w32))
                data[name] = words
            np.savez_compressed(out/f"frame-{num}.npz",**data)
            for key,expected in (("grid_prev",expected_prev),("grid_next",expected_next)):
                np.testing.assert_array_equal(data[key].view("u4").ravel(),expected.view("u4").ravel())
            sent_words = check_transport(data, c, r, tx, ty, e)
            if np.any(data["tsc_end"]>65535) or np.any(data["tsc_start"]>65535):
                raise ValueError("invalid timestamp limb")
            cycles = ((data["tsc_end"].astype(np.int64)-data["tsc_start"].astype(np.int64)) @ np.array([1,65536,1<<32]))%(1<<48)
            if np.any(cycles==0) or np.any(cycles>=1<<32):
                raise ValueError("invalid PE interval")
            physical = data["grid_prev"].reshape(r,c,ty+2,tx+2)[:,:,1:-1,1:-1]
            err = np.abs(physical.astype(np.float64)-tile(fp64,c,r,tx,ty))
            if not np.allclose(physical,tile(fp64,c,r,tx,ty),rtol=2e-5,atol=2e-6):
                raise ValueError("FP64 oracle mismatch")
            trials.append(dict(frame=num,gain=gain,max_cycles=int(cycles.max()),min_cycles=int(cycles.min()),
                               max_fp64_abs=float(err.max()),sent_words=sent_words,
                               output_sha256=hashlib.sha256(data["grid_prev"].tobytes()+data["grid_next"].tobytes()).hexdigest(),
                               raw_sha256=hashlib.sha256((out/f"frame-{num}.npz").read_bytes()).hexdigest()))
    finally:
        runtime.stop()
    if trials[0]["output_sha256"]!=trials[2]["output_sha256"] or trials[0]["output_sha256"]==trials[1]["output_sha256"]:
        raise ValueError("resident input replay mismatch")
    (out/"result.json").write_text(json.dumps(dict(backend="physical" if cmaddr else "SDK simulator",
        params=params,stopped=True,trials=trials),indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("build",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--cmaddr")
    a=p.parse_args()
    execute(a.build.resolve(),a.output.resolve(),a.cmaddr)
