import ctypes
import ctypes.util
import numpy as np
import contract as cyclic

def placements(src, r, mode):
    cap, _ = cyclic.sizes(r)
    if mode == "range":
        return np.divmod(src, cap)
    if mode == "cyclic":
        local, shard = np.divmod(src, r)
        return shard, local
    raise ValueError(mode)


def build(z, r, mode):
    """Accumulate original edges before window sorting, and independently encode all active owner bits."""
    src, dst, weight, panel = (z[k] for k in ("src","dst","weights","features"))
    shard, _ = placements(src, r, mode)
    fma = ctypes.CDLL(ctypes.util.find_library("m")).fmaf
    fma.argtypes = (ctypes.c_float,ctypes.c_float,ctypes.c_float)
    fma.restype = ctypes.c_float
    outputs = []
    for sign in (1.0,-1.0):
        partial = np.zeros((r,cyclic.NODES,cyclic.FEATURES),np.float32)
        for y,s,d,w in zip(shard,src,dst,weight):
            row = partial[y,d]
            for k,value in enumerate(panel[s]):
                row[k] = fma(float(value)*sign, float(w), float(row[k]))
        summed = partial[0].copy()
        for y in range(1,r):
            np.add(summed, partial[y], out=summed)
        outputs.append(summed)
    return np.stack((outputs[0],outputs[1],outputs[0]))
