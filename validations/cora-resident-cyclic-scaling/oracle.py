"""Independent original-COO placement, round-to-nearest FMA, and vertical reduction oracle."""

import ctypes
import ctypes.util
import json
from pathlib import Path

import numpy as np

import contract as cyclic

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "cora-resident-device"


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


def unpack(archive, r, t):
    outputs = []
    for i in range(11):
        owner = archive[f"w{i}_t{t}"].reshape(cyclic.COLS,-1)[:,r*cyclic.HEAD:].view(np.float32)
        outputs.append(owner.reshape(cyclic.COLS,cyclic.ROWS,cyclic.LANES).transpose(1,2,0)
                          .reshape(cyclic.ROWS,cyclic.COLS*cyclic.LANES)[:,:cyclic.FEATURES])
    return np.concatenate(outputs)[:cyclic.NODES]


def wire(z, r):
    src, dst, weights = z["src"], z["dst"], z["weights"]
    bank_list = cyclic.banks(z,r)
    cap, ec = cyclic.sizes(r)
    for y in range(r):
        for local in range(cap):
            oid = local * r + y
            for bank, sign in zip(bank_list, (1,-1)):
                row = bank[y].reshape(cyclic.COLS,cap,cyclic.LANES)[:,local].T.ravel()
                expected = np.zeros(cyclic.COLS*cyclic.LANES,np.float32)
                if oid < cyclic.NODES:
                    expected[:cyclic.FEATURES] = z["features"][oid] * np.float32(sign)
                np.testing.assert_array_equal(row.view(np.uint32), expected.view(np.uint32))
    census = []
    for i in range(11):
        sl = cyclic.window(z,r,i)
        lo, hi = z["bounds"][i:i+2]
        expected_all = np.where((dst>=lo)&(dst<hi))[0]
        roundtrip = []
        for y in range(r):
            # Independent original-ID check of both wire endpoints, exact scalar weights, and stable local order.
            expected = sorted((int(e) for e in expected_all if int(src[e]) % r == y),key=lambda e:int(dst[e]))
            n = int(sl["counts"][y])
            pairs = sl["pairs"][y,:n]
            got_src = (pairs >> 16) // cyclic.LANES * r + y
            got_dst = (pairs & 0xffff) // cyclic.LANES + lo
            np.testing.assert_array_equal(got_src,src[expected])
            np.testing.assert_array_equal(got_dst,dst[expected])
            np.testing.assert_array_equal(sl["values"][y,:n].view(np.uint32),weights[expected].view(np.uint32))
            roundtrip.extend(expected)
        np.testing.assert_array_equal(sorted(roundtrip), expected_all)
        old = np.bincount(placements(src[expected_all],r,"range")[0], minlength=r)
        census.append(dict(index=i,rows=int(hi-lo),edges=len(expected_all),capacity=ec,
                           range_shard_edges=old.tolist(),cyclic_shard_edges=sl["counts"].tolist()))
    return census


if __name__ == "__main__":
    graph = cyclic.load_graph(HERE/"fixture.npz")
    recorded = json.loads((HERE/"order.json").read_text())
    for r in (4,16):
        reconstructed = build(graph,r,"cyclic")
        cyclic.require([cyclic.word_sha(v) for v in reconstructed] == recorded["cases"][str(r)]["cyclic"]["bits"],
                       "full original-COO fused accumulation")
        wire(graph,r)
    print("PASS independent Cora source, bank, weight, edge, reduction and owner bits")
