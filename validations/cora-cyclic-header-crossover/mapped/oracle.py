from pathlib import Path
import ctypes
import ctypes.util
import numpy as np
import contract
HERE=Path(__file__).resolve().parent

def independent(z, r):
    """Recover original source IDs by scanning each frozen bank slot and original COO edge."""
    maps = np.load(HERE/"source-map.npz", allow_pickle=False)
    assignment, slot = maps[f"shard{r}"].astype(int), maps[f"local{r}"].astype(int)
    cap, _ = contract.sizes(r)
    reverse = np.full((r,cap),-1,int)
    for index,(shard,local) in enumerate(zip(assignment,slot)):
        contract.require(reverse[shard,local]==-1,"one original ID per slot")
        reverse[shard,local]=index
    banks = contract.banks(z,r)
    for y in range(r):
        valid=reverse[y]>=0
        for sign,bank in zip((1,-1),banks):
            got=bank[y].reshape(4,cap,2).transpose(1,2,0).reshape(cap,8)
            wanted=np.zeros((cap,8),np.float32)
            wanted[valid,:7]=z["features"][reverse[y,valid]]*np.float32(sign)
            np.testing.assert_array_equal(got.view(np.uint32),wanted.view(np.uint32))
    per_window=[]
    src,dst,weights=(z[k] for k in ("src","dst","weights"))
    for i in range(11):
        lo,hi=z["bounds"][i:i+2]
        active=np.flatnonzero((dst>=lo)&(dst<hi))
        packed=contract.window(z,r,i)
        counts=[]
        for y in range(r):
            original_ids=sorted((n for n in active if assignment[src[n]]==y),key=lambda n:int(dst[n]))
            n=len(original_ids)
            counts.append(n)
            contract.require(n==packed["counts"][y],"actual shard edge count")
            words=packed["pairs"][y,:n]
            np.testing.assert_array_equal(reverse[y,(words>>16)//2],src[original_ids])
            np.testing.assert_array_equal((words&65535)//2+lo,dst[original_ids])
            np.testing.assert_array_equal(packed["values"][y,:n].view(np.uint32),weights[original_ids].view(np.uint32))
        per_window.append(counts)
    fma=ctypes.CDLL(ctypes.util.find_library("m")).fmaf
    fma.argtypes=(ctypes.c_float,ctypes.c_float,ctypes.c_float)
    fma.restype=ctypes.c_float
    expected=[]
    for sign in (1,-1):
        partial=np.zeros((r,2708,7),np.float32)
        for s,d,w in zip(src,dst,weights):
            out=partial[assignment[s],d]
            for k,val in enumerate(z["features"][s]):
                out[k]=fma(float(val)*sign,float(w),float(out[k]))
        acc=partial[0].copy()
        for y in range(1,r):
            np.add(acc,partial[y],out=acc)
        expected.append(acc)
    return np.stack((expected[0],expected[1],expected[0])), per_window


def build(graph,shards,placement):
    assert placement=='mapped'
    return independent(graph,shards)[0]
