"""Frozen real Cora window, independent COO oracle and per-owner PE clocks."""
import hashlib
import json
from pathlib import Path
import numpy as np

PIN = "09860e644f7a4b06477438922ce9c143ff3f1aa65a43db221ec6aec1a81504fe"
FRAMES = 12
OWNERS = 4
WIDTH = 4
ROWS = 64
TAG = 0x41474752
KEYS = ("csl-planned","tungsten-planned","tungsten-unrolled")

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2)+"\n")
    tmp.replace(path)

def load_graph(path):
    require(sha(path)==PIN,"Cora input identity")
    with np.load(path,allow_pickle=False) as npz:
        z={k:npz[k] for k in npz}
    d=z["dest"]
    require(len(d)==320 and z["features"].shape==(267,4) and z["weight"].shape==(4,85),"Cora dimensions")
    for owner in range(OWNERS):
        idx=np.flatnonzero(d//16==owner)
        idx=idx[np.argsort(d[idx],kind="stable")]
        row=np.bincount(d[idx]%16,minlength=16)
        np.testing.assert_array_equal(z["ptr"][owner],np.r_[0,np.cumsum(row)])
        np.testing.assert_array_equal(z["local"][owner,:len(idx)],z["src"][idx])
        np.testing.assert_array_equal(z["weight"][owner,:len(idx)],z["values"][idx])
        require(np.count_nonzero(row==4)==2,"two four-edge rows per owner")
    return z

def panel(z,i):
    require(0<=i<FRAMES,"frame index")
    scale=-.5 if i%3==1 else 1.
    return z["features"]*np.float32(scale)

def ticks(raw):
    t=np.asarray(raw,np.int64)
    require(t.shape==(4,6) and np.all((t>=0)&(t<65536)),"four raw 48-bit clock pairs")
    d=(t @ np.array([-1,-65536,-4294967296,1,65536,4294967296],np.int64))%(1<<48)
    require(np.all((d>0)&(d<100000000)),"bounded per-owner intervals")
    return d.tolist()

def decode(z, trial, raw, direct):
    clocks=np.asarray(raw["clocks"],np.uint32)
    setup=np.asarray(raw["setup"],np.uint32)
    output=np.asarray(raw["output_words"],np.uint32)
    require(output.shape==(64,4) and raw["index"]==trial,"complete frame")
    if direct:
        packet=np.asarray(raw["reply"],np.uint32).reshape(4,78)
        require(np.all(packet[:,0]==trial+1) and np.array_equal(packet[:,1],np.arange(4)),"reply tag and owner order")
        np.testing.assert_array_equal(packet[:,2:8],clocks)
        np.testing.assert_array_equal(packet[:,8:14],setup)
        np.testing.assert_array_equal(packet[:,14:].reshape(64,4),output)
    x=panel(z,trial)
    require(raw["feature_sha256"]==hashlib.sha256(x.tobytes()).hexdigest(),"panel digest")
    y=output.view(np.float32)
    ref=np.zeros((64,4),np.float64)
    np.add.at(ref,z["dest"],z["values"][:,None].astype(np.float64)*x[z["src"]].astype(np.float64))
    np.testing.assert_allclose(y,ref,rtol=1e-5,atol=2e-6)
    error=y.astype(np.float64)-ref
    return dict(compute=ticks(clocks),setup=ticks(setup),max_abs=float(np.abs(error).max()),
                relative_l2=float(np.linalg.norm(error)/np.linalg.norm(ref)),
                owner_sha256=hashlib.sha256(output.tobytes()).hexdigest())

def verify(z, result, role, case):
    require(case in KEYS and result["case"]==case and result["role"]==role,"worker target")
    require(result["stage"]=="PASS" and result["stopped"] and result["input_sha256"]==PIN,"completed worker")
    require(len(result["trials"])==FRAMES,"12 resident frames")
    hashes=[]
    maxima=[]
    expected_setup=None
    for i,frame in enumerate(result["trials"]):
        value=decode(z,i,frame,case.startswith("tungsten"))
        require(frame["numeric"]["owner_sha256"]==value["owner_sha256"] and
                frame["numeric"]["compute"]==value["compute"] and frame["numeric"]["setup"]==value["setup"],"raw replay")
        require(frame["numeric"]["max_abs"]==value["max_abs"] and
                np.isclose(frame["numeric"]["relative_l2"],value["relative_l2"],rtol=1e-12,atol=0),"numerical replay")
        if expected_setup is None:
            expected_setup=frame["setup"]
        require(expected_setup==frame["setup"],"once per resident graph")
        hashes.append(value["owner_sha256"])
        maxima.append(max(value["compute"]))
    a,b=hashes[:2]
    require(a!=b and hashes==[a,b,a]*4,"A/B/A repeated four times")
    return dict(max_owner_compute=maxima,setup_max=max(ticks(expected_setup)),output_sha256=hashes,
                setup_plus_compute_sum=max(ticks(expected_setup))+sum(maxima))
