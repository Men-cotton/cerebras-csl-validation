"""Compare original and hot-table source selection on one actually loaded resident image."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from window_cache import choose, decode as decode_packet, event_packet
from graph import decode, read, require, sha
from rectangles import rectangles
from run import Device

FRAMES = [(False,1.),(True,1.),(True,.5),(False,.5),(True,1.),(False,1.)]


def save(path, data):
    temp = path.with_suffix(".new")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def pack(z, scale, use, slots):
    ids=z["sources"];counts=z["counts"]
    hot=choose(ids[ids>=0],slots) if use else np.empty(0,np.int32)
    where=np.searchsorted(z["original_nodes"],hot)
    np.testing.assert_array_equal(z["original_nodes"][where],hot)
    table=np.ascontiguousarray(z["feature_table"][where]*np.float32(scale)).view(np.uint32)
    bank=np.ascontiguousarray(z["bank"]*np.float32(scale)).view(np.uint32)
    payload=bank.reshape(*bank.shape[:2],-1).copy()
    wire=counts*bank.shape[3]
    hits=np.zeros_like(counts)
    for k in range(len(counts)):
        for pe,count in enumerate(counts[k]):
            rows=ids[k,pe,:count]
            h=int(np.count_nonzero(np.isin(rows,hot)))
            hits[k,pe]=h
            if h:
                packet=event_packet(bank[k,pe,:count],rows,hot)
                np.testing.assert_array_equal(decode_packet(packet,count,bank.shape[3],h,table),bank[k,pe,:count])
                payload[k,pe]=0
                payload[k,pe,:len(packet)]=packet
                wire[k,pe]=len(packet)
    return np.ascontiguousarray(hot),table,wire,payload,hits


def inputs(fixture):
    meta=read(fixture/"fixture.json")
    require(sha(fixture/"fixture.npz")==meta["files"]["fixture.npz"],"original source projection fixture")
    with np.load(fixture/"fixture.npz",allow_pickle=False) as values:
        return meta,dict(values)


def inspect(raw, z, cfg, tag, scale):
    nodes=z["hidden_nodes"]
    groups=[np.searchsorted(nodes,ids[:count]) for ids,count in zip(z["ids"],z["n"])]
    step=dict(mode=1,parts=len(z["counts"]),tag=tag,rows=np.arange(len(nodes)),groups=groups)
    y,clocks,wall=decode(raw,{"b1":np.empty(cfg["Q"])},step,z,cfg)
    ref=z["fp64_a" if scale==1 else "fp64_half"]
    np.testing.assert_allclose(y,ref,rtol=4e-4,atol=3e-4)
    if scale==1:np.testing.assert_array_equal(y.view(np.uint32),z["measured_hidden"].view(np.uint32))
    return y,dict(peak_active_cycles=int(clocks.sum(axis=1).max()),
                  peak_edge_cycles=int(clocks[:,1:-1].sum(axis=1).max()),
                  peak_elapsed_cycles=int(wall.max()),
                  fp64_maximum_absolute_error=float(np.max(np.abs(y.astype(np.float64)-ref))),
                  ordered_output_sha256=hashlib.sha256(y.tobytes()).hexdigest())


class Transfers:
    def __init__(self, target):
        self.t=target
        self.amount=defaultdict(lambda:[0,0,0])

    def copy(self, name, data, x, width, broadcast=False):
        arr=np.ascontiguousarray(data)
        if not arr.size:return
        count=arr.size if broadcast else arr.size//width
        dev=self.t.dev
        fn=dev.memcpy_h2d_rowbcast if broadcast else dev.memcpy_h2d
        begin=time.perf_counter_ns()
        fn(dev.get_id(name),arr.ravel(),x,0,width,1,count,streaming=False,nonblock=False,
           data_type=self.t.dtype,order=self.t.order)
        ns=time.perf_counter_ns()-begin
        row=self.amount[name]
        row[0]+=int(arr.size);row[1]+=1;row[2]+=ns

    def prefixes(self,name,values,counts):
        lanes=int(np.prod(values.shape[2:]))
        for a,b,h in rectangles(counts,lanes):
            self.copy(name,values[a:b,:h],a,b-a)


def execute(image, fixture, out, cmaddr, artifact):
    desc=read(image/"build.json");cfg=desc["params"];p,r,f=(cfg[n] for n in ("P","R","F"))
    original,z=inputs(fixture)
    require(original["original_build_sha256"]==desc["original_build_sha256"] and original["params"]==cfg,"matched resident image and fixture")
    out.mkdir(parents=True,exist_ok=False)
    os.chdir(out)
    target=Device(image,desc,"hardware" if cmaddr else "simulator",cmaddr,artifact)
    file=out/"result.json"
    state=dict(stopped=False,build_sha256=sha(image/"build.json"),fixture_sha256=sha(fixture/"fixture.json"),
               worker_sha256=sha(Path(__file__)),wire_codec_sha256=sha(HERE.parent/"window_cache.py"),
               loaded_image=target.image,platform=target.platform,frames=[])
    save(file,state)
    n=z["n"];parts=len(z["counts"]);peers={}
    for frame,(use,scale) in enumerate(FRAMES,1):
        hot,table,wire,payload,hits=pack(z,scale,use,32)
        ctl=np.zeros((p,5),np.uint32)
        ctl[:,0]=frame;ctl[:,1]=1;ctl[:,2]=parts;ctl[:,3]=n
        tx=Transfers(target)
        if frame==1:tx.copy("weights",z["weights"],0,p,broadcast=True)
        tx.copy("control",ctl,0,p)
        tx.prefixes("indices",z["ids"],n)
        tx.prefixes("inverse",z["inv"],n)
        tx.prefixes("roots",z["root"]*np.float32(scale),n)
        if len(hot):tx.copy("rawx",table,0,p,broadcast=True)
        target.dev.launch("begin_frame",nonblock=False)
        for k in range(parts):
            ctl[:,4]=z["counts"][k] | (hits[k]<<16)
            tx.copy("control",ctl,0,p)
            tx.prefixes("targets",z["ends"][k],n)
            tx.prefixes("features",payload[k],wire[k])
            target.dev.launch("accumulate_cached" if use else "accumulate",nonblock=False)
        target.dev.launch("finish",nonblock=False)
        raw=np.empty((p,4+(parts+2)*6+r*(1+f)),np.uint32)
        target.dev.memcpy_d2h(raw.ravel(),target.dev.get_id("reply"),0,0,p,1,raw.shape[1],streaming=False,
                             nonblock=False,data_type=target.dtype,order=target.order)
        y,metrics=inspect(raw,z,cfg,frame,scale)
        if scale in peers:np.testing.assert_array_equal(y.view(np.uint32),peers[scale])
        else:peers[scale]=y.view(np.uint32).copy()
        name=f"frame-{frame}.npz"
        np.savez_compressed(out/name,reply=raw,original_ids=z["hidden_nodes"],hidden=y,hot=hot,hits=hits,wire=wire)
        state["frames"].append(dict(frame=frame,use_table=use,scale=scale,hits=int(hits.sum()),source_words=int(wire.sum()),
                                    source_table_words=int(table.size),by_symbol=dict(tx.amount),raw_sha256=sha(out/name),**metrics))
        save(file,state)
        print(frame,use,scale,metrics["peak_active_cycles"],tx.amount["features"][:2],flush=True)
    target.stop();state["stopped"]=True;save(file,state)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("image","fixture","out"):parser.add_argument(name,type=Path)
    parser.add_argument("--cmaddr");parser.add_argument("--artifact",type=Path)
    a=parser.parse_args()
    execute(a.image.resolve(),a.fixture.resolve(),a.out.resolve(),a.cmaddr,a.artifact.resolve() if a.artifact else None)
