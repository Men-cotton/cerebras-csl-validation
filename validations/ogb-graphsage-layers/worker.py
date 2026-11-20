"""Run all A, half-A, A frames through one resident OGB layer."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from contract import check_frame, load, pack, payload, sha, write


def run(a):
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkCompileArtifacts,SdkRuntime,SdkTarget,SimfabConfig,get_platform,MemcpyOrder,MemcpyDataType)
    out=a.output.resolve()
    build=a.build.resolve()
    z=load(a.fixture)
    m=json.loads((build/"build.json").read_text())
    if sha(a.fixture)!=m["fixture_sha256"]: raise ValueError("fixture differs from the compiled case")
    cfg,arrays=pack(z,m["params"]["P"],m["placement"],m["edge_capacity"],m["params"].get("S",1))
    if cfg!=m["params"]: raise ValueError("different packed graph")
    overrides={}
    if a.override_x:
        with np.load(a.override_x,allow_pickle=False) as sub:
            if not np.array_equal(sub["node"],z["node"]): raise ValueError("incoming layer node ids differ")
            for trial in (1,2,3):
                x=sub["logits"+str(trial)]
                if x.shape!=z["x"].shape or x.dtype!=np.float32: raise ValueError("incoming layer shape")
                overrides[trial]=x
    p,r,k,q=(cfg[n] for n in ("P","R","K","Q"))
    direct=m["language"]=="tungsten"
    role="hardware" if a.cmaddr else "simulator"
    if a.artifact: a.artifact=a.artifact.resolve()
    out.mkdir(parents=True,exist_ok=False)
    report=dict(language=m["language"],role=role,fixture_sha256=m["fixture_sha256"],params=cfg,
          placement=m["placement"],build_sha256=sha(build/"build.json"),frames=[],stopped=False,
          override_x_sha256=sha(a.override_x) if a.override_x else None)
    def mark(stage):
        report["stage"]=stage
        write(out/"result.json",report)
    mark("starting")
    os.chdir(out)
    platform=get_platform(a.cmaddr,SimfabConfig(num_threads=1,suppress_trace=True),SdkTarget.WSE3)
    if platform.is_system()!=bool(a.cmaddr) or platform.is_simulation()!=bool(not a.cmaddr):
        raise ValueError("unexpected SDK target")
    report["platform"]={"system":bool(platform.is_system()),"simulation":bool(platform.is_simulation())}
    report["host_sha256"]={n:sha(Path(__file__).resolve().parent/n) for n in ("worker.py","contract.py")}
    if direct:
        artifact=a.artifact.resolve() if a.artifact else build/role/"out"
        if sha(artifact/"program.elf")!=m["roles"][role]["elf"] or sha(artifact/"out_port_map.json")!=m["port_map_sha256"]:
            raise ValueError("Direct Link image or port map mismatch")
        report["executed_elf_sha256"]=sha(artifact/"program.elf")
        dev=SdkRuntime(SdkCompileArtifacts(str(artifact)).add_port_mapping(str(artifact/"out_port_map.json")),
                       platform,memcpy_required=False,msg_level="INFO")
        if any(tuple(dev.coord_logical_to_physical((x,0)))!=(x,0) for x in range(p)):
            raise ValueError("unexpected Direct Link coordinate offset")
    else:
        artifact=a.artifact.resolve() if a.artifact else build/"out"
        actual_params=json.loads((artifact/"out.json").read_text())["params"]
        if {k:int(actual_params[k]) for k in cfg}!=cfg:raise ValueError("compiled geometry or workload mismatch")
        if m["role"]!=role or {p.name:sha(p) for p in (artifact/"bin").glob("*.elf")}!=m["elf"]:
            raise ValueError("CSL image role or checksum mismatch")
        report["executed_elf_sha256"]=m["elf"]
        dev=SdkRuntime(str(artifact),cmaddr=a.cmaddr,**({} if a.cmaddr else dict(simfab_numthreads=1,suppress_simfab_trace=True)))
    def io(name,values,put=True):
        ar=np.ascontiguousarray(values).reshape(p,-1)
        func=dev.memcpy_h2d if put else dev.memcpy_d2h
        args=(dev.get_id(name),ar.ravel()) if put else (ar.ravel(),dev.get_id(name))
        func(*args,0,0,p,1,ar.shape[1],streaming=False,nonblock=False,
             order=MemcpyOrder.ROW_MAJOR,data_type=MemcpyDataType.MEMCPY_32BIT)
        return ar
    mark("load")
    dev.load();dev.run()
    raw={}
    try:
        if direct:
            rx=dev.get_port_id("rx_port");tx=dev.get_port_id("tx_port")
        for trial in (1,2,3):
            mark("frame-"+str(trial))
            words=4+6*(k+2+int(cfg.get("S",1)>1))+r*(1+q)
            reply=np.full((p,words),0xbad00bad,np.uint32)
            scale=np.float32(0.5 if trial==2 else 1.)
            if overrides:
                z["x"]=overrides[trial]
                _,arrays=pack(z,p,m["placement"],m["edge_capacity"],m["params"].get("S",1))
                scale=np.float32(1.)
            wire=payload(arrays,trial,scale)
            host_started=time.perf_counter_ns()
            csl_in_words=0
            if direct:
                rec=dev.receive(tx,reply.ravel(),reply.size,nonblock=True)
                send=dev.send(rx,wire,wire.size,nonblock=True)
                dev.task_wait(send);dev.task_wait(rec)
            else:
                if trial==1:
                    io("weights",np.tile(arrays["weights"],(p,1)))
                    csl_in_words+=p*len(arrays["weights"])
                control=np.zeros((p,3),np.uint32);control[:,0]=trial;control[:,1]=arrays["nrows"]
                io("control",control);io("indices",arrays["ids"]);io("inverse",arrays["inv"])
                io("roots",arrays["root"]*scale)
                csl_in_words+=control.size+arrays["ids"].size+arrays["inv"].size+arrays["root"].size
                dev.launch("begin_frame",nonblock=False)
                for j in range(k):
                    control[:,2]=arrays["counts"][j]
                    ids=arrays["rowid"][j]
                    if m.get("row_runs"):
                        # Convert each sorted local block to exclusive CSR row ends.
                        ends=np.zeros_like(ids)
                        for pe in range(p):
                            edges=int(arrays["counts"][j,pe])
                            ends[pe,:r]=np.cumsum(np.bincount(ids[pe,:edges].astype(np.int64),minlength=r))
                        ids=ends
                    io("control",control);io("targets",ids);io("features",arrays["feat"][j]*scale)
                    csl_in_words+=control.size+arrays["rowid"][j].size+arrays["feat"][j].size
                    dev.launch("accumulate",nonblock=False)
                dev.launch("finish",nonblock=False)
                reply=io("reply",reply,False)
            ns=time.perf_counter_ns()-host_started
            logits,ticks,errors=check_frame(reply,z,arrays,cfg,trial,scale)
            key=str(trial);raw["reply"+key]=reply;raw["logits"+key]=logits;raw["cycles"+key]=ticks
            report["frames"].append(dict(trial=trial,clocks=["clear"]+["edge-chunk"]*k+(["reduce"] if cfg.get("S",1)>1 else [])+["project"],
                    peak_sum_pe_cycles=int(ticks.sum(1).max()),sum_phase_max_cycles=int(ticks.max(0).sum()),
                    max_aggregation_cycles=int(ticks[:,1:1+k].sum(1).max()),max_projection_cycles=int(ticks[:,-1].max()),
                    max_reduce_cycles=int(ticks[:,-2].max()) if cfg.get("S",1)>1 else 0,
                    host_input_to_reply_ns=ns,input_word_count=int(wire.size if direct else csl_in_words),
                    output_word_count=int(reply.size),numerics=errors))
    finally:
        mark("stop");dev.stop()
    if not np.array_equal(raw["logits1"].view(np.uint32),raw["logits3"].view(np.uint32)):
        raise ValueError("A replay differs")
    if np.array_equal(raw["logits1"],raw["logits2"]):raise ValueError("changed input produced identical layer")
    raw["node"]=z["seed"]
    np.savez_compressed(out/"raw.npz",**raw)
    report["raw_sha256"]=sha(out/"raw.npz");report["stopped"]=True;mark("PASS")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("build",type=Path);p.add_argument("fixture",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--cmaddr",default=None)
    p.add_argument("--artifact",type=Path,help="runtime artifact directory staged by SdkLauncher")
    p.add_argument("--override-x",type=Path,help="feed prior device layer output aligned by original ID")
    run(p.parse_args())
