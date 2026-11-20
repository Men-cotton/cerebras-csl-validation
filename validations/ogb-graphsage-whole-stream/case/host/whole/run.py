"""Process all destination windows in one resident GraphSAGE final-layer program."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from rectangles import send_prefixes, broadcast_weights
from dispatch import load,read,save,sha,slice_graph,pack,packet
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"device"))
from contract import check_frame


def trimmed(raw,cfg,actual):
    p,r,q,top=[cfg[k] for k in ("P","R","Q","K")]
    raw=raw.reshape(p,4+6*(2+top)+r*(1+q))
    keep=[raw[:,:4+6*(1+actual)],raw[:,4+6*(1+top):]]
    return np.concatenate(keep,axis=1)


def run(a):
    from cerebras.sdk.runtime.sdkruntimepybind import (SdkRuntime,SdkCompileArtifacts,SdkTarget,SimfabConfig,
            get_platform,MemcpyDataType,MemcpyOrder)
    root=a.build.resolve();fixture=a.input.resolve();dest=a.output.resolve()
    m=read(root/"build.json");cfg=m["params"];p,r,q,k=[cfg[key] for key in ("P","R","Q","K")]
    if sha(fixture)!=m["input_sha256"] or sha(root/"windows.json")!=m["windows_sha256"]:raise ValueError("case input binding")
    g=load(fixture);table=read(root/"windows.json")
    total=len(table);guards=m["control_windows"]
    counts=np.array([w["edges"] for w in table])
    initial=sorted(set([int(np.argmin(counts)),int(np.argsort(counts)[len(counts)//2]),int(np.argmax(counts))]))
    order=initial+[i for i in range(total) if i not in initial]
    if a.limit is not None:order=order[:a.limit]
    role="hardware" if a.cmaddr else "simulator"
    dest.mkdir(parents=True,exist_ok=False)
    state=dict(stage="loading",language=m["language"],role=role,params=cfg,build_sha256=sha(root/"build.json"),
          input_sha256=sha(fixture),windows_sha256=m["windows_sha256"],expected_windows=total,requested_order=order,
          selected_guards=guards,completed=[],stopped=False,source={p.name:sha(p) for p in (HERE/"run.py",HERE/"dispatch.py",HERE.parent/"device/contract.py")})
    state["transfer_sha256"] = sha(Path(__file__).with_name("rectangles.py"))
    def statewrite():save(dest/"result.json",state)
    statewrite()
    os.chdir(dest)
    platform=get_platform(a.cmaddr,SimfabConfig(num_threads=1,suppress_trace=True),SdkTarget.WSE3)
    state["platform"]={"system":bool(platform.is_system()),"simulation":bool(platform.is_simulation())}
    if state["platform"]!={"system":bool(a.cmaddr),"simulation":not bool(a.cmaddr)}:raise ValueError("target mismatch")
    direct=m["language"]=="tungsten"
    if direct:
        art=root/role/"out"
        if sha(art/"program.elf")!=m["roles"][role]["elf"] or sha(art/"out_port_map.json")!=m["port_map_sha256"]:raise ValueError("different Direct Link image")
        dev=SdkRuntime(SdkCompileArtifacts(str(art)).add_port_mapping(str(art/"out_port_map.json")),platform,memcpy_required=False,msg_level="INFO")
        if any(tuple(dev.coord_logical_to_physical((x,0)))!=(x,0) for x in range(p)):raise ValueError("logical PE geometry")
        state["elf_sha256"]=m["roles"][role]["elf"]
    else:
        art=root/"out"
        inventory={f.name:sha(f) for f in (art/"bin").glob("*.elf")}
        if inventory!=m["elf"] or role!=m["role"]:raise ValueError("different CSL image")
        state["elf_sha256"]=inventory
        dev=SdkRuntime(str(art),cmaddr=a.cmaddr,**({} if a.cmaddr else {"simfab_numthreads":1,"suppress_simfab_trace":True}))
    def io(name,vals,put=True):
        ar=np.ascontiguousarray(vals).reshape(p,-1)
        fun=dev.memcpy_h2d if put else dev.memcpy_d2h
        args=(dev.get_id(name),ar.ravel()) if put else (ar.ravel(),dev.get_id(name))
        fun(*args,0,0,p,1,ar.shape[1],streaming=False,nonblock=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR)
        return ar
    dev.load();dev.run()
    state["stage"]="running";statewrite()
    wall=time.perf_counter()
    first=True
    try:
        if direct:rx,tx=dev.get_port_id("rx_port"),dev.get_port_id("tx_port")
        for index in order:
            w=table[index];z=slice_graph(g,w);arr=pack(g,z,cfg,w)
            actual=w["parts"];local_cfg=dict(cfg,K=actual)
            name=f"w{index:04d}"
            replies={};frames=[]
            for trial in ((1,2,3) if index in guards else (1,)):
                scale=np.float32(.5 if trial==2 else 1.)
                wire=packet(arr,trial,first)
                rep=np.zeros((p,4+6*(k+2)+r*(1+q)),np.uint32)
                begin=time.perf_counter_ns()
                if direct:
                    rec=dev.receive(tx,rep.ravel(),rep.size,nonblock=True)
                    send=dev.send(rx,wire,wire.size,nonblock=True)
                    dev.task_wait(send);dev.task_wait(rec)
                    inbound=wire.size
                else:
                    inbound=0
                    if first:
                        inbound += broadcast_weights(dev, arr["weights"], p, MemcpyDataType.MEMCPY_32BIT, MemcpyOrder.ROW_MAJOR)
                    control=np.zeros((p,3),np.uint32)
                    control[:,0]=trial;control[:,1]=arr["nrows"]
                    for key,values in (("control",control),("indices",arr["ids"]),("inverse",arr["inv"]),("roots",arr["root"]*scale)):
                        io(key,values);inbound+=values.size
                    dev.launch("begin_frame",nonblock=False)
                    for part in range(actual):
                        control[:,2]=arr["counts"][part]
                        io("control",control);io("targets",arr["rowend"][part])
                        inbound += control.size + arr["rowend"][part].size
                        inbound += send_prefixes(dev, "features", arr["feat"][part]*scale, control[:, 2], MemcpyDataType.MEMCPY_32BIT, MemcpyOrder.ROW_MAJOR)
                        dev.launch("accumulate",nonblock=False)
                    dev.launch("finish",nonblock=False)
                    rep=io("reply",rep,False)
                elapsed=time.perf_counter_ns()-begin;first=False
                rep=trimmed(rep,cfg,actual)
                output,ticks,err=check_frame(rep,z,arr,local_cfg,trial)
                if trial==1:baseline=output.copy()
                if trial==3:np.testing.assert_array_equal(baseline.view(np.uint32),output.view(np.uint32))
                replies["t"+str(trial)]=rep
                frames.append(dict(trial=trial,peak_sum_pe_cycles=int(ticks.sum(1).max()),sum_phase_max_cycles=int(ticks.max(0).sum()),
                     max_aggregation_cycles=int(ticks[:,1:-1].sum(1).max()),max_projection_cycles=int(ticks[:,-1].max()),
                     host_input_to_reply_ns=elapsed,input_words=int(inbound),response_words=p*(4+6*(k+2)+r*(1+q)),
                     max_abs=err["max_abs"],relative_l2=err["relative_l2"],argmax_vs_final_fp64=err["argmax_mismatch"],
                     argmax_vs_original_model=int((output.argmax(1)!=g["original_prediction"][z["seed"]]).sum()) if trial!=2 else None))
            np.savez_compressed(dest/(name+".npz"),**replies)
            state["completed"].append(dict(window=index,first=w["first"],last=w["last"],edges=w["edges"],parts=actual,
                          raw_sha256=sha(dest/(name+".npz")),frames=frames))
            statewrite()
            print(m["language"],index,len(state["completed"]),total,round(time.perf_counter()-wall,1),flush=True)
    finally:dev.stop()
    state["stopped"]=True
    state["stage"]="PASS" if len(order)==total else "PARTIAL"
    state["host_wall_s"]=time.perf_counter()-wall
    statewrite()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("build",type=Path);p.add_argument("input",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--limit",type=int);p.add_argument("--cmaddr")
    run(p.parse_args())
