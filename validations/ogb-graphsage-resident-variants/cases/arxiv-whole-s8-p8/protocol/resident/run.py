"""Drive the one-resident-image GraphSAGE model and replay every phase from raw PE words."""
import argparse
from collections import defaultdict
from pathlib import Path
import os
import sys
import time

import numpy as np

from graph import decode, load_case, oracle, packed, read, require, save, schedule, sha, weights, wire

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "transfer"))
from rectangles import broadcast_weights, send_prefixes


def comparison(actual, expected):
    delta = actual.astype(np.float64) - expected
    diff2 = float(np.square(delta).sum())
    ref2 = float(np.square(expected).sum())
    err = float(np.abs(delta).max())
    require(np.isfinite(actual).all() and np.allclose(actual, expected, rtol=4e-4, atol=3e-4) and
            diff2 <= max(ref2, 1e-20)*1.e-8, f"FP64 discrepancy {err}, relative norm {np.sqrt(diff2/max(ref2,1e-30))}")
    return err, diff2, ref2


class Device:
    def __init__(self, build, desc, role, cmaddr, artifact):
        from cerebras.sdk.runtime.sdkruntimepybind import (SdkRuntime, SdkCompileArtifacts, SdkTarget, SimfabConfig,
                                                          get_platform, MemcpyOrder, MemcpyDataType)
        self.order=MemcpyOrder.ROW_MAJOR
        self.dtype=MemcpyDataType.MEMCPY_32BIT
        self.cfg=desc["params"]
        p=self.cfg["P"]
        self.direct=desc["language"]=="tungsten"
        platform=get_platform(cmaddr,SimfabConfig(num_threads=1,suppress_trace=True),SdkTarget.WSE3)
        self.platform={"system":bool(platform.is_system()),"simulation":bool(platform.is_simulation())}
        require(self.platform=={"system":role=="hardware","simulation":role=="simulator"},"SDK target identity")
        if self.direct:
            art=artifact or build/role/"out"
            self.image=sha(art/"program.elf")
            require(self.image==desc["roles"][role]["elf"] and sha(art/"out_port_map.json")==desc["port_map_sha256"],"Direct Link image binding")
            self.dev=SdkRuntime(SdkCompileArtifacts(str(art)).add_port_mapping(str(art/"out_port_map.json")),
                                platform,memcpy_required=False,msg_level="INFO")
            require(all(tuple(self.dev.coord_logical_to_physical((x,0)))==(x,0) for x in range(p)),"Direct Link owner coordinates")
        else:
            art=artifact or build/"out"
            self.image={p.name:sha(p) for p in (art/"bin").glob("*.elf")}
            require(role==desc["role"] and self.image==desc["elf"] and
                    {k:int(v) for k,v in read(art/"out.json")["params"].items()}==self.cfg,"native image binding")
            self.dev=SdkRuntime(str(art),cmaddr=cmaddr,**({} if cmaddr else {"simfab_numthreads":1,"suppress_simfab_trace":True}))
        self.dev.load();self.dev.run()
        if self.direct:self.rx,self.tx=self.dev.get_port_id("rx_port"),self.dev.get_port_id("tx_port")

    def exchange(self, batch, step, boot):
        cfg=self.cfg;p,r,f,q,k=cfg["P"],cfg["R"],cfg["F"],cfg["Q"],step["parts"]
        fields=(2*f,f,q)[step["mode"]]
        raw=np.zeros((p,4+6*(k+2+int(step.get('shards',1)>1))+r*(1+fields)),np.uint32)
        dev=self.dev
        def unicast(name, values):
            arr=np.ascontiguousarray(values).reshape(p,-1)
            dev.memcpy_h2d(dev.get_id(name),arr.ravel(),0,0,p,1,arr.shape[1],streaming=False,nonblock=False,
                           data_type=self.dtype,order=self.order)
            return arr.size
        def prefixes(name,values,count):
            return send_prefixes(dev,name,values,count,self.dtype,self.order)
        started=time.perf_counter_ns()
        if self.direct:
            packet=wire(batch,step,boot)
            words=packet.size
            rec=dev.receive(self.tx,raw.ravel(),raw.size,nonblock=True)
            send=dev.send(self.rx,packet,packet.size,nonblock=True)
            dev.task_wait(send);dev.task_wait(rec)
        else:
            words=broadcast_weights(dev,boot[1],p,self.dtype,self.order) if boot is not None else 0
            ctl=np.zeros((p,5),np.uint32)
            ctl[:,0]=step["tag"];ctl[:,1]=step.get("wire_mode",step["mode"]);ctl[:,2]=k;ctl[:,3]=batch["n"]
            words+=unicast("control",ctl)
            words+=prefixes("indices",batch["ids"],batch["n"])
            if step["mode"]:words+=prefixes("inverse",batch["inv"],batch["n"])
            words+=prefixes("roots" if step["mode"] else "rawx",batch["root"],batch["n"])
            dev.launch("begin_frame",nonblock=False)
            for part in range(k):
                count=batch["counts"][part]
                ctl[:,4]=count
                words+=unicast("control",ctl)
                # Each live row needs an end even on owners with no edges in this part.
                words+=prefixes("targets",batch["ends"][part],batch["n"])
                words+=prefixes("features",batch["bank"][part],count)
                dev.launch("accumulate",nonblock=False)
            dev.launch("finish",nonblock=False)
            dev.memcpy_d2h(raw.ravel(),dev.get_id("reply"),0,0,p,1,raw.shape[1],streaming=False,nonblock=False,
                           data_type=self.dtype,order=self.order)
        return raw,int(words),time.perf_counter_ns()-started

    def stop(self):
        self.dev.stop()


def run(a):
    build,case=a.build.resolve(),a.case.resolve()
    destination=a.output.resolve()
    record=read(build/"build.json");cfg=record["params"]
    identity,g=load_case(case)
    require(g["x"].shape[1]==cfg["D"] and len(g["b0"])==cfg["F"] and len(g["b1"])<=cfg["Q"],"data/firmware dimensions")
    plan=schedule(g,cfg)
    mats=weights(g,cfg)
    capture=a.replay.resolve() if a.replay else None
    role="hardware" if a.cmaddr else "simulator"
    if capture:
        before=read(capture/"result.json");role=before["role"]
        with (capture/"windows.jsonl").open() as log:
            import json
            observed=[json.loads(line) for line in log]
        require(len(observed)==len(plan) and before["stage"]=="PASS" and before["stopped"] and
                sha(capture/"windows.jsonl")==before["journal_sha256"],"complete returned capture")
    destination.mkdir(parents=True,exist_ok=False)
    os.chdir(destination)
    state={"language":record["language"],"role":role,"dataset":identity["dataset"],"scope":identity["scope"],
           "case_sha256":sha(case/"case.json"),"build_sha256":sha(build/"build.json"),"params":cfg,"stopped":False,
           "planned_frames":len(plan),"completed_frames":0,"host_sha256":{
               p.name:sha(p) for p in (HERE/"run.py",HERE/"graph.py",HERE.parent/"transfer/rectangles.py")}}
    def status(stage):
        state["stage"]=stage;save(destination/"result.json",state)
    status("loading")
    device=None
    if capture:
        require(all(state[k]==before[k] for k in ("build_sha256","case_sha256","planned_frames","language","host_sha256")),"replay bindings")
        expected_img=record["roles"][role]["elf"] if record["language"]=="tungsten" else record["elf"]
        require(before["loaded_image"]==expected_img and before["platform"]=={"system":role=="hardware","simulation":role=="simulator"},"actual executed image and platform")
        state.update(loaded_image=expected_img,platform=before["platform"])
    else:
        device=Device(build,record,role,a.cmaddr,a.artifact.resolve() if a.artifact else None)
        state.update(loaded_image=device.image,platform=device.platform)
    status("running")
    total_parts={}
    base_A={}
    summary={}
    right=np.zeros(3,np.int64);counts=np.zeros(3,np.int64)
    prior_trial=0;own=None;exact=None
    all_fields=(2*cfg["F"],cfg["F"],len(g["b1"]))
    all_sizes=[len(g[k]) for k in ("raw_ids","a_ids","b_ids")]
    try:
        with (destination/"windows.jsonl").open("w",buffering=1) as journal:
            import json
            for index,step in enumerate(plan):
                trial,mode=step["trial"],step["mode"];scale=.5 if trial==2 else 1.
                if trial!=prior_trial:
                    own=[np.full((n,f),np.nan,np.float32) for n,f in zip(all_sizes,all_fields)]
                    exact=[np.full((n,f),np.nan,np.float64) for n,f in zip(all_sizes,all_fields)]
                    prior_trial=trial
                rows=step["rows"];pred=own[mode-1] if mode else None;ref=exact[mode-1] if mode else None
                batch=packed(g,step,cfg,pred,scale)
                name=f't{trial}-s{mode}-w{step["index"]:04d}.npz'
                boot=(len(plan),mats) if index==0 else None
                if capture:
                    prev=observed[index]
                    require(prev["name"]==name and sha(capture/name)==prev["raw_sha256"],"changed PE return")
                    with np.load(capture/name,allow_pickle=False) as z:raw=z["reply"]
                    input_words,elapsed_ns=prev["input_words"],prev["host_ns"]
                    digest=prev["raw_sha256"]
                else:
                    raw,input_words,elapsed_ns=device.exchange(batch,step,boot)
                    np.savez_compressed(destination/name,reply=raw)
                    digest=sha(destination/name)
                y,ticks,elapsed=decode(raw,g,step,batch,cfg)
                local_truth=oracle(g,step,pred,scale)
                full_truth=local_truth if mode==0 else oracle(g,step,ref,scale)
                local_error,_,_=comparison(y,local_truth)
                err,squared,norm=comparison(y,full_truth)
                own[mode][rows]=y;exact[mode][rows]=full_truth
                metrics=dict(name=name,tag=step["tag"],trial=trial,mode=mode,rows=len(rows),edges=step["edges"],parts=step["parts"]+int(step.get("shards",1)>1),edge_parts=step["parts"],shards=step.get("shards",1),
                     input_words=input_words,output_words=raw.size,raw_sha256=digest,host_ns=elapsed_ns,
                     peak_active_cycles=int(ticks.sum(1).max()),sum_phase_max_cycles=int(ticks.max(0).sum()),
                     peak_elapsed_cycles=int(elapsed.max()),max_aggregate_cycles=int(ticks[:,1:1+step["parts"]].sum(1).max()),
                     max_reduction_cycles=int(ticks[:,-2].max()) if step.get("shards",1)>1 else 0,
                     max_project_cycles=int(ticks[:,-1].max()),max_abs=err,local_max_abs=local_error)
                if capture:
                    for k,v in metrics.items():
                        if k in ("max_abs","local_max_abs"):
                            np.testing.assert_allclose(v,prev[k],rtol=1.e-9,atol=3.e-13)
                        else:require(prev[k]==v,"raw replay field "+k)
                journal.write(json.dumps(metrics,allow_nan=False)+"\n")
                key=f"t{trial}-s{mode}"
                if key not in summary:
                    summary[key]={k:0 for k in ("windows","rows","edges","input_words","output_words","sum_window_peak_active",
                                  "sum_window_phase_maxima","sum_window_peak_elapsed","fp64_argmax_mismatches","gpu_argmax_mismatches")}
                    summary[key].update(max_abs=0.,squared_error=0.,squared_reference=0.)
                s=summary[key]
                for key2,v in (("windows",1),("rows",len(rows)),("edges",step["edges"]),("input_words",input_words),
                               ("output_words",raw.size),("sum_window_peak_active",metrics["peak_active_cycles"]),
                               ("sum_window_phase_maxima",metrics["sum_phase_max_cycles"]),("sum_window_peak_elapsed",metrics["peak_elapsed_cycles"])):
                    s[key2]+=v
                s["max_abs"]=max(s["max_abs"],err);s["squared_error"]+=squared;s["squared_reference"]+=norm
                if mode==2:
                    labels=y.argmax(1);s["fp64_argmax_mismatches"]+=int(np.sum(labels!=full_truth.argmax(1)))
                    if trial!=2:s["gpu_argmax_mismatches"]+=int(np.sum(labels!=g["gpu_prediction"][rows]))
                    if trial==1:
                        for partition in range(3):
                            take=(g["split"][rows]==partition)
                            counts[partition]+=int(take.sum())
                            right[partition]+=int(np.sum(labels[take]==g["labels"][rows[take]]))
                if trial==1:
                    base_A[mode]=own[mode]
                if trial==3:
                    np.testing.assert_array_equal(base_A[mode][rows].view(np.uint32),y.view(np.uint32))
                state["completed_frames"]=index+1
                if index%10==0 or index==len(plan)-1:
                    status("running")
                    print(record["language"],trial,mode,step["index"],index+1,len(plan),flush=True)
    finally:
        if device:device.stop()
    for s in summary.values():
        s["relative_l2"]=float(np.sqrt(s.pop("squared_error")/max(s.pop("squared_reference"),1e-30)))
    state.update(stopped=True,journal_sha256=sha(destination/"windows.jsonl"),phases=summary,
                 splits={k:{"correct":int(right[i]),"count":int(counts[i])} for i,k in enumerate(("train","valid","test"))})
    require(state["completed_frames"]==len(plan) and summary["t1-s1"]["edges"]==identity["first_edges"] and
            summary["t1-s2"]["edges"]==identity["final_edges"] and summary["t1-s2"]["rows"]==identity["destinations"],"both actual whole dependency layers")
    status("PASS")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("build",type=Path);p.add_argument("case",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--replay",type=Path);p.add_argument("--cmaddr");p.add_argument("--artifact",type=Path)
    run(p.parse_args())
