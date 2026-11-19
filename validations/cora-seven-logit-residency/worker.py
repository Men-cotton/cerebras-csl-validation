"""Run a 129-frame seven-logit Cora window on an SDK or Direct Link worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

from contract import FRAMES,PIN,TAG,decode,load_graph,panel,require,save,sha,verify

def run(a):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkCompileArtifacts,SdkRuntime,SdkTarget,SimfabConfig,get_platform,MemcpyDataType,MemcpyOrder
    z=load_graph(a.input)
    out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    artifact=a.artifact.resolve()
    bundle=json.loads(a.manifest.read_text())
    role="hardware" if a.cmaddr else "simulator"
    case=a.case
    direct=case!="csl-planned"
    state=dict(case=case,role=role,trials=[],stopped=False,input_sha256=PIN,release_sha256=sha(a.manifest),
               code_sha256={n:sha(Path(__file__).with_name(n)) for n in ("worker.py","contract.py")})
    require(state["code_sha256"]==bundle["host"],"staged worker")
    def checkpoint(stage):
        state["stage"]=stage
        save(out/"result.json",state)
    checkpoint("platform")
    os.chdir(out)
    if direct:
        elf=sha(artifact/"program.elf")
        require(elf==bundle["direct"][case][role]["elf_sha256"] and sha(artifact/"out_port_map.json")==bundle["port_sha256"],"Direct Link code and ports")
        state["elf_sha256"]=elf
        platform=get_platform(a.cmaddr,SimfabConfig(num_threads=1,suppress_trace=True),SdkTarget.WSE3)
        require(bool(platform.is_system())==(role=="hardware") and bool(platform.is_simulation())==(role=="simulator"),"runtime platform")
        rt=SdkRuntime(SdkCompileArtifacts(str(artifact)).add_port_mapping(str(artifact/"out_port_map.json")),platform,memcpy_required=False,msg_level="INFO")
        coords=[list(rt.coord_logical_to_physical((x,y))) for y in range(2) for x in range(4)]
        require(coords==[[x,y] for y in range(2) for x in range(4)],"unshifted owners and gather row")
        state["coordinates"]=coords
    else:
        images={p.name:sha(p) for p in sorted((artifact/"bin").glob("*.elf"))}
        require(bool(images),"CSL application ELF inventory")
        state["csl_elf_sha256"]=images
        if role=="simulator":
            require(images==bundle["simulator_csl_elf_sha256"],"CSL simulated application ELFs")
        params=json.loads((artifact/"out.json").read_text())["params"]
        require(int(params["P"])==4 and int(params["MODE"])==7,"four CSL owners")
        rt=SdkRuntime(str(artifact),cmaddr=a.cmaddr,**({} if a.cmaddr else dict(simfab_numthreads=1,suppress_simfab_trace=True)))
    checkpoint("load")
    rt.load()
    checkpoint("run")
    rt.run()
    def io(name,arr,put=False):
        arr=np.ascontiguousarray(arr)
        owners,n=arr.shape
        cfg=dict(streaming=False,nonblock=False,order=MemcpyOrder.ROW_MAJOR,
                 data_type=MemcpyDataType.MEMCPY_32BIT if arr.dtype==np.float32 else MemcpyDataType.MEMCPY_16BIT)
        sym=rt.get_id(name)
        if put:
            rt.memcpy_h2d(sym,arr.ravel(),0,0,owners,1,n,**cfg)
        else:
            rt.memcpy_d2h(arr.ravel(),sym,0,0,owners,1,n,**cfg)
        return arr
    try:
        checkpoint("graph")
        carriers=(("ptr","ptr"),("src","local"),("weight","weight"))
        if direct:
            rt.report_port_infos()
            rx,tx=rt.get_port_id("rx_port"),rt.get_port_id("tx_port")
            require(rx is not None and tx is not None and rx!=tx,"host ports")
            pre=np.concatenate([z[k][y].view(np.uint32) for y in range(4) for _,k in carriers])
        else:
            for dest,key in carriers:
                io(dest,z[key],True)
            rt.launch("prepare",nonblock=False)
            setup=io("clocks",np.empty((4,6),np.uint32))
        for i in range(FRAMES):
            if i%3==0:
                checkpoint(f"frame-{i+1}")
            x=panel(z,i)
            raw=dict(index=i,feature_sha256=hashlib.sha256(x.tobytes()).hexdigest())
            if direct:
                packet=np.concatenate((np.array([i+1,TAG],np.uint32),x.ravel().view(np.uint32)))
                if i==0:
                    packet=np.concatenate((pre,packet))
                guarded=np.full(4*(14+16*7)+4,0xbad0c0de,np.uint32)
                response=guarded[2:-2]
                start=time.perf_counter_ns()
                recv=rt.receive(tx,response,response.size,nonblock=True)
                send=rt.send(rx,packet,packet.size,nonblock=True)
                rt.task_wait(send)
                rt.task_wait(recv)
                raw["send_to_reply_ns"]=time.perf_counter_ns()-start
                require(np.all(guarded[[0,1,-2,-1]]==0xbad0c0de),"output guards")
                response=response.reshape(4,14+16*7)
                raw.update(reply=response.ravel().tolist(),clocks=response[:,2:8].tolist(),setup=response[:,8:14].tolist(),
                           output_words=response[:,14:].reshape(64,7).tolist())
            else:
                start=time.perf_counter_ns()
                padded=np.pad(x,((0,0),(0,1)))
                io("features",np.tile(padded.ravel(),(4,1)),True)
                rt.launch("aggregate",nonblock=False)
                y=io("out",np.full((4,16*8),np.nan,np.float32))
                raw["input_through_output_ns"]=time.perf_counter_ns()-start
                clocks=io("clocks",np.empty((4,6),np.uint32))
                raw.update(clocks=clocks.tolist(),setup=setup.tolist(),output_words=y.reshape(64,8).view(np.uint32).tolist())
            raw["numeric"]=decode(z,i,raw,direct)
            state["trials"].append(raw)
        if not direct:
            for dest,key in carriers:
                np.testing.assert_array_equal(io(dest,np.empty_like(z[key])),z[key])
    finally:
        checkpoint("stop")
        rt.stop()
        state["stopped"]=True
    state["stage"]="PASS"
    state["summary"]=verify(z,state,role,case)
    checkpoint("PASS")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",required=True,choices=("csl-planned","tungsten-planned","tungsten-unrolled"))
    for name in ("artifact","output","input","manifest"):
        p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--cmaddr")
    run(p.parse_args())
