"""Run and audit the same eight-shard SELL payload with CSL memcpy RPCs."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder
from contract import PIN,check,input_words,packed_graph,require,sha,verify_graph


def run(a):
    with np.load(a.inputs) as z:
        graph={k:z[k] for k in z}
    require(sha(a.inputs)==PIN,"frozen E149 input")
    verify_graph(graph)
    a.output.mkdir(parents=True,exist_ok=False)
    record=dict(backend="hardware" if a.cmaddr else "simulator",input_sha256=PIN,
                worker_sha256=sha(Path(__file__)),trials=[],stopped=False)
    def save(stage):
        record["stage"]=stage
        tmp=a.output/"status.tmp"
        tmp.write_text(json.dumps(record,indent=2)+"\n")
        tmp.replace(a.output/"result.json")
    save("load")
    dev=SdkRuntime(str(a.artifact),cmaddr=a.cmaddr,**({} if a.cmaddr else dict(simfab_numthreads=1,suppress_simfab_trace=True)))
    def transfer(symbol,arr,put=False,y=0):
        h,w,n=arr.shape
        half=arr.dtype==np.uint16
        data=arr.astype(np.uint32) if half else np.ascontiguousarray(arr)
        sym=dev.get_id(symbol)
        args=(sym,data.ravel()) if put else (data.ravel(),sym)
        op=dev.memcpy_h2d if put else dev.memcpy_d2h
        op(*args,0,y,w,h,n,streaming=False,nonblock=False,order=MemcpyOrder.ROW_MAJOR,
           data_type=MemcpyDataType.MEMCPY_16BIT if half else MemcpyDataType.MEMCPY_32BIT)
        return data.astype(np.uint16) if half else data
    dev.load()
    dev.run()
    try:
        immutable=packed_graph(graph)
        for key,arr in immutable.items():
            transfer(key,arr,True)
        for trial in (1,2,3):
            save(f"frame-{trial}")
            panel=input_words(graph,trial)[2:].view(np.float32).reshape(4,2,64)
            transfer("features",panel,True)
            begin=time.perf_counter_ns()
            dev.launch("spmm",nonblock=False)
            completed=transfer("completed",np.empty((4,2,1),np.uint32))
            dt=time.perf_counter_ns()-begin
            require(np.all(completed==trial),"all eight PEs completed wave "+str(trial))
            clocks=transfer("clocks",np.empty((4,2,6),np.uint16))
            owners=transfer("owner",np.empty((1,2,256),np.float32),y=3)[0]
            require(np.array_equal(transfer("features",np.empty_like(panel)),panel),"resident panel")
            reply=[]
            for x in range(2):
                for y in range(4):
                    reply.extend([trial,2*y+x,64,*clocks[y,x].astype(int).tolist()])
                reply.extend(owners[x].view(np.uint32).tolist())
            values=check(graph,trial,reply)
            record["trials"].append(dict(trial=trial,clocks=clocks.tolist(),owner_words=owners.view(np.uint32).tolist(),
                                         completed=completed[:,:,0].tolist(),launch_to_marker_ns=dt,**values))
        for key,arr in immutable.items():
            require(np.array_equal(transfer(key,np.empty_like(arr)),arr),"immutable graph "+key)
    finally:
        save("stop")
        dev.stop()
        record["stopped"]=True
    save("PASS")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("artifact","inputs","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--cmaddr")
    run(p.parse_args())
