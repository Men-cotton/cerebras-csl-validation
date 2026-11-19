"""Execute the frozen 2x4 E149 Direct Link pair and collect complete frames."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkCompileArtifacts, SdkRuntime, SdkTarget, SimfabConfig, get_platform

from contract import PIN, check, input_words, require, sha, verify_graph


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("build","artifact","inputs","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--cmaddr")
    args=p.parse_args()
    role="hardware" if args.cmaddr else "simulator"
    build=json.loads(args.build.read_text())
    artifact=args.artifact.resolve()
    require(sha(artifact/"program.elf")==build["roles"][role]["elf_sha256"],"ELF identity")
    require(sha(artifact/"out_port_map.json")==build["port_map_sha256"],"Direct Link port identity")
    require(sha(args.inputs)==build["input_sha256"]==PIN,"E149 fixture identity")
    with np.load(args.inputs) as z:
        graph={k:z[k] for k in z}
    verify_graph(graph)
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    record=dict(backend=role,elf_sha256=sha(artifact/"program.elf"),input_sha256=PIN,
                build_sha256=sha(args.build),worker_sha256=sha(Path(__file__)),trials=[],stopped=False)
    os.chdir(out)
    def stage(name):
        record["stage"]=name
        tmp=out/"status.tmp"
        tmp.write_text(json.dumps(record,indent=2)+"\n")
        tmp.replace(out/"result.json")
    stage("platform")
    platform=get_platform(args.cmaddr,SimfabConfig(num_threads=1,suppress_trace=True),SdkTarget.WSE3)
    require(bool(platform.is_system())==(role=="hardware") and bool(platform.is_simulation())==(role=="simulator"),"runtime target")
    dev=SdkRuntime(SdkCompileArtifacts(str(artifact)).add_port_mapping(str(artifact/"out_port_map.json")),
                   platform,memcpy_required=False,msg_level="INFO")
    coords=[list(dev.coord_logical_to_physical((x,y))) for y in range(4) for x in range(2)]
    record["physical_coordinates"]=coords
    require(coords==[[x,y] for y in range(4) for x in range(2)],"unshifted physical rectangle for Direct Link")
    stage("load")
    dev.load()
    stage("run")
    dev.run()
    try:
        dev.report_port_infos()
        rx,tx=dev.get_port_id("rx_port"),dev.get_port_id("tx_port")
        require(rx is not None and tx is not None and rx!=tx,"port IDs")
        for trial in (1,2,3):
            frame=input_words(graph,trial)
            guarded=np.full(588,0xbad0c0de,np.uint32)
            reply=guarded[2:-2]
            stage(f"frame-{trial}")
            begin=time.perf_counter_ns()
            receive=dev.receive(tx,reply,len(reply),nonblock=True)
            send=dev.send(rx,frame,len(frame),nonblock=True)
            dev.task_wait(send)
            dev.task_wait(receive)
            elapsed=time.perf_counter_ns()-begin
            require(np.all(guarded[[0,1,-2,-1]]==0xbad0c0de),"reply guards")
            captured=dict(trial=trial,reply=reply.tolist(),host_send_receive_wait_ns=elapsed)
            record["trials"].append(captured)
            stage(f"verify-{trial}")
            captured.update(check(graph,trial,reply))
    finally:
        stage("stop")
        dev.stop()
        record["stopped"]=True
    require(record["trials"][0]["owner_sha256"]==record["trials"][2]["owner_sha256"],"restored full output bytes")
    stage("PASS")


if __name__=="__main__":
    main()
