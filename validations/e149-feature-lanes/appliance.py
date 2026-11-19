"""One E149 physical action for the registered CSL watcher."""
import argparse
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import numpy as np

from contract import PIN, check, require, sha

ROOT=Path(__file__).resolve().parent
PORT="e4a311bc7f0f12ff52503703f4e233182c840339f765d01d49650022fe951d3e"
FLAGS="--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-parallelism=1 -o out"


def read(path):
    return json.loads(path.read_text())


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2)+"\n")


def audit_capture(backend, language, capture):
    build=read(ROOT/"pair/build.json")
    require(capture["backend"]==backend and capture["stage"]=="PASS" and capture["stopped"] is True,"completed backend")
    require(capture["input_sha256"]==PIN and len(capture["trials"])==3,"E149 test sequence")
    worker="worker.py" if language=="tungsten" else "csl_worker.py"
    require(capture["worker_sha256"]==sha(ROOT/worker),"worker bytes")
    with np.load(ROOT/"pair/inputs.npz") as graph:
        cycles=[]
        for trial,frame in enumerate(capture["trials"],1):
            require(frame["trial"]==trial,"trial order")
            if language=="tungsten":
                require(capture["physical_coordinates"]==[[x,y] for y in range(4) for x in range(2)] and
                        capture["elf_sha256"]==build["roles"][backend]["elf_sha256"] and
                        capture["build_sha256"]==sha(ROOT/"pair/build.json"),"Direct Link geometry and ELF")
                reply=frame["reply"]
            else:
                require(np.array_equal(frame["completed"],np.full((4,2),trial)),"per-PE completion markers")
                reply=[]
                for x in range(2):
                    for y in range(4):
                        reply.extend([trial,2*y+x,64,*frame["clocks"][y][x]])
                    reply.extend(frame["owner_words"][x])
            verdict=check(graph,trial,reply)
            require(all(frame[k]==v for k,v in verdict.items()),"raw clock and owner replay")
            cycles.append(verdict["max_pe_cycles"])
        return cycles


def verified():
    pair=ROOT/"pair"
    build=read(pair/"build.json")
    require(sha(pair/"inputs.npz")==build["input_sha256"]==PIN,"E149 weights/panel")
    require(sha(pair/"program.img")==build["machine_image_sha256"],"single machine image")
    for fn,pin in build["source"].items():
        require(Path(fn).name==fn and sha(pair/"source"/fn)==pin,"Tungsten sources")
    require(sha(ROOT/"aggregation.w")==build["source"]["aggregation.w"],"kernel source at build")
    for role,dims in (("hardware",[762,1172]),("simulator",[2,4])):
        target=pair/role/"out"
        require(build["roles"][role]["fabric"]==dims and sha(target/"program.elf")==build["roles"][role]["elf_sha256"],"ELF role")
        require(sha(target/"out_port_map.json")==build["port_map_sha256"]==PORT,"Direct Link assignment")
    return {language:audit_capture("simulator",language,read(ROOT/file)) for language,file in
            (("csl","csl-simulation/result.json"),("tungsten","pair/verification/result.json"))}


def pins():
    return {f:sha(ROOT/f) for f in ("csl/layout.csl","csl/pe.csl","csl_worker.py","worker.py","contract.py","pair/build.json")}


def main(a):
    print("simulator replay", verified(), flush=True)
    if a.action=="check":
        return
    from cerebras.appliance import logger
    from cerebras.sdk.client import SdkCompiler,SdkLauncher
    logging.basicConfig(level=logging.INFO)
    out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    opts=dict(simulator=False,mgmt_namespace=a.namespace,resource_cpu=1000,resource_mem=8<<30,disable_version_check=False)
    if a.action=="compile-csl":
        compile_opts=opts.copy()
        del compile_opts["simulator"]
        with SdkCompiler(**compile_opts) as compiler:
            artifact=Path(compiler.compile(str(ROOT/"csl"),"layout.csl",FLAGS,str(out))).resolve()
        require(artifact.is_file() and artifact.parent==out,"compile archive")
        write(out/"build.json",dict(artifact=artifact.name,sha256=sha(artifact),source_sha256=pins(),flags=FLAGS))
        return
    csl=a.action=="run-csl"
    if csl:
        record=read(a.build)
        artifact=(a.build.parent/record["artifact"]).resolve()
        require(artifact.parent==a.build.resolve().parent and record["sha256"]==sha(artifact)
                and record["source_sha256"]==pins() and record["flags"]==FLAGS,"CSL compile record binding")
        worker="csl_worker.py"
        stage=[ROOT/worker,ROOT/"contract.py",ROOT/"pair/inputs.npz"]
        extra=[]
    else:
        worker="worker.py"
        artifact=out/"bundle"
        (artifact/"out").mkdir(parents=True)
        for fn in ("program.elf","out_port_map.json"):
            shutil.copyfile(ROOT/"pair/hardware/out"/fn,artifact/"out"/fn)
        stage=[ROOT/worker,ROOT/"contract.py",ROOT/"pair/build.json",ROOT/"pair/inputs.npz"]
        extra=["--build","build.json"]
    command=shlex.join(["timeout","--signal=INT","--kill-after=15s","600s","cs_python",worker,"--artifact","out",
                        "--inputs","inputs.npz","--output","capture","--cmaddr","%CMADDR%",*extra])
    command+=' >worker.log 2>&1; code=$?; echo "$code" >worker-exit.txt; exit "$code"'
    state=dict(action=a.action,source_sha256=pins(),command=command,downloads={})
    write(out/"collection.json",state)
    os.chdir(out)
    with SdkLauncher(str(artifact),**opts) as launcher:
        try:
            for path in stage:
                launcher.stage(str(path))
            (out/"response.txt").write_text(str(launcher.run(command)))
        finally:
            for remote in ("capture/result.json","worker-exit.txt","worker.log"):
                dest=out/Path(remote).name
                try:
                    launcher.download_artifact(remote,str(dest))
                    state["downloads"][remote]=dict(sha256=sha(dest))
                except Exception as exc:
                    state["downloads"][remote]=dict(error=repr(exc))
                write(out/"collection.json",state)
    require(all("sha256" in v for v in state["downloads"].values()) and (out/"worker-exit.txt").read_text().strip()=="0","worker and collection")
    state["max_pe_cycles"]=audit_capture("hardware","csl" if csl else "tungsten",read(out/"result.json"))
    write(out/"collection.json",state)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("check","compile-csl","run-csl","run-tungsten"))
    p.add_argument("--out",type=Path)
    p.add_argument("--build",type=Path)
    p.add_argument("--namespace",default="job-operator")
    a=p.parse_args()
    if a.action!="check" and not a.out:
        p.error("--out is required for appliance work")
    if a.action=="run-csl" and not a.build:
        p.error("--build is required for CSL execution")
    main(a)
