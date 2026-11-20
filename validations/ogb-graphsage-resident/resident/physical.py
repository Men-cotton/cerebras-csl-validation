"""Compile, run and audit all phases of a two-layer trained model in one resident image."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from graph import read, require, save, sha
from compare import compare
from control import case_subset, controls, provenance
from limits import limits
from transport import fetch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"device"))
from appliance import canonical_elf,images_in_archive


def unpack(path,dest):
    dest.mkdir()
    with zipfile.ZipFile(path) as z:
        members=z.infolist()
        require(1<len(members)<10000 and len({e.filename for e in members})==len(members),"returned archive inventory")
        size=0
        for m in members:
            require(m.filename in ("result.json","windows.jsonl") or re.fullmatch(r"t[123]-s[012]-w[0-9]{4}\.npz",m.filename),
                    "invalid returned member")
            size+=m.file_size
            require(0<m.file_size<16<<20 and size<768<<20,"returned size bound")
            with z.open(m) as f,(dest/m.filename).open("wb") as g:shutil.copyfileobj(f,g)
    return dest


def replay(build,case,capture,out):
    subprocess.run([sys.executable,"-B",HERE/"run.py",build,case,out,"--replay",capture],check=True,
                   stdout=subprocess.DEVNULL,env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"})
    return read(out/"result.json")


def checked_case(name):
    require(name and Path(name).name==name,"case name")
    folder=ROOT/"cases"/name;rel=read(folder/"release.json")
    for n,h in rel["files"].items():require(sha(folder/n)==h,"changed case file "+n)
    for n,h in rel["host"].items():require(sha(ROOT/n)==h,"changed host file "+n)
    study=folder/"input";pilot=folder/rel["sdk_input"]
    case_subset(study,pilot)
    provenance(study,pilot,folder/"provenance")
    native,direct=read(folder/"csl/build.json"),read(folder/"tungsten/build.json")
    require(native["params"]==direct["params"] and native["dataset"]==direct["dataset"],"matched model geometry")
    require(limits(study,direct["params"],direct.get("nested_gather",False))==rel["tungsten_counts"],"actual graph transfer counts fit the machine")
    for lang,m in (("csl",native),("tungsten",direct)):
        for n,h in m["source"].items():require(sha(folder/lang/"src"/n)==h,"firmware source identity")
    for role,dims in (("simulator",[direct["params"]["P"],2]),("hardware",[762,1172])):
        root=folder/"tungsten"/role/"out"
        require(direct["roles"][role]["fabric"]==dims and sha(root/"program.elf")==direct["roles"][role]["elf"] and
                sha(root/"out_port_map.json")==direct["port_map_sha256"] and
                canonical_elf(root/"program.elf",dims)==direct["canonical_sha256"],"matched physical firmware/geometry")
    with tempfile.TemporaryDirectory() as raw:
        tmp=Path(raw)
        for lang in ("csl","tungsten"):
            b=folder/lang;capture=folder/("sim-"+lang)
            m=replay(b,pilot,capture,tmp/lang)
            require(m["planned_frames"]==rel["sdk_frames"] and m["splits"]==rel["sdk_splits"][lang],"causal SDK replay")
    require(compare(folder/"sim-csl",folder/"sim-tungsten")==rel["sdk_peer"],"frozen cross-language comparison")
    return folder,rel,native,direct


def execute(a):
    folder,rel,native,direct=checked_case(a.case)
    if a.action=="check":return
    from cerebras.sdk.client import SdkCompiler,SdkLauncher
    output=a.out.resolve();output.mkdir(parents=True,exist_ok=False)
    common=dict(mgmt_namespace=a.namespace,resource_cpu=1000,resource_mem=8<<30,disable_version_check=False)
    flags=[("--fabric-dims=762,1172" if f.startswith("--fabric-dims=") else f) for f in native["flags"]]
    key=sha(folder/"release.json")
    if a.action=="compile-csl":
        with SdkCompiler(**common) as compiler:
            archive=Path(compiler.compile(str(folder/"csl/src"),"layout.csl"," ".join(flags),str(output))).resolve()
        require(archive.is_file() and archive.parent==output,"received native compiler archive")
        save(output/"compiled.json",dict(artifact=archive.name,sha256=sha(archive),elf=images_in_archive(archive),
                                        release_sha256=key,flags=flags))
        return
    lang=a.action.removeprefix("run-")
    if lang=="csl":
        receipt=read(a.compiled);artifact=a.compiled.parent/receipt["artifact"]
        require(receipt["release_sha256"]==key and receipt["flags"]==flags and artifact.parent==a.compiled.parent and
                sha(artifact)==receipt["sha256"] and images_in_archive(artifact)==receipt["elf"],"physical CSL compilation identity")
        descriptor=native|{"role":"hardware","elf":receipt["elf"],"flags":flags}
    else:
        artifact=output/"artifact";shutil.copytree(folder/"tungsten/hardware/out",artifact/"out")
        descriptor=direct
    save(output/"build.json",descriptor)
    info=read(folder/"input/case.json")
    command=" && ".join((shlex.join(["cs_python","-B","remote.py","prepare"]),
        shlex.join(["timeout","--signal=INT","--kill-after=30s","7200s","cs_python","-B","resident/run.py",".",".","capture","--cmaddr","%CMADDR%"])))
    command="("+command+') >worker.log 2>&1; rc=$?; echo "$rc" >worker-exit.txt; cs_python -B remote.py collect; exit "$rc"'
    state=dict(release_sha256=key,requested_role="hardware",language=lang,downloads={})
    save(output/"collection.json",state);os.chdir(output)
    with SdkLauncher(str(artifact),simulator=False,**common) as launcher:
        try:
            inputs=[folder/"input"/n for n in ["case.json","graph.npz"]+[c["file"] for c in info["chunks"]]]
            for f in [HERE/"remote.py",HERE/"transport.py",HERE/"run.py",HERE/"graph.py",ROOT/"transfer/rectangles.py",output/"build.json",*inputs]:
                launcher.stage(str(f))
            start=time.perf_counter_ns()
            try:response=launcher.run(command)
            finally:state["launcher_elapsed_ns"]=time.perf_counter_ns()-start
            (output/"response.txt").write_text(str(response))
        finally:
            try:
                state["downloads"]["capture.zip"]=fetch(launcher,output)
            except Exception as exc:state["downloads"]["capture.zip"]={"error":repr(exc)}
            save(output/"collection.json",state)
            for name in ("worker.log","worker-exit.txt"):
                try:
                    launcher.download_artifact(name,str(output/name))
                    state["downloads"][name]={"sha256":sha(output/name)}
                except Exception as exc:state["downloads"][name]={"error":repr(exc)}
                save(output/"collection.json",state)
    require(all("sha256" in x for x in state["downloads"].values()) and
            (output/"worker-exit.txt").read_text().strip()=="0","complete hardware capture")
    with tempfile.TemporaryDirectory() as t:
        tmp=Path(t);actual=unpack(output/"capture.zip",tmp/"raw")
        verified=replay(output,folder/"input",actual,tmp/"replay")
        require(verified["role"]=="hardware" and verified["scope"]==info["scope"] and
                verified["phases"]["t1-s2"]["rows"]==info["destinations"],"actual device/full input coverage")
        state["control_words"]=controls(actual,folder/("sim-"+lang))
        if a.peer:
            other=unpack(a.peer,tmp/"other")
            save(output/"physical-peer.json",compare(other,actual))
        save(output/"replayed-model.json",verified)
    state["numerical_status"]="PASS";save(output/"collection.json",state)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("check","compile-csl","run-csl","run-tungsten"))
    p.add_argument("--case",required=True);p.add_argument("--compiled",type=Path);p.add_argument("--out",type=Path)
    p.add_argument("--peer",type=Path);p.add_argument("--namespace",default="job-operator")
    a=p.parse_args()
    if a.action!="check" and not a.out:p.error("output directory required")
    if a.action=="run-csl" and not a.compiled:p.error("physical compile record required")
    if a.compiled:a.compiled=a.compiled.resolve()
    if a.peer:a.peer=a.peer.resolve()
    execute(a)
