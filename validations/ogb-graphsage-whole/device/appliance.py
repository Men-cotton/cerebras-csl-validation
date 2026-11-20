"""Validate and run an original OGB GraphSAGE layer on the physical CS-3."""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shlex
import shutil
import struct
import tarfile
import time
import numpy as np
from contract import sha,write,load
from audit import read,require,capture,compare

ROOT=Path(__file__).resolve().parent
PORT="e4a311bc7f0f12ff52503703f4e233182c840339f765d01d49650022fe951d3e"


def images_in_archive(path):
    with tarfile.open(path,"r:*") as arc:
        members={PurePosixPath(m.name):m for m in arc if m.isfile()}
        roots={p.parent.parent for p in members if p.parts[-2:]==("west","out.json")}
        roots={p for p in roots if p/"out.json" in members and p/"east/out.json" in members}
        require(len(roots)==1,"compiled archive must contain one application")
        root=next(iter(roots))
        images={p.name:hashlib.sha256(arc.extractfile(m).read()).hexdigest() for p,m in members.items()
                if p.parent==root/"bin" and p.suffix==".elf" and m.size<32<<20}
        require(bool(images),"empty CSL application")
        return images


def canonical_elf(path,dims):
    raw=bytearray(path.read_bytes())
    require(raw[:6]==b"\x7fELF\x02\x01","little-endian ELF64")
    eh=struct.unpack_from("<16sHHIQQQIHHHHHH",raw)
    nnotes,nloads=0,0
    for i in range(eh[10]):
        off=eh[5]+i*eh[9]
        ty,_,start,va,_,size,mem,_=struct.unpack_from("<IIQQQQQQ",raw,off)
        require(start+size<=len(raw),"incomplete segment")
        if ty==1:
            require(not (va<0xe400 and va+mem>0xc000),"invalid WSE SRAM range")
            raw[off+24:off+32]=bytes(8);nloads+=1
        elif ty==4:
            require(size==32 and raw[start:start+21]==struct.pack("<III",9,8,1)+b"Cerebras\0" and
                    raw[start+24:start+32]==struct.pack("<II",*dims),"unexpected fabric geometry")
            raw[start+21:start+32]=bytes(11);nnotes+=1
    require(nnotes==1 and nloads>0,"incomplete image")
    return hashlib.sha256(raw).hexdigest()


def checked_case(name):
    require(name and Path(name).name==name,"case name")
    folder=ROOT/"cases"/name
    release=read(folder/"release.json")
    for n,h in release["files"].items():require(sha(folder/n)==h,"changed case file "+n)
    for n,h in release["host"].items():require(sha(ROOT/n)==h,"changed host file "+n)
    src=read(folder/"native/build.json")
    t=read(folder/"direct/build.json")
    require(src["language"]=="csl" and t["language"]=="tungsten","language identity")
    require(src["fixture_sha256"]==t["fixture_sha256"]==sha(folder/"fixture.npz"),"frozen input")
    require(src["params"]==t["params"] and src["placement"]==t["placement"],"same layer geometry")
    for lang,m,sub in (("csl",src,"native"),("tungsten",t,"direct")):
        for filename,h in m["source"].items():require(sha(folder/sub/"src"/filename)==h,"compiled source")
        capture(folder/("sim-"+lang),folder/"fixture.npz",folder/sub/"build.json","simulator",attested=True)
    require(compare(folder/"sim-csl",folder/"sim-tungsten")==release["cross_language"],"cross-language output identity")
    for role,dims in (("simulator",[t["params"]["P"],2]),("hardware",[762,1172])):
        path=folder/"direct"/role/"out"
        require(t["roles"][role]["fabric"]==dims and sha(path/"program.elf")==t["roles"][role]["elf"] and
                sha(path/"out_port_map.json")==t["port_map_sha256"]==PORT,"hardware pair")
        require(canonical_elf(path/"program.elf",dims)==t["canonical_sha256"],"canonical program bytes")
    return folder,release,src,t


def execute(a):
    folder,release,native,direct=checked_case(a.case)
    if a.action=="check":return
    from cerebras.sdk.client import SdkCompiler,SdkLauncher
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    flags=[("--fabric-dims=762,1172" if s.startswith("--fabric-dims=") else s) for s in native["flags"]]
    common=dict(mgmt_namespace=a.namespace,resource_cpu=1000,resource_mem=8<<30,disable_version_check=False)
    key=sha(folder/"release.json")
    if a.action=="compile-csl":
        with SdkCompiler(**common) as compiler:
            archive=Path(compiler.compile(str(folder/"native/src"),"layout.csl"," ".join(flags),str(out))).resolve()
        require(archive.parent==out and archive.is_file(),"compiler archive")
        write(out/"compiled.json",{"artifact":archive.name,"archive_sha256":sha(archive),"flags":flags,
              "elf":images_in_archive(archive),"release_sha256":key,"case":a.case})
        return
    lang=a.action.removeprefix("run-")
    stage=out/"stage";stage.mkdir()
    if lang=="csl":
        rec=read(a.compiled);artifact=a.compiled.resolve().parent/rec["artifact"]
        require(artifact.parent==a.compiled.resolve().parent and rec["case"]==a.case and rec["release_sha256"]==key
                and flags==rec["flags"] and sha(artifact)==rec["archive_sha256"] and images_in_archive(artifact)==rec["elf"],
                "physical CSL build identity")
        descriptor=native|{"role":"hardware","elf":rec["elf"],"flags":flags}
    else:
        artifact=out/"artifact";shutil.copytree(folder/"direct/hardware/out",artifact/"out")
        descriptor=direct
    write(stage/"build.json",descriptor)
    command=shlex.join(["timeout","--signal=INT","--kill-after=30s","2700s","cs_python","-B","worker.py",
        ".","fixture.npz","capture","--artifact","out","--cmaddr","%CMADDR%"])
    command+=' >worker.log 2>&1; code=$?; echo "$code" >worker-exit.txt; exit "$code"'
    state={"role":"hardware","release_sha256":key,"case":a.case,"language":lang,"downloads":{}}
    write(out/"collection.json",state)
    os.chdir(out)
    with SdkLauncher(str(artifact),simulator=False,**common) as launcher:
        try:
            for f in (ROOT/"worker.py",ROOT/"contract.py",folder/"fixture.npz",stage/"build.json"):
                launcher.stage(str(f))
            before=time.perf_counter_ns()
            try: response=launcher.run(command)
            finally:state["launcher_elapsed_ns"]=time.perf_counter_ns()-before
            (out/"response.txt").write_text(str(response))
        finally:
            for remote in ("capture/result.json","capture/raw.npz","worker.log","worker-exit.txt"):
                dest=out/Path(remote).name
                try:
                    launcher.download_artifact(remote,str(dest))
                    state["downloads"][remote]={"sha256":sha(dest)}
                except Exception as err:state["downloads"][remote]={"error":repr(err)}
                write(out/"collection.json",state)
    require(all("sha256" in item for item in state["downloads"].values()) and
            (out/"worker-exit.txt").read_text().strip()=="0","hardware capture collection")
    shutil.copyfile(stage/"build.json",out/"build.json")
    capture(out,folder/"fixture.npz",out/"build.json","hardware",attested=True)
    with np.load(out/"raw.npz",allow_pickle=False) as hw,np.load(folder/("sim-"+lang)/"raw.npz",allow_pickle=False) as sim:
        for i in (1,2,3):
            np.testing.assert_array_equal(hw["logits"+str(i)].view(np.uint32),sim["logits"+str(i)].view(np.uint32))
    state["numerical_status"]="PASS";write(out/"collection.json",state)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("check","compile-csl","run-csl","run-tungsten"))
    p.add_argument("--case",required=True);p.add_argument("--out",type=Path);p.add_argument("--compiled",type=Path)
    p.add_argument("--namespace",default="job-operator")
    a=p.parse_args()
    if a.action!="check" and not a.out:p.error("output directory required")
    if a.action=="run-csl" and not a.compiled:p.error("physical compile record required")
    execute(a)
