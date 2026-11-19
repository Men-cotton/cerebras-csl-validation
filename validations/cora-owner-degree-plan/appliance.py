"""Compile and collect the same real Cora owner window on a physical CS-3."""
import argparse
import json
import hashlib
import tarfile
import logging
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil

from contract import KEYS,PIN,load_graph,require,save,sha,verify

ROOT=Path(__file__).resolve().parent
FLAGS=("--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 --params=P:4,MODE:7 --memcpy "
       "--channels=1 --width-west-buf=0 --width-east-buf=0 --max-parallelism=1 -o out")

def read(p):
    return json.loads(p.read_text())

def application_elfs(artifact):
    with tarfile.open(artifact,"r:*") as archive:
        regular={PurePosixPath(member.name):member for member in archive if member.isfile()}
        roots={path.parent.parent for path in regular if path.parts[-2:]==("west","out.json")}
        roots={root for root in roots if (root/"east/out.json") in regular and (root/"out.json") in regular}
        require(len(roots)==1,"one compiler output tree")
        directory=next(iter(roots))/"bin"
        result={}
        for path,member in regular.items():
            if path.parent==directory and path.suffix==".elf":
                require(member.size<32<<20,"application ELF size")
                result[path.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
        require(bool(result),"compiled application ELF inventory")
        return result

def check_worker(case,role,z,out,release):
    result=read(out)
    require(result["release_sha256"]==sha(ROOT/"release.json") and result["code_sha256"]==release["host"],"worker and release")
    if case.startswith("tungsten"):
        require(result["elf_sha256"]==release["direct"][case][role]["elf_sha256"],"executed ELF")
    elif role=="simulator":
        require(result["csl_elf_sha256"]==release["simulator_csl_elf_sha256"],"CSL simulated application identity")
    summary=verify(z,result,role,case)
    require(result["summary"]==summary,"full summary replay")
    return summary

def verified():
    release=read(ROOT/"release.json")
    z=load_graph(ROOT/"fixture.npz")
    require(release["input_sha256"]==PIN,"release input")
    for key,pin in release["host"].items():
        require(sha(ROOT/key)==pin,"host source")
    for f,pin in release["csl"].items():
        require(sha(ROOT/"csl"/f)==pin,"CSL source")
    rows={}
    for case in KEYS:
        if case.startswith("tungsten"):
            record=release["direct"][case]
            build=ROOT/case/"build"
            require(sha(build/"program.img")==record["machine_image_sha256"] and
                    sha(build/"build.json")==record["build_sha256"],"Direct Link machine image")
            for f,pin in record["source"].items():
                require(sha(build/"source"/f)==pin,"Direct Link source")
            for role,dim in (("simulator",[4,2]),("hardware",[762,1172])):
                path=build/role/"out"
                require(record[role]["fabric"]==dim and sha(path/"program.elf")==record[role]["elf_sha256"] and
                        sha(path/"out_port_map.json")==release["port_sha256"],"ELF role and ports")
        rows[case]=check_worker(case,"simulator",z,ROOT/"simulator"/case/"result.json",release)
    require(len({tuple(rec["output_sha256"]) for rec in rows.values()})==1,"cross-language owner words")
    return rows

def main(a):
    sim=verified()
    print("simulator", {k:(r["setup_max"],r["max_owner_compute"]) for k,r in sim.items()}, flush=True)
    if a.action=="check":
        return
    from cerebras.sdk.client import SdkCompiler,SdkLauncher
    logging.basicConfig(level=logging.INFO)
    out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    release=read(ROOT/"release.json")
    common=dict(mgmt_namespace=a.namespace,resource_cpu=1000,resource_mem=8<<30,disable_version_check=False)
    if a.action=="compile-csl":
        with SdkCompiler(**common) as compiler:
            archive=Path(compiler.compile(str(ROOT/"csl"),"layout.csl",FLAGS,str(out))).resolve()
        require(archive.parent==out and archive.is_file(),"compiled artifact")
        save(out/"build.json",dict(artifact=archive.name,sha256=sha(archive),release=sha(ROOT/"release.json"),flags=FLAGS,application_elfs=application_elfs(archive)))
        return
    case=a.action.removeprefix("run-")
    if case=="csl-planned":
        build=read(a.build)
        artifact=(a.build.resolve().parent/build["artifact"]).resolve()
        require(artifact.parent==a.build.resolve().parent and sha(artifact)==build["sha256"] and
                build["release"]==sha(ROOT/"release.json") and build["flags"]==FLAGS,"compiled CSL archive")
    else:
        artifact=out/"bundle"
        (artifact/"out").mkdir(parents=True)
        for f in ("program.elf","out_port_map.json"):
            shutil.copyfile(ROOT/case/"build/hardware/out"/f,artifact/"out"/f)
    command=shlex.join(["timeout","--signal=INT","--kill-after=15s","900s","cs_python","worker.py","--case",case,
                        "--artifact","out","--input","fixture.npz","--manifest","release.json","--output","capture","--cmaddr","%CMADDR%"])
    command+=' >worker.log 2>&1; code=$?; echo "$code" >worker-exit.txt; exit "$code"'
    state=dict(case=case,release=sha(ROOT/"release.json"),downloads={},command=command)
    save(out/"collection.json",state)
    os.chdir(out)
    with SdkLauncher(str(artifact),simulator=False,**common) as launcher:
        try:
            for name in ("worker.py","contract.py","fixture.npz","release.json"):
                launcher.stage(str(ROOT/name))
            (out/"reply.txt").write_text(str(launcher.run(command)))
        finally:
            for f in ("capture/result.json","worker.log","worker-exit.txt"):
                dest=out/Path(f).name
                try:
                    launcher.download_artifact(f,str(dest))
                    state["downloads"][f]=dict(sha256=sha(dest))
                except Exception as exc:
                    state["downloads"][f]=dict(error=repr(exc))
                save(out/"collection.json",state)
    require(all("sha256" in rec for rec in state["downloads"].values()) and (out/"worker-exit.txt").read_text().strip()=="0","worker collection")
    state["summary"]=check_worker(case,"hardware",load_graph(ROOT/"fixture.npz"),out/"result.json",release)
    if case=="csl-planned":
        require(read(out/"result.json")["csl_elf_sha256"]==build["application_elfs"],"worker matches compiled CSL ELFs")
    save(out/"collection.json",state)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("check","compile-csl",*["run-"+k for k in KEYS]))
    p.add_argument("--out",type=Path)
    p.add_argument("--build",type=Path)
    p.add_argument("--namespace",default="job-operator")
    a=p.parse_args()
    if a.action!="check" and not a.out:
        p.error("--out is required")
    if a.action=="run-csl-planned" and not a.build:
        p.error("run-csl-planned requires --build")
    main(a)
