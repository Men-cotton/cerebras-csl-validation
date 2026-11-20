"""Replay the remaining three arxiv batches in the original 128-root RCM window."""
import argparse
from pathlib import Path

import appliance as paired

HERE = Path(__file__).resolve().parent
CASES = {f"arxiv-rcm32-b{i}-p32":i for i in (1,2,3)}
HOST = set(paired.HOST) | {HERE / "single.py"}


def checked(name, pin):
    paired.require(name in CASES, "unregistered RCM batch")
    case = HERE/"cases"/name
    single = case/"one"
    release = paired.read(case/"release.json")
    paired.require(paired.sha(case/"release.json") == pin and release["host"] ==
                   {p.relative_to(paired.FAMILY).as_posix():paired.sha(p) for p in HOST}, "host and release hashes")
    for n, h in release["files"].items():
        path=(case/n).resolve()
        paired.require(path.is_relative_to(case.resolve()) and paired.sha(path)==h, "frozen input or image " + n)
    batch=paired.verify_window(case)
    paired.require((batch["index"], paired.read(single/"model.json")["dataset"]) == (CASES[name],"arxiv"),
                   "selected original 32-target subset")
    paired.one.checked_data(single)
    desc=paired.one.images(single)
    paired.require(desc["first"]["P"] == 32 and desc["final"]["Z"]==32
                   and all(desc[n+"_sha256"] == paired.sha(single/(n+".npz")) for n in ("first","last")),
                   "original two layers, all owners and hardware image")
    _, expected=paired.one.observed(single,single/"sdk",None,"simulator",False)
    return case,desc,expected


def replay(case,desc,expected,out):
    info=paired.read(out/"collection.json")
    paired.require(info["role"]=="hardware" and info["release_sha256"]==paired.sha(case/"release.json")
                   and info["build_sha256"]==paired.sha(case/"one/build.json")
                   and info["elf_sha256"]==desc["roles"]["hardware"]["elf_sha256"]==paired.sha(out/"artifact/out/program.elf")
                   and desc["port_sha256"]==paired.sha(out/"artifact/out/out_port_map.json"),
                   "actual installed physical ELF and ports")
    paired.require(info["downloads"]=={n:{"sha256":paired.sha(out/n)} for n in paired.one.FILES}
                   and (out/"worker-exit.txt").read_text().strip()=="0","complete downloads and device stop")
    result,_=paired.one.observed(case/"one",out/"capture",expected,"hardware",True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("check","run"));p.add_argument("--case",required=True,choices=CASES)
    p.add_argument("--release-sha256",required=True);p.add_argument("--out",type=Path)
    p.add_argument("--namespace",default="job-operator");a=p.parse_args()
    case,desc,expected=checked(a.case,a.release_sha256)
    if a.action=="run":
        if a.out is None:p.error("run requires --out")
        out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
        paired.launch(case,"one",out,a.namespace)
        result=replay(case,desc,expected,out);paired.save(out/"replay.json",result)
        info=paired.read(out/"collection.json");info.update(case=a.case,numerical_status="passed",replay_sha256=paired.sha(out/"replay.json"))
        paired.save(out/"collection.json",info)
