"""Execute twelve serial and parallel Direct Link frames on trained OGB data on CS-3."""
import argparse
import json
from pathlib import Path
import shutil

import report_lanes as report
from contract_lanes import TRIAL_PLAN, device

HERE=Path(__file__).resolve().parent
PORTS={1: '140b486961be147490e99a820dfd4019faec3db179764fcdcd69c3845b1898a4', 2: '1b2567f24710dff904290338f78762d1c8741d7a0cb9c68d028f4af023262896', 4: '05e4df8a9bf654f589ad79e4678db519dc7fb895720955b5313c4d206a6cffb9'}
SCRIPTS=("appliance.py","worker_lanes.py","contract_lanes.py","source_table.py","ogb_contract.py","report_lanes.py")


def save(path,value):
    scratch=path.with_suffix(".tmp")
    scratch.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    scratch.replace(path)


def check(case,pin):
    release=case/"release.json"
    if device.sha(release)!=pin:
        raise ValueError("release pin differs")
    doc=json.loads(release.read_text())
    if doc["host_sha256"]!={n:device.sha(HERE/n) for n in SCRIPTS}:
        raise ValueError("host program changed")
    for name,digest in doc["file_sha256"].items():
        path=case/name
        if not path.resolve().is_relative_to(case.resolve()) or device.sha(path)!=digest:
            raise ValueError("case file changed")
    build=json.loads((case/"build.json").read_text())
    if build["roles"]["hardware"]["fabric"]!=[762,1172] or build["port_sha256"]!=PORTS.get(build["lanes"]):
        raise ValueError("physical image or port mapping differs")
    sdk,arrays=report.replay(case/"fixture.npz",case,case/"sdk")
    if sdk["role"]!="simulator" or len(sdk["frames"])!=len(TRIAL_PLAN):
        raise ValueError("missing full SDK controls")
    return build,arrays


def run(case,out,pin,namespace):
    from cerebras.sdk.client import SdkLauncher
    build,expected=check(case,pin)
    out.mkdir(parents=True,exist_ok=False)
    capture=out/"capture"
    capture.mkdir()
    bundle=out/"artifact"
    shutil.copytree(case/"hardware/out",bundle/"out")
    log=dict(case=case.name,release_sha256=pin,elf_sha256=build["roles"]["hardware"]["elf_sha256"],
             namespace=namespace,numerical_status="pending",downloads={})
    save(out/"collection.json",log)
    command=("timeout --signal=INT --kill-after=30s 1500s cs_python -B worker_lanes.py . fixture.npz capture "
             "--artifact out --cmaddr %CMADDR% >worker.log 2>&1; rc=$?; echo \"$rc\" >worker-exit.txt; exit \"$rc\"")
    try:
        with SdkLauncher(str(bundle),simulator=False,mgmt_namespace=namespace,
                         resource_cpu=1000,resource_mem=8<<30,disable_version_check=False) as launcher:
            try:
                for src in (HERE/"worker_lanes.py",HERE/"contract_lanes.py",HERE/"source_table.py",HERE/"ogb_contract.py",
                            case/"build.json",case/"fixture.npz"):
                    launcher.stage(str(src))
                (out/"launcher-response.txt").write_text(str(launcher.run(command)))
            finally:
                files=["worker.log","worker-exit.txt","capture/result.json",*(f"capture/frame-{i}.npz" for i in range(1,len(TRIAL_PLAN)+1))]
                for name in files:
                    try:
                        launcher.download_artifact(name,str(out/name))
                        log["downloads"][name]=dict(sha256=device.sha(out/name))
                    except Exception as exc:
                        log["downloads"][name]=dict(error=repr(exc))
                    save(out/"collection.json",log)
        if any("sha256" not in v for v in log["downloads"].values()) or (out/"worker-exit.txt").read_text().strip()!="0":
            raise ValueError("worker or data collection failed")
        measured,_=report.replay(case/"fixture.npz",case,capture,expected)
        if measured["role"]!="hardware" or len(measured["frames"])!=len(TRIAL_PLAN):
            raise ValueError("missing resident hardware frames")
        save(out/"replay.json",measured)
        log["numerical_status"]="passed"
    except BaseException as exc:
        log["error"]=repr(exc)
        raise
    finally:
        save(out/"collection.json",log)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("check","run"))
    p.add_argument("--case",type=Path,required=True)
    p.add_argument("--release-sha256",required=True)
    p.add_argument("--out",type=Path)
    p.add_argument("--namespace",default="job-operator")
    a=p.parse_args()
    if a.action=="check":
        check(a.case.resolve(),a.release_sha256)
    elif a.out:
        run(a.case.resolve(),a.out.resolve(),a.release_sha256,a.namespace)
    else:
        p.error("run requires --out")
