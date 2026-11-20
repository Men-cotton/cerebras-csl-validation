"""Execute one complete, interleaved trained OGB source-table comparison on CS-3."""
import argparse
import json
from pathlib import Path
import shutil

import report
from source_table import device

HERE=Path(__file__).resolve().parent
PORT="e4a311bc7f0f12ff52503703f4e233182c840339f765d01d49650022fe951d3e"
SCRIPTS=("appliance.py","worker.py","source_table.py","ogb_contract.py","report.py")


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
    if build["roles"]["hardware"]["fabric"]!=[762,1172] or build["port_sha256"]!=PORT:
        raise ValueError("physical image or port mapping differs")
    sdk,arrays=report.replay(case/"fixture.npz",case,case/"sdk",inspect_sram=False)
    if sdk["role"]!="simulator" or len(sdk["frames"])!=6:
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
    command=("timeout --signal=INT --kill-after=30s 1500s cs_python -B worker.py . fixture.npz capture "
             "--artifact out --cmaddr %CMADDR% >worker.log 2>&1; rc=$?; echo \"$rc\" >worker-exit.txt; exit \"$rc\"")
    try:
        with SdkLauncher(str(bundle),simulator=False,mgmt_namespace=namespace,
                         resource_cpu=1000,resource_mem=8<<30,disable_version_check=False) as launcher:
            try:
                for src in (HERE/"worker.py",HERE/"source_table.py",HERE/"ogb_contract.py",
                            case/"build.json",case/"fixture.npz"):
                    launcher.stage(str(src))
                (out/"launcher-response.txt").write_text(str(launcher.run(command)))
            finally:
                files=["worker.log","worker-exit.txt","capture/result.json",*(f"capture/frame-{i}.npz" for i in range(1,7))]
                for name in files:
                    try:
                        launcher.download_artifact(name,str(out/name))
                        log["downloads"][name]=dict(sha256=device.sha(out/name))
                    except Exception as exc:
                        log["downloads"][name]=dict(error=repr(exc))
                    save(out/"collection.json",log)
        if any("sha256" not in v for v in log["downloads"].values()) or (out/"worker-exit.txt").read_text().strip()!="0":
            raise ValueError("worker or data collection failed")
        measured,_=report.replay(case/"fixture.npz",case,capture,expected,inspect_sram=False)
        if measured["role"]!="hardware" or len(measured["frames"])!=6:
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
