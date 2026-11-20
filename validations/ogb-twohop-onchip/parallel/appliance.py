"""Run the complete original two-hop closure through two WEST input ports on one CS-3."""
import argparse
import importlib.util
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
sys.path[:0] = [str(HERE), str(FAMILY)]
import report_parallel
from two_rows import SCHEDULE, TRIALS, device, packets, parts, prepare
from image import equivalence

spec = importlib.util.spec_from_file_location("one_port_original", FAMILY / "appliance.py")
one = importlib.util.module_from_spec(spec)
spec.loader.exec_module(one)
read, save, require = one.read, one.save, one.require
PORT = "1b2567f24710dff904290338f78762d1c8741d7a0cb9c68d028f4af023262896"
FILES = ("worker.log", "worker-exit.txt", "capture/result.json",
         *(f"capture/frame-{i}.npz" for i in range(1, len(SCHEDULE) + 1)))
WORKER = (HERE / "worker_parallel.py", HERE / "two_rows.py",
          FAMILY / "chain.py", FAMILY / "source_table.py", FAMILY / "ogb_contract.py")
HOST = (*WORKER, HERE / "appliance.py", HERE / "report_parallel.py", FAMILY / "report_chain.py",
        FAMILY / "image.py", FAMILY / "originals.py", FAMILY / "appliance.py")
CASES = {"small-arxiv-p2", "arxiv-p16-e24", "products-p16-e24"}


def images(case):
    desc = read(case / "build.json")
    shape = (desc["first"]["P"] // 2, 4)
    require(device.sha(case / "chain.map") == desc["map_sha256"], "two-row machine map")
    require(desc["source_sha256"] == {f.name: device.sha(f) for f in (case/"src").iterdir()
                                     if f.is_file()}, "exact rendered device sources")
    records = []
    for role, fabric in (("simulator", shape), ("hardware", (762, 1172))):
        target = desc["roles"][role]
        image = case / role / "out"
        require(target["fabric"] == list(fabric) and target["elf_sha256"] == device.sha(image/"program.elf")
                and desc["port_sha256"] == device.sha(image/"out_port_map.json") == PORT, "physical image and all four ports")
        records.append(equivalence(image/"program.elf", fabric, shape, (case/role/"ranges.txt").read_text()))
    require(records[0] == records[1] and records[0]["canonical_sha256"] == desc["canonical_sha256"],
            "all compute, bridge and gather PE loads at both geometries")
    return desc


def observed(case, peer, capture, role, attested):
    first, last = peer/"first.npz", peer/"last.npz"
    result = report_parallel.replay(first, last, case, capture, peer, peer/"sdk")
    desc = read(case/"build.json")
    raw = read(capture/"result.json")
    require(result["role"] == role and result["original_role"] == "simulator",
            "target and one-port SDK reference")
    if attested:
        data = prepare(first, last, desc["first"]["P"], desc["first"]["E"])
        lanes = parts(data)
        setup = np.pad(data[8], (0, len(data[8]) % 2))
        require(raw["setup_sha256"] == one.digest(setup)
                and raw["actual_elf_sha256"] == desc["roles"][role]["elf_sha256"]
                and raw["actual_port_sha256"] == PORT
                and raw["host_sha256"] == {p.name:device.sha(p) for p in WORKER}
                and raw["platform"] == {"system":role == "hardware", "simulation":role == "simulator"},
                "actual loaded firmware, worker, setup and SDK target")
        for pos, ((trial, _), frame) in enumerate(zip(SCHEDULE,raw["frames"])):
            require(frame["port_input_sha256"] == [one.digest(x) for x in packets(lanes,pos)],
                    "actual tagged payload on each WEST input")
            echo, _ = TRIALS[trial]
            with np.load(capture/f"frame-{pos+1}.npz", allow_pickle=False) as z:
                require(frame["echo_h2d_sha256"] == (one.digest(z["echo"]) if echo else None),
                        "complete actual host re-entry")
    return result


def checked_case(name, pin):
    require(name in CASES, "registered two-row case")
    case = HERE/"cases"/name
    require(device.sha(case/"release.json") == pin, "two-row release SHA")
    release = read(case/"release.json")
    require(release["host"] == {p.relative_to(FAMILY).as_posix():device.sha(p) for p in HOST}
            and release["reference_sources_at_export"] ==
                {n:device.sha(HERE/"reference-source"/n) for n in ("build_parallel.py","worker_parallel.py")},
            "two-port source and verifier")
    for n, sha in release["files"].items():
        p = (case/n).resolve()
        require(p.is_relative_to(case.resolve()) and device.sha(p) == sha, "frozen two-row file "+n)
    peer, _, original, _ = one.checked_case(name, release["one_port_release_sha256"])
    desc = images(case)
    require(desc["original_build_sha256"] == device.sha(peer/"build.json")
            and all(desc[k] == original[k] for k in ("first","final","first_sha256","last_sha256")),
            "original source assignment, raw features, trained checkpoint and both CSR layers")
    control = observed(case, peer, case/"sdk", "simulator", False)
    if name == "small-arxiv-p2":
        observed(case, peer, case/"bound-sdk", "simulator", True)
    return case, peer, desc, control


def run(name, pin, out, namespace):
    from cerebras.sdk.client import SdkLauncher
    case, peer, desc, _ = checked_case(name, pin)
    out.mkdir(parents=True, exist_ok=False)
    capture = out/"capture"
    capture.mkdir()
    artifact = out/"artifact"
    shutil.copytree(case/"hardware/out", artifact/"out")
    receipt = dict(case=name, release_sha256=pin, one_port_release_sha256=device.sha(peer/"release.json"),
                   build_sha256=device.sha(case/"build.json"), expected_elf_sha256=desc["roles"]["hardware"]["elf_sha256"],
                   namespace=namespace, downloads={}, numerical_status="pending")
    save(out/"collection.json", receipt)
    command = shlex.join(["timeout", "--signal=INT", "--kill-after=30s","3000s","cs_python","-B",
                          "worker_parallel.py",".","first.npz","last.npz","capture","--artifact","out","--cmaddr","%CMADDR%"])
    try:
        with SdkLauncher(str(artifact),simulator=False,mgmt_namespace=namespace,resource_cpu=1000,
                         resource_mem=8<<30,disable_version_check=False) as launcher:
            try:
                for source in (*WORKER,case/"build.json",peer/"first.npz",peer/"last.npz"):
                    launcher.stage(str(source))
                response = launcher.run(command+' >worker.log 2>&1; rc=$?; echo "$rc" >worker-exit.txt; exit "$rc"')
                (out/"launcher-response.txt").write_text(str(response))
            finally:
                for n in FILES:
                    try:
                        launcher.download_artifact(n,str(out/n))
                        receipt["downloads"][n] = {"sha256":device.sha(out/n)}
                    except Exception as exc:
                        receipt["downloads"][n] = {"error":repr(exc)}
                    save(out/"collection.json",receipt)
        require(all("sha256" in d for d in receipt["downloads"].values())
                and (out/"worker-exit.txt").read_text().strip()=="0", "complete physical return and normal worker exit")
        metrics = observed(case,peer,capture,"hardware",True)
        save(out/"replay.json",metrics)
        receipt["numerical_status"] = "passed"
    finally:
        save(out/"collection.json",receipt)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("check","run"));p.add_argument("--case",required=True)
    p.add_argument("--release-sha256",required=True);p.add_argument("--out",type=Path)
    p.add_argument("--namespace",default="job-operator")
    a=p.parse_args()
    if a.action=="check":checked_case(a.case,a.release_sha256)
    elif a.out:run(a.case,a.release_sha256,a.out.resolve(),a.namespace)
    else:p.error("run requires --out")
