"""Run the original arxiv GraphSAGE window on two WEST lanes separated by 18 rows."""
import argparse
import json
from pathlib import Path
import re
import shutil

import numpy as np

from contract_lanes import TRIAL_PLAN, TRIALS, configured_parts, decode_clock, device, evaluate
from layout import equivalence
import report_lanes as report

HERE = Path(__file__).resolve().parent
CASE = HERE / "cases/arxiv-128-p16-l2-pitch18"
COMPACT = HERE / "controls/compact"
SINGLE = HERE / "controls/single-port-cs3"
HOST = ("appliance.py", "worker_lanes.py", "contract_lanes.py", "source_table.py", "ogb_contract.py", "layout.py", "report_lanes.py")
ROWS = (0, 1, 18, 19)
COMPACT_PIN = "230089c18d0fca8806df3fc7bed4ef6cff492ec3d6404e2286f4d26260deb52d"


def need(test, text):
    if not test:
        raise ValueError(text)


def read(p):
    return json.loads(p.read_bytes())


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def mapping(desc):
    buses, ports = desc["buses"], desc["lvds_ports"]
    need(len(buses) == len(ports) == 4 and {b["lvds_id"] for b in buses} == set(range(4)), "four unique WEST endpoints")
    return [(b["port_name"], b["io_type"], ports[b["lvds_id"]]["side"], ports[b["lvds_id"]]["loc"],
             ports[b["lvds_id"]]["color"]) for b in buses]


def check_layout(case):
    desc = read(case / "build.json")
    old = read(COMPACT / "build.json")
    need(desc["lanes"] == 2 and desc["owners_per_lane"] == 8 and desc["lane_pitch"] == 18
         and desc["entries"] == 128 and desc["lane_params"] == old["lane_params"]
         and desc["fixture_sha256"] == old["fixture_sha256"] == device.sha(case / "fixture.npz"), "same trained P16 geometry")
    views = []
    ports = [(f"{name}{i}_port", direction, "WEST", ROWS[2*i+k], 2*i+k)
             for i in range(2) for k, (name, direction) in enumerate((("rx","IN"),("tx","OUT")))]
    for role, dims in (("simulator",(8,20)), ("hardware",(762,1172))):
        base = case / role
        elf, mapping_file = base / "out/program.elf", base / "out/out_port_map.json"
        need(desc["roles"][role] == {"fabric": list(dims), "elf_sha256":device.sha(elf)}
             and desc["port_sha256"] == device.sha(mapping_file) and mapping(read(mapping_file)) == ports,
             "physical and small-size images with the prescribed WEST placement")
        views.append(equivalence(elf, dims, (8,20), base / "ranges.txt", active_rows=ROWS))
    need(views[0] == views[1] and views[0]["canonical_sha256"] == desc["canonical_sha256"],
         "all PE data, code and router loads must match across fabric sizes")
    for name, pin in desc["source_sha256"].items():
        source = case / "src" / name
        need(device.sha(source) == pin, "rendered source SHA")
        content = source.read_bytes()
        if name == "program.paint":
            pattern = rb"(let lane1[og][0-9]+=tile\([0-7] )1([89])\)"
            content, count = re.subn(pattern,lambda m: m[1]+{b"8":b"2",b"9":b"3"}[m[2]]+b")",content)
            need(count == 16, "the eight second-lane compute and gather positions moved")
        need(content == (COMPACT / "src" / name).read_bytes(), "identical underlying computation and links")
    return desc


def checked_controls():
    release = COMPACT / "release.json"
    need(device.sha(release) == COMPACT_PIN, "original registered compact L2 control")
    for name, pin in read(release)["file_sha256"].items():
        need(device.sha(COMPACT / name) == pin, "compact SDK and image inventory")
    reference, outputs = report.replay(COMPACT / "fixture.npz", COMPACT, COMPACT / "sdk")
    need(reference["role"] == "simulator" and len(reference["frames"]) == 12, "complete compact SDK control")
    fixture = device.load(COMPACT / "fixture.npz")
    peer = read(SINGLE / "provenance.json")
    need(peer["role"] == "hardware" and peer["fixture_sha256"] == device.sha(COMPACT / "fixture.npz")
         and peer["jobs"] and len(peer["frames"]) == 2, "independent physical single-port reference")
    for row in peer["frames"]:
        raw = SINGLE / f"frame-{row['trial']}.npz"
        need(device.sha(raw) == row["raw_sha256"], "physical single-port words")
        with np.load(raw, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["node"], fixture["seed"])
            np.testing.assert_array_equal(z["logits"].view("u4"), outputs[row["scale"]].view("u4"))
    return outputs


def check_sdk(case, desc, expected):
    root = case / "sdk-at-release"
    state = read(root / "result.json")
    total = len(state["frames"])
    need(0 <= total <= 12 and state["role"] == "simulator" and
         state["fixture_sha256"] == desc["fixture_sha256"] and state["build_sha256"] == device.sha(case/"build.json"),
         "actual pitched SDK capture and release image")
    if state["stopped"]:
        measured, _ = report.replay(case / "fixture.npz", case, root, expected)
        need(len(measured["frames"]) == 12, "normal SDK stop requires twelve frames")
        return
    z = device.load(case / "fixture.npz")
    parts = configured_parts(z, desc, "contiguous")
    for i, row in enumerate(state["frames"]):
        logical, concurrent = TRIAL_PLAN[i]
        mode, scale = TRIALS[logical]
        raw = root / f"frame-{i+1}.npz"
        need((row["frame"],row["method"],row["scale"],row["concurrent"],row["raw_sha256"]) ==
             (i+1, mode, scale, concurrent, device.sha(raw)), "ordered physical map SDK prefix")
        with np.load(raw,allow_pickle=False) as data:
            out, stats = evaluate(z, parts, [data[f"lane{k}"] for k in range(2)], logical)
            np.testing.assert_array_equal(out.view("u4"), data["logits"].view("u4"))
            np.testing.assert_array_equal(out.view("u4"), expected[scale].view("u4"))
            np.testing.assert_array_equal(data["nodes"], z["seed"])
            np.testing.assert_array_equal(data["ack"], [i+1,0xC510A11])
            need(decode_clock(data["timer"], i+1) == row["common_clock_cycles"] and
                 row["lane_compute_peaks"] == [r["peak_first_to_last"] for r in stats], "returned per-lane and fence clocks")


def check(case, pin):
    need(case.resolve() == CASE.resolve() and device.sha(case / "release.json") == pin, "registered pitch-18 release")
    cert = read(case / "release.json")
    need(cert["host"] == {n:device.sha(HERE/n) for n in HOST}, "complete host program")
    for name, digest in cert["files"].items():
        file = HERE / name
        need(file.resolve().is_relative_to(HERE) and device.sha(file) == digest, "registered layout, system and SDK controls")
    desc = check_layout(case)
    expected = checked_controls()
    check_sdk(case, desc, expected)
    return desc, expected


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
