"""Replay original GraphSAGE destinations across three assignments on one four-port CS-3 image."""

import argparse
import json
from pathlib import Path
import shutil
from hashlib import sha256
from image import equivalence
from partition import assignments

import report_lanes as report
from contract_lanes import TRIAL_PLAN, device

HERE = Path(__file__).resolve().parent
PORTS = {
    1: "140b486961be147490e99a820dfd4019faec3db179764fcdcd69c3845b1898a4",
    2: "1b2567f24710dff904290338f78762d1c8741d7a0cb9c68d028f4af023262896",
    4: "05e4df8a9bf654f589ad79e4678db519dc7fb895720955b5313c4d206a6cffb9",
}
POLICIES = ("contiguous", "balanced", "dictionary")
SCRIPTS = (
    "appliance.py",
    "partition.py",
    "image.py",
    "worker_lanes.py",
    "contract_lanes.py",
    "source_table.py",
    "ogb_contract.py",
    "report_lanes.py",
)


def save(path, value):
    scratch = path.with_suffix(".tmp")
    scratch.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    scratch.replace(path)


def check(case, pin):
    release = case / "release.json"
    if device.sha(release) != pin:
        raise ValueError("release pin differs")
    doc = json.loads(release.read_text())
    if doc["host_sha256"] != {n: device.sha(HERE / n) for n in SCRIPTS}:
        raise ValueError("host program changed")
    for name, digest in doc["file_sha256"].items():
        path = case / name
        if not path.resolve().is_relative_to(case.resolve()) or device.sha(path) != digest:
            raise ValueError("case file changed")
    build = json.loads((case / "build.json").read_text())
    if build["roles"]["hardware"]["fabric"] != [762, 1172] or build["port_sha256"] != PORTS.get(
        build["lanes"]
    ):
        raise ValueError("physical image or port mapping differs")
    if (build["lanes"], build["owners_per_lane"]) != (4, 4):
        raise ValueError("wrong compute deployment")
    z = device.load(case / "fixture.npz")
    if build["partitions"] != assignments(z, 4, 4, build["entries"]):
        raise ValueError("unable to reproduce original row assignments")
    proof = []
    for role, fabric in [("simulator", (4, 8)), ("hardware", (762, 1172))]:
        image = case / role / "out/program.elf"
        if (
            device.sha(image) != build["roles"][role]["elf_sha256"]
            or device.sha(image.parent / "out_port_map.json") != PORTS[4]
        ):
            raise ValueError("paired ELF or port mapping changed")
        proof.append(equivalence(image, fabric, (4, 8), (case / role / "ranges.txt").read_text()))
    if proof[0] != proof[1] or proof[0]["canonical_sha256"] != build["canonical_sha256"]:
        raise ValueError("CS-3 application loads differ from the SDK control")
    arrays = None
    for name in POLICIES:
        sdk, arrays = report.replay(case / "fixture.npz", case, case / "sdk" / name, arrays)
        if (
            sdk["role"] != "simulator"
            or sdk["partition"] != name
            or len(sdk["frames"]) != len(TRIAL_PLAN)
        ):
            raise ValueError("missing named SDK controls")
        actual = {str(k): sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}
        if actual != doc["original_single_port_logits_sha256"]:
            raise ValueError("original classifications differ")
    return build, arrays


def run(case, out, pin, namespace, policy):
    if policy not in POLICIES:
        raise ValueError("unknown assignment")
    from cerebras.sdk.client import SdkLauncher

    build, expected = check(case, pin)
    out.mkdir(parents=True, exist_ok=False)
    capture = out / "capture"
    capture.mkdir()
    bundle = out / "artifact"
    shutil.copytree(case / "hardware/out", bundle / "out")
    log = dict(
        case=case.name,
        release_sha256=pin,
        elf_sha256=build["roles"]["hardware"]["elf_sha256"],
        partition=policy,
        namespace=namespace,
        numerical_status="pending",
        downloads={},
    )
    save(out / "collection.json", log)
    command = (
        "timeout --signal=INT --kill-after=30s 1500s cs_python -B worker_lanes.py . fixture.npz capture "
        f'--partition {policy} --artifact out --cmaddr %CMADDR% >worker.log 2>&1; rc=$?; echo "$rc" >worker-exit.txt; exit "$rc"'
    )
    try:
        with SdkLauncher(
            str(bundle),
            simulator=False,
            mgmt_namespace=namespace,
            resource_cpu=1000,
            resource_mem=8 << 30,
            disable_version_check=False,
        ) as launcher:
            try:
                for src in (
                    HERE / "worker_lanes.py",
                    HERE / "contract_lanes.py",
                    HERE / "source_table.py",
                    HERE / "ogb_contract.py",
                    case / "build.json",
                    case / "fixture.npz",
                ):
                    launcher.stage(str(src))
                (out / "launcher-response.txt").write_text(str(launcher.run(command)))
            finally:
                files = [
                    "worker.log",
                    "worker-exit.txt",
                    "capture/result.json",
                    *(f"capture/frame-{i}.npz" for i in range(1, len(TRIAL_PLAN) + 1)),
                ]
                for name in files:
                    try:
                        launcher.download_artifact(name, str(out / name))
                        log["downloads"][name] = dict(sha256=device.sha(out / name))
                    except Exception as exc:
                        log["downloads"][name] = dict(error=repr(exc))
                    save(out / "collection.json", log)
        if (
            any("sha256" not in v for v in log["downloads"].values())
            or (out / "worker-exit.txt").read_text().strip() != "0"
        ):
            raise ValueError("worker or data collection failed")
        measured, _ = report.replay(case / "fixture.npz", case, capture, expected)
        if (
            measured["role"] != "hardware"
            or measured["partition"] != policy
            or len(measured["frames"]) != len(TRIAL_PLAN)
        ):
            raise ValueError("missing resident hardware frames")
        save(out / "replay.json", measured)
        log["numerical_status"] = "passed"
    except BaseException as exc:
        log["error"] = repr(exc)
        raise
    finally:
        save(out / "collection.json", log)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("check", "run"))
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--release-sha256", required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--namespace", default="job-operator")
    p.add_argument("--partition", choices=POLICIES)
    a = p.parse_args()
    if a.action == "check":
        check(a.case.resolve(), a.release_sha256)
    elif a.out and a.partition:
        run(a.case.resolve(), a.out.resolve(), a.release_sha256, a.namespace, a.partition)
    else:
        p.error("run requires --out and --partition")
