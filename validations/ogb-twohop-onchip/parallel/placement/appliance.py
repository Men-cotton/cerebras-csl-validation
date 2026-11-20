"""Compare two owner placements by changing only the input streams of a single physical ELF."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PARALLEL = HERE.parent
FAMILY = PARALLEL.parent
sys.path[:0] = [str(HERE), str(FAMILY)]
from balance_owners import search
from report_parallel import replay
from two_rows import SCHEDULE, TRIALS, device, packets, prepare, trial_inputs

CASE = "arxiv-p16-e24"
IMAGE = PARALLEL / "cases" / CASE
PEER = FAMILY / "cases" / CASE
RAW = ("result.json", *(f"frame-{n}.npz" for n in range(1, 10)))
DOWNLOADS = ("worker.log", "worker-exit.txt", *("capture/" + n for n in RAW))
WORKER = (HERE / "worker_parallel.py", HERE / "two_rows.py", FAMILY / "chain.py",
          FAMILY / "source_table.py", FAMILY / "ogb_contract.py")
HOST = (*WORKER, HERE / "appliance.py", HERE / "report_parallel.py", HERE / "balance_owners.py",
        FAMILY / "report_chain.py", PARALLEL / "appliance.py")


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def words_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def observed(capture, plan, role, attest):
    first, last = PEER / "first.npz", PEER / "last.npz"
    result = replay(first, last, IMAGE, capture, PEER, PEER / "sdk")
    raw = read(capture / "result.json")
    require(result["role"] == role and result["original_role"] == "simulator"
            and result["compared_owner_order"] == plan["order"]
            and all(r["concurrent"] for r in result["frames"]), "wrong placement trial or platform")
    for tag, frame in enumerate(result["frames"][1:], 1):
        original = SCHEDULE[tag][1]
        require(frame["input_words_by_port"] == plan["original_words" if original else "balanced_words"],
                "payload differs from the selected placement")
    if attest:
        desc = read(IMAGE / "build.json")
        require(raw["actual_elf_sha256"] == desc["roles"][role]["elf_sha256"]
                and raw["actual_port_sha256"] == desc["port_sha256"]
                and raw["host_sha256"] == {p.name: device.sha(p) for p in WORKER}
                and raw["platform"] == {"system": role == "hardware", "simulation": role == "simulator"},
                "worker or installed ELF does not identify the requested run")
        data = prepare(first, last, desc["first"]["P"], desc["first"]["E"])
        setup = np.pad(data[8], (0, len(data[8]) % 2))
        require(raw["setup_sha256"] == words_sha(setup), "initial setup hash")
        for i, ((logical, _, _, rows), frame) in enumerate(zip(trial_inputs(IMAGE, data, plan["order"]), raw["frames"])):
            require(frame["port_input_sha256"] == [words_sha(v) for v in packets(rows, i)], "actual two port input")
            with np.load(capture / f"frame-{i+1}.npz", allow_pickle=False) as z:
                require(frame["echo_h2d_sha256"] == (words_sha(z["echo"]) if TRIALS[logical][0] else None),
                        "actual intermediate echo hash")
    return result


def same_replay(actual, expected):
    require(set(actual) == set(expected) and len(actual["frames"]) == len(expected["frames"]), "replay fields")
    for key in actual.keys() - {"frames"}:
        require(actual[key] == expected[key], "replay metadata " + key)
    for frame, saved in zip(actual["frames"], expected["frames"]):
        require(set(frame) == set(saved), "replay frame fields")
        for key, value in frame.items():
            if key.endswith("max_abs") and value is not None:
                np.testing.assert_allclose(value, saved[key], rtol=1e-8, atol=1e-13)
            else:
                require(value == saved[key], "replay measurement " + key)


def checked(name, pin):
    require(name == CASE, "unknown four-root placement case")
    base = HERE / "cases" / name
    require(device.sha(base / "release.json") == pin, "release hash")
    release = read(base / "release.json")
    require(release["host"] == {p.relative_to(FAMILY).as_posix():device.sha(p) for p in HOST},
            "changed host source")
    for n, digest in release["files"].items():
        path = (base / n).resolve()
        require(path.is_relative_to(base.resolve()) and device.sha(path) == digest, "frozen evidence " + n)
    # The previously released family's modules have the same filenames and fixed hashes.
    # Verify that release in its original import environment before importing the new trials.
    subprocess.run([sys.executable, "-B", str(PARALLEL / "appliance.py"), "check", "--case", name,
                    "--release-sha256", release["two_port_release_sha256"]], check=True)
    plan = read(base / "plan.json")
    require(plan == search(PEER / "first.npz", PEER / "last.npz", IMAGE), "owner search from original data")
    same_replay(observed(base / "sdk", plan, "simulator", False), read(base / "sdk-replay.json"))
    return plan


def run(name, pin, out, namespace):
    from cerebras.sdk.client import SdkLauncher
    plan = checked(name, pin)
    out.mkdir(parents=True, exist_ok=False)
    capture = out / "capture"
    capture.mkdir()
    artifact = out / "artifact"
    shutil.copytree(IMAGE / "hardware/out", artifact / "out")
    receipt = dict(case=name, release_sha256=pin, namespace=namespace,
                   expected_hardware_elf_sha256=device.sha(artifact / "out/program.elf"),
                   order=plan["order"], downloads={}, numerical_status="pending")
    save(out/"collection.json", receipt)
    cmd = shlex.join(["timeout", "--signal=INT", "--kill-after=30s", "3000s", "cs_python", "-B",
                      "worker_parallel.py", ".", "first.npz", "last.npz", "capture", "--artifact", "out",
                      "--cmaddr", "%CMADDR%", "--compare-owner-order", *map(str, plan["order"])])
    try:
        with SdkLauncher(str(artifact), simulator=False, mgmt_namespace=namespace, resource_cpu=1000,
                         resource_mem=8 << 30, disable_version_check=False) as launcher:
            try:
                for source in (*WORKER, IMAGE / "build.json", *(IMAGE/"src"/f"owner{k}.w" for k in range(2)),
                               PEER/"first.npz", PEER/"last.npz"):
                    launcher.stage(str(source))
                response = launcher.run(cmd + ' >worker.log 2>&1; rc=$?; echo "$rc" >worker-exit.txt; exit "$rc"')
                (out/"launcher-response.txt").write_text(str(response))
            finally:
                for n in DOWNLOADS:
                    try:
                        launcher.download_artifact(n, str(out/n))
                        receipt["downloads"][n] = {"sha256":device.sha(out/n)}
                    except Exception as exc:
                        receipt["downloads"][n] = {"error":repr(exc)}
                    save(out/"collection.json", receipt)
        require(all("sha256" in x for x in receipt["downloads"].values())
                and (out/"worker-exit.txt").read_text().strip() == "0", "incomplete physical return")
        save(out/"replay.json", observed(capture, plan, "hardware", True))
        receipt["numerical_status"] = "passed"
    finally:
        save(out/"collection.json", receipt)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("check","run"))
    parser.add_argument("--case",required=True)
    parser.add_argument("--release-sha256",required=True)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--namespace",default="job-operator")
    a=parser.parse_args()
    if a.action == "check":
        checked(a.case,a.release_sha256)
    elif a.out:
        run(a.case,a.release_sha256,a.out.resolve(),a.namespace)
    else:
        parser.error("run requires --out")
