"""Run all three trained-graph assignments on one eight-port image."""

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("eight_partition_driver", HERE / "appliance.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
SCRIPTS = (*driver.SCRIPTS, "eight.py")
POLICIES = driver.POLICIES
device = driver.device
save = driver.save
PORT = "a86df9927b082418eeebdee80225858b45b29ec6f4e22e5960d456dcdf430dd9"


def check(case, pin):
    release = case / "release.json"
    if device.sha(release) != pin:
        raise ValueError("release pin differs")
    doc = json.loads(release.read_text())
    if doc["host_sha256"] != {name: device.sha(HERE / name) for name in SCRIPTS}:
        raise ValueError("host program changed")
    for name, digest in doc["file_sha256"].items():
        path = case / name
        if not path.resolve().is_relative_to(case.resolve()) or device.sha(path) != digest:
            raise ValueError("case file changed")
    build = json.loads((case / "build.json").read_text())
    if (build["lanes"], build["owners_per_lane"], build["port_sha256"]) != (8, 2, PORT):
        raise ValueError("wrong eight-port physical deployment")
    fixture = case / "fixture.npz"
    if build["fixture_sha256"] != device.sha(fixture):
        raise ValueError("compiled for a different graph window")
    if build["partitions"] != driver.assignments(device.load(fixture), 8, 2, build["entries"]):
        raise ValueError("original assignments disagree with the linked image")
    proof = []
    for role, fabric in [("simulator", (2, 16)), ("hardware", (762, 1172))]:
        image = case / role / "out/program.elf"
        if build["roles"][role] != {"fabric": list(fabric), "elf_sha256": device.sha(image)}:
            raise ValueError("firmware differs")
        if device.sha(image.parent / "out_port_map.json") != PORT:
            raise ValueError("physical or SDK port map differs")
        proof.append(driver.equivalence(image, fabric, (2, 16), (case / role / "ranges.txt").read_text()))
    if proof[0] != proof[1] or proof[0]["canonical_sha256"] != build["canonical_sha256"]:
        raise ValueError("hardware and SDK application loads differ")
    reference = None
    for name in POLICIES:
        sdk, reference = driver.report.replay(fixture, case, case / "sdk" / name, reference)
        if sdk["role"] != "simulator" or sdk["partition"] != name or len(sdk["frames"]) != 12:
            raise ValueError("missing assignment control")
        actual = {str(k): sha256(a.tobytes()).hexdigest() for k, a in reference.items()}
        if actual != doc["original_single_port_logits_sha256"]:
            raise ValueError("class scores differ from the trained single-port control")
    return build, reference


# Use the public four-port launch and collection protocol with this release's geometry checks.
driver.check = check

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--namespace", default="job-operator")
    parser.add_argument("--partition", choices=POLICIES)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "check":
        check(args.case.resolve(), args.release_sha256)
    elif args.out and args.partition:
        driver.run(args.case.resolve(), args.out.resolve(), args.release_sha256, args.namespace, args.partition)
    else:
        parser.error("run requires --out and --partition")
