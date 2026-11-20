"""Run only the five one-port products transactions after replaying both local route controls."""

import argparse
from pathlib import Path
import appliance as paired

NAME = "products-rcm32-b8-p32"


def checked(name, pin, driver):
    paired.require(
        name == NAME and paired.sha(Path(__file__)) == driver,
        "registered one-port products worker",
    )
    case, expected = paired.checked_case(name, pin)
    return case, paired.read(case / "one/build.json"), expected


def replay(case, desc, expected, out):
    receipt = paired.read(out / "collection.json")
    paired.require(
        receipt["role"] == "hardware"
        and receipt["release_sha256"] == paired.sha(case / "release.json")
        and receipt["build_sha256"] == paired.sha(case / "one/build.json")
        and receipt["elf_sha256"]
        == desc["roles"]["hardware"]["elf_sha256"]
        == paired.sha(out / "artifact/out/program.elf")
        and desc["port_sha256"] == paired.sha(out / "artifact/out/out_port_map.json"),
        "five-frame experiment used the registered one-port system ELF and ports",
    )
    paired.require(
        receipt["downloads"]
        == {n: {"sha256": paired.sha(out / n)} for n in paired.one.FILES}
        and (out / "worker-exit.txt").read_text().strip() == "0",
        "complete one-port device return",
    )
    result, _ = paired.one.observed(
        case / "one", out / "capture", expected, "hardware", True
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("check", "run"))
    p.add_argument("--case", required=True)
    p.add_argument("--release-sha256", required=True)
    p.add_argument("--driver-sha256", required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--namespace", default="job-operator")
    a = p.parse_args()
    case, desc, expected = checked(a.case, a.release_sha256, a.driver_sha256)
    if a.action == "run":
        if not a.out:
            p.error("run requires --out")
        out = a.out.resolve()
        out.mkdir(parents=True, exist_ok=False)
        paired.launch(case, "one", out, a.namespace)
        result = replay(case, desc, expected, out)
        paired.save(out / "replay.json", result)
        record = paired.read(out / "collection.json")
        record.update(
            numerical_status="passed",
            case=a.case,
            driver_sha256=a.driver_sha256,
            replay_sha256=paired.sha(out / "replay.json"),
        )
        paired.save(out / "collection.json", record)
