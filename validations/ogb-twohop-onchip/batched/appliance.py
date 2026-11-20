"""Compare the same 32 complete trained roots across a one-port and two-port CS-3 application."""

import argparse
import importlib.util
from pathlib import Path
import shlex
import shutil

from origin import FAMILY, verify_window

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "batched_parallel", FAMILY / "parallel/appliance.py"
)
two = importlib.util.module_from_spec(spec)
spec.loader.exec_module(two)
one = two.one
read, save, require, sha = one.read, one.save, one.require, one.device.sha
CASES = {"arxiv-rcm32-b0-p32": ("arxiv", 0), "products-rcm32-b8-p32": ("products", 8)}
HOST = {
    HERE / "appliance.py",
    HERE / "origin.py",
    *two.HOST,
    *(FAMILY / n for n in one.HOST),
    *(
        p / n
        for p, n in [
            (FAMILY / "reference-source", "worker_chain.py"),
            (FAMILY / "reference-source", "build_chain.py"),
            (FAMILY / "parallel/reference-source", "build_parallel.py"),
            (FAMILY / "parallel/reference-source", "worker_parallel.py"),
        ]
    ),
}


def checked_case(name, pin):
    require(name in CASES, "registered 32-root batch")
    case = HERE / "cases" / name
    meta = read(case / "release.json")
    require(sha(case / "release.json") == pin, "frozen batch release")
    require(
        meta["host"] == {p.relative_to(FAMILY).as_posix(): sha(p) for p in HOST},
        "original one-port, two-port, collector and independent audit",
    )
    inventory = {
        p.relative_to(case).as_posix()
        for p in case.rglob("*")
        if p.is_file() and p != case / "release.json"
    }
    require(inventory == set(meta["files"]), "complete 32-root case inventory")
    for path, expected in meta["files"].items():
        actual = (case / path).resolve()
        require(
            actual.is_relative_to(case.resolve()) and sha(actual) == expected,
            "different input or image " + path,
        )
    batch = verify_window(case)
    solo, pair = case / "one", case / "two"
    require(
        CASES[name] == (read(solo / "model.json")["dataset"], batch["index"]),
        "registered dataset and group",
    )
    one.checked_data(solo)
    base = one.images(solo)
    actual = two.images(pair)
    require(
        base["first"]["P"] == 32
        and actual["original_build_sha256"] == sha(solo / "build.json")
        and all(
            base[k] == actual[k]
            for k in ("first", "final", "first_sha256", "last_sha256")
        )
        and all(
            base[k + "_sha256"] == sha(solo / (k + ".npz")) for k in ("first", "last")
        ),
        "identical original arrays, trained weights and 32 first-layer owners",
    )
    _, expected = one.observed(solo, solo / "sdk", None, "simulator", False)
    two.observed(pair, solo, pair / "sdk", "simulator", False)
    return case, expected


def replay(case, out, expected):
    solo, pair = case / "one", case / "two"
    for route, cfg in [("one", one), ("two", two)]:
        output = out / route
        receipt = read(output / "collection.json")
        desc = read(case / route / "build.json")
        require(
            receipt["elf_sha256"] == desc["roles"]["hardware"]["elf_sha256"]
            and receipt["elf_sha256"] == sha(output / "artifact/out/program.elf")
            and desc["port_sha256"] == sha(output / "artifact/out/out_port_map.json")
            and receipt["build_sha256"] == sha(case / route / "build.json")
            and receipt["release_sha256"] == sha(case / "release.json")
            and receipt["role"] == "hardware",
            "returned images, firmware and role",
        )
        require(
            receipt["downloads"] == {n: {"sha256": sha(output / n)} for n in cfg.FILES}
            and (output / "worker-exit.txt").read_text().strip() == "0",
            "complete physical return and normal stop",
        )
    first, _ = one.observed(solo, out / "one/capture", expected, "hardware", True)
    two.observed(pair, solo, out / "two/capture", "hardware", True)
    second = two.report_parallel.replay(
        solo / "first.npz",
        solo / "last.npz",
        pair,
        out / "two/capture",
        solo,
        out / "one/capture",
    )
    require(
        second["role"] == second["original_role"] == "hardware",
        "numerical and cycle comparison of both physical runs",
    )
    return dict(one_port=first, two_port=second, batch=read(case / "batch.json"))


def launch(case, route, out, namespace):
    from cerebras.sdk.client import SdkLauncher

    cfg = one if route == "one" else two
    worker = [FAMILY / n for n in one.WORKER] if route == "one" else list(two.WORKER)
    build = case / route
    (out / "capture").mkdir(parents=True)
    artifact = out / "artifact"
    shutil.copytree(build / "hardware/out", artifact / "out")
    desc = read(build / "build.json")
    receipt = dict(
        namespace=namespace,
        role="hardware",
        release_sha256=sha(case / "release.json"),
        build_sha256=sha(build / "build.json"),
        elf_sha256=desc["roles"]["hardware"]["elf_sha256"],
        downloads={},
    )
    save(out / "collection.json", receipt)
    command = shlex.join(
        [
            "timeout",
            "--signal=INT",
            "--kill-after=30s",
            "3600s",
            "cs_python",
            "-B",
            worker[0].name,
            ".",
            "first.npz",
            "last.npz",
            "capture",
            "--artifact",
            "out",
            "--cmaddr",
            "%CMADDR%",
        ]
    )
    with SdkLauncher(
        str(artifact),
        simulator=False,
        mgmt_namespace=namespace,
        resource_cpu=1000,
        resource_mem=8 << 30,
        disable_version_check=False,
    ) as launcher:
        try:
            for source in [
                *worker,
                build / "build.json",
                case / "one/first.npz",
                case / "one/last.npz",
            ]:
                launcher.stage(str(source))
            response = launcher.run(
                command
                + ' >worker.log 2>&1; rc=$?; echo "$rc" >worker-exit.txt; exit "$rc"'
            )
            (out / "launcher-response.txt").write_text(str(response))
        finally:
            for n in cfg.FILES:
                try:
                    launcher.download_artifact(n, str(out / n))
                    receipt["downloads"][n] = {"sha256": sha(out / n)}
                except Exception as exc:
                    receipt["downloads"][n] = {"error": repr(exc)}
                save(out / "collection.json", receipt)
    require(
        all("sha256" in r for r in receipt["downloads"].values())
        and (out / "worker-exit.txt").read_text().strip() == "0",
        route + " collection and exit",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["check", "run", "replay"])
    p.add_argument("--case", required=True, choices=sorted(CASES))
    p.add_argument("--release-sha256", required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--namespace", default="job-operator")
    a = p.parse_args()
    case, expected = checked_case(a.case, a.release_sha256)
    if a.action == "check":
        return
    if not a.out:
        p.error("--out is required")
    out = a.out.resolve()
    if a.action == "replay":
        replay(case, out, expected)
        print(
            "Audited 5 and 9 physical frames, 32 original roots and all hidden output words."
        )
        return
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        case=a.case,
        release_sha256=a.release_sha256,
        namespace=a.namespace,
        numerical_status="pending",
        collected=[],
    )
    save(out / "collection.json", receipt)
    try:
        for route in ("one", "two"):
            launch(case, route, out / route, a.namespace)
            receipt["collected"].append(route)
            save(out / "collection.json", receipt)
        result = replay(case, out, expected)
        save(out / "replay.json", result)
        receipt["numerical_status"] = "passed"
        receipt["replay_sha256"] = sha(out / "replay.json")
    except Exception as error:
        receipt["numerical_status"] = "failed"
        receipt["error"] = repr(error)
        raise
    finally:
        save(out / "collection.json", receipt)


if __name__ == "__main__":
    main()
