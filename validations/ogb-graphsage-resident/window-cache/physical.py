"""Check a returned two-layer-derived products window and run dense/row-table delivery on one CS-3 program."""

import argparse
import hashlib
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from bootstrap import HOT, MODULES, arrange
from inventory import actual, program_bytes, read, require, save, sha

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
FILES = [
    "worker.log",
    "worker-exit.txt",
    "loaded.json",
    "capture/result.json",
    *[f"capture/frame-{i}.npz" for i in range(1, 7)],
]


def hosts():
    pairs = {
        name: (
            FAMILY / "resident" / name
            if name in ("graph.py", "run.py")
            else (
                FAMILY / "transfer" / name
                if name == "rectangles.py"
                else HERE
                / "host"
                / ("hot-window" if directory.endswith("hot-window") else "")
                / name
            )
        )
        for name, directory in MODULES.items()
    }
    pairs.update({name: HERE / name for name in ("bootstrap.py", "inventory.py")})
    return pairs


def verified_tree(root, record):
    expected = read(record)
    require(
        set(expected)
        == {
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and p != record
        },
        "frozen tree contents",
    )
    for name, pin in expected.items():
        path = (root / name).resolve()
        require(
            path.is_relative_to(root.resolve()) and sha(path) == pin, "frozen " + name
        )


def origin(case, scratch):
    base = HERE / "base"
    shared = base / "upstream"
    src = scratch / "sources"
    arrange(hosts(), src)
    root = shared / "case"
    identity = read(root / "case.json")
    model = read(shared / "model.json")
    require(
        identity["original_checkpoint_sha256"]
        == model["weights_sha256"]
        == sha(shared / "weights.npz")
        and identity["original_graph"] == model["graph"],
        "trained weights and original topology",
    )
    with (
        np.load(root / "graph.npz", allow_pickle=False) as graph,
        np.load(shared / "weights.npz", allow_pickle=False) as weights,
    ):
        live = graph["live"]
        mappings = {
            "wr0": weights["root0.weight"][live].T,
            "wn0": weights["neigh0.weight"][live].T,
            "b0": weights["neigh0.bias"][live],
            "wr1": weights["root1.weight"][:, live].T,
            "wn1": weights["neigh1.weight"][:, live].T,
            "b1": weights["neigh1.bias"],
        }
        for key, original in mappings.items():
            np.testing.assert_array_equal(
                graph[key].view(np.uint32), original.view(np.uint32)
            )
    full = scratch / "full"
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(src / HOT / "prepare.py"),
            str(root),
            str(shared / "original"),
            str(shared / "capture"),
            str(full),
        ],
        check=True,
    )
    compare = full
    if case.name == "products-2hidden-p32":
        compare = scratch / "subset"
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(src / HOT / "calibration.py"),
                str(full),
                str(compare),
            ],
            check=True,
        )
    for name in ("fixture.npz", "fixture.json"):
        require(
            sha(compare / name) == sha(case / name),
            "regenerated exact upstream fixture " + name,
        )
    return src


def observed(image, fixture, capture, sources=None, completed_prefix=False):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        if sources is None:
            sources = work / "sources"
            arrange(hosts(), sources)
        result = work / "audit.json"
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(sources / HOT / "report.py"),
                str(image),
                str(fixture),
                str(capture),
                str(result),
                *(["--completed-prefix"] if completed_prefix else []),
            ],
            check=True,
        )
        return read(result)


def check(key, pin):
    require(
        key in ("products-2hidden-p32", "products-203hidden-p32"),
        "registered original-ID window",
    )
    case = HERE / "cases" / key
    base = HERE / "base"
    rel = read(case / "release.json")
    sdk = read(base / "sim/build.json")
    require(
        sha(case / "release.json") == pin
        and rel["key"] == key
        and rel["base_sha256"] == sha(base / "files.json"),
        "frozen base and case",
    )
    verified_tree(base, base / "files.json")
    require(
        rel["files"]
        == {
            p.relative_to(case).as_posix(): sha(p)
            for p in case.rglob("*")
            if p.is_file() and p != case / "release.json"
        },
        "complete calibration raw files",
    )
    require(
        rel["host_sha256"] == {n: sha(p) for n, p in hosts().items()},
        "checked host and wire implementation",
    )
    local = read(base / "physical/build.json")
    cfg = sdk["params"]
    flags = [
        "--fabric-dims=762,1172" if s.startswith("--fabric-dims=") else s
        for s in sdk["flags"]
    ]
    require(
        cfg["P"] == 32
        and local["role"] == "hardware"
        and local["flags"] == flags
        and sdk["source"]
        == local["source"]
        == {f.name: sha(f) for f in (base / "src").iterdir()},
        "local physical compile, source and owner count",
    )
    for name, desc in (("sim", sdk), ("physical", local)):
        inv = actual(base / name / "out")
        require(
            desc["elf"]
            == {
                Path(k).name: v
                for k, v in inv.items()
                if Path(k).parent == Path("bin") and k.endswith(".elf")
            },
            "local core ELF inventory",
        )
    require(
        read(base / "capacity.json")["cached_physical"]["last_byte"] == 47296,
        "P32 locally linked endpoint",
    )
    with tempfile.TemporaryDirectory() as work:
        sources = origin(case, Path(work))
        partial = rel.get("local_calibration") == "same-image-full-input-prefix"
        if partial:
            require(key == "products-203hidden-p32", "full captured hidden window")
            peer, peer_rel, peer_sdk, _ = check(
                "products-2hidden-p32", rel["same_image_peer"]
            )
            require(
                peer_sdk == sdk
                and peer_rel["base_sha256"] == rel["base_sha256"]
                and peer_rel["host_sha256"] == rel["host_sha256"],
                "completely verified same-image six-frame SDK calibration",
            )
            state = read(case / "sdk/result.json")
            require(
                not state["stopped"]
                and state["build_sha256"] == sha(base / "sim/build.json")
                and state["fixture_sha256"] == sha(case / "fixture.json")
                and state["platform"] == {"system": False, "simulation": True}
                and state["loaded_image"] == sdk["elf"]
                and state["worker_sha256"] == sha(hosts()["worker.py"])
                and state["wire_codec_sha256"] == sha(hosts()["window_cache.py"]),
                "exact larger-input simulator startup checkpoint",
            )
            replay = (
                observed(base / "sim", case, case / "sdk", sources, True)
                if state["frames"]
                else dict(
                    role="simulator",
                    observed_frames=0,
                    required_frames=6,
                    stopped=False,
                    frames=[],
                    capture_sha256=sha(case / "sdk/result.json"),
                )
            )
            require(
                replay["observed_frames"] == len(state["frames"])
                and replay == read(case / "audit.json"),
                "actual full-input simulator coverage",
            )
        else:
            replay = observed(base / "sim", case, case / "sdk", sources)
            require(
                replay == read(case / "audit.json")
                and replay["stopped"]
                and replay["role"] == "simulator"
                and len(replay["frames"]) == 6,
                "independent calibrated resident comparison",
            )
    return case, rel, sdk, flags


def bind(record, case, rel, sdk, flags):
    comp = read(record)
    name = comp["archive"]
    require(Path(name).name == name, "compiler archive name")
    archive = record.parent / name
    pin = sha(case / "release.json")
    require(
        comp["case"] == case.name
        and comp["release_sha256"] == pin
        and comp["source"] == sdk["source"]
        and comp["flags"] == flags
        and sha(archive) == comp["archive_sha256"],
        "original compiler receipt",
    )
    files = program_bytes(archive, sdk["source"], flags, sdk["params"])
    inv = {n: hashlib.sha256(v).hexdigest() for n, v in files.items()}
    require(inv == comp["inventory"], "actual system compiler firmware")
    desc = sdk | dict(
        role="hardware",
        flags=flags,
        elf={
            Path(n).name: v
            for n, v in inv.items()
            if Path(n).parent == Path("bin") and n.endswith(".elf")
        },
        inventory=inv,
        fixture_sha256=sha(case / "fixture.json"),
        host_sha256=rel["host_sha256"],
        release_sha256=pin,
    )
    return archive, desc, files


def validate_loaded(out, desc, expected):
    got = read(out / "loaded.json")
    require(
        got
        == dict(
            role="hardware",
            build_sha256=sha(out / "build.json"),
            inventory=desc["inventory"],
            host_sha256=desc["host_sha256"],
            fixture_sha256=desc["fixture_sha256"],
        )
        and actual(out / "out") == desc["inventory"],
        "launch artifact and returned host identity",
    )
    require(
        expected["role"] == "hardware"
        and expected["stopped"]
        and len(expected["frames"]) == 6,
        "six system captures and normal stop",
    )


def execute(a):
    case, rel, sdk, flags = check(a.case, a.release_sha256)
    if a.action == "check":
        return
    from cerebras.sdk.client import SdkCompiler, SdkLauncher

    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    options = dict(
        mgmt_namespace=a.namespace,
        resource_cpu=1000,
        resource_mem=8 << 30,
        disable_version_check=False,
    )
    if a.action == "compile":
        with SdkCompiler(**options) as compiler:
            archive = Path(
                compiler.compile(
                    str(HERE / "base/src"), "layout.csl", " ".join(flags), str(out)
                )
            ).resolve()
        require(
            archive.parent == out and archive.is_file(),
            "returned system compiler archive",
        )
        programs = program_bytes(archive, sdk["source"], flags, sdk["params"])
        save(
            out / "compiled.json",
            dict(
                case=case.name,
                release_sha256=a.release_sha256,
                source=sdk["source"],
                flags=flags,
                archive=archive.name,
                archive_sha256=sha(archive),
                inventory={
                    n: hashlib.sha256(v).hexdigest() for n, v in programs.items()
                },
            ),
        )
        return
    archive, desc, programs = bind(a.compiled.resolve(), case, rel, sdk, flags)
    for n, data in programs.items():
        p = out / "out" / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    save(out / "build.json", desc)
    (out / "capture").mkdir()
    status = dict(
        role="hardware",
        case=case.name,
        release_sha256=a.release_sha256,
        build_sha256=sha(out / "build.json"),
        archive_sha256=sha(archive),
        downloads={},
        numerical_status="pending",
    )
    save(out / "collection.json", status)
    command = shlex.join(
        [
            "timeout",
            "--signal=INT",
            "--kill-after=30s",
            "3600s",
            "cs_python",
            "-B",
            "bootstrap.py",
            "--cmaddr",
            "%CMADDR%",
        ]
    )
    try:
        with SdkLauncher(str(archive), simulator=False, **options) as launcher:
            try:
                for src in [
                    *hosts().values(),
                    out / "build.json",
                    case / "fixture.npz",
                    case / "fixture.json",
                ]:
                    launcher.stage(str(src))
                response = launcher.run(
                    command
                    + ' >worker.log 2>&1; rc=$?; echo "$rc" >worker-exit.txt; exit "$rc"'
                )
                (out / "launcher-response.txt").write_text(str(response))
            finally:
                for n in FILES:
                    try:
                        launcher.download_artifact(n, str(out / n))
                        status["downloads"][n] = {"sha256": sha(out / n)}
                    except Exception as error:
                        status["downloads"][n] = {"error": repr(error)}
                    save(out / "collection.json", status)
        require(
            all("sha256" in x for x in status["downloads"].values())
            and (out / "worker-exit.txt").read_text().strip() == "0",
            "complete physical return",
        )
        measured = observed(out, case, out / "capture")
        validate_loaded(out, desc, measured)
        sdk_audit = read(case / "audit.json")
        require(
            [x["output_sha256"] for x in measured["frames"][: len(sdk_audit["frames"])]]
            == [x["output_sha256"] for x in sdk_audit["frames"]],
            "original-ID device and SDK hidden vectors",
        )
        save(out / "replay.json", measured)
        status["numerical_status"] = "passed"
    finally:
        save(out / "collection.json", status)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("check", "compile", "run"))
    p.add_argument("--case", required=True)
    p.add_argument("--release-sha256", required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--compiled", type=Path)
    p.add_argument("--namespace", default="job-operator")
    a = p.parse_args()
    if a.action != "check" and not a.out:
        p.error("--out required")
    if a.action == "run" and not a.compiled:
        p.error("--compiled required")
    execute(a)
