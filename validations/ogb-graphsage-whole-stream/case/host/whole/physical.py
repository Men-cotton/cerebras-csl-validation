"""Launch or independently replay the complete, original ogbn-arxiv final layer."""
import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np

from dispatch import read, save, sha
from replay import audit, require

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "device"))
from appliance import canonical_elf, images_in_archive


from stream_variant import checked_native


def checked_case():
    return checked_native(ROOT)


def unpack(archive, dest):
    dest.mkdir()
    total = 0
    with zipfile.ZipFile(archive) as file:
        members = file.infolist()
        require(1 <= len(members) <= 10000 and len({m.filename for m in members}) == len(members), "capture members")
        for item in members:
            require(item.filename == "result.json" or re.fullmatch(r"w[0-9]{4}\.npz", item.filename), "unknown capture path")
            total += item.file_size
            require(0 < item.file_size <= 32 << 20 and total <= 512 << 20, "capture size")
            with file.open(item) as source, (dest / item.filename).open("wb") as output:
                shutil.copyfileobj(source, output)
    return dest


def execute(a):
    case, release, native, direct = checked_case()
    if a.action == "check":
        return
    from cerebras.sdk.client import SdkCompiler, SdkLauncher
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    common = dict(mgmt_namespace=a.namespace, resource_cpu=1000, resource_mem=8 << 30, disable_version_check=False)
    flags = [("--fabric-dims=762,1172" if item.startswith("--fabric-dims=") else item) for item in native["flags"]]
    key = sha(case / "release.json")
    if a.action == "compile-csl":
        with SdkCompiler(**common) as compiler:
            artifact = Path(compiler.compile(str(case / "csl/src"), "layout.csl", " ".join(flags), str(out))).resolve()
        require(artifact.is_file() and artifact.parent == out, "CSL compiler artifact")
        save(out / "compiled.json", dict(artifact=artifact.name, archive_sha256=sha(artifact), elf=images_in_archive(artifact),
                                         release_sha256=key, flags=flags))
        return
    lang = a.action.removeprefix("run-")
    if lang == "csl":
        rec = read(a.compiled)
        artifact = a.compiled.resolve().parent / rec["artifact"]
        require(artifact.parent == a.compiled.resolve().parent and rec["release_sha256"] == key and
                sha(artifact) == rec["archive_sha256"] and flags == rec["flags"] and
                images_in_archive(artifact) == rec["elf"], "physical compilation identity")
        descriptor = native | {"role": "hardware", "elf": rec["elf"], "flags": flags}
    else:
        artifact = out / "artifact"
        shutil.copytree(case / "tungsten/hardware/out", artifact / "out")
        descriptor = direct
    save(out / "build.json", descriptor)
    shutil.copyfile(case / lang / "windows.json", out / "windows.json")
    state = dict(release_sha256=key, language=lang, role="hardware", downloads={})
    save(out / "collection.json", state)
    os.chdir(out)
    command = " && ".join((shlex.join(["cs_python", "-B", "remote.py", "prepare"]),
                  shlex.join(["timeout", "--signal=INT", "--kill-after=30s", "7200s", "cs_python", "-B",
                              "whole/run.py", ".", "whole.npz", "capture", "--cmaddr", "%CMADDR%"])))
    command = "(" + command + ') >worker.log 2>&1; rc=$?; echo "$rc" >worker-exit.txt; cs_python -B remote.py collect; exit "$rc"'
    with SdkLauncher(str(artifact), simulator=False, **common) as launcher:
        try:
            for f in (HERE / "remote.py", HERE / "run.py", HERE / "dispatch.py", HERE / "rectangles.py", ROOT / "device/contract.py",
                      case / "whole.npz", out / "build.json", out / "windows.json"):
                launcher.stage(str(f))
            start = time.perf_counter_ns()
            try:
                value = launcher.run(command)
            finally:
                state["launcher_elapsed_ns"] = time.perf_counter_ns() - start
            (out / "response.txt").write_text(str(value))
        finally:
            for name in ("capture.zip", "worker.log", "worker-exit.txt"):
                try:
                    launcher.download_artifact(name, str(out / name))
                    state["downloads"][name] = {"sha256": sha(out / name)}
                except Exception as err:
                    state["downloads"][name] = {"error": repr(err)}
                save(out / "collection.json", state)
    require(all("sha256" in value for value in state["downloads"].values()) and
            (out / "worker-exit.txt").read_text().strip() == "0", "incomplete physical capture")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        local = unpack(out / "capture.zip", tmp / "local")
        other = unpack(a.peer.resolve(), tmp / "peer") if a.peer else None
        measured = audit(out, case / "whole.npz", local, role="hardware", peer=other)
        require(measured["graph_complete"], "incomplete original graph")
        # A simulator and physical run of the identical firmware must reproduce every tested logit bit.
        before = read(case / ("sim-" + lang) / "result.json")["completed"]
        before = {item["window"]: item for item in before}
        for window in release["representative_windows"]:
            name = f"w{window:04d}.npz"
            with np.load(case / ("sim-" + lang) / name) as sim, np.load(local / name) as hw:
                k = before[window]["parts"]
                start = 4 + 6 * (k + 2)
                for t in sim.files:
                    lhs, rhs = sim[t], hw[t]
                    np.testing.assert_array_equal(lhs[:, :4], rhs[:, :4])
                    r, q = descriptor["params"]["R"], descriptor["params"]["Q"]
                    for pe, count in enumerate(lhs[:, 2]):
                        n = int(count)
                        np.testing.assert_array_equal(lhs[pe, start:start+n], rhs[pe, start:start+n])
                        first = start + r
                        np.testing.assert_array_equal(lhs[pe, first:first+n*q], rhs[pe, first:first+n*q])
        save(out / "whole-graph-audit.json", measured)
    state["numerical_status"] = "PASS"
    save(out / "collection.json", state)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("check", "compile-csl", "run-csl", "run-tungsten"))
    p.add_argument("--out", type=Path)
    p.add_argument("--compiled", type=Path)
    p.add_argument("--peer", type=Path)
    p.add_argument("--namespace", default="job-operator")
    a = p.parse_args()
    if a.action != "check" and not a.out:
        p.error("output directory required")
    if a.action == "run-csl" and not a.compiled:
        p.error("physical compilation receipt required")
    if a.compiled:
        a.compiled = a.compiled.resolve()
    if a.peer:
        a.peer = a.peer.resolve()
    execute(a)
