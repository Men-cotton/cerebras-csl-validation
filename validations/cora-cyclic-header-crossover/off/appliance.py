"""Compile or run the complete resident Cora final aggregation with the validation watcher."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import tarfile
import time
import numpy as np
from contract import audit, load_graph, require, save, sha
from compile_csl import flags

ROOT = Path(__file__).resolve().parent
PORT = "e4a311bc7f0f12ff52503703f4e233182c840339f765d01d49650022fe951d3e"


def read(path):
    return json.loads(path.read_text())


def application_elfs(artifact):
    with tarfile.open(artifact, "r:*") as archive:
        regular = {PurePosixPath(m.name): m for m in archive if m.isfile()}
        roots = {p.parent.parent for p in regular if p.parts[-2:] == ("west", "out.json")}
        roots = {p for p in roots if p / "east/out.json" in regular and p / "out.json" in regular}
        require(len(roots) == 1, "compiler program tree")
        folder = next(iter(roots)) / "bin"
        images = {}
        for p, m in regular.items():
            if p.parent == folder and p.suffix == ".elf":
                require(m.size < 32 << 20, "reasonable CSL ELF size")
                images[p.name] = hashlib.sha256(archive.extractfile(m).read()).hexdigest()
        require(bool(images), "compiled application ELFs")
        return images


def check_capture(root, language, shards, role, release, graph, compiled=None):
    state = read(root / "result.json")
    require(
        state["language"] == language
        and state["role"] == role
        and state["shards"] == shards
        and state["release_sha256"] == sha(ROOT / "release.json")
        and state["input_sha256"] == release["input_sha256"]
        and state["host"] == release["host"]
        and state["platform"] == {"system": role == "hardware", "simulation": role == "simulator"}
        and sha(root / "raw.npz") == state["raw_sha256"],
        "captured program, input, role, raw bytes",
    )
    if language == "tungsten":
        require(
            state["elf_sha256"] == release["direct"][str(shards)]["roles"][role]["elf_sha256"]
            and state["coords"] == [[x, y] for y in range(shards) for x in range(4)],
            "actual ELF and unshifted Direct Link coordinates",
        )
    else:
        images = release["csl_simulator"][str(shards)]["elf"] if role == "simulator" else compiled
        require(
            images is not None and state["elf_sha256"] == images,
            "CSL executed application agrees with compiler artifact",
        )
    with np.load(root / "raw.npz", allow_pickle=False) as raw:
        summary = audit(graph, state, raw)
    recorded = state["summary"]
    require(set(recorded) == set(summary), "complete summary fields")
    for name, value in summary.items():
        if name != "fp64_error":
            require(value == recorded[name], "raw summary replay " + name)
    for actual, expected in zip(summary["fp64_error"], recorded["fp64_error"], strict=True):
        require(actual["max_abs"] == expected["max_abs"], "exact maximum FP64 error")
        # The appliance and analysis hosts can use different FP64 BLAS norm reductions.
        np.testing.assert_allclose(actual["relative_l2"], expected["relative_l2"], rtol=1e-12, atol=0)
    from contract import original_plan,window,word_sha
    for w in state["windows"]:
        i=w["index"]; expected=original_plan(graph,window(graph,shards,i),i,shards,release["mode"])
        np.testing.assert_array_equal(w["expected_plan"],expected)
        if language=="csl":
            actual=np.array(w["device_plan_all_pe"],np.uint32)
            require(actual.shape==(shards,4,3) and word_sha(actual)==w["device_plan_sha256"],"all native plan responses")
            np.testing.assert_array_equal(actual,np.broadcast_to(expected[:,None,:],actual.shape))
    if role=="hardware":
        ref=ROOT/"sim"/f"{language}-r{shards}"/"raw.npz"
        with np.load(ref,allow_pickle=False) as reference,np.load(root/"raw.npz",allow_pickle=False) as actual:
            require(set(actual.files)==set(reference.files),"all physical panels")
            for key in actual.files:
                np.testing.assert_array_equal(actual[key].reshape(4,-1)[:,20*shards:],reference[key].reshape(4,-1)[:,20*shards:])
    return summary


def verified():
    from oracle import build
    from contract import original_plan,window,word_sha
    release=read(ROOT/"release.json"); r=release["shards"]
    graph=load_graph(ROOT/"fixture.npz")
    require(release["mode"]=="all-fmac" and release["input_sha256"]==sha(ROOT/"fixture.npz") and
            release["provenance_sha256"]==sha(ROOT/"provenance.json") and release["port_sha256"]==PORT,"trained input and professor map")
    require(release["host"]=={f:sha(ROOT/f) for f in ("contract.py","worker.py")} and
            release["csl"]=={p.name:sha(p) for p in (ROOT/"csl").glob("*.csl")},"bound host and native sources")
    native=release["csl_simulator"][str(r)]
    require(native["flags"]==flags(r) and native["source"]==release["csl"] and
            native["elf"]=={p.name:sha(p) for p in (ROOT/f"csl-r{r}/out/bin").glob("*.elf")},"all native simulator PEs")
    direct=release["direct"][str(r)];base=ROOT/f"direct-r{r}"
    compiled=read(base/"build.json")
    require(compiled["equivalence"] and direct["build_sha256"]==sha(base/"build.json") and
            direct["machine_image_sha256"]==sha(base/"program.img") and sha(ROOT/"aggregation.w")==direct["source"]["aggregation.w"],"same Direct Link machine")
    for f,hashval in direct["source"].items():require(sha(base/"source"/f)==hashval,"compiler input "+f)
    for role,dimensions in (("simulator",[4,r]),("hardware",[762,1172])):
        require(sha(base/role/"out/program.elf")==direct["roles"][role]["elf_sha256"] and
                sha(base/role/"out/out_port_map.json")==PORT and direct["roles"][role]["fabric"]==dimensions,"actual target dimensions and image")
    expected=[word_sha(row) for row in build(graph,r,"cyclic")]
    comparison=read(ROOT/"simulator-comparison.json");results={}
    for lang in ("csl","tungsten"):
        key=f"{lang}-r{r}";capture=ROOT/"sim"/key
        result=check_capture(capture,lang,r,"simulator",release,graph)
        require(result["owner_sha256"]==expected and result["validation"]==[402,500] and
                comparison["cases"][lang]["raw_sha256"]==sha(capture/"raw.npz") and
                comparison["cases"][lang]["result_sha256"]==sha(capture/"result.json"),"all original Cora logits")
        results[key]=result
    return results


def main(a):
    sim = verified()
    print(
        "simulator per-panel maxima by case", {k: v["compute_sum"] for k, v in sim.items()}, flush=True
    )
    if a.action == "check":
        return
    from cerebras.sdk.client import SdkCompiler, SdkLauncher

    logging.basicConfig(level=logging.INFO)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    release = read(ROOT / "release.json")
    r = a.shards
    key = sha(ROOT / "release.json")
    common = dict(
        mgmt_namespace=a.namespace, resource_cpu=1000, resource_mem=8 << 30, disable_version_check=False
    )
    if a.action == "compile-csl":
        with SdkCompiler(**common) as compiler:
            archive = Path(
                compiler.compile(str(ROOT / "csl"), "layout.csl", flags(r, True), str(out))
            ).resolve()
        require(archive.parent == out and archive.is_file(), "compiled archive")
        save(
            out / "build.json",
            dict(
                artifact=archive.name,
                sha256=sha(archive),
                release_sha256=key,
                flags=flags(r, True),
                application_elfs=application_elfs(archive),
                shards=r,
            ),
        )
        return
    lang = a.action.removeprefix("run-")
    compiled = None
    if lang == "csl":
        record = read(a.build)
        artifact = (a.build.resolve().parent / record["artifact"]).resolve()
        compiled = record["application_elfs"]
        require(
            artifact.parent == a.build.resolve().parent
            and sha(artifact) == record["sha256"]
            and record["shards"] == r
            and record["flags"] == flags(r, True)
            and record["release_sha256"] == key
            and application_elfs(artifact) == compiled,
            "bound CSL archive",
        )
    else:
        artifact = out / "bundle"
        (artifact / "out").mkdir(parents=True)
        for f in ("program.elf", "out_port_map.json"):
            shutil.copyfile(ROOT / f"direct-r{r}/hardware/out" / f, artifact / "out" / f)
    command = shlex.join(
        [
            "timeout",
            "--signal=INT",
            "--kill-after=30s",
            "1800s",
            "cs_python",
            "worker.py",
            "--language",
            lang,
            "--shards",
            str(r),
            "--artifact",
            "out",
            "--input",
            "fixture.npz",
            "--manifest",
            "release.json",
            "--output",
            "capture",
            "--cmaddr",
            "%CMADDR%",
        ]
    )
    command += ' >worker.log 2>&1; code=$?; echo "$code" >worker-exit.txt; exit "$code"'
    state = dict(
        backend="hardware", language=lang, shards=r, release_sha256=key, command=command, downloads={}
    )
    save(out / "collection.json", state)
    os.chdir(out)
    with SdkLauncher(str(artifact), simulator=False, **common) as launcher:
        try:
            for f in ("worker.py", "contract.py", "fixture.npz", "release.json"):
                launcher.stage(str(ROOT / f))
            launched = time.perf_counter_ns()
            try:
                response = launcher.run(command)
            finally:
                state["launcher_run_elapsed_ns"] = time.perf_counter_ns() - launched
            (out / "response.txt").write_text(str(response))
        finally:
            for remote in ("capture/result.json", "capture/raw.npz", "worker.log", "worker-exit.txt"):
                dest = out / Path(remote).name
                try:
                    launcher.download_artifact(remote, str(dest))
                    state["downloads"][remote] = dict(sha256=sha(dest))
                except Exception as exc:
                    state["downloads"][remote] = dict(error=repr(exc))
                save(out / "collection.json", state)
    require(
        all("sha256" in v for v in state["downloads"].values())
        and (out / "worker-exit.txt").read_text().strip() == "0",
        "physical completion and collection",
    )
    state["summary"] = check_capture(
        out, lang, r, "hardware", release, load_graph(ROOT / "fixture.npz"), compiled
    )
    require(
        state["summary"]["owner_sha256"] == sim[f"{lang}-r{r}"]["owner_sha256"],
        "hardware against all simulator logits",
    )
    save(out / "collection.json", state)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("check", "compile-csl", "run-csl", "run-tungsten"))
    p.add_argument("--shards", type=int, choices=(16,))
    p.add_argument("--namespace", default="job-operator")
    p.add_argument("--out", type=Path)
    p.add_argument("--build", type=Path)
    a = p.parse_args()
    if a.action != "check" and (not a.shards or not a.out):
        p.error("appliance actions require --shards and --out")
    if a.action == "run-csl" and not a.build:
        p.error("run-csl requires --build")
    main(a)
