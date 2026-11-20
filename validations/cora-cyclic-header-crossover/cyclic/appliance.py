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
    from contract import window,word_sha
    from plan import expected_plan
    target=read(ROOT/"target.json")
    for entry in state["windows"]:
        i=entry["index"];sl=window(graph,shards,i)
        require(entry["packed_graph_sha256"]==word_sha(sl["wire"]),"actual original window graph")
        if target["method"]!="original":
            expected=expected_plan(graph,sl,i,shards,target["method"])
            if language=="csl":
                actual=np.asarray(entry["device_plan_all_pe"],np.uint32)
                require(actual.shape==(shards,4,3) and word_sha(actual)==entry["device_plan_sha256"],"all device plan words")
                np.testing.assert_array_equal(actual,np.broadcast_to(expected[:,None,:],actual.shape))
    if role=="hardware":
        with np.load(ROOT/"sim/raw.npz",allow_pickle=False) as reference,np.load(root/"raw.npz",allow_pickle=False) as actual:
            require(set(actual.files)==set(reference.files),"all physical panels")
            for key in actual.files:
                np.testing.assert_array_equal(actual[key].reshape(4,-1)[:,20*shards:],reference[key].reshape(4,-1)[:,20*shards:])
    return summary


def verified():
    from contract import word_sha
    from oracle import build
    target=read(ROOT/"target.json");lang=target["language"];release=read(ROOT/"release.json")
    graph=load_graph(ROOT/"fixture.npz")
    require(release["input_sha256"]==sha(ROOT/"fixture.npz") and release["provenance_sha256"]==sha(ROOT/"provenance.json")
            and target["release_sha256"]==sha(ROOT/"release.json") and target["input_sha256"]==release["input_sha256"],"original Cora and selected release")
    require(release["host"]=={key:sha(ROOT/key) for key in ("worker.py","contract.py")},"selected host source")
    if lang=="csl":
        m=release["csl_simulator"]["16"];images={p.name:sha(p) for p in (ROOT/"csl-r16/out/bin").glob("*.elf")}
        require(m["flags"]==flags(16) and m["source"]==release["csl"]=={p.name:sha(p) for p in (ROOT/"csl").glob("*.csl")}
                and m["elf"]==images and len(images)==64,"actual native input, flags and simulator executables")
    else:
        m=release["direct"]["16"];d=ROOT/"direct-r16";info=read(d/"build.json")
        require(info["equivalence"] and info["port_sha256"]==release["port_sha256"]==PORT and m["build_sha256"]==sha(d/"build.json")
                and m["machine_image_sha256"]==sha(d/"program.img") and m["source"]["aggregation.w"]==sha(ROOT/"aggregation.w"),"matching Tungsten compiler input and image")
        for f,h in m["source"].items():require(sha(d/"source"/f)==h,"exact compiler input "+f)
        for role,dims in (("simulator",[4,16]),("hardware",[762,1172])):
            require(m["roles"][role]["fabric"]==dims and sha(d/role/"out/program.elf")==m["roles"][role]["elf_sha256"]
                    and sha(d/role/"out/out_port_map.json")==PORT,"actual machine target and professor port map")
    cap=ROOT/"sim";summary=check_capture(cap,lang,16,"simulator",release,graph)
    original=[word_sha(v) for v in build(graph,16,"cyclic")]
    require(target["raw_sha256"]==sha(cap/"raw.npz") and target["result_sha256"]==sha(cap/"result.json") and
            summary["owner_sha256"]==original and summary["words"]==56868 and summary["validation"]==[402,500],"complete original fused logits")
    if target.get("pointer_sram_sha256"):
        require(target["pointer_sram_sha256"]==sha(cap/"pointer-sram.npz")==read(cap/"pointer-sram.json")["raw_sha256"],"stopped simulator SRAM arrays")
    return {lang+"-r16":summary}


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
    p.add_argument("action", choices=("check","run-tungsten"))
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
