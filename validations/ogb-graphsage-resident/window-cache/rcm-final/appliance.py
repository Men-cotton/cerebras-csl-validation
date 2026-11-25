"""Measure final-layer source delivery on the returned native 32-PE system application."""
import argparse
import hashlib
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent
sys.path.insert(0, str(SHARED))
from bootstrap import HOT, MODULES
from inventory import program_bytes, read, require, save, sha
from physical import hosts
from session import verify

DOWNLOADS = ("worker.log", "worker-exit.txt", "loaded.json", "capture/result.json",
             *(f"capture/frame-{i}.npz" for i in range(1, 7)))


def sources():
    family_hosts = hosts()
    files = {f"runtime/{folder}/{name}": family_hosts[name] for name, folder in MODULES.items()}
    for name in ("contract.py", "worker_final.py", "audit_final.py"):
        files[str(Path("runtime") / HOT / "rcm-final" / name)] = HERE / "host" / name
    files.update({"session.py": HERE / "session.py", "inventory.py": SHARED / "inventory.py"})
    return files


def load_segments(elf):
    require(elf[:7] == b"\x7fELF\x02\x01\x01", "little-endian ELF64")
    entry, offset = struct.unpack_from("<QQ", elf, 24)
    size, count = struct.unpack_from("<HH", elf, 54)
    segments = []
    for index in range(count):
        kind, flags, start, vaddr, paddr, length, memory_size, alignment = struct.unpack_from(
            "<IIQQQQQQ", elf, offset + index * size)
        if kind == 1:
            require(start + length <= len(elf), "complete PT_LOAD")
            segments.append((flags, vaddr, paddr, memory_size, alignment, elf[start:start + length]))
    require(segments, "loadable PE image")
    return entry, segments


def audit(stage, image, fixture, capture, destination, partial=False):
    auditor = stage / "runtime" / HOT / "rcm-final/audit_final.py"
    subprocess.run([sys.executable, "-B", str(auditor), str(image), str(fixture), str(capture),
                    str(destination), *(["--partial"] if partial else [])], check=True)
    return read(destination)


def prepare(stage, pin):
    release = HERE / "release.json"
    require(sha(release) == pin, "immutable RCM request")
    rel = read(release)
    for key, expected in rel["files"].items():
        require(sha(HERE / key) == expected, "package " + key)
    runtime = sources()
    require({n: sha(path) for n, path in runtime.items()} == rel["runtime"], "measured host sources")
    for name, path in runtime.items():
        dst = stage / name
        dst.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(path, dst)
    for name, src in (("release.json", release), ("build.json", HERE / "physical/build.json"),
                      ("fixture.json", HERE / "physical/fixture.json"), ("fixture.npz", HERE / "fixture.npz")):
        shutil.copyfile(src, stage / name)

    desc = read(stage / "build.json")
    fixture = read(HERE / "fixture.json")
    local = SHARED / "base/physical"
    require(sha(local / "build.json") == fixture["physical_size_sha256"]
            and sha(SHARED / "base/sim/build.json") == fixture["simulator_sha256"]
            and read(stage / "fixture.json") == fixture | {"physical_size_sha256": sha(stage / "build.json")},
            "same input with separate measured local and physical image pins")
    receipt = read(HERE / "physical/compiled.json")
    archive = HERE / "physical" / receipt["archive"]
    require(sha(archive) == receipt["archive_sha256"] == desc["archive_sha256"]
            and receipt["source"] == desc["source"] == read(local / "build.json")["source"]
            and receipt["flags"] == desc["flags"] == read(local / "build.json")["flags"]
            and desc["params"] == fixture["params"], "physical compiler output, sources and options")
    binaries = program_bytes(archive, desc["source"], desc["flags"], desc["params"])
    inventory = {name: hashlib.sha256(data).hexdigest() for name, data in binaries.items()}
    measured = read(HERE / "physical/collection.json")
    previous = read(HERE / "physical/loaded.json")
    require(receipt["case"] == measured["case"] == "products-203hidden-p32"
            and previous["role"] == "hardware" and measured["numerical_status"] == "passed"
            and receipt["archive_sha256"] == measured["archive_sha256"]
            and sha(HERE / "physical/loaded.json") == measured["downloads"]["loaded.json"]["sha256"]
            and previous["build_sha256"] == measured["build_sha256"]
            and inventory == receipt["inventory"] == desc["inventory"] == previous["inventory"],
            "system archive matches the earlier verified system launch")
    different = {n for n, data in binaries.items() if n.endswith(".elf") and
                 load_segments(data) != load_segments((local / "out" / n).read_bytes())}
    require(different == {"east/bin/out_1_0.elf"} and
            set(desc["elf"]) == {f"out_{i}_0.elf" for i in range(32)} and
            all(inventory["bin/" + n] == digest for n, digest in desc["elf"].items()),
            "32 identical loaded application images and known system memcpy helper")
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        artifact = work / "out"
        for name, blob in binaries.items():
            dst = artifact / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(blob)
        verify(stage, artifact, pin)
        sdk = audit(stage, SHARED / "base/sim", HERE, HERE / "sdk", work / "prefix.json", partial=True)
        require(sdk == read(HERE / "sdk/audit.json") and sdk["observed"] == 2 and not sdk["stopped"],
                "independently replayed two original-input simulator frames")
    return archive, sdk


def run(args):
    with tempfile.TemporaryDirectory(prefix="rcm4789-") as tmp:
        stage = Path(tmp) / "rcm-case"
        stage.mkdir()
        archive, sdk = prepare(stage, args.release)
        if args.action == "check":
            return
        from cerebras.sdk.client import SdkLauncher
        out = args.out.resolve()
        out.mkdir(parents=True, exist_ok=False)
        (out / "capture").mkdir()
        state = dict(release_sha256=args.release, archive_sha256=sha(archive), downloads={}, numerical_status="pending")
        save(out / "collection.json", state)
        for name in ("build.json", "fixture.json"):
            shutil.copyfile(stage / name, out / name)
        # SDK directory staging can leave a tarball in the remote application root.
        # Stage one ordinary file and let the command that consumes it unpack it.
        payload = Path(shutil.make_archive(str(Path(tmp) / "rcm-payload"), "zip", stage))
        unpack = shlex.join(["cs_python", "-m", "zipfile", "-e", payload.name, stage.name])
        command = shlex.join(["timeout", "--signal=INT", "--kill-after=30s", "3600s",
                              "cs_python", "-B", stage.name + "/session.py", "--release", args.release,
                              "--artifact", "out", "--cmaddr", "%CMADDR%"])
        with SdkLauncher(str(archive), simulator=False, mgmt_namespace=args.namespace,
                         resource_cpu=1000, resource_mem=8 << 30, disable_version_check=False) as launcher:
            try:
                launcher.stage(str(payload))
                response = launcher.run(f"({unpack} && {command}) >worker.log 2>&1; "
                                        'rc=$?; echo "$rc" >worker-exit.txt; exit "$rc"')
                (out / "launcher-response.txt").write_text(str(response))
            finally:
                for name in DOWNLOADS:
                    try:
                        remote = name if name in ("worker.log", "worker-exit.txt") else stage.name + "/" + name
                        launcher.download_artifact(remote, str(out / name))
                        state["downloads"][name] = {"sha256": sha(out / name)}
                    except Exception as error:
                        state["downloads"][name] = {"error": repr(error)}
                    save(out / "collection.json", state)
        require(all("sha256" in r for r in state["downloads"].values())
                and (out / "worker-exit.txt").read_text().strip() == "0", "six raw frames and successful worker")
        descriptor = read(stage / "build.json")
        require(read(out / "loaded.json") == dict(release_sha256=args.release,
                fixture_sha256=sha(stage / "fixture.json"), build_sha256=sha(stage / "build.json"),
                inventory=descriptor["inventory"]), "actual CS-3 firmware receipt")
        result = audit(stage, stage, stage, out / "capture", out / "replay.json")
        require(result["role"] == "hardware" and result["observed"] == 6 and result["stopped"]
                and [r["ordered_logits_sha256"] for r in result["frames"][:2]]
                == [r["ordered_logits_sha256"] for r in sdk["frames"]], "all physical frames and SDK prefix")
        state["numerical_status"] = "passed"
        save(out / "collection.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--release", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--namespace", default="job-operator")
    args = parser.parse_args()
    if args.action == "run" and not args.out:
        parser.error("run requires --out")
    run(args)
