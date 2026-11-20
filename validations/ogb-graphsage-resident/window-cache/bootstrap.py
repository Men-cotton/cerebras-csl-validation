"""Bind the staged native application, memcpy firmwares and worker before launching the measured window."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from inventory import actual, read, require, save, sha

# The research runtime resolves its sibling modules relative to these directories.
MODULES = {
    "graph.py": "ogb-graphsage/resident",
    "run.py": "ogb-graphsage/resident",
    "rectangles.py": "ogb-graphsage/transfer",
    "traffic.py": "host-device-transfer/ogb-traffic",
    "sparse_events.py": "host-device-transfer/ogb-traffic",
    "window_cache.py": "host-device-transfer/ogb-traffic",
    "uniform.py": "host-device-transfer/ogb-traffic",
    "codec.py": "host-device-transfer/ogb-traffic",
    "worker.py": "host-device-transfer/ogb-traffic/hot-window",
    "report.py": "host-device-transfer/ogb-traffic/hot-window",
    "prepare.py": "host-device-transfer/ogb-traffic/hot-window",
    "calibration.py": "host-device-transfer/ogb-traffic/hot-window",
}
HOT = Path("host-device-transfer/ogb-traffic/hot-window")


def arrange(paths, dest):
    dest.mkdir(parents=True, exist_ok=False)
    for name, parent in MODULES.items():
        target = dest / parent / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths[name], target)


def verified(stage, artifact):
    desc = read(stage / "build.json")
    names = set(MODULES) | {"inventory.py", "bootstrap.py"}
    require(set(desc["host_sha256"]) == names, "complete measured worker interface")
    files = {n: stage / n for n in names}
    require(
        {n: sha(p) for n, p in files.items()} == desc["host_sha256"],
        "staged host input hashes",
    )
    inv = actual(artifact)
    require(
        inv == desc["inventory"]
        and desc["elf"]
        == {
            Path(n).name: v
            for n, v in inv.items()
            if Path(n).parent == Path("bin") and n.endswith(".elf")
        },
        "actual core, RPC and memcpy firmware",
    )
    require(
        desc["fixture_sha256"] == sha(stage / "fixture.json")
        and sha(stage / "fixture.npz")
        == read(stage / "fixture.json")["files"]["fixture.npz"],
        "actual complete original-ID inputs",
    )
    return desc, files, inv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cmaddr")
    p.add_argument("--artifact", type=Path, default=Path("out"))
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    stage = Path.cwd()
    artifact = a.artifact.resolve()
    desc, paths, inventory = verified(stage, artifact)
    if a.verify_only:
        return
    require(
        bool(a.cmaddr) == (desc["role"] == "hardware"), "actual CS-3 target selection"
    )
    arrange(paths, stage / "runtime")
    save(
        stage / "loaded.json",
        dict(
            role=desc["role"],
            build_sha256=sha(stage / "build.json"),
            inventory=inventory,
            host_sha256=desc["host_sha256"],
            fixture_sha256=desc["fixture_sha256"],
        ),
    )
    command = [
        sys.executable,
        "-B",
        str(stage / "runtime" / HOT / "worker.py"),
        str(stage),
        str(stage),
        str(stage / "capture"),
        "--artifact",
        str(artifact),
    ]
    if a.cmaddr:
        command.extend(["--cmaddr", a.cmaddr])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
