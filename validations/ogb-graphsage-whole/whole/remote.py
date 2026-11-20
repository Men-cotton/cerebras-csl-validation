"""Run or collect a staged whole-layer program within an appliance launcher."""
import argparse
import os
from pathlib import Path
import re
import sys
import zipfile


def prepare():
    base = Path.cwd()
    for folder, files in (("whole", ("run.py", "dispatch.py")), ("device", ("contract.py",))):
        dest = base / folder
        dest.mkdir()
        for name in files:
            os.link(base / name, dest / name)
    # The launcher exposes the Direct Link image and compiled CSL application as ./out.
    (base / "hardware").mkdir()
    (base / "hardware/out").symlink_to("../out", target_is_directory=True)


def collect():
    src = Path("capture")
    members = [f for f in sorted(src.glob("*")) if f.name == "result.json" or re.fullmatch(r"w[0-9]{4}\.npz", f.name)]
    with zipfile.ZipFile("capture.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        for path in members:
            archive.write(path, path.name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "collect"))
    a = p.parse_args()
    (prepare if a.action == "prepare" else collect)()
