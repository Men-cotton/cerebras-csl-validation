"""Place the staged hardware worker and preserve all returned raw windows."""
import argparse
import os
from pathlib import Path
import re
import zipfile
from transport import parts


def prepare():
    base=Path.cwd()
    for dirname,names in (("resident",("run.py","graph.py")),("transfer",("rectangles.py",))):
        here=base/dirname;here.mkdir()
        for name in names:os.link(base/name,here/name)
    (base/"hardware").mkdir()
    (base/"hardware/out").symlink_to("../out",target_is_directory=True)


def collect():
    with zipfile.ZipFile("capture.zip","w",compression=zipfile.ZIP_STORED) as archive:
        for p in sorted(Path("capture").glob("*")):
            if p.name in ("result.json","windows.jsonl") or re.fullmatch(r"t[123]-s[012]-w[0-9]{4}\.npz",p.name):
                archive.write(p,p.name)
    parts(Path("capture.zip"),Path.cwd())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("action",choices=("prepare","collect"))
    a=parser.parse_args();(prepare if a.action=="prepare" else collect)()
