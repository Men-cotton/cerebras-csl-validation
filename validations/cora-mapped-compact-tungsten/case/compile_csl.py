"""Build the CSL simulator fixture with the same source, params and SDK flags as the appliance."""

import argparse
from pathlib import Path
import subprocess
from contract import sha, sizes, save

HERE = Path(__file__).resolve().parent


def flags(r, hardware=False):
    cap, ec = sizes(r)
    dims = "762,1172" if hardware else f"11,{r + 2}"
    return f"--arch=wse3 --fabric-dims={dims} --fabric-offsets=4,1 --params=R:{r},F:{cap * 2},E:{ec} --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-parallelism=1 -o out"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("shards", type=int)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    dest = a.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    argv = ["cslc", str(HERE / "csl/layout.csl"), *flags(a.shards).split()]
    with (dest / "compile.log").open("w") as log:
        subprocess.run(argv, cwd=dest, stdout=log, stderr=subprocess.STDOUT, check=True)
    save(
        dest / "build.json",
        dict(
            source={p.name: sha(p) for p in (HERE / "csl").glob("*.csl")},
            flags=flags(a.shards),
            elf={p.name: sha(p) for p in (dest / "out/bin").glob("*.elf")},
        ),
    )
