"""Measure two order-preserving lowerings on one PE with identical FP32 arrays."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(n, seed):
    rng = np.random.default_rng(seed)
    a,b,c = [rng.standard_normal(n).astype("<f4") for _ in range(3)]
    # The two parenthesizations differ on these entries in single precision.
    a[::7], b[::7], c[::7] = 2**24, 1, -(2**24)
    if not np.all(((a+b)+c)[::7] == 0) or not np.all((a+(b+c))[::7] == 1):
        raise ValueError("host FP32 order discriminator failed")
    return a,b,c


def run(case, sdk, mode, n, seed):
    case.mkdir(parents=True,exist_ok=False)
    env = dict(os.environ, PATH=str(sdk/"bin")+":"+os.environ["PATH"])
    def tool(name, *args):
        proc = subprocess.run([sdk/"bin"/name,*args],cwd=case,env=env,text=True,
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        if proc.returncode:
            raise RuntimeError(f"{name} failed in {case}\n{proc.stdout}")
        return proc.stdout
    for name in ("sum.w","bench.paint"):
        shutil.copyfile(HERE/name,case/name)
    version = tool("carbide","version").strip()
    tool("paint","source=bench.paint","lib=.","arch=sdr","annotate","image=sum.obj",
         "map=sum.map",f"set=N={n}",f"set=MODE={mode}")
    a,b,c = inputs(n,seed)
    previous = "sum.obj"
    for name,values in (("a",a),("b",b),("c",c)):
        values.tofile(case/(name+".bin"))
        target = name+".img"
        tool("miset","image="+previous,"out="+target,"map=sum.map","symbol=pe."+name,
             "order=n _word","data="+name+".bin")
        previous = target
    tool("mielf","image="+previous,"elf=sum.elf")
    log=tool("simfabric-sdk","elf=sum.elf","core=sum.elf-core","json=0","threads=1","cycles=500000")
    if "simfabric stopped: IDLE" not in log or "All router state is clean" not in log:
        raise ValueError("incomplete simulator run")
    tool("elfmi","elf=sum.elf-core","image=sim.core")
    for name,order in (("out","n _word"),("clocks","n")):
        tool("miget","image=sim.core","map=sum.map","symbol=pe."+name,"order="+order,"data="+name+".bin")
    raw=np.fromfile(case/"out.bin","<f4")
    np.testing.assert_array_equal(raw.view("u4"),((a+b)+c).view("u4"))
    clock_words=np.fromfile(case/"clocks.bin","<u2").reshape(2,3)
    t = clock_words.astype(np.int64) @ np.array([1,65536,1<<32])
    cycles=int((t[1]-t[0])%(1<<48))
    if not 0<cycles<500000:raise ValueError("invalid PE interval")
    layout=(case/"sum.map").read_text()
    addresses={k:int(re.search(r"^pe\."+k+r"\s+\[[^\n]+?\]\s+\(\s*(0x[0-9a-f]+)",layout,re.M)[1],16)
               for k in ("a","b","c","out")}
    listing=next((case/"sum_artefacts").glob("*.yeti"))
    adds=[line.split("//",1)[0].strip() for line in listing.read_text().splitlines() if line.strip().startswith("fadds ")]
    return dict(mode=("expression","staged")[mode],elements=n,seed=seed,version=version,
                cycles=cycles,clock_words=clock_words.tolist(),word_addresses=addresses,adds=adds,
                sha256={f:sha(case/f) for f in ("a.bin","b.bin","c.bin","out.bin","sum.elf","sum.map")},
                listing_sha256=sha(listing))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--sdk",type=Path,default=HERE.parents[2]/"sdr-toolset")
    p.add_argument("--sizes",type=int,nargs="+",default=(17,64,257,1024,2048))
    args=p.parse_args()
    root = args.output.resolve()
    rows=[]
    for n in args.sizes:
        if n<1 or n>2048:p.error("expected 1..2048 elements within the measured PE SRAM budget")
        for seed in (88457,27):
            pair=[run(root/f"n{n}-s{seed}-m{m}",args.sdk.resolve(),m,n,seed) for m in (0,1)]
            if pair[0]["word_addresses"]!=pair[1]["word_addresses"]:
                raise ValueError("layout confounds the comparison")
            if any(pair[0]["sha256"][f]!=pair[1]["sha256"][f] for f in ("a.bin","b.bin","c.bin","out.bin")):
                raise ValueError("input/output identities differ")
            rows.extend(pair)
    (root/"comparison.json").write_text(json.dumps(rows,indent=2)+"\n")
    for row in rows:print(row["elements"],row["seed"],row["mode"],row["cycles"])


if __name__=="__main__":
    main()
