"""Compare the returned 64-PE all-FMAC Direct Link job to the published header-control hardware jobs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parent


def main(a):
    case=json.loads(subprocess.check_output([sys.executable,"-B",str(ROOT/"case/verify_return.py"),str(a.run)],text=True))
    assert case["backend"]=="hardware" and case["language"]=="tungsten"
    out=dict(backend="hardware",mapped_tungsten=case,prior_hardware_available=bool(a.prior),
             scope="sum of each window's maximum individual PE setup plus N compute intervals; excludes transfers and host dispatch")
    if a.prior:
        prior=json.loads(a.prior.read_text());assert prior["backend"]=="hardware" and prior["shared_physical_csl_image"]
        old=prior["cases"]
        assert old["cyclic-tungsten"]["physical_elf_sha256"]==case["physical_elf_sha256"]
        assert old["mapped-tungsten"]["summary"]["owner_sha256"]==case["summary"]["owner_sha256"]
        measurements={}
        for name,result in old.items():
            a0,b0=result["summary"],case["summary"]
            measurements[name]=dict(steady_difference=a0["compute_sum"][0]-b0["compute_sum"][0],
                setup_difference=a0["setup_sum"]-b0["setup_sum"],
                inclusive_difference=[x-y for x,y in zip(a0["max_owner_cumulative_sum"],b0["max_owner_cumulative_sum"])],
                prior_collection_sha256=result["collection_sha256"])
        out["prior_minus_new_cycles"]=measurements
    a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);p.add_argument("out",type=Path);p.add_argument("--prior",type=Path)
    main(p.parse_args())
