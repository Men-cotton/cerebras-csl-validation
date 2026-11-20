"""Compare returned strided-self hardware geometries and optionally the earlier physical cyclic-source report."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent

def ratio(old,new):
    return {k:(np.asarray(old[k])/np.asarray(new[k])).tolist() for k in
            ("setup_sum","compute_sum","max_owner_cumulative_sum","phase_max_cumulative_sum")}

def main(a):
    program={}
    for r,root in ((4,a.r4),(16,a.r16)):
        staged=HERE/f"r{r}"
        if not staged.exists():continue
        if root is None:raise ValueError(f"missing returned steps for {4*r} PEs")
        text=subprocess.check_output([sys.executable,"-B",str(staged/"return_one.py"),str(root)],text=True)
        program[str(r)]=json.loads(text)
    parent=None
    if a.parent_cyclic_report:
        parent=json.loads(a.parent_cyclic_report.read_text())
        assert parent["backend"]=="hardware" and parent["release_sha256"]==json.loads((HERE/"comparators.json").read_text())["release_sha256"]
    gains={}
    for r,case in program.items():
        results=case["cases"]
        gains[r]=dict(csl_vs_tungsten=ratio(results["tungsten"]["summary"],results["csl"]["summary"]))
        c,w=(results[lang]["summary"]["max_owner_cumulative_sum"] for lang in ("csl","tungsten"))
        gains[r]["first_measured_csl_gain_through_three"]=next((n+1 for n in range(3) if all(c[j]<w[j] for j in range(n,3))),None)
        if parent:
            for lang,record in results.items():
                old=parent["cases"][f"{lang}-r{r}"]
                assert record["summary"]["owner_sha256"]==old["owner_sha256"]
                gains[r][lang+"_vs_prior"]=ratio(old,record["summary"])
    scaling={}
    if set(program)=={"4","16"}:
        for lang in ("csl","tungsten"):
            scaling[lang]=ratio(program["4"]["cases"][lang]["summary"],program["16"]["cases"][lang]["summary"])
    a.output.write_text(json.dumps(dict(backend="hardware",programs=program,comparisons=gains,strong_scaling=scaling,
                timing="sum of per-window longest individual PE interval; host and window dispatch excluded"),indent=2,sort_keys=True)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("r4","r16","parent-cyclic-report"):p.add_argument("--"+key,type=Path)
    p.add_argument("--output",type=Path,required=True);main(p.parse_args())
