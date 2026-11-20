"""Compare returned native CSL and Direct Link Tungsten from the selected original-order Cora request."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
HERE=Path(__file__).resolve().parent


def ratio(a,b):
    result={k:(np.asarray(a[k])/np.asarray(b[k])).tolist() for k in
            ("setup_sum","compute_sum","max_owner_cumulative_sum","phase_max_cumulative_sum")}
    x,y=b["max_owner_cumulative_sum"],a["max_owner_cumulative_sum"]
    result["first_sampled_inclusive_gain_through_three"]=next((n+1 for n in range(3) if all(x[j]<y[j] for j in range(n,3))),None)
    return result


def main(a):
    results={}
    for lang,root in (("csl",a.csl),("tungsten",a.tungsten)):
        argv=[sys.executable,"-B",str(HERE/lang/"verify_return.py"),str(root)]
        if lang=="csl":argv+=["--build",str(a.build)]
        results[lang]=json.loads(subprocess.check_output(argv,text=True))
    c,w=(results[k]["summary"] for k in ("csl","tungsten"))
    assert c["owner_sha256"]==w["owner_sha256"] and c["words"]==w["words"]==56868
    comparisons={"selected_csl_vs_selected_tungsten":ratio(w,c)}
    for name,path in (("prior_range",a.prior_range),("prior_cyclic",a.prior_cyclic)):
        if path is None:continue
        prior=json.loads(path.read_text())
        anchor=json.loads((HERE/"comparators.json").read_text())[name]
        assert prior["backend"]=="hardware" and prior["release_sha256"]==anchor["release_sha256"]
        comparisons[name]={}
        for language,cap in results.items():
            old=prior["cases"][language+"-r16"]
            if name=="prior_range": assert old["owner_sha256"]==cap["summary"]["owner_sha256"]
            assert old["validation"]==[402,500] and old["original_class_mismatches"]==[0,0,0]
            comparisons[name][language]=ratio(old,cap["summary"])
    a.out.write_text(json.dumps(dict(backend="hardware",active_pe=64,selected=results,comparisons=comparisons,
        metric="sum of each window maximum individual PE interval; excludes host and inter-window dispatch"),indent=2,sort_keys=True)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("csl","tungsten","build","out"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--prior-range",type=Path);p.add_argument("--prior-cyclic",type=Path)
    main(p.parse_args())
