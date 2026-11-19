"""Compare all simulated or hardware-returned 129-panel PE-clock prefixes."""
import argparse
import hashlib
import json
import numpy as np
from pathlib import Path
from contract import FRAMES,KEYS,load_graph,require,save,sha,verify

def compare(release_dir, role, captures, compile_record=None):
    require("csl-planned" in captures and len(captures)>=2,"CSL and at least one Tungsten target")
    release=json.loads((release_dir/"release.json").read_text())
    z=load_graph(release_dir/"fixture.npz")
    result={}
    hashes=[]
    for case,path in captures.items():
        rec=json.loads(path.read_text())
        require(case in KEYS and rec["code_sha256"]==release["host"] and rec["release_sha256"]==sha(release_dir/"release.json"),"release/worker identity")
        if case.startswith("tungsten"):
            require(rec["elf_sha256"]==release["direct"][case][role]["elf_sha256"],"executed Direct Link program")
        elif role=="simulator":
            require(rec["csl_elf_sha256"]==release["simulator_csl_elf_sha256"],"executed simulator CSL")
        else:
            require(compile_record,"hardware build record")
            build=json.loads(compile_record.read_text())
            require(rec["csl_elf_sha256"]==build["application_elfs"] and build["release"]==rec["release_sha256"],"compiled hardware CSL")
        verdict=verify(z,rec,role,case)
        hashes.append(verdict["output_sha256"])
        s=np.array(rec["trials"][0]["numeric"]["setup"],np.int64)
        c=np.array([t["numeric"]["compute"] for t in rec["trials"]],np.int64)
        total=s+np.cumsum(c,axis=0)
        both={"max_owner_total":total.max(1),"sum_phase_max":s.max()+np.cumsum(c.max(1))}
        sampled={str(n):{k:int(seq[n-1]) for k,seq in both.items()} for n in (1,12,93,94,95,128,129)}
        result[case]=dict(setup_by_owner=s.tolist(),min_compute_max=int(c.max(1).min()),max_compute_max=int(c.max(1).max()),
                          final_by_owner=total[-1].tolist(),sampled=sampled,
                          **{k:v.tolist() for k,v in both.items()},raw_sha256=sha(path))
    require(all(h==hashes[0] for h in hashes),"all 57,792 logits bitwise identical")
    leads={}
    for metric in ("max_owner_total","sum_phase_max"):
        a=np.array(result["csl-planned"][metric])
        tungsten={case:np.array(v[metric]) for case,v in result.items() if case.startswith("tungsten")}
        tungsten["best-tungsten-by-prefix"]=np.min(list(tungsten.values()),axis=0)
        leads[metric]={case:next((n+1 for n in range(FRAMES) if np.all(a[n:]<arr[n:])),None)
                       for case,arr in tungsten.items()}
    return dict(role=role,owner_count=4,frames=FRAMES,active_classes=7,source_ids=z["original_source"].astype(int).tolist(),
                destination_ids=z["destination_ids"].tolist(),checked_logits_per_backend=FRAMES*64*7,
                class_mismatches_vs_original=0,all_owner_words_equal=True,
                results=result,first_strict_lead_through_last_sample=leads)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("release",type=Path)
    p.add_argument("role",choices=("simulator","hardware"))
    p.add_argument("output",type=Path)
    p.add_argument("--csl-planned",required=True,type=Path)
    p.add_argument("--tungsten-planned",type=Path)
    p.add_argument("--tungsten-unrolled",type=Path)
    p.add_argument("--compile-record",type=Path)
    a=p.parse_args()
    captures={k:getattr(a,k.replace("-","_")) for k in KEYS}
    save(a.output,compare(a.release,a.role,{k:v for k,v in captures.items() if v},a.compile_record))
