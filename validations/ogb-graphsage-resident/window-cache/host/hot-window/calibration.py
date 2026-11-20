"""Make a short same-image check using two complete hidden rows with observed overlapping sources."""
import argparse
import json
from pathlib import Path

import numpy as np

from worker import inputs, sha


def pair(original, out):
    info,z=inputs(original)
    p,r,f=z["root"].shape
    e=z["bank"].shape[2]
    all_rows=[]
    ends=z["ends"].astype(np.int64).sum(axis=0)
    for pe in range(p):
        ids=np.concatenate([a[:c] for a,c in zip(z["sources"][:,pe],z["counts"][:,pe])])
        values=np.concatenate([a[:c] for a,c in zip(z["bank"][:,pe],z["counts"][:,pe])])
        start=0
        for row,node in enumerate(z["ids"][pe,:z["n"][pe]]):
            last=ends[pe,row]
            if 8<=last-start<=50:
                all_rows.append((int(node),pe,row,ids[start:last],values[start:last]))
            start=last
    candidates=[]
    for k,record in enumerate(all_rows):
        for previous in all_rows[:k]:
            if record[1]==previous[1]:continue
            shared=len(set(record[3])&set(previous[3]));size=len(record[3])+len(previous[3])
            if shared:
                candidates.append((shared/size,shared,-size,-record[0],-previous[0],record,previous))
    best=max(candidates,key=lambda item:item[:5])
    rows=sorted(best[-2:],key=lambda record:record[0])
    chosen=np.array([a[0] for a in rows],dtype=z["hidden_nodes"].dtype)
    select=np.searchsorted(z["hidden_nodes"],chosen)
    ids=np.full_like(z["ids"],0xffffffff);n=np.zeros_like(z["n"])
    roots=np.zeros_like(z["root"]);inv=np.zeros_like(z["inv"])
    counts=np.zeros((1,p),z["counts"].dtype);stops=np.zeros((1,p,r),z["ends"].dtype)
    source=np.full((1,p,e),-1,z["sources"].dtype)
    bank=np.zeros((1,p,e,f),z["bank"].dtype)
    for node,pe,old,neighbors,v in rows:
        n[pe]=1;ids[pe,0]=node;roots[pe,0]=z["root"][pe,old];inv[pe,0]=z["inv"][pe,old]
        counts[0,pe]=len(neighbors);stops[0,pe,0]=len(neighbors)
        source[0,pe,:len(neighbors)]=neighbors;bank[0,pe,:len(neighbors)]=v
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/"fixture.npz",n=n,ids=ids,root=roots,inv=inv,counts=counts,ends=stops,sources=source,bank=bank,
                        original_nodes=z["original_nodes"],feature_table=z["feature_table"],weights=z["weights"],hidden_nodes=chosen,
                        measured_hidden=z["measured_hidden"][select],fp64_a=z["fp64_a"][select],fp64_half=z["fp64_half"][select])
    rec=dict(info,original_arcs=int(counts.sum()),original_hidden_words=2*f,parts=1,files={"fixture.npz":sha(out/"fixture.npz")},
             source_fixture_sha256=sha(original/"fixture.json"),selected_actual_hidden_rows=chosen.tolist(),owners=[a[1] for a in rows],
             selection="highest overlap fraction for two distinct original PE owners, complete 8..50-neighbor rows, deterministic original-ID ties",
             distinct_shared_sources=best[1])
    (out/"fixture.json").write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__);a.add_argument("original",type=Path);a.add_argument("out",type=Path)
    args=a.parse_args();pair(args.original,args.out)
