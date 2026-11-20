"""Prove each dynamically counted Tungsten socket loop fits the signed 16-bit PE counter."""
import numpy as np
from graph import load_case,require,schedule,sha


def limits(case,cfg,nested=False):
    _,g=load_case(case)
    plan=schedule(g,cfg)
    p,r,f,q,d,e=(cfg[k] for k in ("P","R","F","Q","D","E"))
    degree=[None,np.diff(g["a_ptr"]).astype(np.int64),np.diff(g["b_ptr"]).astype(np.int64)]
    maximum={"forwarded_words":-1}
    max_local=0;max_parts=0
    for step in plan:
        mode=step["mode"];parts=step["parts"]
        fields=(2*f,f,q)[mode]
        owner=4+6*(2+parts)+r*(1+fields)
        count=owner*(p-1)
        if mode:
            max_local=max(max_local,max(int(degree[mode][rows].sum()) for rows in step["groups"]))
        if count>maximum["forwarded_words"]:
            maximum=dict(forwarded_words=count,owner_words=owner,tag=step["tag"],
                         trial=step["trial"],mode=mode,parts=parts,signed_16_bit_limit=32767)
        max_parts=max(max_parts,parts)
        loop_bound=max(p-1,owner) if nested else max(count,owner)
        require(max(loop_bound,step["tag"],max_local)<=32767,"16-bit collector, frame or owner edge count")
    require(max(len(plan),r*d,e*f,r*q,d*2*f+f+(2*f+1)*q)<=32767,"16-bit frame or payload counter")
    return dict(case_sha256=sha(case/"case.json"),frames=len(plan),maximum_collector=maximum,nested_gather=nested,
                max_owner_edges=max_local,max_edge_parts=max_parts,params=cfg)
