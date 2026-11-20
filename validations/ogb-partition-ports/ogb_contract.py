"""Pack original OGB neighbourhoods and decode all PE clocks and model outputs."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2,allow_nan=False)+"\n")


def load(path):
    with np.load(path,allow_pickle=False) as data:
        z=dict(data)
    n, nodes, edges, f, c = map(int,z["shape"])
    assert z["x"].shape==(nodes,f) and z["wn"].shape==z["wr"].shape==(f,c)
    assert np.array_equal(z["node"][z["root"]],z["seed"])
    assert np.all(np.diff(z["ptr"])>=0) and int(z["ptr"][-1])==len(z["src"])==edges
    assert len(set(z["seed"]))==n and np.all((z["src"]>=0)&(z["src"]<nodes))
    for r in range(n):
        ns=z["node"][z["src"][z["ptr"][r]:z["ptr"][r+1]]]
        if np.any(ns[1:]<=ns[:-1]):
            raise ValueError("unexpected OGB adjacency order or duplicates")
    return z


def place(z,p,policy):
    n,_,_,f,c=map(int,z["shape"])
    degree=np.diff(z["ptr"])
    if policy=="contiguous":
        return [x.tolist() for x in np.array_split(np.arange(n),p)]
    if policy=="cyclic":
        return [list(range(pe,n,p)) for pe in range(p)]
    if policy!="balanced":
        raise ValueError(policy)
    # Measured comparison changes only row assignment. Include both the
    # edge-vector adds and two dense feature-by-class transforms in the cost.
    groups=[[] for _ in range(p)]
    cost=np.zeros(p,np.int64)
    for r in np.argsort(-degree,kind="stable"):
        dest=int(cost.argmin())
        groups[dest].append(int(r))
        cost[dest]+= int(degree[r])*f + 2*f*c
    return [sorted(rows) for rows in groups]


def pack(z,p,policy,edge_capacity=None,shards=1):
    n,_,_,f,c=map(int,z["shape"])
    q=(c+3)//4*4
    if shards<1 or p%shards or (shards>1 and f>q):raise ValueError("unsupported source-shard geometry")
    destinations=place(z,p//shards,policy)
    groups=[destinations[x//shards] for x in range(p)]
    r=max(map(len,groups))
    # Aim below 36 KiB of static data, preserving a margin for the program.
    weights=2*f*q+q
    static=weights+2*r*f+r*q+4*r+64
    max_e=(9000-static)//(f+1)
    if max_e<1: raise ValueError(f"per-owner dense state alone exceeds SRAM: {static}")
    edges=[]
    for x,group in enumerate(groups):
        rank=x%shards
        local=[]
        for slot,row in enumerate(group):
            b,e=map(int,z["ptr"][row:row+2])
            degree=e-b
            local.extend((slot,int(src)) for src in z["src"][b+degree*rank//shards:b+degree*(rank+1)//shards])
        edges.append(local)
    ec=min(max_e,max(1,max(map(len,edges))))
    if edge_capacity is not None: ec=min(ec,edge_capacity)
    k=max(1,max(math.ceil(len(ed)/ec) for ed in edges))
    params={"P":p,"R":r,"F":f,"Q":q,"E":ec,"K":k,"RELU":int(z["relu"])}
    if shards>1:params["S"]=shards
    matrices=np.zeros((2*f+1,q),np.float32)
    matrices[:f,:c]=z["wr"]
    matrices[f:2*f,:c]=z["wn"]
    matrices[-1,:c]=z["b"]
    root=np.zeros((p,r,f),np.float32)
    ids=np.full((p,r),0xffffffff,np.uint32)
    inv=np.zeros((p,r),np.float32)
    nrows=np.array(list(map(len,groups)),np.uint32)
    counts=np.zeros((k,p),np.uint32)
    rowid=np.zeros((k,p,ec),np.uint32)
    feat=np.zeros((k,p,ec,f),np.float32)
    for pe,(group,ed) in enumerate(zip(groups,edges)):
        if group:
            ids[pe,:len(group)]=z["seed"][group]
            root[pe,:len(group)]=z["x"][z["root"][group]]
            inv[pe,:len(group)]=1./np.maximum(np.diff(z["ptr"])[group],1)
        for j in range(k):
            part=ed[j*ec:(j+1)*ec]
            counts[j,pe]=len(part)
            if part:
                owners,src=zip(*part)
                rowid[j,pe,:len(part)]=owners
                feat[j,pe,:len(part)]=z["x"][list(src)]
    assert sum(map(len,edges))==z["shape"][2]
    return params,dict(weights=matrices.ravel(),root=root,ids=ids,inv=inv,nrows=nrows,counts=counts,
                            rowid=rowid,feat=feat,groups=groups)


def payload(a,trial,scale=None):
    scale=np.float32(0.5 if trial==2 else 1.) if scale is None else np.float32(scale)
    words=[np.array([trial],np.uint32)]
    if trial==1: words.append(a["weights"].view(np.uint32))
    for i,nrows in enumerate(a["nrows"]):
        n=int(nrows)
        words.extend([np.array([n],np.uint32),a["ids"][i,:n],a["inv"][i,:n].view(np.uint32),
                    (a["root"][i,:n]*scale).reshape(-1).view(np.uint32)])
    for j,counts in enumerate(a["counts"]):
        for i,nn in enumerate(counts):
            n=int(nn)
            words.extend([np.array([n],np.uint32),a["rowid"][j,i,:n],
                       (a["feat"][j,i,:n]*scale).reshape(-1).view(np.uint32)])
    return np.concatenate(words)


def oracle(z,scale):
    mat=z["x"].astype(np.float64)*scale
    xroot=mat[z["root"]]
    agg=np.zeros_like(xroot)
    for r in range(len(xroot)):
        sub=z["src"][z["ptr"][r]:z["ptr"][r+1]]
        if len(sub): agg[r]=mat[sub].mean(axis=0)
    y=xroot@z["wr"].astype(np.float64)+agg@z["wn"].astype(np.float64)+z["b"].astype(np.float64)
    return np.maximum(y,0) if z["relu"] else y


def check_frame(reply,z,batch,params,trial,input_scale=None):
    p,r,f,q,e,k=[params[key] for key in ("P","R","F","Q","E","K")]
    n,_,_,_,c=map(int,z["shape"])
    shards=params.get("S",1)
    d=6*(2+k+(shards>1))
    a=reply.reshape(p,4+d+r+r*q)
    assert np.all(a[:,0]==trial) and np.array_equal(a[:,1],np.arange(p))
    assert np.array_equal(a[:,2],batch["nrows"]) and np.array_equal(a[:,3],batch["counts"].sum(axis=0))
    # Check only live entries. The padding never contributes to the workload.
    logits=np.empty((n,c),np.float32)
    for pe,group in enumerate(batch["groups"]):
        assert np.array_equal(a[pe,4+d:4+d+len(group)],z["seed"][group])
        all_out=a[pe,4+d+r:].view(np.float32).reshape(r,q)
        if group:
            if pe%shards==shards-1:
                logits[group]=all_out[:len(group),:c]
                assert np.all(all_out[:len(group),c:]==0)
            else:
                scale=(.5 if trial==2 else 1.) if input_scale is None else input_scale
                for slot,row in enumerate(group):
                    first,end=map(int,z["ptr"][row:row+2]);degree=end-first
                    rank=pe%shards
                    src=z["src"][first+degree*rank//shards:first+degree*(rank+1)//shards]
                    expected=z["x"][src].astype(np.float64).sum(axis=0)*scale
                    if not np.allclose(all_out[slot,:f],expected,rtol=1e-4,atol=1e-3):
                        raise ValueError("source-shard partial differs from its exact original edge interval")
                    assert np.all(all_out[slot,f:]==0)
    clocks=a[:,4:4+d].astype(np.int64).reshape(p,d//6,6)
    assert np.all(clocks<65536)
    ticks=(clocks@np.array([-1,-65536,-(1<<32),1,65536,1<<32]))%(1<<48)
    assert np.all(ticks>0) and np.all(ticks<1<<36)
    truth=oracle(z,(.5 if trial==2 else 1.) if input_scale is None else input_scale)
    diff=logits.astype(np.float64)-truth
    rel=float(np.linalg.norm(diff)/(np.linalg.norm(truth)+1e-30))
    if not np.allclose(logits,truth,rtol=3e-4,atol=2e-4) or rel>3e-5:
        raise ValueError(("device layer vs independent FP64",float(np.abs(diff).max()),rel))
    return logits,ticks,dict(max_abs=float(np.abs(diff).max()),relative_l2=rel,
           argmax_mismatch=int(np.count_nonzero(logits.argmax(1)!=truth.argmax(1))),
           whole_checkpoint_max_abs=float(np.max(np.abs(logits-z["fp64"]))) if trial!=2 else None)
