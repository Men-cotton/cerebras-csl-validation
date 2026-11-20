"""Bound each destination window by fixed local row and edge capacities."""
from pathlib import Path
import hashlib,json,math
import numpy as np


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


def load(path):
    with np.load(path,allow_pickle=False) as z:g=dict(z)
    n,f=g["x"].shape
    if len(g["ptr"])!=n+1 or g["ptr"][0]!=0 or g["ptr"][-1]!=len(g["src"]) or np.any(np.diff(g["ptr"])<0):
        raise ValueError("incomplete graph traversal")
    if not(np.all(g["src"]>=0) and np.all(g["src"]<n) and g["wn"].shape==g["wr"].shape==(f,len(g["b"]))):
        raise ValueError("invalid feature or weight dimensions")
    return g


def assign(degrees,p,r,f,q):
    parts=[[] for _ in range(p)];cost=np.zeros(p,np.int64);cnt=np.zeros(p,np.int64)
    for row in np.argsort(-degrees,kind="stable"):
        valid=np.where(cnt<r)[0]
        pe=int(valid[cost[valid].argmin()])
        parts[pe].append(int(row));cnt[pe]+=1;cost[pe]+=int(degrees[row])*f+2*f*q
    return [sorted(group) for group in parts]


def windows(g,p,r):
    deg=np.diff(g["ptr"]);f=g["x"].shape[1];q=(len(g["b"])+3)//4*4
    words=(2*f+1)*q+2*r*f+r*q+4*r+64
    e=(9000-words)//(f+1)
    if e<r:raise ValueError("fixed per-PE edge bank too small for root ingress")
    work=[]
    for first in range(0,len(deg),p*r):
        last=min(len(deg),first+p*r)
        local=deg[first:last]
        groups=assign(local,p,r,f,q)
        counts=[int(local[group].sum()) if group else 0 for group in groups]
        work.append(dict(first=first,last=last,edges=sum(counts),per_owner=counts,
                         parts=max(1,math.ceil(max(counts)/e))))
    active=[w["parts"] for w in work]
    guards=sorted({0,len(work)//2,len(work)-1,int(np.argmax(active))})
    cfg=dict(P=p,R=r,F=f,Q=q,E=e,K=max(active),RELU=0)
    return cfg,work,guards


def slice_graph(data,w):
    first,last=w["first"],w["last"]
    ptr=data["ptr"][first:last+1];raw_edges=data["src"][ptr[0]:ptr[-1]]
    seeds=np.arange(first,last,dtype=np.int32)
    nodes=np.unique(np.r_[seeds,raw_edges])
    q=len(data["b"]);x=data["x"][nodes]
    z=dict(seed=seeds,node=nodes,src=np.searchsorted(nodes,raw_edges).astype(np.int32),ptr=ptr-ptr[0],
           x=x,root=np.searchsorted(nodes,seeds).astype(np.int32),wn=data["wn"],wr=data["wr"],b=data["b"],
           relu=np.int32(0),shape=np.array([len(seeds),len(nodes),len(raw_edges),x.shape[1],q]))
    values=x.astype(np.float64);summed=np.zeros((len(seeds),x.shape[1]),np.float64)
    rows=np.flatnonzero(np.diff(z["ptr"])>0)
    if len(rows):
        summed[rows]=np.add.reduceat(values[z["src"]],z["ptr"][rows],axis=0)
    z["fp64"]=values[z["root"]]@z["wr"].astype(np.float64)+(summed/np.maximum(np.diff(z["ptr"]),1)[:,None])@z["wn"].astype(np.float64)+z["b"]
    return z


def pack(g,z,cfg,w):
    p,r,e,f,q=[cfg[n] for n in ("P","R","E","F","Q")]
    c=len(g["b"]);k=w["parts"]
    groups=assign(np.diff(z["ptr"]),p,r,f,q)
    weights=np.zeros((2*f+1,q),np.float32);weights[:f,:c]=z["wr"];weights[f:2*f,:c]=z["wn"];weights[-1,:c]=z["b"]
    roots=np.zeros((p,r,f),np.float32);ids=np.full((p,r),0xffffffff,np.uint32);inv=np.zeros((p,r),np.float32)
    ptr=np.zeros((k,p,r),np.uint32);feat=np.zeros((k,p,e,f),np.float32);counts=np.zeros((k,p),np.uint32)
    for pe,group in enumerate(groups):
        dg=np.diff(z["ptr"])[group].astype(np.int64)
        ns=len(group)
        if ns:
            ids[pe,:ns]=z["seed"][group];inv[pe,:ns]=1/np.maximum(dg,1)
            roots[pe,:ns]=z["x"][z["root"][group]]
            src=np.concatenate([z["src"][z["ptr"][j]:z["ptr"][j+1]] for j in group])
            ends=dg.cumsum()
            for part in range(k):
                offset=part*e;v=src[offset:offset+e];counts[part,pe]=len(v)
                ptr[part,pe,:ns]=np.clip(ends-offset,0,e)
                if ns<r:ptr[part,pe,ns:]=len(v)
                if len(v):feat[part,pe,:len(v)]=z["x"][v]
    if counts.sum()!=w["edges"]:raise ValueError("edge coverage")
    return dict(weights=weights.ravel(),root=roots,ids=ids,inv=inv,nrows=np.array(list(map(len,groups)),np.uint32),
              counts=counts,rowend=ptr,feat=feat,groups=groups)


def packet(arr,trial,first):
    scale=np.float32(.5 if trial==2 else 1.)
    words=[np.array([trial,len(arr["counts"])],np.uint32)]
    if first:words.append(arr["weights"].view(np.uint32))
    for pe,n0 in enumerate(arr["nrows"]):
        n=int(n0)
        words.extend((np.array([n],np.uint32),arr["ids"][pe,:n],arr["inv"][pe,:n].view(np.uint32),
                      (arr["root"][pe,:n]*scale).reshape(-1).view(np.uint32)))
    for part,count in enumerate(arr["counts"]):
        for pe,n0 in enumerate(count):
            n=int(n0);nr=int(arr["nrows"][pe])
            words.extend((np.array([n,nr],np.uint32),arr["rowend"][part,pe,:nr],
                          (arr["feat"][part,pe,:n]*scale).reshape(-1).view(np.uint32)))
    return np.concatenate(words)
