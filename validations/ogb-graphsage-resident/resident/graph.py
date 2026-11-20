"""Plan exact original two-hop dependencies and build only the device's live input words."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_case(root):
    info=read(root/"case.json")
    require(sha(root/"graph.npz")==info["graph_sha256"],"frozen topology or weights")
    with np.load(root/"graph.npz",allow_pickle=False) as z:
        g=dict(z)
    x=[]
    for item in info["chunks"]:
        name=item["file"]
        require(Path(name).name==name and sha(root/name)==item["sha256"],"frozen raw features")
        with np.load(root/name,allow_pickle=False) as z:
            require(z["x"].shape==(item["rows"],info["features"]),"feature chunk shape")
            x.append(z["x"])
    g["x"]=np.concatenate(x)
    require(g["x"].dtype==np.float32 and len(g["x"])==len(g["raw_ids"])==info["raw_nodes"],"raw node coverage")
    if info["scope"]=="whole":
        for field in ("ptr","src","root"):
            g["b_"+field]=g["a_"+field]
    for layer,parent in (("a","raw_ids"),("b","a_ids")):
        ids=g[layer+"_ids"];offset=g[layer+"_ptr"];indices=g[layer+"_src"];own=g[layer+"_root"]
        require(len(offset)==len(ids)+1 and offset[0]==0 and offset[-1]==len(indices) and np.all(np.diff(offset)>=0),
                "CSR row extents")
        require(np.all(indices>=0) and np.all(indices<len(g[parent])) and np.array_equal(g[parent][own],ids),"original node mapping")
    require(all(np.all(np.diff(g[k])>0) for k in ("raw_ids","a_ids","b_ids")),"unique increasing original IDs")
    require(g["x"].shape[1]==g["wr0"].shape[0] and g["wr0"].shape==g["wn0"].shape and g["wr1"].shape==g["wn1"].shape
            and g["wr0"].shape[1]==g["wr1"].shape[0]==len(g["b0"]) and len(g["b1"])==g["wr1"].shape[1],"checkpoint dimensions")
    return info,g


def rowsources(g,prefix,rows):
    ptr=g[prefix+"_ptr"]; src=g[prefix+"_src"]
    return np.concatenate([src[ptr[i]:ptr[i+1]] for i in rows]) if len(rows) else np.empty(0,np.int32)


def upstream(g,prefix,rows):
    return np.unique(np.r_[g[prefix+"_root"][rows],rowsources(g,prefix,rows)]).astype(np.int32)


def assign(rows,degree,p,r,mode,f,q):
    if mode==0:
        return [np.asarray(a,np.int32) for a in np.array_split(rows,p)]
    groups=[[] for _ in range(p)];cost=np.zeros(p,np.int64);counts=np.zeros(p,np.int64)
    for local in np.argsort(-degree[rows],kind="stable"):
        row=int(rows[local]);eligible=np.flatnonzero(counts<r)
        pe=int(eligible[cost[eligible].argmin()])
        groups[pe].append(row);counts[pe]+=1
        cost[pe]+=int(degree[row])*f+(f if mode==1 else 2*f*q)
    return [np.array(sorted(group),np.int32) for group in groups]


def schedule(g,cfg):
    p,r,f,q,e=(cfg[k] for k in ("P","R","F","Q","E"))
    batch=p*r
    plan=[]
    for trial in (1,2,3):
        output=np.arange(len(g["b_ids"]),dtype=np.int32) if trial==1 else g["control"]
        middle=upstream(g,"b",output); source=upstream(g,"a",middle)
        if trial==1:
            require(len(source)==len(g["raw_ids"]) and len(middle)==len(g["a_ids"]),"frozen graph contains unused nodes")
        for mode,selected,prefix in ((0,source,None),(1,middle,"a"),(2,output,"b")):
            degree=np.diff(g[prefix+"_ptr"]).astype(np.int64) if prefix else None
            for start in range(0,len(selected),batch):
                rows=selected[start:start+batch]
                groups=assign(rows,degree,p,r,mode,f,q)
                counts=[int(degree[group].sum()) if prefix else 0 for group in groups]
                parts=math.ceil(max(counts)/e) if prefix else 0
                require(parts<=cfg["K"],"chip edge bank clock capacity")
                plan.append(dict(trial=trial,mode=mode,index=start//batch,groups=groups,
                                 rows=rows,edges=sum(counts),parts=parts,tag=len(plan)+1))
    return plan


def weights(g,cfg):
    f,q=cfg["F"],cfg["Q"]
    w=np.zeros((2*f+1,q),np.float32)
    c=len(g["b1"]);w[:f,:c]=g["wr1"];w[f:2*f,:c]=g["wn1"];w[-1,:c]=g["b1"]
    return np.r_[np.c_[g["wr0"],g["wn0"]].ravel(),g["b0"],w.ravel()].astype(np.float32)


def packed(g,step,cfg,previous,scale):
    p,r,d,f,q,e=(cfg[k] for k in ("P","R","D","F","Q","E"))
    mode=step["mode"];k=step["parts"]
    ids=np.full((p,r),0xffffffff,np.uint32);ns=np.array(list(map(len,step["groups"])),np.uint32)
    root=np.zeros((p,r,d if mode==0 else f),np.float32)
    inv=np.zeros((p,r),np.float32)
    ends=np.zeros((k,p,r),np.uint32);counts=np.zeros((k,p),np.uint32)
    bank=np.zeros((k,p,e,f),np.float32)
    original=g[("raw_ids","a_ids","b_ids")[mode]]
    prefix="a" if mode==1 else "b"
    for pe,rows in enumerate(step["groups"]):
        n=len(rows)
        if n==0:continue
        ids[pe,:n]=original[rows]
        if mode==0:
            root[pe,:n]=g["x"][rows]*np.float32(scale)
            continue
        ptr=g[prefix+"_ptr"]
        degree=ptr[rows+1]-ptr[rows]
        own=previous[g[prefix+"_root"][rows]]
        root[pe,:n]=own[:,:f]
        inv[pe,:n]=1./np.maximum(degree,1)
        sources=rowsources(g,prefix,rows)
        data=previous[sources, f:2*f] if mode==1 else previous[sources,:f]
        require(np.isfinite(data).all() and np.isfinite(own).all(),"nonresident predecessor value")
        edge_end=np.cumsum(degree,dtype=np.int64)
        for j in range(k):
            block=data[j*e:(j+1)*e]
            counts[j,pe]=len(block);bank[j,pe,:len(block)]=block
            ends[j,pe,:n]=np.clip(edge_end-j*e,0,e)
    require(counts.sum()==step["edges"],"edge coverage per transfer")
    return dict(n=ns,ids=ids,root=root,inv=inv,counts=counts,ends=ends,bank=bank)


def wire(batch,step,boot=None):
    data=[]
    if boot is not None:
        frames,mat=boot
        data.extend((np.array([frames],np.uint32),mat.view(np.uint32)))
    data.append(np.array([step["tag"],step["mode"],step["parts"]],np.uint32))
    for pe,count in enumerate(batch["n"]):
        n=int(count);data.extend((np.array([n],np.uint32),batch["ids"][pe,:n]))
        if step["mode"]:data.append(batch["inv"][pe,:n].view(np.uint32))
        data.append(batch["root"][pe,:n].ravel().view(np.uint32))
    for j in range(step["parts"]):
        for pe,n0 in enumerate(batch["n"]):
            count=int(batch["counts"][j,pe]);n=int(n0)
            data.extend((np.array([count,n],np.uint32),batch["ends"][j,pe,:n],
                         batch["bank"][j,pe,:count].ravel().view(np.uint32)))
    return np.concatenate(data)


def oracle(g,step,previous,scale):
    mode=step["mode"];rows=step["rows"];f=len(g["b0"])
    if mode==0:
        values=g["x"][rows].astype(np.float64)*scale
        return values@np.c_[g["wr0"],g["wn0"]].astype(np.float64)
    key="a" if mode==1 else "b"
    root=previous[g[key+"_root"][rows],:f].astype(np.float64)
    ptr=g[key+"_ptr"];src=g[key+"_src"]
    agg=np.zeros_like(root)
    for j,row in enumerate(rows):
        neighbors=src[ptr[row]:ptr[row+1]]
        if len(neighbors):
            vals=previous[neighbors,f:2*f] if mode==1 else previous[neighbors,:f]
            agg[j]=vals.mean(axis=0,dtype=np.float64)
    if mode==1:return np.maximum(root+agg+g["b0"],0)
    return root@g["wr1"].astype(np.float64)+agg@g["wn1"].astype(np.float64)+g["b1"]


def decode(raw,g,step,batch,cfg):
    p,r,f,q,k=cfg["P"],cfg["R"],cfg["F"],cfg["Q"],step["parts"]
    mode=step["mode"];fields=(2*f,f,q)[mode];w=4+(k+2)*6
    require(raw.dtype==np.uint32 and raw.shape==(p,w+r+r*fields),"device packet width")
    require(np.all(raw[:,0]==step["tag"]) and np.array_equal(raw[:,1],np.arange(p)) and
            np.array_equal(raw[:,2],batch["n"]) and np.array_equal(raw[:,3],batch["counts"].sum(0)),"device frame identity")
    clocks=raw[:,4:w].astype(np.int64).reshape(p,k+2,6)
    require(np.all(clocks<65536),"48-bit clock words")
    base=np.array([1,65536,1<<32])
    ticks=((clocks[:,:,3:]-clocks[:,:,:3])@base)%(1<<48)
    elapsed=((clocks[:,-1,3:]-clocks[:,0,:3])@base)%(1<<48)
    require(np.all(ticks>0) and np.all(ticks<1<<36) and np.all(elapsed>=ticks.sum(1)),"device clock intervals")
    count=(2*f,f,len(g["b1"]))[mode]
    y=np.empty((len(step["rows"]),count),np.float32)
    for pe,rows in enumerate(step["groups"]):
        n=len(rows)
        require(np.array_equal(raw[pe,w:w+n],batch["ids"][pe,:n]),"returned original IDs")
        decoded=raw[pe,w+r:].view(np.float32).reshape(r,fields)[:n]
        if count<fields:require(np.all(decoded[:,count:]==0),"class padding")
        y[np.searchsorted(step["rows"],rows)]=decoded[:,:count]
    return y,ticks,elapsed
