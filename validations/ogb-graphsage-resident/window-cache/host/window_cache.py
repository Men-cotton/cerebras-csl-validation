"""Count an exact, storage-bounded source table broadcast once per resident destination window."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from sparse_events import build_lib, dma, selftest
from traffic import OGB, assign, full_csr, prefixes, read, require, sha
sys.path.insert(0,str(OGB/"source-tiles"))



def choose(sources, capacity):
    """Rank reused sources by descending count and original ID for deterministic ties."""
    ids,counts=np.unique(sources,return_counts=True)
    reused=np.flatnonzero(counts>1)
    rank=np.lexsort((ids[reused],-counts[reused]))
    return ids[reused[rank[:capacity]]]


def event_packet(words, ids, hot_ids):
    """Ordered dense misses and two uint16 (position, slot) descriptors per word."""
    lookup={int(n):i for i,n in enumerate(hot_ids)}
    events=[];misses=[]
    for pos,(src,v) in enumerate(zip(ids,words)):
        slot=lookup.get(int(src))
        if slot is None:misses.extend(v)
        else:events.append((pos<<8)|slot)
    if len(events)%2:events.append(0)
    lanes=np.asarray(events,dtype="<u2").view("<u4")
    return np.concatenate((np.asarray(misses,dtype=np.uint32),lanes))


def decode(packet,n,f,hit_count,hot_words):
    nmiss=n-hit_count
    live=packet[:nmiss*f].reshape(nmiss,f)
    tail=np.ascontiguousarray(packet[nmiss*f:],dtype="<u4").view("<u2")
    require(len(tail)==2*((hit_count+1)//2),"descriptor extent")
    if hit_count%2:require(tail[-1]==0,"event padding")
    events=tail[:hit_count];pos=events>>8;slots=events&255
    require(np.all(pos<n) and np.all(slots<len(hot_words)) and np.all(np.diff(pos.astype(int))>0),"valid sorted source hits")
    result=np.empty((n,f),np.uint32)
    misses=np.ones(n,bool);misses[pos]=False
    result[misses]=live;result[pos]=hot_words[slots]
    return result


def audit_codec():
    rng=np.random.default_rng(3941)
    words=rng.integers(0,1<<32,(1300,16),dtype=np.uint32)
    for n in (1,127,128,192):
        ids=rng.integers(0,300,n,dtype=np.int32)
        for cap in (16,32,128,256):
            hot=choose(ids,cap)
            packet=event_packet(words[ids],ids,hot)
            hits=np.count_nonzero(np.isin(ids,hot))
            np.testing.assert_array_equal(decode(packet,n,16,hits,words[hot]),words[ids])
            require(len(packet)==(n-hits)*16+(hits+1)//2,"byte count")


def census(a,lib):
    from window_order import row_order
    dataset=a.dataset;source=read(a.baseline);cfg=source["params"]
    p,r,e,f,q=(cfg[k] for k in ("P","R","E","F","Q"))
    require(source["dataset"]==dataset and source["scope"]=="whole" and e<=256,"fixed resident geometry")
    model=read(OGB/"evidence"/dataset/"model.json")
    ptr,index=full_csr(a.cache/dataset,model["graph"],model["nodes"])
    degree=np.diff(ptr).astype(np.int64,copy=False)
    order=np.asarray(row_order(a.order,ptr,index,degree,1031),dtype="<i4")
    official=read(OGB/"source-tiles/evidence/census"/(dataset+".json"))
    order_sha=hashlib.sha256(order.tobytes()).hexdigest()
    require(order_sha==official["orders"][a.order]["order_sha256"],"frozen destination order")
    capacities=(16,32,64,128,256)
    ranks=np.full(len(degree),65535,np.uint16)
    owners=p*r
    costs={m:dict(baseline=[],hits={n:[] for n in capacities},metadata=0) for m in (1,2)}
    windows=0;loads={n:0 for n in capacities};calls={n:0 for n in capacities};table_hash={n:hashlib.sha256() for n in capacities}
    for begin in range(0,len(order),owners):
        selected=order[begin:begin+owners]
        arcs=np.concatenate([index[ptr[j]:ptr[j+1]] for j in selected])
        hot=choose(arcs,max(capacities))
        ranks[hot]=np.arange(len(hot),dtype=np.uint16)
        for cap in capacities:
            k=min(cap,len(hot))
            loads[cap]+=k*f;calls[cap]+=int(k!=0)
            table_hash[cap].update(np.asarray([windows,k],dtype="<i4").tobytes()+hot[:k].astype("<i4").tobytes())
        windows+=1
        for mode in (1,2):
            groups=assign(selected,degree,p,r,mode,f,q)
            sizes=tuple(map(len,groups))
            lengths=np.array([degree[g].sum() for g in groups],dtype=np.int64)
            banks=(int(lengths.max())+e-1)//e
            dense=np.clip(lengths-np.arange(banks)[:,None]*e,0,e).astype(np.uint16)
            costs[mode]["baseline"].append(dense)
            counts={cap:np.zeros_like(dense) for cap in capacities}
            for pe,group in enumerate(groups):
                if not lengths[pe]:continue
                ids=np.concatenate([index[ptr[j]:ptr[j+1]] for j in group])
                lane=ranks[ids];begins=np.arange(0,len(lane),e)
                for cap in capacities:
                    counts[cap][:len(begins),pe]=np.add.reduceat(lane<cap,begins)
            row_ctl=prefixes(sizes,1)
            costs[mode]["metadata"]+=prefixes(sizes,f)+(2+banks)*row_ctl+(1+banks)*p*5
            for cap in capacities:costs[mode]["hits"][cap].append(counts[cap])
        ranks[hot]=65535
        if windows%2000==0:print(dataset,a.order,windows,flush=True)
    out=dict(dataset=dataset,order=a.order,order_sha256=order_sha,checkpoint_sha256=model["weights_sha256"],
             original_graph=model["graph"],baseline_sha256=sha(a.baseline),params=cfg,windows=windows,
             projection_with_boot_words=source["phases"]["project"]["csl_words"],phases={})
    for mode,c in costs.items():
        dense=np.concatenate(c["baseline"])
        base=dma(lib,dense,f)
        require(int(dense.sum())==len(index),"every original arc exactly once")
        if a.order=="original":
            old=source["phases"][("aggregate_hidden","aggregate_logits")[mode-1]]
            require(base["words"]==old["csl_edge_words"] and c["metadata"]+base["words"]==old["csl_words"],"original native totals")
        summary=[]
        for cap in capacities:
            hits=np.concatenate(c["hits"][cap])
            require(np.all(hits<=dense),"valid edge count")
            packed=(dense-hits)*f+(hits+1)//2
            fixed=dma(lib,dense,f,payload=packed)
            variable=dma(lib,packed,1)
            summary.append(dict(slots=cap,hits=int(hits.sum()),event_words=int(((hits+1)//2).sum()),
                            broadcast_words=loads[cap],broadcast_calls=calls[cap],table_sha256=table_hash[cap].hexdigest(),
                            cache_bytes_per_pe=cap*f*4,logical_edge_and_table_words=int(packed.sum())+loads[cap],
                            fixed_edge_and_table_words=fixed["words"]+loads[cap],fixed_edge_and_table_calls=fixed["calls"]+calls[cap],
                            variable_edge_and_table_words=variable["words"]+loads[cap],variable_edge_and_table_calls=variable["calls"]+calls[cap]))
        out["phases"][str(mode)]=dict(original_arcs=len(index),edge_banks=len(dense),nonedge_words=c["metadata"],dense=base,points=summary)
    a.out.write_text(json.dumps(out,allow_nan=False,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("dataset",choices=("arxiv","products"))
    p.add_argument("order",choices=("original","rcm","minhash"))
    p.add_argument("--cache",type=Path,required=True);p.add_argument("--baseline",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    with tempfile.TemporaryDirectory() as work:
        lib=build_lib(Path(work));selftest(lib);audit_codec();census(args,lib)
