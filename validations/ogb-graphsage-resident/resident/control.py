"""Bind a two-hop SDK control to the larger original graph and compare its exact PE outputs."""
from pathlib import Path
import json
import numpy as np
from graph import load_case, read, require, sha


def case_subset(full, small):
    a,g=load_case(full);b,h=load_case(small)
    require(a["dataset"]==b["dataset"] and a["original_graph"]==b["original_graph"] and
            a["original_checkpoint_sha256"]==b["original_checkpoint_sha256"],"original graph/checkpoint identity")
    for key in ("wr0","wn0","b0","wr1","wn1","b1","live"):
        np.testing.assert_array_equal(g[key],h[key])
    def index(ids, subset):
        ix=np.searchsorted(ids,subset)
        require(np.all(ix<len(ids)) and np.array_equal(ids[ix],subset),"original node membership")
        return ix
    np.testing.assert_array_equal(g["x"][index(g["raw_ids"],h["raw_ids"])],h["x"])
    for prefix,source in (("a","raw_ids"),("b","a_ids")):
        rows=index(g[prefix+"_ids"],h[prefix+"_ids"])
        src_ids=g[source]; small_ids=h[source]
        for row,full_row in enumerate(rows):
            big_offset=g[prefix+"_ptr"][full_row:full_row+2]
            small_offset=h[prefix+"_ptr"][row:row+2]
            np.testing.assert_array_equal(src_ids[g[prefix+"_src"][slice(*big_offset)]],
                                          small_ids[h[prefix+"_src"][slice(*small_offset)]])
    require(np.array_equal(g["b_ids"][g["control"]],h["b_ids"]) and
            np.array_equal(h["control"],np.arange(len(h["b_ids"]))),"matching A/half-A/A target closure")


def words(path,entry,cfg):
    require(sha(path/entry["name"])==entry["raw_sha256"],"raw window identity")
    mode=entry["mode"];r=cfg["R"];fields=(2*cfg["F"],cfg["F"],cfg["Q"])[mode]
    header=4+6*(2+entry["parts"])
    with np.load(path/entry["name"],allow_pickle=False) as z:
        arr=z["reply"]
        rows=[];values=[]
        for pe,n0 in enumerate(arr[:,2]):
            n=int(n0)
            if n:
                rows.append(arr[pe,header:header+n])
                values.append(arr[pe,header+r:header+r+n*fields].reshape(n,fields))
        return np.concatenate(rows),np.concatenate(values)


def controls(actual,sample):
    left,right=read(actual/"result.json"),read(sample/"result.json")
    require(left["language"]==right["language"] and left["params"]==right["params"] and
            right["stage"]==left["stage"]=="PASS","same complete firmware geometry")
    with (actual/"windows.jsonl").open() as a,(sample/"windows.jsonl").open() as b:
        full=[json.loads(line) for line in a];sub=[json.loads(line) for line in b]
    totals={}
    for trial in (1,2,3):
        for mode in (0,1,2):
            selected=lambda e:e["trial"]==trial and e["mode"]==mode
            ref=[words(sample,e,right["params"]) for e in sub if selected(e)]
            nodes=np.concatenate([n for n,_ in ref]);values=np.concatenate([v for _,v in ref])
            order=np.argsort(nodes);nodes=nodes[order];values=values[order]
            require(np.all(np.diff(nodes.astype(np.int64))>0),"unique original control nodes")
            seen=np.zeros(len(nodes),bool)
            for e in full:
                if not selected(e):continue
                ids,vals=words(actual,e,left["params"])
                ix=np.searchsorted(nodes,ids)
                keep=(ix<len(nodes)) & (nodes[np.minimum(ix,len(nodes)-1)]==ids)
                target=ix[keep]
                require(not seen[target].any(),"repeated control row")
                np.testing.assert_array_equal(vals[keep],values[target])
                seen[target]=True
            require(seen.all(),"missing control row")
            totals[f"t{trial}-s{mode}"]={"node_ids":len(nodes),"exact_words":int(values.size)}
    return totals


def provenance(study,pilot,archive):
    """Bind the live matrices and real two-hop control to the archived full-model reference."""
    import io, hashlib
    info,full=load_case(study)
    _,small=load_case(pilot)
    model=read(archive/"model.json");cpu=read(archive/"fp64.json");compact=read(archive/"compaction.json")
    checkpoint=sha(archive/"weights.npz")
    require(checkpoint==model["weights_sha256"]==cpu["checkpoint_sha256"]==compact["weights_sha256"]==
            info["original_checkpoint_sha256"],"trained checkpoint identity")
    require(info["original_graph"]==model["graph"] and info["original_gpu_logits_sha256"]==model["gpu_logits_sha256"],
            "whole graph and prediction identity")
    live=full["live"];np.testing.assert_array_equal(live,compact["live_channels"])
    with np.load(archive/"weights.npz",allow_pickle=False) as z:
        for k,other in (("wr0",z["root0.weight"][live].T),("wn0",z["neigh0.weight"][live].T),
                        ("b0",z["neigh0.bias"][live]),("wr1",z["root1.weight"][:,live].T),
                        ("wn1",z["neigh1.weight"][:,live].T),("b1",z["neigh1.bias"])):
            np.testing.assert_array_equal(full[k],other)
    for layer,prefix,source in (("first-closure","a","raw_ids"),("final-4","b","a_ids")):
        path=archive/(layer+".npz")
        require(sha(path)==cpu["fixtures"][path.name],"original reference fixture")
        with np.load(path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z["seed"],small[prefix+"_ids"])
            np.testing.assert_array_equal(z["node"],small[source])
            for rowfield in ("ptr","src","root"):
                np.testing.assert_array_equal(z[rowfield],small[prefix+"_"+rowfield])
            if prefix=="a":np.testing.assert_array_equal(z["x"],small["x"])
            else:
                np.testing.assert_array_equal(z["gpu"].argmax(1),small["gpu_prediction"])
                np.testing.assert_array_equal(z["labels"],small["labels"])
    if info["scope"]=="whole":
        for key,field in (("a_ptr","ptr_sha256"),("a_src","index_sha256")):
            output=io.BytesIO()
            np.save(output,full[key],allow_pickle=False)
            require(hashlib.sha256(output.getbuffer()).hexdigest()==model["graph"][field],"complete original CSR order")
        for k,name in enumerate(("train","valid","test")):
            take=full["split"]==k
            require(int(take.sum())==cpu["splits"][name]["count"] and
                    int(np.sum(full["gpu_prediction"][take]==full["labels"][take]))==cpu["splits"][name]["correct"],
                    "complete original dataset metrics")
