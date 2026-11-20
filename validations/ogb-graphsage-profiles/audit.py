"""Reconstruct OGB outputs and clock intervals directly from the archived PE words."""
import json
from pathlib import Path
import numpy as np
from contract import sha, load, pack, check_frame


def read(path):
    return json.loads(Path(path).read_text())


def require(condition,message):
    if not condition: raise ValueError(message)


def capture(path,fixture,descriptor,role,override=None,attested=False):
    path=Path(path)
    m=read(descriptor)
    r=read(path/"result.json")
    require(r["stage"]=="PASS" and r["stopped"] and r["role"]==role and r["language"]==m["language"],
            "incomplete capture or incorrect target")
    require(sha(descriptor)==r["build_sha256"] and sha(fixture)==m["fixture_sha256"]==r["fixture_sha256"],
            "actual build and input binding")
    require(sha(path/"raw.npz")==r["raw_sha256"] and len(r["frames"])==3,"raw capture identity")
    z=load(fixture);cfg=m["params"];p=cfg["P"];k=cfg["K"];s=cfg.get("S",1)
    sources={}
    if override:
        require(sha(override)==r["override_x_sha256"],"previous device layer identity")
        with np.load(override,allow_pickle=False) as o:
            np.testing.assert_array_equal(o["node"],z["node"])
            sources={t:o["logits"+str(t)] for t in (1,2,3)}
    else: require(not r["override_x_sha256"],"missing preceding layer")
    if attested:
        require(r["platform"]=={"system":role=="hardware","simulation":role=="simulator"},"actual SDK target")
        image=m["roles"][role]["elf"] if m["language"]=="tungsten" else m["elf"]
        require(r["executed_elf_sha256"]==image,"actual loaded images")
        require(r["host_sha256"]=={n:sha(Path(__file__).parent/n) for n in ("worker.py","contract.py")},
                "actual runtime source identity")
    with np.load(path/"raw.npz",allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw["node"],z["seed"])
        for trial,frame in enumerate(r["frames"],1):
            if sources:z["x"]=sources[trial]
            cfg2,b=pack(z,p,m["placement"],m["edge_capacity"],s)
            require(cfg2==cfg and cfg==r["params"],"physical shape and complete occurrence count")
            y,ticks,errs=check_frame(raw["reply"+str(trial)],z,b,cfg,trial,1. if sources else None)
            np.testing.assert_array_equal(y.view(np.uint32),raw["logits"+str(trial)].view(np.uint32))
            np.testing.assert_array_equal(ticks,raw["cycles"+str(trial)])
            for n,v in (("peak_sum_pe_cycles",ticks.sum(1).max()),("sum_phase_max_cycles",ticks.max(0).sum()),
                       ("max_aggregation_cycles",ticks[:,1:k+1].sum(1).max()),("max_projection_cycles",ticks[:,-1].max())):
                require(int(v)==frame[n],"raw clock replay "+n)
            for n,v in errs.items():
                if v is not None:
                    # FP64 dot products from different host BLAS libraries can
                    # differ at ~1e-15. PE words and every clock above remain exact.
                    np.testing.assert_allclose(v,frame["numerics"][n],rtol=1e-9,atol=2e-13)
        np.testing.assert_array_equal(raw["logits1"].view(np.uint32),raw["logits3"].view(np.uint32))
    return r


def compare(left,right):
    with np.load(Path(left)/"raw.npz",allow_pickle=False) as x, np.load(Path(right)/"raw.npz",allow_pickle=False) as y:
        np.testing.assert_array_equal(x["node"],y["node"])
        values=[]
        for i in (1,2,3):
            a,b=x["logits"+str(i)],y["logits"+str(i)]
            require(np.allclose(a,b,rtol=4e-4,atol=3e-4),"matched CSL/Tungsten output")
            values.append(dict(max_abs=float(np.max(np.abs(a-b))),different_words=int((a.view(np.uint32)!=b.view(np.uint32)).sum()),
                     argmax_mismatches=int(np.count_nonzero(a.argmax(1)!=b.argmax(1)))))
        return values
