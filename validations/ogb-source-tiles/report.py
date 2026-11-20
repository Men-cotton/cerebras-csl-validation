"""Replay the original-ID graph, wire input sizes, all PE replies and A/B clocks."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from source_table import TRIALS, HERE, assign, audit, device, source_tiles, upload


def storage(build, meta):
    if "csl_params" in meta:
        sys.path.insert(0, str(HERE.parent/"device"))
        from storage import csl_elf
        return max(csl_elf(p)["sram_end"] for p in (build/"out/bin").glob("out_*_0.elf"))
    sys.path.insert(0, str(HERE.parents[1]/"gcn-e236/tools"))
    from metrics import map_resources
    names = {"weight","roots","sum","table","out","inverse","ids","clocks"}
    return map_resources(build/"program.map", names)["mapped_static_endpoint_bytes"]


def replay(fixture, build, output, reference=None, inspect_sram=True):
    meta = json.loads((build/"build.json").read_text())
    result = json.loads((output/"result.json").read_text())
    role = result["role"]
    native = "csl_params" in meta
    if not (result["stopped"] and result["build_sha256"] == device.sha(build/"build.json")
            and result["fixture_sha256"] == meta["fixture_sha256"] == device.sha(fixture)):
        raise ValueError("incomplete or mismatched capture")
    for name,h in meta["source_sha256"].items():
        if device.sha(build/"src"/name)!=h:
            raise ValueError("changed source")
    if native:
        if role!=meta["role"] or meta["elf_sha256"]!={p.name:device.sha(p) for p in (build/"out/bin").glob("*.elf")}:
            raise ValueError("wrong native artifact")
    elif (meta["roles"][role]["elf_sha256"] != device.sha(build/role/"out/program.elf")
          or meta["port_sha256"] != device.sha(build/role/"out/out_port_map.json")):
        raise ValueError("wrong paired ELF or port mapping")
    z = device.load(fixture)
    cfg,g,owner,slot = assign(z,meta["params"]["P"],meta["params"]["E"])
    if result["params"]!=cfg or meta["params"]!=cfg or len(result["frames"]) not in (2,6):
        raise ValueError("geometry or frame count differs")
    schedules={m:source_tiles(z,cfg,owner,slot,m) for m,_ in TRIALS[:2]}
    previous = {}
    frames = []
    for i,frame in enumerate(result["frames"]):
        method,scale=TRIALS[i]
        path=output/f"frame-{i+1}.npz"
        if frame["raw_sha256"]!=device.sha(path) or (frame["trial"],frame["method"],frame["scale"])!=(i+1,method,scale):
            raise ValueError("different raw frame or ordering")
        with np.load(path,allow_pickle=False) as raw:
            logits, measured=audit(raw["reply"],z,cfg,g,i)
            np.testing.assert_array_equal(raw["logits"].view(np.uint32),logits.view(np.uint32))
            np.testing.assert_array_equal(raw["node"],z["seed"])
            words=logits.view(np.uint32)
            if scale in previous:
                np.testing.assert_array_equal(words,previous[scale])
            else:
                previous[scale]=words
            if reference is not None:
                np.testing.assert_array_equal(words,reference[scale])
            if native:
                np.testing.assert_array_equal(frame["peak_active_edge_cycles"],raw["active_edge_cycles"].max())
        for key,v in measured.items():
            # FP64 BLAS reductions differ by roundoff across the SDK and host.
            # Raw logits, integer counters and timestamp reconstruction stay exact.
            if key.startswith("fp64_") and isinstance(v,float):
                np.testing.assert_allclose(frame[key],v,rtol=1e-8,atol=1e-13)
            elif frame[key]!=v:
                raise ValueError("incorrect reported "+key)
        tiles=schedules[method]
        count=sum(len(src) for src,_ in tiles)
        pe,rows,f,q=(cfg[k] for k in ("P","R","F","Q"))
        if native:
            factor=1 if result["distribution"]=="row" else pe
            weights=factor*((2*f+1)*q) if i==0 else 0
            controls=pe*(3+rows*(2+f)+3*len(tiles))
            selectors=pe*sum(max(map(len,refs)) for _,refs in tiles)
            source_words=factor*count*f
            expected=weights+controls+selectors+source_words
        else:
            source_words=count*f
            expected=len(upload(z,cfg,g,tiles,i))
        if frame["input_words"]!=expected or frame["source_words"]!=source_words or frame["blocks"]!=len(tiles):
            raise ValueError("reported transfer count disagrees with the original IDs")
        frames.append(dict(trial=i+1,method=method,scale=scale,total_words=expected,
                           source_vectors=count,wire_source_words=source_words,blocks=len(tiles),
                           peak_interval=frame["peak_first_to_last"],peak_edge_interval=frame["peak_edge_interval"],
                           peak_active_edge_cycles=frame.get("peak_active_edge_cycles"),
                           fp64_max_abs=measured["fp64_max_abs"],fp64_argmax_disagreements=measured["argmax_differences"],
                           logits_sha256=sha256(logits.tobytes()).hexdigest(),raw_sha256=device.sha(path)))
    summary = dict(role=role,language="CSL" if native else "Tungsten",delivery=result.get("distribution","Direct Link"),
                params=cfg,build_sha256=device.sha(build/"build.json"),fixture_sha256=device.sha(fixture),
                nodes=len(z["seed"]),arcs=len(z["src"]),
                original_source_vertices=int(len(np.unique(z["src"]))),frames=frames)
    if inspect_sram:
        summary["mapped_endpoint_bytes"] = storage(build,meta)
    return summary, previous


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("fixture",type=Path)
    p.add_argument("build",type=Path)
    p.add_argument("capture",type=Path)
    p.add_argument("--peer-build",type=Path)
    p.add_argument("--peer-capture",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    if (a.peer_build is None)!=(a.peer_capture is None):
        p.error("--peer-build and --peer-capture must be supplied together")
    report,ref=replay(a.fixture,a.build,a.capture)
    if a.peer_build:
        peer,other=replay(a.fixture,a.peer_build,a.peer_capture,ref)
        report["bit_exact_peer"]=peer
    data=json.dumps(report,indent=2,allow_nan=False)+"\n"
    if a.output:a.output.write_text(data)
    else:print(data,end="")
