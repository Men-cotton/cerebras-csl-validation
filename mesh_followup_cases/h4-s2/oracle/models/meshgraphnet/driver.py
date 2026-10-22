"""Reproducible direct SDK experiments. Run commands through run_with_budget.py."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import data
import mapping
import reference
import resident_validation
import selective_schedule
import selective_validation
import colored_schedule
import colored_validation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("elf_audit", HERE.parent/"hierarchical_gnn/grouped_csr/driver.py")
elf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(elf)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def sources():
    files = [HERE/n for n in ("driver.py","mapping.py","data.py","reference.py","resident_validation.py","selective_schedule.py","selective_validation.py","colored_schedule.py","colored_validation.py","sdk_runner.py","device/pe.csl","device/layout.csl","device/no_halo.csl")]
    files += [ROOT/n for n in ("cslc","cs_python","sdk_debug_shell","sdk-2.10.2/cslc","sdk-2.10.2/cs_python","sdk-2.10.2/sdk_debug_shell","sdk-2.10.2/sha256sum.txt")]
    files += [Path(elf.__file__), ROOT/"kernels/communication/sparse-halo/pe.csl"]
    return {str(f.relative_to(ROOT)):sha(f) for f in files}


def schemas(p):
    l,e,h,w = (p[k] for k in ("L","E","H","W"))
    ins = dict(counts=(2,"uint32"),velocity=(l*2,"float32"),node_type=(l,"uint32"),
               edge_features=(e*3,"float32"),src_owner=(e,"uint32"),src_slot=(e,"uint32"),
               src_begin=(p["C"]*p["R"]+1,"uint32"),
               dst_slot=(e,"uint32"),weights=(w,"float32"))
    outs = dict(h=(l*h,"float32"),e=(e*h,"float32"),cached_e0=(e*h,"float32"),
                sender_cache=(e*h,"float32"),
                aggregate=(l*h,"float32"),panel=(l*h,"float32"),stats=(16,"uint32"),clocks=(6,"uint16"),
                lifecycle=(4,"uint32"))
    if p.get("MODE",0) > 0:
        ins.update(tx_begin=(p["C"]*p["R"]+1,"uint32"),tx_slots=(p["T"],"uint32"),
                   rx_count=(p["C"]*p["R"],"uint32"),rx_slot=(e,"uint32"))
        outs["comm_stats"]=(12,"uint32")
    if p.get("MODE",0) == 2:
        ins["round_sources"]=(p["Q"],"uint32")
    return ins,outs


def prepare(a):
    case = a.case.resolve()
    if case.exists() or ROOT/"research/runs" not in case.parents:
        raise ValueError("fresh case under research/runs required")
    if not 2 <= a.hidden <= 32 or min(a.columns,a.rows) < 2 or min(a.blocks,a.steps) < 1:
        raise ValueError("replicated baseline requires H2..32, C/R>=2, positive blocks/steps")
    owners = a.columns*a.rows
    if owners > 32767 or 8*owners*(owners+1) > 64*1024**2:
        raise ValueError("owner directories exceed 64 MiB preparation budget")
    if a.transport == "colored" and owners > colored_schedule.MAX_OWNERS:
        raise ValueError("colored preparation currently bounds ownership to 128 PEs")
    if a.transport != "broadcast" and not (2 <= a.hidden <= 27 and (a.batch is None or 1 <= a.batch <= 27//a.hidden)):
        raise ValueError("selective requires H2..27 and 1<=batch<=floor(27/H)")
    if a.transport == "broadcast" and a.batch is not None:
        raise ValueError("--batch requires selective or colored transport")
    chunks = [a.steps] if a.chunks is None else [int(value) for value in a.chunks.split(",")]
    if not chunks or any(value <= 0 for value in chunks) or sum(chunks) != a.steps:
        raise ValueError("positive chunk sizes must sum to --steps")
    graph = data.fixture() if a.graph is None else dict(np.load(a.graph, allow_pickle=False))
    if a.adversarial:
        if a.graph is not None:
            raise ValueError("adversarial fixture cannot replace known graph silently")
        graph = selective_schedule.adversarial_graph()
    if a.crossing:
        if a.graph is not None or a.adversarial or a.limit is not None or (a.columns,a.rows,a.partition) != (4,2,"id"):
            raise ValueError("crossing fixture requires 4x2 ID placement, no graph/limit/adversarial override")
        graph = colored_schedule.crossing_graph()
    original_counts=dict(nodes=len(graph["mesh_pos"]),edges=len(graph["senders"]))
    provenance=None
    if a.graph is not None:
        sidecar=a.graph.with_suffix(".json")
        provenance=json.loads(sidecar.read_text())
        if provenance["npz_sha256"]!=sha(a.graph) or not provenance["framing_crc32c_verified"]:
            raise ValueError("data provenance mismatch")
    graph = mapping.induced(graph, a.limit)
    weights = reference.make_weights(a.hidden, a.blocks)
    if a.edge_encoder_probe != "normal":
        # Numerical stress only: isolate LayerNorm at zero/epsilon-scale variance.
        fields = reference.unpack_weights(weights, a.hidden, a.blocks)["edge_encoder"]
        fields["w3"][:] = 0
        fields["b3"][:] = 1
        if a.edge_encoder_probe == "near-constant":
            fields["b3"][0::2] -= np.float32(2**-9)
            fields["b3"][1::2] += np.float32(2**-9)
    arrays, ni, ei, census = mapping.pack(graph, a.columns, a.rows, a.hidden, weights, a.partition)
    p = dict(C=a.columns,R=a.rows,H=a.hidden,S=a.blocks,STEPS=a.steps,NORM=int(a.norm=="invsqrt"),
             L=ni.shape[1],E=ei.shape[1],W=len(weights),MODE={"broadcast":0,"selective":1,"colored":2}[a.transport],T=1,K=1,Q=a.columns*a.rows)
    transport_census=None
    if p["MODE"] > 0:
        batch = a.batch if a.batch is not None else 27//a.hidden
        schedule,transport_census=selective_schedule.build(graph,ni,ei,a.columns,a.hidden,batch)
        census["selective_scope"] = "analytical XY payload baseline; implemented selective packet counts are in transport_census; actual router order unmeasured"
        arrays.update(schedule)
        p.update(T=transport_census["T"],K=batch)
        selective_validation.validate_schedule(graph,dict(node_ids=ni,edge_ids=ei),arrays,p)
        if p["MODE"] == 2:
            round_schedule,color_census=colored_schedule.build(arrays,a.columns)
            arrays.update(round_schedule)
            p["Q"]=arrays["round_sources"].shape[1]
            colored_validation.validate_rounds(arrays,p)
            transport_census["coloring"]=color_census
    if max(p.values()) > 32767 or max(p["L"]*p["H"],p["E"]*p["H"],p["E"]*3) > 32767:
        raise ValueError("ABI bound")
    if np.any(expected_macs(p, arrays["counts"]) >= 2**32):
        raise ValueError("MAC counter bound")
    case.mkdir(parents=True)
    (case/"source").mkdir()
    for name in ("pe.csl","layout.csl","no_halo.csl"):
        shutil.copyfile(HERE/"device"/name, case/"source"/name)
    shutil.copyfile(ROOT/"kernels/communication/sparse-halo/pe.csl", case/"source/sparse_halo.csl")
    shutil.copyfile(HERE/"sdk_runner.py", case/"sdk_runner.py")
    np.savez(case/"inputs.npz", **arrays)
    np.savez(case/"original.npz", **graph)
    np.savez(case/"mapping.npz", node_ids=ni,edge_ids=ei)
    ins,outs = schemas(p)
    files = ("inputs.npz","original.npz","mapping.npz","source/pe.csl","source/layout.csl","source/sparse_halo.csl","source/no_halo.csl","sdk_runner.py")
    m = dict(experiment="E189",transport=a.transport,transport_census=transport_census,adversarial=a.adversarial,crossing=a.crossing,params=p,inputs=ins,outputs=outs,census=census,partition=a.partition,
             residency=dict(schema=1,chunks=chunks),
             edge_encoder_probe=a.edge_encoder_probe,
             sources=sources(),files={f:sha(case/f) for f in files},
             model="deterministic untrained public CFD operator probe; identity input norms; output std .01",
             graph_source=None if a.graph is None else dict(path=str(a.graph.resolve()),sha256=sha(a.graph)),
             graph_limit=a.limit,original_graph_counts=original_counts,data_provenance=provenance,
             graph_scope="full" if len(graph["mesh_pos"])==original_counts["nodes"] else "prefix_induced")
    save(case/"case.json",m)
    return m


def load(case):
    m=json.loads((case/"case.json").read_text())
    if m["params"].get("MODE") not in (0,1,2) or m.get("transport") != {0:"broadcast",1:"selective",2:"colored"}[m["params"]["MODE"]]:
        raise ValueError("transport identity differs")
    if m["sources"] != sources():
        raise ValueError("source differs from prepared case; prepare a new case")
    if any(sha(case/f)!=s for f,s in m["files"].items()):
        raise ValueError("prepared file changed")
    return m


def expected_macs(p,counts):
    n,e=counts[:,0].astype(np.uint64),counts[:,1].astype(np.uint64)
    h,s,t=p["H"],p["S"],p["STEPS"]
    return e*(3*h+2*h*h)+n*t*(4*h*h+13*h)+s*t*(5*e+4*n)*h*h


def check_abi(case,m):
    p=json.loads((case/"build/out.json").read_text())["params"]
    if {k:int(v) for k,v in p.items()} != m["params"]:
        raise ValueError("compiled params differ")
    actual=json.loads((case/"build/bin/out_rpc.json").read_text())["rpc_symbols"]
    expected={n:("Var",{"float32":"[*]f32","uint32":"[*]u32","uint16":"[*]u16"}[t])
              for n,(_,t) in dict(m["inputs"],**m["outputs"]).items()}
    expected["run"]=("Func","void")
    if len(actual)!=len(expected) or {s["name"]:(s["kind"],s["type"]) for s in actual} != expected:
        raise ValueError("compiled symbol ABI differs")
    entry=next(symbol for symbol in actual if symbol["name"]=="run")
    if entry.get("inputs") != [{"name":"steps","type":"u16"}]:
        raise ValueError("compiled run argument must be a single u16 step count")


def compile_case(case):
    m=load(case)
    if (case/"build").exists():
        raise ValueError("fresh build required")
    p=m["params"]
    command=[str(ROOT/"cslc"),"--arch=wse3",str(case/"source/layout.csl"),
             "--fabric-dims=%d,%d"%(p["C"]+7,p["R"]+2),"--fabric-offsets=4,1","--memcpy","--channels=1",
             "--max-parallelism=1","--params="+",".join("%s:%d"%(k,v) for k,v in p.items()),"-o",str(case/"build")]
    start=time.monotonic()
    result=dict(status="FAILED",command=command,cwd=str(case),case_sha256=sha(case/"case.json"))
    try:
        ret=subprocess.run(command,cwd=case,env=dict(os.environ,CSL_IMPORT_PATH=str(case/"source")),check=False)
        result["returncode"]=ret.returncode
        if ret.returncode:
            raise RuntimeError("compiler failed")
        check_abi(case,m)
        result.update(elf.build_inventory(case))
        result["status"]="PASS"
    except Exception as error:
        result["error"]=str(error)
        raise
    finally:
        result["wall_seconds"]=time.monotonic()-start
        save(case/"build.json",result)
    return result


def audit(case):
    m=load(case)
    p=m["params"]
    build=json.loads((case/"build.json").read_text())
    if build["status"]!="PASS" or build["case_sha256"]!=sha(case/"case.json"):
        raise ValueError("build incomplete or changed")
    check_abi(case,m)
    inventory=elf.build_inventory(case,build["artifacts_sha256"])
    for key in ("max_static_sram_end","artifact_bytes"):
        if inventory[key] != build[key]:
            raise ValueError("recorded resource metadata differs")
    execution=json.loads((case/"execution.json").read_text())
    if execution["status"]!="COMPLETED" or execution["raw_sha256"]!=sha(case/"raw.npz") or execution["case_sha256"]!=sha(case/"case.json"):
        raise ValueError("capture incomplete or changed")
    raw=dict(np.load(case/"raw.npz",allow_pickle=False))
    arrays=dict(np.load(case/"inputs.npz",allow_pickle=False))
    graph=dict(np.load(case/"original.npz",allow_pickle=False))
    maps=dict(np.load(case/"mapping.npz",allow_pickle=False))
    if p["MODE"] > 0:
        selective_validation.validate_schedule(graph,maps,arrays,p)
        if p["MODE"] == 2:
            colored_validation.validate_rounds(arrays,p)
    checkpoints=resident_validation.validate_capture(m,arrays,raw,execution)
    reports=[]
    for checkpoint in checkpoints:
        report=audit_checkpoint(p,arrays,graph,maps,checkpoint)
        if p["MODE"] > 0:
            report["communication"]=selective_validation.validate_comm(p,arrays,checkpoint["raw"],checkpoint["steps"],checkpoint["cumulative_steps"])
        reports.append(report)
    return dict(status="PASS",checkpoints=reports,params=p,census=m["census"],partition=m["partition"],
                residency=m["residency"],transport=m["transport"],transport_census=m["transport_census"],
                compile_wall_seconds=build["wall_seconds"],runtime_wall_seconds=execution["wall_seconds"],
                max_static_sram_end=build["max_static_sram_end"],artifact_bytes=build["artifact_bytes"],
                cycles_scope="each PE local chunk timestamp span; first chunk includes geometry encoding; local SIM only",
                raw_sha256=sha(case/"raw.npz"),case_sha256=sha(case/"case.json"))


def audit_checkpoint(p,arrays,graph,maps,checkpoint):
    raw=checkpoint["raw"]
    out=reference.rollout(graph["velocity"],graph["node_type"],graph["edge_features"],graph["senders"],graph["receivers"],
                          arrays["weights"][0],p["H"],p["S"],checkpoint["cumulative_steps"],record_history=True)
    errors={}
    for name in ("velocity","h","e","aggregate","cached_e0"):
        ids=maps["edge_ids" if name in ("e","cached_e0") else "node_ids"]
        width=2 if name=="velocity" else p["H"]
        packed=raw[name].reshape(*ids.shape,width)
        actual=np.empty_like(out[name])
        if sorted(ids[ids>=0].tolist())!=list(range(len(actual))):
            raise ValueError("ownership coverage")
        actual[ids[ids>=0]]=packed[ids>=0]
        np.testing.assert_allclose(actual,out[name],rtol=3e-4,atol=3e-4,err_msg=name)
        np.testing.assert_array_equal(packed[ids<0],0,err_msg="padding "+name)
        if name=="velocity":
            fixed=~np.isin(graph["node_type"].ravel(),[0,5])
            np.testing.assert_array_equal(actual[fixed],graph["velocity"][fixed])
        if not np.isfinite(packed).all():
            raise ValueError("nonfinite readback "+name)
        errors[name]=float(np.max(np.abs(actual-out[name]))) if actual.size else 0.
    previous=(out["history"][-1]["blocks"][-2]["h"] if p["S"]>1
              else out["history"][-1]["encoded_h"])
    last_ids=maps["node_ids"][-1]
    last_panel=np.zeros((p["L"],p["H"]),np.float32)
    last_panel[last_ids>=0]=previous[last_ids[last_ids>=0]]
    if p.get("MODE",0) == 0:
        np.testing.assert_allclose(raw["panel"],np.tile(last_panel.ravel(),(p["C"]*p["R"],1)),rtol=3e-4,atol=3e-4)
    else:
        np.testing.assert_array_equal(raw["panel"],0)
    edge_ids=maps["edge_ids"]
    expected_senders=np.zeros((*edge_ids.shape,p["H"]),np.float32)
    expected_senders[edge_ids>=0]=previous[graph["senders"][edge_ids[edge_ids>=0]]]
    np.testing.assert_allclose(raw["sender_cache"].reshape(expected_senders.shape),expected_senders,rtol=3e-4,atol=3e-4)
    np.testing.assert_array_equal(raw["sender_cache"].reshape(expected_senders.shape)[edge_ids<0],0)
    clocks=raw["clocks"].astype(np.uint64)
    if np.any(clocks>65535):
        raise ValueError("u16 clock carrier corruption")
    ticks=clocks[:,0]+(clocks[:,1]<<16)+(clocks[:,2]<<32)
    end=clocks[:,3]+(clocks[:,4]<<16)+(clocks[:,5]<<32)
    cycles=(end-ticks)&((1<<48)-1)
    return dict(status="PASS",errors=errors,steps=checkpoint["steps"],
                cumulative_steps=checkpoint["cumulative_steps"],
                local_cycles_min=int(cycles.min()),local_cycles_max=int(cycles.max()),
                geometry_initializations=1)


def simulate(case):
    m=load(case)
    build=json.loads((case/"build.json").read_text())
    if build["status"]!="PASS" or (case/"execution.json").exists() or (case/"raw.npz").exists():
        raise ValueError("fresh completed build required")
    if build["case_sha256"]!=sha(case/"case.json"):
        raise ValueError("build belongs to another case")
    check_abi(case,m)
    elf.build_inventory(case,build["artifacts_sha256"])
    command=[str(ROOT/"cs_python"),str(case/"sdk_runner.py"),"--case",str(case)]
    start=time.monotonic()
    result=dict(status="FAILED",command=command,cwd=str(case))
    try:
        ret=subprocess.run(command,cwd=case,check=False)
        result["returncode"]=ret.returncode
        if ret.returncode:
            raise RuntimeError("SDK process failed")
        result.update(audit(case))
    except Exception as error:
        result["error"]=str(error)
        raise
    finally:
        result["wall_seconds"]=time.monotonic()-start
        save(case/"result.json",result)
    return result


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=("prepare","compile","simulate","audit"))
    ap.add_argument("--case",type=Path,required=True)
    ap.add_argument("--graph",type=Path)
    ap.add_argument("--transport",choices=("broadcast","selective","colored"),default="broadcast")
    ap.add_argument("--batch",type=int,help="selective vertices per packet; default floor(27/H)")
    ap.add_argument("--crossing",action="store_true",help="simultaneous disjoint endpoints sharing a physical link on ID4x2")
    ap.add_argument("--adversarial",action="store_true",help="duplicate directed edges, fanout and isolated vertices")
    ap.add_argument("--limit",type=int)
    ap.add_argument("--columns",type=int,default=2)
    ap.add_argument("--rows",type=int,default=2)
    ap.add_argument("--hidden",type=int,default=8)
    ap.add_argument("--blocks",type=int,default=2)
    ap.add_argument("--steps",type=int,default=2)
    ap.add_argument("--chunks",help="optional checkpoint chunks, e.g. 2,2; sum must equal --steps")
    ap.add_argument("--norm",choices=("sqrt","invsqrt"),default="sqrt",
                    help="LayerNorm sqrt/divide baseline or one inverse scale per vector")
    ap.add_argument("--edge-encoder-probe",choices=("normal","constant","near-constant"),default="normal",
                    help="optional untrained LayerNorm stress fixture; normal leaves seeded weights unchanged")
    ap.add_argument("--partition",choices=("id","spatial"),default="id")
    args=ap.parse_args()
    result=prepare(args) if args.command=="prepare" else {"compile":compile_case,"simulate":simulate,"audit":audit}[args.command](args.case.resolve())
    print(json.dumps(result,sort_keys=True,allow_nan=False))
