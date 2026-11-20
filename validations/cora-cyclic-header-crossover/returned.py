"""Recheck every returned job and compare the same physical CSL archive across runtime graph headers."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parent
RUNS={"off-csl":("off","csl","run-csl"),"on-csl":("on","csl","run-csl-enabled"),
      "off-tungsten":("off","tungsten","run-tungsten"),"cyclic-tungsten":("cyclic","tungsten","run-cyclic-tungsten"),
      "mapped-tungsten":("mapped","tungsten","run-mapped-tungsten")}


def single(case,lang,folder,build):
    sys.path.insert(0,str(case))
    from appliance import ROOT as target,check_capture,application_elfs,verified
    from contract import load_graph,sha,require
    from compile_csl import flags
    read=lambda p:json.loads(p.read_text())
    release=read(target/"release.json");state=read(folder/"collection.json")
    require(state["backend"]=="hardware" and state["language"]==lang and state["shards"]==16 and state["launcher_run_elapsed_ns"]>0
            and state["release_sha256"]==sha(target/"release.json"),"physical job and frozen release")
    for f in ("capture/raw.npz","capture/result.json","worker.log","worker-exit.txt"):
        require(state["downloads"][f]["sha256"]==sha(folder/Path(f).name),"downloaded "+f)
    require((folder/"worker-exit.txt").read_text().strip()=="0","completed physical worker")
    compiled=None
    if lang=="csl":
        meta=read(build);archive=(build.parent/meta["artifact"]).resolve()
        require(meta["shards"]==16 and meta["flags"]==flags(16,True) and meta["release_sha256"]==sha(ROOT/"off/release.json")
                and archive.parent==build.resolve().parent,"shared physical compilation")
        compiled=meta["application_elfs"]
        if archive.exists():require(sha(archive)==meta["sha256"] and application_elfs(archive)==compiled,"physical archive bytes")
    summary=check_capture(folder,lang,16,"hardware",release,load_graph(target/"fixture.npz"),compiled)
    ref=verified()[lang+"-r16"]
    require(summary["owner_sha256"]==ref["owner_sha256"] and summary["words"]==56868,"physical original-COO outputs")
    return dict(summary=summary,physical_elf_sha256=read(folder/"result.json")["elf_sha256"],
                launch_ns=state["launcher_run_elapsed_ns"],collection_sha256=sha(folder/"collection.json"),raw_sha256=sha(folder/"raw.npz"))


def main(a):
    output=a.output.resolve();build=output/"compile-csl/build.json"
    cases={}
    for name,(case,lang,step) in RUNS.items():
        argv=[sys.executable,"-B",str(ROOT/"returned.py"),"--single",str(ROOT/case),lang,str(output/step),str(build)]
        cases[name]=json.loads(subprocess.check_output(argv,text=True))
    assert cases["off-csl"]["physical_elf_sha256"]==cases["on-csl"]["physical_elf_sha256"]
    same=[cases[k]["summary"]["owner_sha256"] for k in ("off-csl","on-csl","off-tungsten","cyclic-tungsten")]
    assert all(x==same[0] for x in same)
    c=cases["off-csl"]["summary"]["max_owner_cumulative_sum"]
    comparisons={}
    for name,case in cases.items():
        w=case["summary"]["max_owner_cumulative_sum"]
        comparisons[name]=dict(cycles_minus_off_csl=[x-y for x,y in zip(w,c)],
                               faster_than_off_csl_at=[j+1 for j in range(3) if w[j]<c[j]])
    if a.prior_enabled_tungsten:
        earlier=json.loads(a.prior_enabled_tungsten.read_text())
        assert earlier["backend"]=="hardware" and earlier["release_sha256"]==json.loads((ROOT/"comparison.json").read_text())["enabled_release_sha256"]
        old=earlier["cases"]["tungsten"]
        assert old["physical_elf_sha256"]==cases["off-tungsten"]["physical_elf_sha256"] and old["summary"]["owner_sha256"]==same[0]
        comparisons["previously_published_tungsten_enabled_minus_disabled"]=[x-y for x,y in zip(old["summary"]["max_owner_cumulative_sum"],cases["off-tungsten"]["summary"]["max_owner_cumulative_sum"])]
    a.out.write_text(json.dumps(dict(backend="hardware",cases=cases,comparisons=comparisons,
                    shared_physical_csl_image=True,scope="sum of per-window largest individual PE interval; excludes host and dispatch"),indent=2)+"\n")


if __name__=="__main__":
    if sys.argv[1:2]==["--single"]:
        print(json.dumps(single(Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]),Path(sys.argv[5]))))
    else:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument("output",type=Path);p.add_argument("out",type=Path)
        p.add_argument("--prior-enabled-tungsten",type=Path);main(p.parse_args())
