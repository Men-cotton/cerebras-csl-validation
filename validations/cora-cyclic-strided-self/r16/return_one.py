"""Recheck one returned physical compile and both job receipts against this program's actual local outputs."""
from pathlib import Path
import argparse
import json
from appliance import ROOT,read,application_elfs,verified,check_capture
from compile_csl import flags
from contract import require,sha,load_graph


def run(output):
    release=read(ROOT/"release.json")
    r=release["shards"];graph=load_graph(ROOT/"fixture.npz")
    ref=verified()
    build_file=output/"compile-csl/build.json"
    build=read(build_file)
    archive=(build_file.parent/build["artifact"]).resolve()
    require(archive.parent==build_file.resolve().parent and build["shards"]==r and build["flags"]==flags(r,True)
            and build["release_sha256"]==sha(ROOT/"release.json"),"physical compile and geometry")
    if archive.exists():
        require(sha(archive)==build["sha256"] and application_elfs(archive)==build["application_elfs"],"actual compiled CSL archive")
    cases={}
    for lang in ("csl","tungsten"):
        folder=output/("run-"+lang)
        state=read(folder/"collection.json")
        cap=check_capture(folder,lang,r,"hardware",release,graph,build["application_elfs"] if lang=="csl" else None)
        require(state["backend"]=="hardware" and state["language"]==lang and state["shards"]==r and
                state["release_sha256"]==sha(ROOT/"release.json") and state["launcher_run_elapsed_ns"]>0,"actual launcher role")
        for key in ("capture/raw.npz","capture/result.json","worker.log","worker-exit.txt"):
            require(state["downloads"][key]["sha256"]==sha(folder/Path(key).name),"download "+key)
        require((folder/"worker-exit.txt").read_text().strip()=="0" and cap["owner_sha256"]==ref[f"{lang}-r{r}"]["owner_sha256"],
                "whole-graph physical completion and original-COO oracle")
        cases[lang]=dict(summary=cap,launcher_ns=state["launcher_run_elapsed_ns"],capture_sha256=sha(folder/"raw.npz"),
                         result_sha256=sha(folder/"result.json"),collection_sha256=sha(folder/"collection.json"),
                         physical_elf_sha256=read(folder/"result.json")["elf_sha256"],all_padded_owner_words_equal_to_simulator=True)
    require(cases["csl"]["summary"]["owner_sha256"]==cases["tungsten"]["summary"]["owner_sha256"],"all trained words on both devices")
    return dict(backend="hardware",mode=release["mode"],shards=r,pe=r*4,release_sha256=sha(ROOT/"release.json"),
                build_sha256=sha(build_file),archive_sha256=build["sha256"],archive_present=archive.exists(),cases=cases)

if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__);a.add_argument("output",type=Path)
    print(json.dumps(run(a.parse_args().output),sort_keys=True))
