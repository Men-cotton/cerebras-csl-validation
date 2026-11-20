"""Audit one returned physical selected implementation using the watcher downloads and actual ELF."""
import argparse
import json
from pathlib import Path
from appliance import ROOT,read,verified,check_capture,application_elfs
from compile_csl import flags
from contract import require,sha,load_graph


def main(a):
    target=read(ROOT/"target.json");lang=target["language"]
    release=read(ROOT/"release.json");graph=load_graph(ROOT/"fixture.npz")
    result=a.output
    compiler=None
    if lang=="csl":
        require(a.build is not None,"actual selected CSL physical compilation")
        meta=read(a.build)
        archive=(a.build.parent/meta["artifact"]).resolve()
        require(archive.parent==a.build.resolve().parent and meta["shards"]==16 and meta["flags"]==flags(16,True)
                and meta["release_sha256"]==sha(ROOT/"release.json"),"actual compile configuration")
        compiler=meta["application_elfs"]
        if archive.exists():require(sha(archive)==meta["sha256"] and compiler==application_elfs(archive),"actual downloaded compile archive")
    state=read(result/"collection.json")
    require(state["backend"]=="hardware" and state["language"]==lang and state["shards"]==16 and
            state["release_sha256"]==sha(ROOT/"release.json") and state["launcher_run_elapsed_ns"]>0,"physical launcher and release")
    for file in ("capture/raw.npz","capture/result.json","worker.log","worker-exit.txt"):
        require(sha(result/Path(file).name)==state["downloads"][file]["sha256"],"downloaded "+file)
    require((result/"worker-exit.txt").read_text().strip()=="0","physical worker exit")
    summary=check_capture(result,lang,16,"hardware",release,graph,compiler)
    reference=verified()[f"{lang}-r16"]
    require(summary["owner_sha256"]==reference["owner_sha256"] and len(summary["windows"])==11,"all physical words against original ordered COO")
    return dict(backend="hardware",language=lang,implementation=target["name"],summary=summary,collection_sha256=sha(result/"collection.json"),
                capture_sha256=sha(result/"raw.npz"),result_sha256=sha(result/"result.json"),launcher_ns=state["launcher_run_elapsed_ns"],
                build_record_sha256=sha(a.build) if lang=="csl" else None,
                physical_elf_sha256=read(result/"result.json")["elf_sha256"],
                compile_archive_sha256=meta["sha256"] if lang=="csl" else None,
                compile_archive_present=archive.exists() if lang=="csl" else None,
                all_padded_owner_words_equal_to_simulator=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("output",type=Path);p.add_argument("--build",type=Path)
    print(json.dumps(main(p.parse_args()),sort_keys=True))
