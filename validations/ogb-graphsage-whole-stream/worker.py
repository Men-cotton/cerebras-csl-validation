"""Run the streamed native final layer using the immutable original arxiv watcher fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(check,message):
    if not check:
        raise ValueError(message)


def relative(name):
    path=Path(name)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix()==name,"package-relative file")
    return path


def assemble(family,out):
    local=family/"case"
    release=read(local/"release.json")
    parent=family.parent/"ogb-graphsage-whole"
    source=parent/"case"
    require(digest(source/"release.json")==release["parent_release_sha256"],"published original full-layer release")
    base=read(source/"release.json")
    require(release["representative_windows"]==base["representative_windows"] and release["windows"]==base["windows"]
            and release["frames"]==base["frames"],"same original graph windows and controls")
    for kind,old_root,old_map,dest in (("host",parent,base["host"],out),("payload",source,base["payload"],out/"case")):
        for name,item in release[kind].items():
            file=relative(name)
            if item["source"]=="parent":
                require(old_map.get(name)==item["sha256"],"original released file "+name)
                src=old_root/file
            elif item["source"]=="worker":
                require(kind=="host" and name=="whole/stream_variant.py","native package binder")
                src=family/"worker.py"
            else:
                require(item["source"]=="variant","known file origin")
                src=local/kind/file
            require(digest(src)==item["sha256"],"payload identity "+name)
            path=dest/file
            path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,path)
    shutil.copyfile(local/"release.json",out/"case/release.json")


def checked_native(root):
    from replay import audit
    case=root/"case";release=read(case/"release.json")
    for kind,folder in (("host",root),("payload",case)):
        for name,entry in release[kind].items():
            require(digest(folder/relative(name))==entry["sha256"],"assembled source or original input "+name)
    native=read(case/"csl/build.json");source=read(case/"provenance.json")
    require(native["language"]=="csl" and native["role"]=="simulator" and native["params"]==release["params"] and
            native["windows"]==release["windows"] and native["full_frames"]==release["frames"] and
            native["aggregation"]=="one auto-advancing CSR edge source DSR per part" and
            native["input_sha256"]==source["input_sha256"]==digest(case/"whole.npz"),"fixed whole native program and model")
    require(native["source"]=={f.name:digest(f) for f in (case/"csl/src").iterdir() if f.is_file()} and
            native["elf"]=={f.name:digest(f) for f in (case/"csl/out/bin").glob("*.elf")} and
            native["windows_sha256"]==digest(case/"csl/windows.json"),"matched native source, image and global row order")
    proof=audit(case/"csl",case/"whole.npz",case/"sim-csl",release["representative_windows"],peer=case/"sim-tungsten")
    require(all(proof[k]==v for k,v in release["native_sdk"].items()),"original PE replies, FP64 classes and measured windows")
    return case,release,native,None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("check","compile-csl","run-csl"))
    p.add_argument("--out",type=Path);p.add_argument("--compiled",type=Path);p.add_argument("--namespace",default="job-operator")
    a=p.parse_args()
    if a.action!="check" and a.out is None:p.error("output directory required")
    if a.action=="run-csl" and a.compiled is None:p.error("CSL compile receipt required")
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp)/"validated"
        assemble(Path(__file__).resolve().parent,out)
        args=[sys.executable,"-B",out/"whole/physical.py",a.action,"--namespace",a.namespace]
        if a.out:args.extend(("--out",a.out.resolve()))
        if a.compiled:args.extend(("--compiled",a.compiled.resolve()))
        subprocess.run(args,check=True)


if __name__=="__main__":
    main()
