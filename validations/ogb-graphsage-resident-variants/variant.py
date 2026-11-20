"""Run a changed resident firmware against an existing immutable watcher input and model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(check,message):
    if not check:raise ValueError(message)


def assemble(root,case,destination):
    require(case==Path(case).name and case,"case identifier")
    local=root/"cases"/case
    link=read(local/"parent.json")
    parent_name=link["base_case"]
    require(Path(parent_name).name==parent_name and parent_name,"original case identifier")
    base=root.parent/"ogb-graphsage-resident"
    old=base/"cases"/parent_name
    require(digest(old/"release.json")==link["base_release_sha256"] and
            digest(local/"release.json")==link["release_sha256"],"original and tuned release identities")
    original=read(old/"release.json");updated=read(local/"release.json")
    roles=set(link["changed_languages"])
    require(roles and roles<={"csl","tungsten"},"firmware language")
    changed_host=set(link.get("changed_host",[]))
    allow_host={"resident/graph.py","resident/run.py","resident/limits.py"}
    require(changed_host<=allow_host and (not changed_host or roles=={"csl","tungsten"}) and
            set(updated["host"])==set(original["host"]) and
            changed_host=={k for k in original["host"] if updated["host"][k]!=original["host"][k]},
            "host protocol overrides and paired firmware")
    require(updated["name"]==case and
            updated["dataset"]==original["dataset"] and updated["scope"]==original["scope"] and
            updated["destinations"]==original["destinations"] and updated["sdk_input"]==original["sdk_input"],
            "same immutable input and model scope")
    for name,sha in original["host"].items():
        require(digest(base/name)==sha,"original host source "+name)
        source=local/"protocol"/name if name in changed_host else base/name
        require(digest(source)==updated["host"][name],"tuned host source "+name)
        out=destination/name
        out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,out)
    mounted=destination/"cases"/case;mounted.mkdir(parents=True)
    roots={"input","provenance",updated["sdk_input"],"csl","tungsten","sim-csl","sim-tungsten"}
    changed=roles|{"sim-"+lang for lang in roles}
    for name in roots:
        source=local/name if name in changed else old/name
        require(source.is_dir(),"missing bound case component "+name)
        if name not in changed:
            prefix=name+"/"
            parent_hash={k:v for k,v in original["files"].items() if k.startswith(prefix)}
            child_hash={k:v for k,v in updated["files"].items() if k.startswith(prefix)}
            require(parent_hash and child_hash==parent_hash,"different inherited input or control "+name)
        (mounted/name).symlink_to(source.resolve(),target_is_directory=True)
    shutil.copyfile(local/"release.json",mounted/"release.json")
    return destination/"resident/physical.py",roles


def execute(args):
    root=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as raw:
        entry,languages=assemble(root,args.case,Path(raw))
        if args.action!="check":
            require(args.action.split("-")[-1] in languages,"requested unmodified firmware")
        command=[sys.executable,"-B",entry,args.action,"--case",args.case,"--namespace",args.namespace]
        for field in ("compiled","peer","out"):
            value=getattr(args,field)
            if value:command+=["--"+field,value.resolve()]
        subprocess.run(command,check=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("check","compile-csl","run-csl","run-tungsten"))
    parser.add_argument("--case",required=True);parser.add_argument("--namespace",default="job-operator")
    for name in ("out","compiled","peer"):parser.add_argument("--"+name,type=Path)
    execute(parser.parse_args())
