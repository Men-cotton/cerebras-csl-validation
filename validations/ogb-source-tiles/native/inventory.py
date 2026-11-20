"""Bind the application and both native memcpy helpers to one compiler result."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import tarfile


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def require(valid,message):
    if not valid:raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path,record):
    path=Path(path)
    scratch=path.with_suffix(path.suffix+'.tmp')
    scratch.write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n')
    scratch.replace(path)


def semantic(name):
    p=PurePosixPath(name)
    return (name in ('out.json','bin/out_rpc.json','east/out.json','west/out.json') or
            (p.suffix=='.elf' and str(p.parent) in ('bin','west/bin','east/bin')))


def actual(folder):
    selected={}
    for sub in ('','west','east'):
        root=folder/sub
        for path in sorted(root.glob('bin/*.elf')):
            selected[path.relative_to(folder).as_posix()]=sha(path)
        require(any(p.startswith((sub+'/' if sub else '')+'bin/') and p.endswith('.elf') for p in selected),
                'missing application or memcpy firmware '+sub)
    for key in ('out.json','bin/out_rpc.json','east/out.json','west/out.json'):
        selected[key]=sha(folder/key)
    return selected


def program_bytes(tarpath,expected_source,flags,params):
    with tarfile.open(tarpath,'r:*') as arc:
        items={}
        for m in arc:
            p=PurePosixPath(m.name)
            require(not p.is_absolute() and '..' not in p.parts and p not in items,'unsafe or duplicate compiler record')
            if m.isfile():items[p]=m
        tops={p.parent.parent for p in items if p.parts[-2:]==('west','out.json')}
        tops={p for p in tops if p/'out.json' in items and p/'east/out.json' in items}
        require(len(tops)==1,'unique compiled application')
        top=tops.pop()
        def data(p):
            entry=items[p]
            require(0<entry.size<32<<20,'compiler record size')
            return arc.extractfile(entry).read()
        files={p.relative_to(top).as_posix():data(p) for p in items if p.is_relative_to(top) and semantic(p.relative_to(top).as_posix())}
        pre=top.parent
        sources=[p.parent for p in items if p.name=='layout.csl' and p.is_relative_to(pre) and
                 all(p.parent/n in items and hashlib.sha256(data(p.parent/n)).hexdigest()==digest
                     for n,digest in expected_source.items())]
        require(len(sources)==1,'returned physical application sources differ')
        cmd=shlex.split(data(pre/'compile_command.sh').decode())
        for flag in flags:
            if flag.startswith('--params='):
                candidates=[v for v in cmd if v.startswith('--params=')]
                require(len(candidates)==1 and sorted(candidates[0][9:].split(','))==sorted(flag[9:].split(',')),
                        'archived compiler parameter record')
            else:require(flag in cmd,'archived compiler flag '+flag)
        require({k:int(v) for k,v in json.loads(files['out.json'])['params'].items()}==params,'compiled owner geometry')
    require(all(any((PurePosixPath(n).parent.as_posix()==sub and n.endswith('.elf')) for n in files)
                for sub in ('bin','west/bin','east/bin')) and
            all(n in files for n in ('out.json','bin/out_rpc.json','east/out.json','west/out.json')),'complete native compute and memcpy inventory')
    return files
