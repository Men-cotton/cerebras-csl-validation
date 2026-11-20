"""Compile and run distinct row broadcasts on one native rectangle on CS-3."""

import argparse
from hashlib import sha256
from pathlib import Path
import shlex
import shutil
import sys

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
sys.path.insert(0,str(FAMILY))
import appliance
from report_rectangle import replay
from inventory import program_bytes, read, require, save, sha

NATIVE=('verified_worker.py','worker_rectangle.py','inventory.py')
SHARED=('contract_lanes.py','source_table.py','ogb_contract.py')

def check(key,pin):
    require(Path(key).name==key and key.startswith(('arxiv-','products-')),'expected OGB case')
    case=HERE/'cases'/key
    rel=read(case/'release.json')
    require(sha(case/'release.json')==pin,'native release pin')
    for root,pins in ((case,rel['files']),(FAMILY,rel['host'])):
        for name,digest in pins.items():
            path=(root/name).resolve()
            require(path.is_relative_to(root.resolve()) and sha(path)==digest,'changed frozen '+name)
    cousin=FAMILY/'cases'/key
    t,expected=appliance.check(cousin,rel['tungsten_release_sha256'])
    orig=cousin/'fixture.npz'
    sdk=read(case/'build.json')
    require(sdk['params']['P']==t['owners_per_lane'] and sdk['params']['L']==t['lanes'] and
            sdk['fixture_sha256']==t['fixture_sha256']==sha(orig),'matched allocation on original input')
    raw,values=replay(orig,case,case/'sdk',expected)
    require(raw['role']=='simulator' and len(raw['frames'])==12,'native simulator completion')
    return case,rel,sdk,orig,expected

def execute(a):
    case,rel,sdk,orig,expected=check(a.case,a.release_sha256)
    if a.action=='check':return
    from cerebras.sdk.client import SdkCompiler,SdkLauncher
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    flags=['--fabric-dims=762,1172' if f.startswith('--fabric-dims=') else f for f in sdk['flags']]
    resources=dict(mgmt_namespace=a.namespace,resource_mem=8<<30,resource_cpu=1000,disable_version_check=False)
    if a.action=='compile':
        with SdkCompiler(**resources) as service:
            archive=Path(service.compile(str(case/'src'),'layout.csl',' '.join(flags),str(out))).resolve()
        require(archive.parent==out and archive.is_file(),'returned one physical compiler archive')
        data=program_bytes(archive,sdk['source_sha256'],flags,sdk['params'])
        save(out/'compiled.json',dict(case=case.name,release_sha256=a.release_sha256,flags=flags,
             source_sha256=sdk['source_sha256'],archive=archive.name,archive_sha256=sha(archive),inventory={n:sha256(v).hexdigest() for n,v in data.items()}))
        return
    record=a.compiled.resolve();compiled=read(record)
    archive=record.parent/compiled['archive']
    require(archive.name==compiled['archive'] and sha(archive)==compiled['archive_sha256'] and
            compiled['flags']==flags and compiled['case']==case.name and compiled['release_sha256']==a.release_sha256 and
            compiled['source_sha256']==sdk['source_sha256'],'requested physical compiler result')
    data=program_bytes(archive,sdk['source_sha256'],flags,sdk['params'])
    inv={n:sha256(v).hexdigest() for n,v in data.items()}
    require(inv==compiled['inventory'],'original compiler firmware')
    workers=[*[HERE/n for n in NATIVE],*[FAMILY/n for n in SHARED]]
    desc=sdk|dict(role='hardware',flags=flags,inventory=inv,elf_sha256={Path(n).name:v for n,v in inv.items() if
                  Path(n).parent==Path('bin') and n.endswith('.elf')},worker_sha256={p.name:sha(p) for p in workers},release_sha256=a.release_sha256)
    save(out/'build.json',desc)
    shutil.copytree(case/'src',out/'src')
    for n,v in data.items():
        p=out/'out'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
    state=dict(role='hardware',release_sha256=a.release_sha256,compiler_archive_sha256=sha(archive),downloads={},numerical_status='pending')
    save(out/'collection.json',state);(out/'capture').mkdir()
    files=['worker.log','worker-exit.txt','capture/result.json',*(f'capture/frame-{i}.npz' for i in range(1,13))]
    command=shlex.join(['timeout','--signal=INT','--kill-after=30s','2700s','cs_python','-B','verified_worker.py','--cmaddr','%CMADDR%'])
    try:
        with SdkLauncher(str(archive),simulator=False,**resources) as runtime:
            try:
                for p in [*workers,out/'build.json',orig]:runtime.stage(str(p))
                (out/'launcher-response.txt').write_text(str(runtime.run(command+' >worker.log 2>&1; rc=$?; echo "$rc" >worker-exit.txt; exit "$rc"')))
            finally:
                for name in files:
                    try:
                        runtime.download_artifact(name,str(out/name));state['downloads'][name]={'sha256':sha(out/name)}
                    except Exception as exc:state['downloads'][name]={'error':repr(exc)}
                    save(out/'collection.json',state)
        require(all('sha256' in v for v in state['downloads'].values()) and (out/'worker-exit.txt').read_text().strip()=='0','complete native capture')
        measured,_=replay(orig,out,out/'capture',expected)
        require(measured['role']=='hardware' and len(measured['frames'])==12,'native hardware completion')
        save(out/'replay.json',measured);state['numerical_status']='passed'
    finally:save(out/'collection.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('compile','run','check'));p.add_argument('--case',required=True);p.add_argument('--release-sha256',required=True)
    p.add_argument('--out',type=Path);p.add_argument('--compiled',type=Path);p.add_argument('--namespace',default='job-operator')
    a=p.parse_args()
    if a.action!='check' and not a.out:p.error('--out required')
    if a.action=='run' and not a.compiled:p.error('--compiled required')
    execute(a)
