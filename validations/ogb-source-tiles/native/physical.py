"""Compile one native program and measure two data distributions in one system session."""
import argparse
import hashlib
from pathlib import Path
import shlex
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent
sys.path.insert(0,str(FAMILY))
import report
from compare_delivery import checked
from inventory import program_bytes, read, require, save, sha

HOST=('paired_worker.py','inventory.py')
SHARED=('source_table.py','ogb_contract.py')
NAMES=('arxiv-rcm128-q50-p16-e128','products-rcm128-q50-p16-e128')


def checked_case(name,pin):
    require(name in NAMES,'registered native case')
    case=HERE/'cases'/name
    require(sha(case/'release.json')==pin,'native release pin')
    rel=read(case/'release.json')
    for path,digest in rel['files'].items():
        where=(case/path).resolve()
        require(where.is_relative_to(case.resolve()) and sha(where)==digest,'native case file '+path)
    for path,digest in rel['host'].items():
        where=(FAMILY/path).resolve()
        require(where.is_relative_to(FAMILY.resolve()) and sha(where)==digest,'native host source '+path)
    cousin=FAMILY/'cases'/name
    require(sha(cousin/'release.json')==rel['tungsten_release_sha256'],'matched Tungsten physical release')
    sdk=read(case/'build.json');fixture=cousin/'fixture.npz'
    require(sha(fixture)==sdk['fixture_sha256']==rel['fixture_sha256'],'matched original graph and trained weights')
    require(sdk['params']==read(cousin/'build.json')['params'],'native/Tungsten distribution and geometry')
    window=read(case/'window.json');model=read(case/'model.json')
    require(window['source_graph']==model['graph'] and window['checkpoint_sha256']==model['weights_sha256']==
            rel['model_checkpoint_sha256'] and window['complete_gpu_hidden_sha256']==model['gpu_hidden_sha256'] and
            window['complete_gpu_logits_sha256']==model['gpu_logits_sha256'],'frozen original model and CSR provenance')
    with np.load(fixture,allow_pickle=False) as z,np.load(case/'weights.npz',allow_pickle=False) as w:
        def digest(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
        require(digest(z['node'][z['src']])==window['original_source_arc_sequence_sha256'] and
                digest(z['ptr'])==window['original_rowptr_sha256'] and digest(z['x'])==window['hidden_rows_sha256'] and
                digest(z['gpu'])==window['gpu_logits_sha256'] and window['original_ids']==rel['original_ids'],
                'original source features, arc sequence, rows and reference')

        for field,key in (('wr','root1.weight'),('wn','neigh1.weight'),('b','neigh1.bias')):
            np.testing.assert_array_equal(z[field],w[key].T)
        require(len(z['seed'])==128 and np.array_equal(z['seed'],rel['original_ids']) and
                sha(case/'weights.npz')==rel['model_checkpoint_sha256'],'original checkpoint and IDs')
    _,expected=report.replay(fixture,cousin,cousin/'sdk',inspect_sram=False)
    summaries={}
    for mode,frame_count in (('row',6),('replicated',2)):
        metrics,_=report.replay(fixture,case,case/('sdk-'+mode),expected,inspect_sram=False)
        require(metrics['role']=='simulator' and metrics['delivery']==mode and len(metrics['frames'])==frame_count,
                'two native SDK calibration deliveries')
        summaries[mode]=metrics
    return case,rel,sdk,fixture,expected


def compiler_args(sdk):
    return ['--fabric-dims=762,1172' if s.startswith('--fabric-dims=') else s for s in sdk['flags']]


def bound_artifact(record_path,case,rel,sdk,pin):
    compiled=read(record_path)
    path=record_path.parent/compiled['artifact']
    require(path.name==compiled['artifact'] and path.is_file() and sha(path)==compiled['archive_sha256'] and
            compiled['release_sha256']==pin and compiled['source_sha256']==sdk['source_sha256'] and
            compiled['flags']==compiler_args(sdk) and compiled['case']==case.name,'actual compile record')
    files=program_bytes(path,sdk['source_sha256'],compiled['flags'],sdk['csl_params'])
    inv={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
    require(inv==compiled['inventory'],'actual archived native firmware')
    workers={p.name:sha(p) for p in (*[HERE/n for n in HOST],*[FAMILY/n for n in SHARED])}
    native=sdk|dict(role='hardware',flags=compiled['flags'],inventory=inv,worker_sha256=workers,
                    elf_sha256={Path(n).name:h for n,h in inv.items() if n.startswith('bin/') and n.endswith('.elf')},
                    release_sha256=pin,compiler_archive_sha256=compiled['archive_sha256'])
    return path,native


def execute(a):
    case,rel,sdk,fixture,expected=checked_case(a.case,a.release_sha256)
    if a.action=='check':return
    from cerebras.sdk.client import SdkCompiler,SdkLauncher
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    service=dict(mgmt_namespace=a.namespace,resource_cpu=1000,resource_mem=8<<30,disable_version_check=False)
    if a.action=='compile':
        flags=compiler_args(sdk)
        with SdkCompiler(**service) as compiler:
            archive=Path(compiler.compile(str(case/'src'),'layout.csl',' '.join(flags),str(out))).resolve()
        require(archive.parent==out and archive.is_file(),'one returned physical compiler archive')
        files=program_bytes(archive,sdk['source_sha256'],flags,sdk['csl_params'])
        save(out/'compiled.json',dict(case=case.name,release_sha256=a.release_sha256,flags=flags,
              artifact=archive.name,archive_sha256=sha(archive),source_sha256=sdk['source_sha256'],
              inventory={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}))
        return
    archive,desc=bound_artifact(a.compiled.resolve(),case,rel,sdk,a.release_sha256)
    save(out/'build.json',desc);cap=out/'capture';cap.mkdir()
    state=dict(role='hardware',case=case.name,release_sha256=a.release_sha256,build_sha256=sha(out/'build.json'),
               compiler_archive_sha256=sha(archive),namespace=a.namespace,downloads={},numerical_status='pending')
    save(out/'collection.json',state)
    expected_files=['worker.log','worker-exit.txt','capture/result.json',*(f'capture/frame-{i:02d}.npz' for i in range(1,13))]
    command=shlex.join(['timeout','--signal=INT','--kill-after=30s','1800s','cs_python','-B','paired_worker.py',
                       '.','fixture.npz','capture','--artifact','out','--cmaddr','%CMADDR%'])
    try:
        with SdkLauncher(str(archive),simulator=False,**service) as runtime:
            try:
                for path in (*[HERE/n for n in HOST],*[FAMILY/n for n in SHARED],fixture,out/'build.json'):
                    runtime.stage(str(path))
                response=runtime.run(command+' >worker.log 2>&1; rc=$?; echo "$rc" >worker-exit.txt; exit "$rc"')
                (out/'launcher-response.txt').write_text(str(response))
            finally:
                for name in expected_files:
                    try:
                        runtime.download_artifact(name,str(out/name))
                        state['downloads'][name]={'sha256':sha(out/name)}
                    except Exception as exc:state['downloads'][name]={'error':repr(exc)}
                    save(out/'collection.json',state)
        require(all('sha256' in v for v in state['downloads'].values()) and
                (out/'worker-exit.txt').read_text().strip()=='0','complete physical frame collection')
        audit=checked(out,fixture,cap,expected,'hardware')
        save(out/'replay.json',audit)
        state['numerical_status']='passed'
    finally:save(out/'collection.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','compile','run'));parser.add_argument('--case',required=True)
    parser.add_argument('--release-sha256',required=True);parser.add_argument('--out',type=Path)
    parser.add_argument('--compiled',type=Path);parser.add_argument('--namespace',default='job-operator')
    a=parser.parse_args()
    if a.action!='check' and not a.out:parser.error('--out is required')
    if a.action=='run' and not a.compiled:parser.error('--compiled is required')
    execute(a)
