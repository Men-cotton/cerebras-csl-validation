"""Compile the registered H16 source once, then load it once for both physical placements."""
import argparse
from pathlib import Path
import sys

from paired_worker import save, sha
from contract import audit, frame, loaded, read, references, require

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent/'h16-hub-cora'
NAMESPACE = 'job-operator'


def check(case, pin):
    require(sha(case/'release.json')==pin, 'release identity')
    release=read(case/'release.json')
    for group, directory in (('files',case), ('host_files',HERE), ('baseline_host', FAMILY)):
        for name, expected in release[group].items():
            require(sha(directory/name)==expected, 'frozen file '+name)
    doc,ctx,bindings=references(case,FAMILY)
    import portable
    base=FAMILY/'cases'/doc['baseline_case']
    require(doc['flags']==portable.compiler_arguments(doc['params'],'hardware',1),'physical compiler arguments')
    evidence=read(case/'local.json')
    require(evidence['baseline_case']==doc['baseline_case_sha256'] and evidence['source']==
            {n:sha(base/'source'/n) for n in ('layout.csl','pe.csl')},'same registered source')
    portable.admit_compiled_inventory(evidence['sdk_inventory'])
    target=read(case/'target.json')
    require(target['registered_case_sha256']==doc['baseline_case_sha256'] and target['fabric']==[762,1172]
            and target['flags']==doc['flags'] and target['source']==evidence['source'], 'physical-sized source and compiler flags')
    portable.admit_compiled_inventory(target['inventory'])
    probe=read(case/'ordinary-sdk.json')
    require(probe['status']=='PASS' and probe['backend']=='WSE3 SDK simulator' and
            (probe['compute_pes'],probe['frame_count'],probe['startups'])==(4,9,1) and len(probe['readbacks'])==12
            and probe['worker_sha256']==sha(HERE/'paired_worker.py') and probe['launcher_sha256']==sha(HERE/'paired_launch.py'),
            'ordinary SDK transfers, restoration and numerical probe')
    for mode,(arrays,expected,nodes) in bindings.items():
        proof=evidence[mode]
        require(proof['pe_count']==len(nodes) and proof['input_symbols']==len(arrays)
                and proof['matched_input_bytes']==sum(v.nbytes for v in arrays.values())
                and proof['writes_outside_input_arrays']==0 and proof['zero_initial_stats'], 'verified startup')
        numeric=frame(loaded(case/(mode+'-sdk.npz')),expected,nodes,ctx,1.)
        require(numeric==evidence['measurements'][mode], 'recomputed simulator result '+mode)
    before=evidence['measurements']['original']
    after=evidence['measurements']['swap']
    require(before['original_pre']==after['original_pre'] and before['original_logits']==after['original_logits'],
            'matched original-ID simulator layer bits')
    return doc,release,base


def compile_case(case,pin,out):
    from cerebras.sdk.client import SdkCompiler
    doc,_,base=check(case,pin)
    out.mkdir(parents=True,exist_ok=False)
    with SdkCompiler(mgmt_namespace=NAMESPACE,resource_cpu=1000,resource_mem=8<<30,
                     disable_version_check=False) as compiler:
        artifact=Path(compiler.compile(str(base/'source'),'layout.csl',doc['flags'],str(out))).resolve()
    require(artifact.is_file() and artifact.parent==out, 'returned physical compiler archive')
    save(out/'build.json',dict(release=pin,case=sha(case/'case.json'),namespace=NAMESPACE,flags=doc['flags'],
                               artifact=artifact.name,sha256=sha(artifact)))


def run_case(case,pin,out,build):
    from cerebras.sdk.client import SdkLauncher
    doc,release,base=check(case,pin)
    info=read(build)
    artifact=build.parent/info['artifact']
    require(artifact.resolve().parent==build.parent and info['release']==pin and info['case']==sha(case/'case.json')
            and info['flags']==doc['flags'] and info['namespace']==NAMESPACE and sha(artifact)==info['sha256'],
            'original physical compiler archive and command')
    out.mkdir(parents=True,exist_ok=False)
    capture=out/'capture'
    capture.mkdir()
    files=['capture/capture.json','linked.json','worker.log','worker-exit.txt','staged.sha256']
    files.extend(f'capture/{mode}-{suffix}.npz' for mode in doc['mode_order'] for suffix in ('initial','1','2','3'))
    staged={name:HERE/name for name in ('paired_worker.py','paired_launch.py')}
    staged.update({name:FAMILY/name for name in release['baseline_host']})
    staged.update({'case.json':case/'case.json','inputs.npz':base/'inputs.npz','swap.npz':case/'swap.npz'})
    if 'secondary' in doc['mode_order']:
        staged['secondary.npz']=case/'secondary.npz'
    command=('sha256sum '+' '.join(staged)+' > staged.sha256 && '
             'timeout --signal=INT --kill-after=15s 3600s cs_python paired_launch.py out case.json capture '
             '--cmaddr %CMADDR% > worker.log 2>&1; code=$?; echo "$code" >worker-exit.txt; exit "$code"')
    record=dict(release=pin,artifact_sha256=info['sha256'],downloads={},status='pending')
    try:
        with SdkLauncher(str(artifact),simulator=False,mgmt_namespace=NAMESPACE,resource_cpu=1000,
                         resource_mem=8<<30,disable_version_check=False) as launcher:
            for path in staged.values():
                launcher.stage(str(path))
            try:
                (out/'launcher-response.txt').write_text(str(launcher.run(command)))
            finally:
                for name in files:
                    try:
                        launcher.download_artifact(name,str(out/name))
                        record['downloads'][name]=dict(sha256=sha(out/name))
                    except Exception as e:
                        record['downloads'][name]=dict(error=repr(e))
                    save(out/'collection.json',record)
        require(all('sha256' in r for r in record['downloads'].values()) and
                (out/'worker-exit.txt').read_text().strip()=='0', 'complete worker and capture')
        actual={n:v for v,n in (line.split() for line in (out/'staged.sha256').read_text().splitlines())}
        require(actual=={n:sha(p) for n,p in staged.items()},'staged source and input hashes')
        linked=read(out/'linked.json')
        require('runtime_created' in linked and linked['case_sha256']==sha(case/'case.json') and
                linked['worker_sha256']==sha(HERE/'paired_worker.py'),
                'physical ABI and worker identity')
        import portable
        portable.admit_compiled_inventory(linked['elf'])
        results=audit(case,FAMILY,capture,'hardware')
        save(out/'fp64.json',results)
        record.update(status='passed',cycles={mode:[r['max_cycles'] for r in rows] for mode,rows in results.items()})
    except BaseException as error:
        record.update(status='failed',error=repr(error))
        raise
    finally:
        save(out/'collection.json',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','compile','run'])
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--release',required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--build',type=Path)
    a=parser.parse_args()
    if a.action=='check':
        check(a.case.resolve(),a.release)
    elif a.action=='compile':
        compile_case(a.case.resolve(),a.release,a.out.resolve())
    else:
        run_case(a.case.resolve(),a.release,a.out.resolve(),a.build.resolve())
