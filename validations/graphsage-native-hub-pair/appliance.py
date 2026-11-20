"""Compile a native GraphSAGE case once and audit both hub assignments on that artifact."""
import argparse
import json
from pathlib import Path

from paired_worker import sha, save
from paired_audit import audit, require

HERE = Path(__file__).resolve().parent
NAMESPACE = 'job-operator'


def load(path):
    return json.loads(path.read_text())


def references(case, pin):
    release=case/'release.json'
    require(sha(release)==pin,'release pin')
    cert=load(release)
    require(sha(case/'case.json')==cert['case_sha256'],'case pin')
    doc=load(case/'case.json')
    for name,h in doc['files'].items():
        require(sha(case/name)==h,'frozen fixture '+name)
    original_family=HERE.parent/'graphsage-resident-templates'
    original=original_family/'cases'/doc['original_case']
    for name,h in cert['host'].items():
        require(sha(HERE/name)==h,'paired host '+name)
    for name,h in cert['original_host'].items():
        require(sha(original_family/name)==h,'original host '+name)
    for name,h in cert['simulator'].items():
        require(sha(case/'sim'/name)==h,'paired simulator '+name)
    require(sha(case/'linked-sdk.json')==cert['linked_sdk_sha256'],'SDK ELF inventory pin')
    import sys
    sys.path.insert(0,str(original_family))
    import portable
    portable.admit_compiled_inventory(load(case/'linked-sdk.json'))
    require(sha(case/'sdk-audit.json')==cert['sdk_audit_sha256'],'original simulator audit pin')
    recorded=load(case/'sdk-audit.json')
    kind = cert.get('sdk_measurement', 'six resident frames')
    if kind == 'two stopped-core A frames':
        from sample_audit import audit_samples
        required = {f'{mode}-1.{ext}' for mode in ('control', 'split') for ext in ('json', 'npz')}
        require(set(cert['simulator']) == required, 'incomplete SDK A witnesses')
        target_file = case/'linked-target.json'
        require(sha(target_file) == cert['target_inventory_sha256'], 'physical-sized ELF inventory changed')
        target = load(target_file)
        require(target['fabric'] == [762,1172] and target['compute'] == [doc['params']['C'], doc['params']['R']]
                and target['case_sha256'] == cert['case_sha256'] and target['source'] == doc['files']['src/pe.csl'],
                'physical-sized link uses different dimensions or source')
        portable.admit_compiled_inventory(target['linked'])
        numeric = audit_samples(case, original)
        count = 1
    else:
        require(kind == 'six resident frames', 'unrecognized SDK evidence')
        numeric = audit(case,case/'sim',original,'simulator')
        count = 3
    for mode in ('control','split'):
        require(len(recorded['modes'][mode])==count,'missing saved SDK frames')
        for now,old in zip(numeric['modes'][mode],recorded['modes'][mode]):
            for name in ('frame','scale','raw_sha256','moved_partial_word_hops','predictions','per_pe_cycles'):
                require(now[name]==old[name],f'{mode} simulator {name}')
    return doc,cert,original


def compile_case(case,pin,out):
    from cerebras.sdk.client import SdkCompiler
    doc,_,_=references(case,pin)
    out.mkdir(parents=True,exist_ok=False)
    with SdkCompiler(mgmt_namespace=NAMESPACE,resource_cpu=1000,resource_mem=8<<30,disable_version_check=False) as compiler:
        target=Path(compiler.compile(str(case/'src'),'layout.csl',' '.join(doc['flags']),str(out))).resolve()
    require(target.is_file() and target.parent==out,'returned compiler archive')
    save(out/'build.json',dict(release=pin,case=sha(case/'case.json'),namespace=NAMESPACE,
                               flags=doc['flags'],artifact=target.name,sha256=sha(target)))


def run_case(case,pin,out,build):
    from cerebras.sdk.client import SdkLauncher
    doc,cert,original=references(case,pin)
    compiled=load(build)
    artifact=build.parent/compiled['artifact']
    require(artifact.resolve().parent==build.parent and compiled['release']==pin and
            compiled['case']==cert['case_sha256'] and compiled['flags']==doc['flags'] and
            compiled['namespace']==NAMESPACE and sha(artifact)==compiled['sha256'],'compile-to-run binding')
    out.mkdir(parents=True,exist_ok=False)
    dest=out/'capture'
    dest.mkdir()
    files=['capture/capture.json','linked.json','worker.log','worker-exit.txt','staged.sha256']
    files.extend(f'capture/{m}-{i}.npz' for m in ('control','split') for i in (1,2,3))
    staged={name:HERE/name for name in ('paired_worker.py','paired_launch.py')}
    staged.update({n:original.parents[1]/n for n in cert['original_host']})
    staged.update({name:case/name for name in ('case.json','control.npz','split.npz')})
    command=('sha256sum '+ ' '.join(staged) + ' > staged.sha256 && '
             'timeout --signal=INT --kill-after=15s 3600s cs_python paired_launch.py out case.json capture '
             '--cmaddr %CMADDR% > worker.log 2>&1; code=$?; echo "$code" >worker-exit.txt; exit "$code"')
    info=dict(release=pin,archive_sha256=compiled['sha256'],downloads={},status='pending')
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
                        info['downloads'][name]=dict(sha256=sha(out/name))
                    except Exception as error:
                        info['downloads'][name]=dict(error=repr(error))
                    save(out/'collection.json',info)
        require(all('error' not in x for x in info['downloads'].values()) and
                (out/'worker-exit.txt').read_text().strip()=='0','complete physical worker and downloads')
        pins=dict((n,h) for h,n in (line.split() for line in (out/'staged.sha256').read_text().splitlines()))
        require(pins=={n:sha(path) for n,path in staged.items()},'staged files')
        linked=load(out/'linked.json')
        require(linked['case_sha256']==cert['case_sha256'] and linked['worker_sha256']==sha(HERE/'paired_worker.py'),
                'loaded physical ABI/case binding')
        import sys
        sys.path.insert(0,str(original.parents[1]))
        import portable
        portable.admit_compiled_inventory(linked['elf'])
        if cert.get('sdk_measurement') == 'two stopped-core A frames':
            from sample_audit import compare_returned_A
            result=audit(case,dest,original,'hardware')
            compare_returned_A(case,dest)
        else:
            result=audit(case,dest,original,'hardware',sdk=case/'sim')
        save(out/'fp64.json',result)
        info.update(status='passed',per_mode_max_cycles={k:[f['local_interval_min_max'][1] for f in v]
                                                       for k,v in result['modes'].items()})
    except BaseException as error:
        info.update(status='failed',error=repr(error))
        raise
    finally:
        save(out/'collection.json',info)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','compile','run'])
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--release',required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--build',type=Path)
    a=p.parse_args()
    case=a.case.resolve()
    if a.action=='check':
        references(case,a.release)
    elif a.action=='compile':
        compile_case(case,a.release,a.out.resolve())
    else:
        run_case(case,a.release,a.out.resolve(),a.build.resolve())
