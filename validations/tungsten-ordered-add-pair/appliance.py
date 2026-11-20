"""Check the calibrated same-image comparison or execute it once on CS-3."""
import argparse
import json
from pathlib import Path
import shlex
import shutil

from contract import PORT, read, require, sha, verify

HERE=Path(__file__).resolve().parent
CAPTURE=('worker.log','worker-exit.txt','capture/capture.json','capture/raw.npz','capture/audit.json')


def save(path,record):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)


def verify_case(case,pin):
    require(sha(case/'release.json')==pin,'registered release hash differs')
    release=read(case/'release.json')
    for name,expected in release['files'].items():
        target=case/name
        require(target.resolve().is_relative_to(case.resolve()) and target.is_file() and sha(target)==expected,'changed case file '+name)
    for name,expected in release['host'].items():
        require(name in ('appliance.py','contract.py','worker.py') and sha(HERE/name)==expected,'changed host source')
    m=read(case/'build.json')
    require(m['source_sha256']=={p.name:sha(p) for p in (case/'src').iterdir()} and sha(case/'pair.img')==m['machine_image_sha256']
            and sha(case/'compute.yeti')==m['compiler_listing_sha256'] and sha(case/'baseline/sum.w')==m['baseline_source_sha256']
            and sha(case/'baseline/reproduce.py')==m['input_generator_sha256'],'machine image, source or listing differs')
    for role,dims in [('simulator',[1,2]),('hardware',[762,1172])]:
        artifact=case/role/'out'
        require(m['roles'][role]['fabric']==dims and sha(artifact/'program.elf')==m['roles'][role]['elf_sha256'] and
                sha(artifact/'out_port_map.json')==m['ports_sha256']==PORT,'ELF or Direct Link ports differ')
    require(verify(case/'build.json',case/'fixture.npz',case/'sdk','simulator')==read(case/'sdk/audit.json'),'SDK recalibration differs')
    return m


def run(case,pin,out,namespace):
    from cerebras.sdk.client import SdkLauncher
    m=verify_case(case,pin)
    out.mkdir(parents=True,exist_ok=False)
    artifact=out/'artifact';shutil.copytree(case/'hardware/out',artifact/'out')
    record=dict(case=case.name,release_sha256=pin,build_sha256=sha(case/'build.json'),elf_sha256=m['roles']['hardware']['elf_sha256'],
                namespace=namespace,numerical_status='pending',downloads={})
    command=shlex.join(['timeout','--signal=INT','--kill-after=30s','300s','cs_python','-B','worker.py','build.json',
                       'fixture.npz','out','capture','--cmaddr','%CMADDR%'])
    command+=' >worker.log 2>&1; rc=$?; echo "$rc" >worker-exit.txt; exit "$rc"'
    (out/'capture').mkdir()
    save(out/'collection.json',record)
    try:
        with SdkLauncher(str(artifact),simulator=False,mgmt_namespace=namespace,
                         resource_cpu=1000,resource_mem=8<<30,disable_version_check=False) as launcher:
            try:
                for path in (HERE/'worker.py',HERE/'contract.py',case/'build.json',case/'fixture.npz'):launcher.stage(str(path))
                response=launcher.run(command);(out/'response.txt').write_text(str(response))
            finally:
                for name in CAPTURE:
                    try:
                        launcher.download_artifact(name,str(out/name))
                        record['downloads'][name]=dict(sha256=sha(out/name),bytes=(out/name).stat().st_size)
                    except Exception as exc:record['downloads'][name]=dict(error=repr(exc))
                    save(out/'collection.json',record)
        require(all('sha256' in item for item in record['downloads'].values()) and
                (out/'worker-exit.txt').read_text().strip()=='0','incomplete collection or worker failure')
        independent=verify(case/'build.json',case/'fixture.npz',out/'capture','hardware')
        require(independent==read(out/'capture/audit.json'),'replayed output or PE clocks differ')
        record.update(numerical_status='passed',median_expression=independent['median_expression'],median_staged=independent['median_staged'])
    except BaseException as exc:
        record['error']=repr(exc)
        raise
    finally:save(out/'collection.json',record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('check','run'))
    p.add_argument('--case',type=Path,required=True);p.add_argument('--release-sha256',required=True)
    p.add_argument('--out',type=Path);p.add_argument('--namespace',default='job-operator')
    a=p.parse_args()
    if a.action=='check':verify_case(a.case.resolve(),a.release_sha256)
    elif a.out is None:p.error('run requires --out')
    else:run(a.case.resolve(),a.release_sha256,a.out.resolve(),a.namespace)
