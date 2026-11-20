"""Identify automatic versus fixed CS-3 Direct Link endpoints with two four-word controls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
HOST = ('appliance.py', 'probe.py', 'fixed_worker.py', 'driver.py', 'echo0.csl', 'echo1.csl')
EC = (0x61a72d90, 0x16c599af)
INPUT = ((1,0x80000000,19,0xffffffff), (0x40000000,7,0,31))
HISTORICAL = 'be48ea564742296b6d966b4828832a6fab21bdfbefe753d9e83ab2bf6a150f96'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def replay(root, role, build):
    if json.loads((root/'driver.json').read_bytes()) != {'automatic':0,'fixed':0}:
        raise ValueError('both stopped children are required')
    tests = {}
    for name in ('automatic', 'fixed'):
        folder = root/name
        data = json.loads((folder/'trace.json').read_bytes())
        if data['last_phase'] != 'stopped' or data['backend'] != role or data['masks'] != list(EC):
            raise ValueError('unstopped or unexpected SDK execution')
        if sha(folder/'port-map.json') != data['port_map_sha256']:
            raise ValueError('returned port map digest differs')
        raw = json.loads((folder/'port-map.json').read_bytes())
        measured = {}
        for bus in raw['buses']:
            port = raw['lvds_ports'][bus['lvds_id']]
            measured[bus['port_name']] = dict(io_type=bus['io_type'], **port)
        if measured != data['endpoints'] or len(measured)!=4 or len(data['names'])!=2:
            raise ValueError('endpoint mapping differs from the returned SDK map')
        expected = [dict(pair=i,input=list(words),output=[word^EC[i] for word in words])
                    for i,words in enumerate(INPUT)]
        if expected != data['transactions']:
            raise ValueError('eight returned data words differ')
        if name=='fixed':
            if (data['port_map_sha256']!=build['ports_sha256']
                    or data['physical_or_simulated_elf_sha256']!=build['roles'][role]['elf_sha256']
                    or data['historical_fabric_sha256']!=HISTORICAL):
                raise ValueError('fixed image, routes or historical descriptor differ')
        elif (not data['native_elf_sha256'] or
              data['sources']!={f'echo{i}.csl':sha(HERE/f'echo{i}.csl') for i in range(2)}):
            raise ValueError('unidentified native code')
        if 'fabric_report_sha256' in data and sha(folder/'fabric-connections.json')!=data['fabric_report_sha256']:
            raise ValueError('returned current fabric differs')
        tests[name] = dict(endpoints=measured, transactions=expected, trace_sha256=sha(folder/'trace.json'))
    return dict(role=role, controls=tests)


def check(case, pin):
    release = case/'release.json'
    if sha(release)!=pin:
        raise ValueError('release pin differs')
    manifest=json.loads(release.read_bytes())
    if manifest['host_sha256']!={name:sha(HERE/name) for name in HOST}:
        raise ValueError('changed probe programs')
    for name,checksum in manifest['file_sha256'].items():
        path=case/name
        if not path.resolve().is_relative_to(case.resolve()) or sha(path)!=checksum:
            raise ValueError('case file differs')
    build=json.loads((case/'build.json').read_bytes())
    if build['roles']['hardware']['dimensions']!=[762,1172] or build['historical_fabric_sha256']!=HISTORICAL:
        raise ValueError('unexpected physical target')
    if sha(case/'hardware/out/program.elf')!=build['roles']['hardware']['elf_sha256']:
        raise ValueError('the frozen physical image differs')
    if sha(case/'hardware/out/out_port_map.json')!=build['ports_sha256']:
        raise ValueError('physical route map differs')
    replay(case/'sdk','simulator',build)
    return build


def run(case, out, pin, namespace):
    from cerebras.sdk.client import SdkLauncher

    build=check(case,pin)
    out.mkdir(parents=True,exist_ok=False)
    bundle=out/'artifact'
    shutil.copytree(case/'hardware/out',bundle/'out')
    log=dict(namespace=namespace,release_sha256=pin,downloads={},outcome='pending')
    save(out/'collection.json',log)
    command=('timeout --signal=INT --kill-after=15s 1050s cs_python -B driver.py . out build.json capture '
             '--cmaddr %CMADDR% >worker.log 2>&1; rc=$?; echo "$rc" >worker-exit.txt; exit "$rc"')
    files=['worker-exit.txt','worker.log','capture/driver.json']
    files.extend('capture/'+name for name in
                 ('automatic.log','fixed.log','automatic/trace.json','fixed/trace.json',
                  'automatic/port-map.json','fixed/port-map.json','automatic/fabric-connections.json'))
    try:
        with SdkLauncher(str(bundle),simulator=False,mgmt_namespace=namespace,
                         resource_cpu=1000,resource_mem=8<<30,disable_version_check=False) as launcher:
            try:
                for name in HOST[1:]:
                    launcher.stage(str(HERE/name))
                launcher.stage(str(case/'build.json'))
                response=launcher.run(command)
                (out/'launcher-response.txt').write_text(str(response))
            finally:
                for name in files:
                    local=out/name
                    local.parent.mkdir(parents=True,exist_ok=True)
                    try:
                        launcher.download_artifact(name,str(local))
                        log['downloads'][name]=dict(sha256=sha(local))
                    except Exception as exc:
                        log['downloads'][name]=dict(error=repr(exc))
                    save(out/'collection.json',log)
        if (out/'worker-exit.txt').read_text().strip()!='0':
            raise RuntimeError('worker failed; individual phase markers and diagnostics are retained')
        save(out/'replay.json',replay(out/'capture','hardware',build))
        log['outcome']='passed'
    except BaseException as exc:
        log.update(outcome='failed',error=repr(exc))
        raise
    finally:
        save(out/'collection.json',log)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('check','run'))
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--release-sha256',required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--namespace',default='job-operator')
    a=p.parse_args()
    if a.action=='check':
        check(a.case.resolve(),a.release_sha256)
    elif a.out:
        run(a.case.resolve(),a.out.resolve(),a.release_sha256,a.namespace)
    else:
        p.error('run requires --out')
