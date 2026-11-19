"""One physical run of a replay-verified trained GCN, with complete frame collection."""
import argparse
import json
import logging
import os
from pathlib import Path
import shlex
import shutil

import numpy as np

from worker import sha, unpack, wire_words

HERE = Path(__file__).resolve().parent
PORT_SHA = 'e4a311bc7f0f12ff52503703f4e233182c840339f765d01d49650022fe951d3e'
FRAMES = [f'frame-{n}.npz' for n in range(1,4)]


def read(path):
    return json.loads(path.read_text())


def worker_hash(case):
    path = case/'run/worker.sha256'
    if path.is_file():
        return path.read_text().strip().split()[0]
    return read(case/'run/capture.json')['worker_sha256']


def check_capture(build_path, ref_path, folder, role):
    build = read(build_path)
    captured = read(folder/'capture.json')
    if (captured['stage'] != 'complete' or captured['stop_success'] is not True
            or captured['backend'] != role or captured['case'] != build['case']
            or captured['build_sha256'] != sha(build_path)):
        raise ValueError('capture backend, model, build, or runtime completion differs')
    if [r['trial'] for r in captured['trials']] != [1,2,3]:
        raise ValueError('incomplete frame sequence')
    if 'worker_sha256' in captured and captured['worker_sha256'] != sha(HERE/'worker.py'):
        raise ValueError('capture worker source changed')
    if sha(ref_path) != build['comparison_reference_sha256']:
        raise ValueError('incorrect frozen reference')
    results = []
    p = build['params']
    reply_words = wire_words(p)
    with np.load(ref_path, allow_pickle=False) as reference:
        for name, trial in zip(FRAMES,captured['trials']):
            with np.load(folder/name, allow_pickle=False) as raw:
                if raw.files != ['reply']:
                    raise ValueError('unexpected frame array members')
                words = raw['reply']
            if words.dtype != np.dtype('<u4') or words.shape != (reply_words,):
                raise ValueError('received frame type or dimensions differ from ABI')
            actual = unpack(words, trial['trial'], p, reference)
            if any(trial[k] != v for k, v in actual.items()) or not trial['host_frame_ns'] > 0:
                raise ValueError('saved frame summary differs from collected payload or timing')
            results.append(dict(trial=actual['trial'], max_pe_cycles=actual['max_pe_cycles'],
                                reply_words=actual['reply_words'], reply_sha256=actual['reply_sha256'],
                                host_frame_ns=trial['host_frame_ns']))
    return results


def verify_case(case, build_sha):
    build_path = case/'build.json'
    if sha(build_path) != build_sha:
        raise ValueError('request and paired build manifest differ')
    build = read(build_path)
    p = build['params']
    if build['roles']['hardware']['fabric'] != [762,1172] or build['roles']['simulator']['fabric'] != [p['C'],p['R']]:
        raise ValueError('incorrect physical or simulator fabric')
    if sha(case/'gcn.img') != build['machine_image'] or build['ordered_loads'] <= 0:
        raise ValueError('paired machine image differs')
    if set(build['source']) != {'gcn.w','bench.paint'}:
        raise ValueError('incomplete rendered source')
    for name, pin in build['source'].items():
        if sha(case/'src'/name) != pin:
            raise ValueError('rendered kernel changed')
    for role in ('simulator','hardware'):
        out = case/role/'out'
        if (sha(out/'program.elf') != build['roles'][role]['elf_sha256'] or
                sha(out/'out_port_map.json') != build['ports_sha256'] or build['ports_sha256'] != PORT_SHA):
            raise ValueError('paired image or Direct Link port map changed')
    if {f.name for f in (case/'hardware/out').iterdir()} != {'program.elf','out_port_map.json'}:
        raise ValueError('extra hardware bundle members')
    if sha(HERE/'worker.py') != worker_hash(case):
        raise ValueError('worker changed since the verified simulator run')
    return check_capture(build_path,case/'reference.npz',case/'run','simulator')


def run(a):
    case, out = a.case.resolve(), a.out.resolve()
    verify_case(case, a.build_sha256)
    out.mkdir(parents=True, exist_ok=False)
    stage = out/'stage'
    stage.mkdir()
    payload = out/'bundle'/'out'
    payload.mkdir(parents=True)
    build = read(case/'build.json')
    staged = [(case/'hardware/out/program.elf',payload,build['roles']['hardware']['elf_sha256']),
              (case/'hardware/out/out_port_map.json',payload,PORT_SHA),
              (HERE/'worker.py',stage,worker_hash(case)),
              (case/'reference.npz',stage,build['comparison_reference_sha256']),
              (case/'build.json',stage,a.build_sha256)]
    for src, dest, expected in staged:
        target = dest/src.name
        shutil.copyfile(src,target)
        if sha(target) != expected:
            raise ValueError(f'staged {src.name} changed')
    os.chdir(out)
    from cerebras.appliance import logger  # Ensure SDK job creation goes through the normal log.
    from cerebras.sdk.client import SdkLauncher
    logging.basicConfig(level=logging.INFO)
    command = shlex.join(['timeout','--signal=INT','--kill-after=15s',f'{a.worker_timeout}s',
                          'cs_python','worker.py','.','capture','--artifact','out','--cmaddr','%CMADDR%'])
    command += ' >worker.log 2>&1; code=$?; echo "$code" >worker-exit.txt; exit "$code"'
    state = dict(case=build['case'], build_sha256=a.build_sha256, worker_sha256=worker_hash(case),
                 namespace=a.namespace, command=command, downloads={}, numerical_status='pending')
    def save():
        (out/'collection.tmp').write_text(json.dumps(state, indent=2)+'\n')
        (out/'collection.tmp').replace(out/'collection.json')
    save()
    try:
        with SdkLauncher(str(payload.parent), simulator=False, mgmt_namespace=a.namespace,
                         resource_cpu=1000, resource_mem=8 << 30, disable_version_check=False) as launcher:
            try:
                for src, dest, expected in staged:
                    if sha(dest/src.name) != expected:
                        raise ValueError(f'{src.name} changed during staging')
                    if dest == stage:
                        launcher.stage(str(dest/src.name))
                (out/'launcher-response.txt').write_text(str(launcher.run(command)))
            finally:
                for name in ('worker.log','worker-exit.txt','capture/capture.json',*['capture/'+f for f in FRAMES]):
                    dst = out/name
                    dst.parent.mkdir(parents=True,exist_ok=True)
                    try:
                        launcher.download_artifact(name,str(dst))
                        state['downloads'][name]=dict(bytes=dst.stat().st_size, sha256=sha(dst))
                    except Exception as exc:
                        state['downloads'][name]=dict(error=repr(exc))
                    save()
        if any('error' in d for d in state['downloads'].values()) or (out/'worker-exit.txt').read_text().strip() != '0':
            raise ValueError('hardware worker failed or artifact collection incomplete')
        state['trials'] = check_capture(stage/'build.json',stage/'reference.npz',out/'capture','hardware')
        state['numerical_status'] = 'passed'
    except BaseException as exc:
        state['error'] = repr(exc)
        raise
    finally:
        save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['run','check'])
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--build-sha256', required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--namespace', default='job-operator')
    p.add_argument('--worker-timeout', type=int, default=1800)
    a = p.parse_args()
    if a.action == 'check':
        print(json.dumps(verify_case(a.case.resolve(),a.build_sha256)))
    elif a.out is None or not 1 <= a.worker_timeout <= 86400:
        p.error('run requires a fresh output path and a 1..86400s worker timeout')
    else:
        run(a)
