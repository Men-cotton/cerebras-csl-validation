"""Run one previously verified ELF on one physical CS-3 via the CSL validation watcher."""
import argparse
import json
import logging
import os
from pathlib import Path
import shlex
import shutil

from audit import HERE, check_capture, load_verified, sha
from contract import WIDTHS


def run(a):
    root = a.build.resolve()
    if a.build_sha256 and sha(root/'build.json') != a.build_sha256:
        raise ValueError('request references a different paired build')
    build = load_verified(root, a.owners)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from cerebras.appliance import logger  # SDK job creation records are consumed by csl_validation.
    from cerebras.sdk.client import SdkLauncher
    logging.basicConfig(level=logging.INFO)
    os.chdir(out)
    command = shlex.join(['timeout','--signal=INT','--kill-after=10s','180s','cs_python','worker.py',
                          '--build','build.json','--artifact','out','--output','capture','--cmaddr','%CMADDR%'])
    command += ' >worker.log 2>&1; code=$?; echo "$code" > worker-exit.txt; exit "$code"'
    state = dict(owners=a.owners, namespace=a.namespace, artifact_sha256=build['roles']['hardware']['elf_sha256'],
                 build_sha256=sha(root/'build.json'), command=command, downloads={}, numerical_status='pending')
    def save():
        (out/'collection.json').write_text(json.dumps(state, indent=2)+'\n')
    save()
    # Only the exact two ELF/map inputs are passed to the SDK's bundle uploader.
    payload = out/'bundle'/'out'
    payload.mkdir(parents=True)
    for name in ('program.elf','out_port_map.json'):
        shutil.copyfile(root/'hardware/out'/name, payload/name)
    try:
        with SdkLauncher(str(payload.parent), simulator=False, mgmt_namespace=a.namespace,
                         resource_cpu=1000, resource_mem=8 << 30, disable_version_check=False) as launcher:
            try:
                for path in (HERE/'worker.py', HERE/'contract.py', root/'build.json'):
                    launcher.stage(str(path))
                response = launcher.run(command)
                (out/'launcher-response.txt').write_text(str(response))
            finally:
                for remote in ('worker.log','worker-exit.txt','capture/result.json'):
                    name = Path(remote).name
                    dest = out/name
                    try:
                        launcher.download_artifact(remote, str(dest))
                        state['downloads'][remote] = dict(file=name, sha256=sha(dest))
                    except Exception as exc:
                        state['downloads'][remote] = dict(error=repr(exc))
                    save()
        if any('error' in d for d in state['downloads'].values()):
            raise ValueError('incomplete worker capture')
        if (out/'worker-exit.txt').read_text().strip() != '0':
            raise ValueError('worker returned failure')
        capture = json.loads((out/'result.json').read_text())
        state['max_compute_cycles'] = check_capture(build, capture, 'hardware')
        state['numerical_status'] = 'passed'
    except BaseException as exc:
        state['error'] = repr(exc)
        raise
    finally:
        save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['run','check'])
    p.add_argument('--owners', choices=WIDTHS, type=int, required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--build-sha256')
    p.add_argument('--namespace', default='job-operator')
    a = p.parse_args()
    if a.action == 'check':
        if a.build_sha256 and sha(a.build/'build.json') != a.build_sha256:
            raise ValueError('build manifest changed')
        load_verified(a.build.resolve(), a.owners)
        print(f'P={a.owners} numerical and artifact checks passed')
    elif not a.out:
        p.error('run requires --out')
    else:
        run(a)
