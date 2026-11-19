"""Compile one layout or run one previously compiled owner-aggregation image."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
FILES = ('layout.csl', 'pe.csl', 'worker.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flags(owners, mode, physical):
    dims = (762,1172) if physical else (owners+7,3)
    return ['--arch=wse3', f'--fabric-dims={dims[0]},{dims[1]}', '--fabric-offsets=4,1',
            f'--params=P:{owners},MODE:{mode}', '--memcpy', '--channels=1', '--max-parallelism=1',
            '--width-west-buf=0', '--width-east-buf=0', '-o', 'out']


def options(args):
    return dict(mgmt_namespace=args.namespace, resource_cpu=1000, resource_mem=8<<30,
                disable_version_check=False)


def compile_case(args):
    args.out.mkdir(parents=True, exist_ok=False)
    args_list = flags(args.owners, args.mode, not args.local)
    record = dict(owners=args.owners, mode=args.mode, local=args.local, flags=args_list,
                  source_sha256={name: sha(ROOT/name) for name in FILES})
    if args.local:
        cmd = [args.compiler, str(ROOT/'layout.csl'), *args_list]
        with (args.out/'compile.log').open('wb') as log:
            subprocess.run(cmd, cwd=args.out, env=dict(os.environ, CSL_IMPORT_PATH=str(ROOT)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        artifact = args.out/'out'
        record['files'] = {str(p.relative_to(artifact)): sha(p) for p in sorted(artifact.rglob('*')) if p.is_file()}
    else:
        from cerebras.sdk.client import SdkCompiler
        source = args.out/'source'
        source.mkdir()
        for name in FILES[:2]:
            (source/name).write_bytes((ROOT/name).read_bytes())
        with SdkCompiler(**options(args)) as compiler:
            artifact = Path(compiler.compile(str(source), 'layout.csl', ' '.join(args_list), str(args.out))).resolve()
        if artifact.parent != args.out or not artifact.is_file():
            raise ValueError('unexpected compiler artifact')
        record.update(artifact_sha256=sha(artifact), namespace=args.namespace)
    record['artifact'] = artifact.name
    (args.out/'build.json').write_text(json.dumps(record, indent=2)+'\n')


def run_case(args):
    record = json.loads(args.build.read_text())
    if (record['local'],record['owners'],record['mode']) != (args.local,args.owners,args.mode):
        raise ValueError('compiled parameters or backend differ')
    if record['source_sha256'] != {name: sha(ROOT/name) for name in FILES}:
        raise ValueError('source has changed since compilation')
    artifact = args.build.parent/record['artifact']
    args.out.mkdir(parents=True, exist_ok=False)
    os.chdir(args.out)
    if args.local:
        if record['files'] != {str(p.relative_to(artifact)): sha(p) for p in artifact.rglob('*') if p.is_file()}:
            raise ValueError('compiled tree has changed')
        import worker
        worker.execute(artifact, args.out/'result.json')
    else:
        from cerebras.sdk.client import SdkLauncher
        if record['namespace'] != args.namespace or sha(artifact) != record['artifact_sha256']:
            raise ValueError('compiled hardware artifact or namespace differs')
        with SdkLauncher(str(artifact), simulator=False, **options(args)) as launcher:
            launcher.stage(str(ROOT/'worker.py'))
            command = ('sha256sum worker.py > worker-sha256.txt && '
                       'timeout --signal=INT --kill-after=10s 300s cs_python worker.py '
                       'out result.json --cmaddr %CMADDR% > worker.log 2>&1; '
                       'code=$?; echo "$code" > worker-exit.txt; exit "$code"')
            saved = {}
            try:
                launcher.run(command)
            finally:
                for name in ('worker.log', 'worker-exit.txt', 'worker-sha256.txt', 'result.json'):
                    dest = args.out/name
                    try:
                        launcher.download_artifact(name, str(dest))
                        saved[name] = {'sha256': sha(dest)}
                    except Exception as error:
                        saved[name] = {'error': repr(error)}
                (args.out/'collection.json').write_text(json.dumps(saved, indent=2)+'\n')
        if any('error' in entry for entry in saved.values()):
            raise ValueError('incomplete CS-3 capture')
        if (args.out/'worker-exit.txt').read_text().strip() != '0':
            raise ValueError('worker failed')
        if (args.out/'worker-sha256.txt').read_text().split()[0] != record['source_sha256']['worker.py']:
            raise ValueError('different worker executed')
    result = json.loads((args.out/'result.json').read_text())
    verify(result, args.owners, args.mode, args.local)


def verify(result, owners, mode, local):
    import numpy as np
    import worker
    if (result['backend'], result['owners'], result['mode']) != ('local-sim' if local else 'hardware', owners, mode):
        raise ValueError('worker backend or topology differs')
    if [t['trial'] for t in result['trials']] != list(range(1,12+1)):
        raise ValueError('missing or repeated trial')
    setup = result.get('setup_cycles')
    if mode >= 5 and (not isinstance(setup, list) or len(setup) != owners or min(setup) <= 0):
        raise ValueError('directory setup time missing')
    if mode < 5 and setup is not None:
        raise ValueError('unexpected directory setup time')
    for t in result['trials']:
        x = worker.features(t['trial'])
        y = np.asarray(t['output'], dtype=np.float32)
        np.testing.assert_array_equal(t['features'], x)
        np.testing.assert_array_equal(y.view(np.uint32), worker.oracle(x).view(np.uint32))
        clocks = np.array(t['clocks'], dtype=np.int64)
        if clocks.shape != (owners,6) or np.any((clocks < 0) | (clocks > 65535)):
            raise ValueError('bad clock words')
        cycles = clocks @ np.array([-1,-65536,-(1<<32),1,65536,1<<32]) % (1<<48)
        np.testing.assert_array_equal(t['cycles'], cycles)
        if not np.all(cycles):
            raise ValueError('empty timed interval')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('compile','run'))
    p.add_argument('--owners', type=int, choices=(1,2,4,8,16), required=True)
    p.add_argument('--mode', type=int, choices=range(8), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--build', type=Path)
    p.add_argument('--local', action='store_true')
    p.add_argument('--compiler', default='cslc')
    p.add_argument('--namespace', default='job-operator')
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.action == 'run' and not a.build:
        p.error('run requires --build')
    if a.build:
        a.build = a.build.resolve()
    logging.basicConfig(level=logging.INFO)
    if not a.local:
        from cerebras.appliance import logger  # Attach SDK creation log for the CSL watcher.
    (compile_case if a.action == 'compile' else run_case)(a)
