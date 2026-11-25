"""Submit one resident runtime or replay its six raw completion and diagnostic frames."""
import argparse
import json
import logging
import shutil
import statistics
import uuid
from pathlib import Path

import numpy as np
from worker import CALLS, diagnostics, sha, witnesses

HERE = Path(__file__).resolve().parent
FILES = ['worker.log', 'worker-exit.txt', 'capture/capture.json', 'capture/initial.npy',
         *[f'capture/frame-{i}.npz' for i in range(1, CALLS + 1)]]


def read(path):
    return json.loads(path.read_text())


def verify(pin):
    release = HERE/'release.json'
    if sha(release) != pin:
        raise ValueError('incorrect release')
    for name, expected in read(release)['files'].items():
        if sha(HERE/name) != expected:
            raise ValueError(name)
    return read(HERE/'build.json')


def replay(folder, build, role, reference, build_hash, worker_hash):
    r = read(folder/'capture.json')
    if (r['role'], r['build_sha256'], r['worker_sha256'], r['stopped'], r['warmup'], r['repeats']) != (
            role, build_hash, worker_hash, True, 1, 5) or len(r['samples']) != CALLS:
        raise ValueError('runtime identity, stop, or repetition schedule')
    if str(uuid.UUID(r['runtime_id'])) != r['runtime_id']:
        raise ValueError('invalid runtime identifier')
    p = build['params']
    witnesses(np.load(folder/'initial.npy', allow_pickle=False), p, 0)
    times, peaks = [], []
    last_end = -1
    with np.load(reference, allow_pickle=False) as ref:
        for epoch, sample in enumerate(r['samples'], 1):
            with np.load(folder/f'frame-{epoch}.npz', allow_pickle=False) as raw:
                witnesses(raw['witness'], p, epoch)
                peak = diagnostics(raw['reply'], p, epoch, ref)
            start, dispatch, end = (sample[k] for k in ('start_ns', 'dispatched_ns', 'completed_ns'))
            if (sample['epoch'] != epoch or sample['peak_pe_cycles'] != peak
                    or not last_end < start <= dispatch <= end or start == end):
                raise ValueError('raw frames, epoch or host clocks disagree')
            last_end = end
            if epoch > 1:
                times.append(end-start)
                peaks.append(peak)
    return dict(runtime_id=r['runtime_id'], dispatch_to_all_pe_ns=times, median_ns=statistics.median(times),
                max_pe_cycles=peaks, median_max_pe_cycles=statistics.median(peaks))


def audit(out, build):
    return replay(out/'capture', build, 'hardware', HERE/'reference.npz',
                  sha(HERE/'build.json'), sha(HERE/'worker.py'))


def run(out, build):
    from cerebras.appliance import logger  # Install SDK job-ID logging.
    from cerebras.sdk.client import SdkLauncher
    logging.basicConfig(level=logging.INFO)
    out.mkdir(parents=True, exist_ok=False)
    # The SDK writes a sibling tarball when uploading a directory.
    bundle = out/'bundle'
    shutil.copytree(HERE/'bundle', bundle)
    for name, expected in (('program.elf', build['roles']['hardware']['elf_sha256']),
                           ('out_port_map.json', build['ports_sha256'])):
        if sha(bundle/'out'/name) != expected:
            raise ValueError(f'staged {name} differs')
    command = 'cs_python worker.py . capture --artifact out --cmaddr %CMADDR% >worker.log 2>&1; rc=$?; echo "$rc" >worker-exit.txt; exit "$rc"'
    with SdkLauncher(str(bundle), simulator=False, mgmt_namespace='job-operator',
                     resource_cpu=1000, resource_mem=8 << 30,
                     disable_version_check=False) as launcher:
        for name in ('worker.py', 'wire.py', 'reference.npz', 'build.json'):
            launcher.stage(str(HERE/name))
        missing = {}
        try:
            (out/'launcher-response.txt').write_text(str(launcher.run(command)))
        finally:
            for name in FILES:
                dest = out/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    launcher.download_artifact(name, str(dest))
                except Exception as error:
                    missing[name] = repr(error)
            (out/'missing.json').write_text(json.dumps(missing, indent=2)+'\n')
    if missing or (out/'worker-exit.txt').read_text().strip() != '0':
        raise RuntimeError('worker did not finish or captured files are incomplete')
    (out/'audit.json').write_text(json.dumps(audit(out, build), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check', 'run', 'summarize'])
    p.add_argument('--release-sha256', required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    build = verify(a.release_sha256)
    if a.action == 'run':
        run(a.output.resolve(), build)
    elif a.action == 'summarize':
        reports = [audit(a.output/f'runtime-{k}', build) for k in (1, 2)]
        if reports[0]['runtime_id'] == reports[1]['runtime_id']:
            raise ValueError('two distinct runtimes required')
        (a.output/'completion-summary.json').write_text(json.dumps(reports, indent=2)+'\n')
