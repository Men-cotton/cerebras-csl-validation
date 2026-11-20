"""Measure four- and eight-row native broadcasts on a fixed GraphSAGE window."""

import argparse
import importlib.util
from pathlib import Path
import sys

FAMILY = Path(__file__).resolve().parent
VALIDATIONS = FAMILY.parent
BASE = VALIDATIONS / 'ogb-parallel-ports'
NATIVE = BASE / 'native'
sys.path[:0] = [str(NATIVE), str(BASE)]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = load('row_scale_native', NATIVE / 'physical.py')
eight = load('row_scale_eight', VALIDATIONS / 'ogb-eight-ports/physical.py')


def check(key, pin):
    if Path(key).name != key:
        raise ValueError('expected a case name')
    case = FAMILY / 'cases' / key
    release = native.read(case / 'release.json')
    native.require(native.sha(case / 'release.json') == pin, 'native release pin')
    for root, files in ((case, release['files']), (VALIDATIONS, release['host'])):
        for name, expected in files.items():
            p = (root / name).resolve()
            native.require(p.is_relative_to(root) and native.sha(p) == expected, 'changed frozen ' + name)
    sdk = native.read(case / 'build.json')
    rows = sdk['params']['L']
    if rows == 4:
        peer_dir, driver = BASE, native.appliance
    elif rows == 8:
        peer_dir, driver = VALIDATIONS / 'ogb-eight-ports', eight.driver
    else:
        raise ValueError('expected four or eight native rows')
    peer = peer_dir / 'cases' / key
    desc, expected = driver.check(peer, release['tungsten_release_sha256'])
    fixture = peer / 'fixture.npz'
    native.require(desc['lanes'] == rows and desc['owners_per_lane'] == sdk['params']['P'] and rows * sdk['params']['P'] == 16, 'matched 16-PE allocation')
    native.require(desc['fixture_sha256'] == sdk['fixture_sha256'] == native.sha(fixture), 'matched original fixture')
    audit, _ = native.replay(fixture, case, case/'sdk', expected)
    native.require(audit['role'] == 'simulator' and len(audit['frames']) == 12, 'complete native SDK control')
    return case, release, sdk, fixture, expected


native.check = check

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('compile','run','check'))
    parser.add_argument('--case', required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--compiled', type=Path)
    parser.add_argument('--namespace', default='job-operator')
    args = parser.parse_args()
    if args.action != 'check' and not args.out:
        parser.error('compile and run need an output directory')
    if args.action == 'run' and not args.compiled:
        parser.error('run needs the original compiled record')
    native.execute(args)
