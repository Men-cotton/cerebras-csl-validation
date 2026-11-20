"""Admit the frozen two-frame SDK prefix and its three-frame cross-geometry control."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from worker import sha, unpack, words_per_frame


def verify_prefix(case, prefix_sha, standard_family):
    evidence = case/'prefix.json'
    if sha(evidence) != prefix_sha:
        raise ValueError('request does not bind this prefix')
    proof = json.loads(evidence.read_text())
    if (proof['local_completed_trials'] != [1,2] or proof['hardware_required_trials'] != [1,2,3]
            or proof['hardware_numerical_references'] != [1,2,1]):
        raise ValueError('incorrect interrupted-prefix contract')
    expected = {'run/capture.json','run/frame-1.npz','run/frame-2.npz','run/replay.json'}
    if set(proof['local_files']) != expected or (case/'run/frame-3.npz').exists():
        raise ValueError('expected exactly the two archived local replies')
    for name, pin in proof['local_files'].items():
        if sha(case/name) != pin:
            raise ValueError('prefix archive changed: '+name)
    build = json.loads((case/'build.json').read_text())
    summary = json.loads((case/'run/capture.json').read_text())
    interrupted = json.loads((case/'run/replay.json').read_text())
    if (summary['backend'] != 'simulator' or summary['case'] != build['case']
            or summary['build_sha256'] != sha(case/'build.json') or summary['stage'] != 'frame-3'
            or summary.get('stop_success') is not False
            or [t['trial'] for t in summary['trials']] != [1,2]
            or interrupted['capture_sha256'] != sha(case/'run/capture.json')
            or interrupted['trials'] != 2 or interrupted['stage'] != 'frame-3'
            or interrupted['image_sha256'] != build['roles']['simulator']['elf_sha256']
            or interrupted['build_sha256'] != summary['build_sha256']
            or interrupted['worker_sha256'] != summary['worker_sha256']):
        raise ValueError('stored capture does not identify the interrupted third trial')
    control = proof['three_frame_control']
    if control['case'] != 'full-c16r32-t8-stream':
        raise ValueError('unexpected control geometry')
    if set(control['family_sources']) != {'appliance.py','worker.py','oracle.py'}:
        raise ValueError('incomplete control runtime inventory')
    control_files = {'build.json','reference.npz','run/capture.json',
                     'run/frame-1.npz','run/frame-2.npz','run/frame-3.npz'}
    if set(control['files']) != control_files:
        raise ValueError('incomplete control frame inventory')
    for name, pin in control['family_sources'].items():
        if sha(standard_family/name) != pin:
            raise ValueError('published complete-control runtime changed: '+name)
    if (summary['worker_sha256'] != sha(standard_family/'worker.py')
            or summary['worker_sha256'] != sha(Path(__file__).with_name('worker.py'))):
        raise ValueError('cross-geometry control did not use the same worker')
    regular_case = standard_family/'cases'/control['case']
    for name, pin in control['files'].items():
        if sha(regular_case/name) != pin:
            raise ValueError('published three-frame control changed: '+name)
    original = json.loads((regular_case/'build.json').read_text())
    for field in ('source','original_checkpoint_sha256','nodes','edges','readout','stream_packets'):
        if build[field] != original[field]:
            raise ValueError('control and target differ: '+field)
    if (build['params']['C'],build['params']['R'],build['params']['T']) != (16,64,8):
        raise ValueError('unexpected target geometry')
    if (original['params']['C'],original['params']['R'],original['params']['T']) != (16,32,8):
        raise ValueError('unexpected completed geometry')
    spec = importlib.util.spec_from_file_location('published_complete_graphsage',standard_family/'appliance.py')
    complete = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(complete)
    complete.verify_case(regular_case,control['files']['build.json'])
    with np.load(case/'reference.npz',allow_pickle=False) as frozen:
        for trial in summary['trials']:
            with np.load(case/'run'/f"frame-{trial['trial']}.npz",allow_pickle=False) as data:
                if data.files != ['reply']:
                    raise ValueError('unexpected raw reply arrays')
                words = data['reply']
                if words.dtype != np.dtype('<u4') or words.shape != (words_per_frame(build['params']),):
                    raise ValueError('malformed reply')
                actual = unpack(words,trial['trial'],build['params'],frozen)
            if any(trial[k] != val for k,val in actual.items() if k != 'errors') or trial['host_frame_ns'] <= 0:
                raise ValueError('incomplete archived trial or metadata mismatch')
            for tensor in ('pre','hidden','logits'):
                if trial['errors'][tensor]['max_abs'] != actual['errors'][tensor]['max_abs']:
                    raise ValueError('archived absolute error differs: '+tensor)
                np.testing.assert_allclose(trial['errors'][tensor]['relative_l2'],
                                           actual['errors'][tensor]['relative_l2'],rtol=1e-12,atol=0)
    return dict(local_completed_trials=[1,2],local_stage='interrupted during trial 3',
                reference_control=control['case'],control_completed_trials=[1,2,3],
                hardware_required_trials=[1,2,3],trials=summary['trials'])
