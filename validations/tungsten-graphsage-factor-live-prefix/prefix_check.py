"""Verify a live two-frame factor-1024 SDK snapshot and its complete fixed-source control."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from worker import sha,unpack


def verify_prefix(case,prefix_sha,standard_family):
    def read(path):
        return json.loads(path.read_text())
    if sha(case/'prefix.json')!=prefix_sha:
        raise ValueError('the manifest does not pin this snapshot')
    proof=read(case/'prefix.json')
    local=read(case/'run/capture.json')
    b=read(case/'build.json')
    files=('capture.json','frame-1.npz','frame-2.npz')
    if (case.name!='full-c16r64-t8-cluster-mean-factor' or (case/'run/frame-3.npz').exists()
            or proof['local_completed_trials']!=[1,2] or proof['hardware_required_trials']!=[1,2,3]
            or proof['hardware_numerical_references']!=[1,2,1] or proof['sdk_status_at_snapshot']!='frame-3'
            or set(proof['local_files'])!=set(files) or any(sha(case/'run'/n)!=proof['local_files'][n] for n in files)
            or local['stage']!='frame-3' or local['stop_success'] is not False
            or local['backend']!='simulator' or local['case']!=b['case']
            or [t['trial'] for t in local['trials']]!=[1,2]
            or local['worker_sha256']!=sha(Path(__file__).with_name('worker.py'))
            or local['build_sha256']!=sha(case/'build.json') or local['build_sha256']!=proof['build_sha256']):
        raise ValueError('invalid two-frame live simulator snapshot')
    p=b['params'];ctl=proof['complete_control']
    if (p['C'],p['R'],p['T'],b['nodes'],b['edges'],b['stream_packets'])!=(16,64,8,2708,10556,False):
        raise ValueError('unexpected cluster target or packet transport')
    if (b['canonical_sha256']!=proof['paired_canonical_sha256'] or b['ordered_loads']!=proof['ordered_loads']
            or b['ordered_loads']<=0 or proof['max_linked_bytes']>44*1024
            or b['original_result_sha256']!=proof['bare_result_sha256']):
        raise ValueError('paired ELF or bare comparison proof differs')
    if ctl['rows'] not in (16,32) or ctl['case']!=f"full-c16r{ctl['rows']}-t8-cluster-mean-factor":
        raise ValueError('unexpected completed cluster factor control')
    target=standard_family/'cases'/ctl['case']
    control_files=('build.json','reference.npz','run/capture.json','run/frame-1.npz','run/frame-2.npz','run/frame-3.npz')
    family_files=('appliance.py','worker.py','oracle.py')
    if (set(ctl['files'])!=set(control_files) or set(ctl['family_sources'])!=set(family_files)
            or any(sha(target/n)!=ctl['files'][n] for n in control_files)
            or any(sha(standard_family/n)!=ctl['family_sources'][n] for n in family_files)
            or ctl['family_sources']['worker.py']!=local['worker_sha256']):
        raise ValueError('completed control or frozen family changed')
    old=read(target/'build.json')
    for k in ('source','original_source','original_checkpoint_sha256','nodes','edges','readout','stream_packets'):
        if b[k]!=old[k]:raise ValueError('control source, original model or packet transport differs: '+k)
    if (old['params']['C'],old['params']['R'],old['params']['T'])!=(16,ctl['rows'],8):
        raise ValueError('wrong completed control geometry')
    spec=importlib.util.spec_from_file_location('complete_factor_control',standard_family/'appliance.py')
    complete=importlib.util.module_from_spec(spec);spec.loader.exec_module(complete)
    complete.verify_case(target,ctl['files']['build.json'])
    with np.load(case/'reference.npz',allow_pickle=False) as ref:
        for t in local['trials']:
            with np.load(case/'run'/f"frame-{t['trial']}.npz",allow_pickle=False) as doc:
                if doc.files!=['reply']:raise ValueError('unexpected local packet members')
                actual=unpack(doc['reply'],t['trial'],p,ref)
            if any(t[k]!=v for k,v in actual.items() if k!='errors') or t['host_frame_ns']<=0:
                raise ValueError('local packet metadata or timing differs')
            for k in actual['errors']:
                if t['errors'][k]['max_abs']!=actual['errors'][k]['max_abs']:
                    raise ValueError('local layer error differs')
                np.testing.assert_allclose(t['errors'][k]['relative_l2'],actual['errors'][k]['relative_l2'],rtol=1e-12,atol=0)
    return dict(local_completed_trials=[1,2],local_stage='frame-3 in progress at snapshot',
                reference_control=ctl['case'],control_completed_trials=[1,2,3],hardware_required_trials=[1,2,3],
                trials=local['trials'])
