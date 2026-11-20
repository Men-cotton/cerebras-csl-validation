"""Verify a trained two-hop partial-sum target and the two actual SDK controls."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from oracle import validate_reference
from worker import sha,unpack

HERE=Path(__file__).resolve().parent
BASE='full-c16r32-t8-cluster-mean-factor'
SPLIT=BASE+'-hub-split'


def module(name,source):
    spec=importlib.util.spec_from_file_location(name,source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def common():
    return module('published_fused_frame_checks',HERE.parent/'tungsten-graphsage-fused-scale/scale_check.py')


def check_physical_outputs(case,physical):
    common().check_physical_outputs(case,physical)


def verify_scale(case,proof_sha,standard_family):
    read=lambda p:json.loads(p.read_text())
    if sha(case/'proof.json')!=proof_sha:raise ValueError('hub request does not pin its proof')
    proof=read(case/'proof.json');b=read(case/'build.json');p=b['params']
    rows=p['R'];pes=16*rows;split=b['split_plan']
    if (rows not in (32,64) or case.name!=f'full-c16r{rows}-t8-cluster-mean-factor-hub-split'
            or (p['C'],p['T'],b['nodes'],b['edges'])!=(16,8,2708,10556)
            or split['original_hub_id']!=163 or split['helper_xy']!=[0,2] or split['actual_partial_words']!=23
            or split['moved']!={32:105,64:76}[rows] or proof['required_hardware_trials']!=[1,2,3]
            or proof['build_sha256']!=sha(case/'build.json') or proof['canonical_elf_sha256']!=b['canonical_sha256']
            or proof['ordered_loads']!=b['ordered_loads'] or not 0<proof['linked_bytes']<=44*1024):
        raise ValueError('wrong model, split layout, build or physical proof')
    shared_family=HERE.parent/'tungsten-graphsage-fused-scale'
    if (sha(shared_family/'scale_check.py')!=proof['frozen_frame_checker_sha256']
            or sha(shared_family/'worker.py')!=sha(HERE/'worker.py')):
        raise ValueError('published three-frame verification runtime changed')
    ctl=proof['complete_unsplit_control']
    full=standard_family/'cases'/BASE
    expected={'build.json','reference.npz','run/capture.json',*(f'run/frame-{i}.npz' for i in (1,2,3))}
    if (ctl['case']!=BASE or set(ctl['files'])!=expected or any(sha(full/n)!=h for n,h in ctl['files'].items())
            or set(ctl['runtime'])!={'appliance.py','worker.py','oracle.py'}
            or any(sha(standard_family/n)!=h for n,h in ctl['runtime'].items())
            or ctl['runtime']['worker.py']!=sha(HERE/'worker.py')):
        raise ValueError('published three-input unsplit control changed')
    module('registered_standard_hub_control',standard_family/'appliance.py').verify_case(full,ctl['files']['build.json'])
    original=read(full/'build.json')
    for k in ('original_checkpoint_sha256','nodes','edges','stream_packets','readout'):
        if b[k]!=original[k]:raise ValueError('graph, weights or packet mode differ from completed control')
    if rows==32 and original['node_ids_sha256']!=b['node_ids_sha256']:
        raise ValueError('512-PE split changed original node ownership')
    peer=HERE/'cases'/SPLIT
    sibling=proof['split_512_sdk_control']
    if (sibling['case']!=SPLIT or set(sibling['files'])!={'build.json','reference.npz','run/frame-1.npz'}
            or any(sha(peer/n)!=h for n,h in sibling['files'].items())):
        raise ValueError('verified split SDK control changed')
    pb=read(peer/'build.json')
    if (pb['source']!=b['source'] or pb['original_source']!=b['original_source']
            or pb['original_checkpoint_sha256']!=b['original_checkpoint_sha256']):
        raise ValueError('modified compute or physical readout differs from verified 512-PE split source')
    validate_reference(peer)
    with np.load(peer/'run/frame-1.npz',allow_pickle=False) as f,np.load(peer/'reference.npz',allow_pickle=False) as ref:
        verified=unpack(f['reply'],1,pb['params'],ref)
    if verified['reply_sha256']!=sibling['reply_sha256'] or verified['max_pe_cycles']!=sibling['cycles']:
        raise ValueError('calibrated split SDK frame differs')
    checks=proof['initialized_input_readbacks']
    if set(checks)!=set(b['input_buffers_sha256']) or 'split_info' not in checks:raise ValueError('incomplete input inventory')
    for name,c in checks.items():
        if (c['original']!=b['input_buffers_sha256'][name] or c['readback']!=c['expected']
                or (name!='tx_hdr' and c['original']!=c['expected'])):raise ValueError('split data changed: '+name)
    bare=case/'bare';result=read(bare/'result.json');measured=read(bare/'split-result.json')
    if (sha(bare/'result.json')!=b['original_result_sha256'] or sha(bare/'split-result.json')!=split['bare_result_sha256']
            or result['src_sha256']!=b['original_source'] or result['split_plan']!=split or measured['status']!='PASS'
            or not measured['split'] or measured['max_pe_cycles']!=result['maximum'] or result['maximum']!=b['original_cycles']
            or measured['partial_wire_words']!=23 or measured['argmax_disagreements']!=0):
        raise ValueError('bare calibrated split or input plan differs')
    for n in ('out-pre.bin','out-hidden.bin','out-logits.bin','out-clocks.bin','out-split_words.bin','out-split_info.bin'):
        if sha(bare/n)!=result['capture_sha256'][n] or sha(bare/n)!=measured['data_sha256'][n]:
            raise ValueError('bare retained capture changed: '+n)
    with np.load(case/'reference.npz',allow_pickle=False) as ref:
        for n in ('pre','hidden','logits'):
            val=np.fromfile(bare/('out-'+n+'.bin'),'<f4').reshape(pes,-1)
            np.testing.assert_array_equal(val.view('u4'),ref['bare_'+n].view('u4'))
            np.testing.assert_allclose(val,ref[n+'_1'],atol=2e-5,rtol=2e-4)
    clocks=np.fromfile(bare/'out-clocks.bin','<u2').reshape(pes,2,3).astype('i8')@np.array([1,65536,1<<32])
    ticks=(clocks[:,1]-clocks[:,0])%(1<<48)
    np.testing.assert_array_equal(ticks,result['cycles'])
    partial=np.zeros(pes,int);partial[[0,32]]=23
    np.testing.assert_array_equal(np.fromfile(bare/'out-split_words.bin','<u2'),partial)
    roles=np.fromfile(bare/'out-split_info.bin','<u2').reshape(pes,3)
    if roles[0,0]!=1 or roles[32,0]!=2 or int(roles[32,2])-int(roles[32,1])!=split['moved']:
        raise ValueError('helper occurrence interval differs')
    count=common().check_progress(case,b,proof)
    return dict(local_completed_trials=list(range(1,count+1)),local_stage=proof['local_stage'],
                complete_control=BASE,complete_control_trials=[1,2,3],split_control=SPLIT,split_control_trial=1,
                bare_cycles=int(ticks.max()),required_hardware_trials=[1,2,3],verified_inputs=len(checks))
