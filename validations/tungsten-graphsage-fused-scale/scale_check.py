"""Check source-equivalent fused scaling with the completed 256-PE SDK control."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from worker import sha,unpack,words_per_frame


CONTROL='full-c16r16-t8-cluster-mean-fused-44k'
CONTROL_FILES=('build.json','reference.npz','run/capture.json','run/frame-1.npz','run/frame-2.npz','run/frame-3.npz')
FAMILY_FILES=('appliance.py','worker.py','oracle.py')


def check_progress(case,b,proof):
    record=json.loads((case/'run/capture.json').read_text())
    count=len(record['trials'])
    if (record['backend']!='simulator' or record['case']!=b['case'] or record['build_sha256']!=sha(case/'build.json')
            or record['worker_sha256']!=sha(Path(__file__).with_name('worker.py'))
            or [t['trial'] for t in record['trials']]!=list(range(1,count+1)) or count>3
            or proof['local_completed_trials']!=list(range(1,count+1)) or proof['local_stage']!=record['stage']
            or proof['local_stop_success']!=record['stop_success']):
        raise ValueError('local SDK snapshot does not match its physical-scale record')
    if count<3 and (record['stage']!=f'frame-{count+1}' or record['stop_success'] is not False):
        raise ValueError('unfinished local SDK trial misreported')
    if count==3 and (record['stage']!='complete' or record['stop_success'] is not True):
        raise ValueError('local completion requires a proper runtime stop')
    expected={'capture.json'}|{f'frame-{i+1}.npz' for i in range(count)}
    if (set(proof['local_files'])!=expected or {f.name for f in (case/'run').iterdir()}!=expected
            or any(sha(case/'run'/n)!=proof['local_files'][n] for n in expected)):
        raise ValueError('local snapshot inventory differs')
    with np.load(case/'reference.npz',allow_pickle=False) as ref:
        for t in record['trials']:
            with np.load(case/'run'/f"frame-{t['trial']}.npz",allow_pickle=False) as f:
                now=unpack(f['reply'],t['trial'],b['params'],ref)
            if any(t[k]!=v for k,v in now.items() if k!='errors'):raise ValueError('SDK reply metadata differs')
            for k in now['errors']:
                np.testing.assert_allclose([t['errors'][k]['max_abs'],t['errors'][k]['relative_l2']],
                                           [now['errors'][k]['max_abs'],now['errors'][k]['relative_l2']],rtol=1e-12,atol=0)
    return count


def verify_scale(case,proof_sha,standard_family):
    if sha(case/'proof.json')!=proof_sha:raise ValueError('the request does not pin this proof')
    rec=json.loads((case/'proof.json').read_text())
    b=json.loads((case/'build.json').read_text())
    p=b['params'];rows=p['R'];npes=p['C']*rows
    if (rows not in (32,64) or case.name!=f'full-c16r{rows}-t8-cluster-mean-fused-44k'
            or (p['C'],p['T'],b['nodes'],b['edges'])!=(16,8,2708,10556)
            or rec['required_hardware_trials']!=[1,2,3] or rec['build_sha256']!=sha(case/'build.json')
            or rec['canonical_elf_sha256']!=b['canonical_sha256'] or rec['ordered_loads']!=b['ordered_loads']
            or rec['linked_bytes']<=0 or rec['linked_bytes']>44*1024 or not b['shared_row_scratch']
            or b['stream_packets'] or b['row_weight_storage']['verified_edges']!=b['edges']):
        raise ValueError('unexpected geometry, algorithm or linked memory proof')
    ctl=rec['complete_control']
    path=standard_family/'cases'/CONTROL
    if (ctl['case']!=CONTROL or set(ctl['files'])!=set(CONTROL_FILES) or set(ctl['runtime'])!=set(FAMILY_FILES)
            or any(sha(path/n)!=ctl['files'][n] for n in CONTROL_FILES)
            or any(sha(standard_family/n)!=ctl['runtime'][n] for n in FAMILY_FILES)
            or ctl['runtime']['worker.py']!=sha(Path(__file__).with_name('worker.py'))):
        raise ValueError('completed 256-PE SDK control changed')
    control=json.loads((path/'build.json').read_text())
    for k in ('original_source','source','original_checkpoint_sha256','nodes','edges','shared_row_scratch','readout','stream_packets'):
        if b[k]!=control[k]:raise ValueError('target and completed SDK source or model differ: '+k)
    if (control['params']['C'],control['params']['R'],control['params']['T'])!=(16,16,8):
        raise ValueError('control geometry differs')
    spec=importlib.util.spec_from_file_location('fused_complete_control',standard_family/'appliance.py')
    complete=importlib.util.module_from_spec(spec);spec.loader.exec_module(complete)
    complete.verify_case(path,ctl['files']['build.json'])
    input_keys=set(b['input_buffers_sha256'])
    if set(rec['initialized_input_readbacks'])!=input_keys:raise ValueError('incomplete initialized PE input set')
    for k,check in rec['initialized_input_readbacks'].items():
        if (check['original']!=b['input_buffers_sha256'][k] or check['expected']!=check['readback']
                or (k not in ('edge_weight','tx_hdr') and check['expected']!=check['original'])
                or (k=='edge_weight' and check['expected']!=b['row_weight_storage']['staged_sha256'])):
            raise ValueError('initial SRAM readback disagrees with input: '+k)
    bare=case/'bare'
    original=json.loads((bare/'result.json').read_text())
    if (sha(bare/'result.json')!=b['original_result_sha256'] or original['status']!='PASS' or original['row_mean']!='fused'
            or original['case_sha256']!=b['original_fixture_sha256'] or original['src_sha256']!=b['original_source']
            or original['params']!=p or original['maximum']!=b['original_cycles']):
        raise ValueError('incorrect retained bare calibration')
    with np.load(case/'reference.npz',allow_pickle=False) as ref:
        for k in ('pre','hidden','logits'):
            name='out-'+k+'.bin'
            if sha(bare/name)!=original['capture_sha256'][name]:raise ValueError('bare layer changed: '+k)
            fp=np.fromfile(bare/name,'<f4').reshape(npes,-1)
            np.testing.assert_array_equal(fp.view('u4'),ref['bare_'+k].view('u4'))
            np.testing.assert_allclose(fp,ref[k+'_1'],rtol=2e-4,atol=2e-5)
    if sha(bare/'out-clocks.bin')!=original['capture_sha256']['out-clocks.bin']:raise ValueError('bare clock capture changed')
    clk=np.fromfile(bare/'out-clocks.bin','<u2').reshape(npes,2,3).astype('i8')@np.array([1,65536,1<<32])
    d=(clk[:,1]-clk[:,0])%(1<<48)
    np.testing.assert_array_equal(d,original['cycles'])
    if int(d.max())!=original['maximum']:raise ValueError('bare clock maximum differs')
    completed=check_progress(case,b,rec)
    return dict(local_completed_trials=list(range(1,completed+1)),local_stage=rec['local_stage'],
                original_max_pe_cycles=int(d.max()),verified_input_symbols=len(input_keys),required_hardware_trials=[1,2,3],
                complete_control=CONTROL,complete_control_trials=[1,2,3])


def check_physical_outputs(case,physical):
    b=json.loads((case/'build.json').read_text())
    p=b['params'];width=words_per_frame(p)//(p['C']*p['R'])
    def numeric(folder,trial):
        with np.load(folder/f'frame-{trial}.npz',allow_pickle=False) as raw:
            return raw['reply'].reshape(-1,width)[:,9:].copy()
    np.testing.assert_array_equal(numeric(physical,1),numeric(physical,3))
    if np.array_equal(numeric(physical,1),numeric(physical,2)):
        raise ValueError('half input did not change the physical output')
    local=json.loads((case/'run/capture.json').read_text())
    for t in local['trials']:
        np.testing.assert_array_equal(numeric(physical,t['trial']),numeric(case/'run',t['trial']))
