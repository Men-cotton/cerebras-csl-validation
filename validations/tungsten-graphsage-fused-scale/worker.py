"""Drive three resident GraphSAGE frames and validate every returned PE record."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words_per_frame(p):
    return p['C']*p['R']*(10+p['L']*(2*p['H']+p['O']))


def unpack(reply,trial,p,ref):
    c,r,l,h,o=(p[k] for k in ('C','R','L','H','O'))
    if reply.dtype!=np.dtype('<u4') or reply.shape!=(words_per_frame(p),):
        raise ValueError('unexpected GraphSAGE frame shape or dtype')
    records=reply.reshape(c*r,-1)
    idx=np.arange(c*r)
    np.testing.assert_array_equal(records[:,0],np.full(c*r,trial))
    np.testing.assert_array_equal(records[:,1],idx%c)
    np.testing.assert_array_equal(records[:,2],idx//c)
    np.testing.assert_array_equal(records[:,9],np.ones(c*r))
    clk=records[:,3:9].astype(np.int64)
    if np.any(clk>65535):raise ValueError('invalid 48-bit clock chunk')
    duration=clk@np.array([-1,-65536,-(1<<32),1,65536,1<<32])%(1<<48)
    if not np.all((duration>0)&(duration<8_000_000)):raise ValueError('invalid inference interval')
    pos=10
    errors={}
    nodes=ref['nodes']
    valid=nodes>=0
    for name,width in [('pre',h),('hidden',h),('logits',o)]:
        got=records[:,pos:pos+l*width].copy().view('<f4')
        expected=ref[f'{name}_{2 if trial==2 else 1}']
        if trial!=2:
            np.testing.assert_array_equal(got.view('<u4'),ref['bare_'+name].view('<u4'))
        np.testing.assert_allclose(got,expected,atol=2e-5,rtol=2e-4)
        delta=got.astype('f8')-expected
        errors[name]=dict(max_abs=float(np.max(abs(delta))),relative_l2=float(np.linalg.norm(delta)/max(1e-30,np.linalg.norm(expected))))
        np.testing.assert_array_equal(got.reshape(*nodes.shape,width)[~valid].view('<u4'),0)
        if name=='logits':
            got_prediction=np.empty(int(valid.sum()),dtype=np.int64)
            got_prediction[nodes[valid]]=got.reshape(*nodes.shape,o)[valid].argmax(1)
            want_prediction=expected.reshape(*nodes.shape,o)[valid].argmax(1)
            np.testing.assert_array_equal(got_prediction[nodes[valid]],want_prediction)
        pos+=l*width
    scores={key:[int(np.count_nonzero(got_prediction[ref[key]]==ref['labels'][ref[key]])),len(ref[key])]
            for key in ('train_idx','val_idx','test_idx')}
    return dict(trial=trial,per_pe_cycles=duration.tolist(),max_pe_cycles=int(duration.max()),reply_words=reply.size,
                reply_sha256=hashlib.sha256(reply.tobytes()).hexdigest(),
                numerical_words_sha256=hashlib.sha256(records[:,10:].copy().tobytes()).hexdigest(),
                errors=errors,all_node_argmax_disagreements=0,accuracy=scores)


def run(root,out,cmaddr,artifact=None):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkCompileArtifacts, SdkRuntime, SdkTarget, SimfabConfig, get_platform
    record=json.loads((root/'build.json').read_text())
    role='hardware' if cmaddr else 'simulator'
    artifact = artifact or root/role/'out'
    if sha(artifact/'program.elf') != record['roles'][role]['elf_sha256'] or sha(artifact/'out_port_map.json') != record['ports_sha256']:
        raise ValueError('staged ELF or port map hash mismatch')
    if sha(root/'reference.npz') != record['comparison_reference_sha256']:
        raise ValueError('frozen reference mismatch')
    out.mkdir(parents=True,exist_ok=False)
    os.chdir(out)
    result = dict(backend=role,case=record['case'],build_sha256=sha(root/'build.json'),worker_sha256=sha(Path(__file__).resolve()),trials=[],stop_success=False)
    def stage(name):
        result['stage']=name
        (out/'progress.tmp').write_text(json.dumps(result,indent=2)+'\n')
        (out/'progress.tmp').replace(out/'capture.json')
    platform = get_platform(cmaddr,SimfabConfig(num_threads=1,suppress_trace=True),SdkTarget.WSE3)
    if bool(platform.is_system()) != bool(cmaddr) or bool(platform.is_simulation()) == bool(cmaddr):
        raise ValueError('runtime platform mismatch')
    image = SdkCompileArtifacts(str(artifact)).add_port_mapping(str(artifact/'out_port_map.json'))
    stage('constructor')
    dev = SdkRuntime(image,platform,memcpy_required=False,msg_level='INFO')
    p = record['params']
    c,r=p['C'],p['R']
    if any(tuple(dev.coord_logical_to_physical((x,y))) != (x,y) for y in range(r) for x in range(c)):
        raise ValueError('packet header coordinates do not match the physical allocation')
    stage('load')
    dev.load()
    dev.run()
    try:
        dev.report_port_infos()
        rx,tx=dev.get_port_id('rx_port'),dev.get_port_id('tx_port')
        n=words_per_frame(p)
        with np.load(root/'reference.npz') as ref:
            for trial,factor in enumerate([1,0.5,2],1):
                stage(f'frame-{trial}')
                send=np.array([factor],dtype='<f4').view('<u4')
                guarded=np.full(n+4,0xdeadc0de,np.uint32)
                reply=guarded[2:-2]
                start=perf_counter_ns()
                read=dev.receive(tx,reply,n,nonblock=True)
                write=dev.send(rx,send,1,nonblock=True)
                dev.task_wait(write)
                dev.task_wait(read)
                end=perf_counter_ns()
                if not np.all(guarded[[0,1,-2,-1]]==0xdeadc0de):
                    raise ValueError('receive guards overwritten')
                np.savez_compressed(out/f'frame-{trial}.npz',reply=reply)
                result['trials'].append(dict(unpack(reply,trial,p,ref),host_frame_ns=end-start))
    finally:
        stage('stop')
        dev.stop()
        result['stop_success']=True
        stage('stopped')
    stage('complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build',type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cmaddr')
    parser.add_argument('--artifact',type=Path)
    a=parser.parse_args()
    run(a.build.resolve(),a.output.resolve(),a.cmaddr,a.artifact.resolve() if a.artifact else None)
