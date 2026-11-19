"""Drive three resident GCN frames and validate every returned PE record."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpack(reply, trial, p, ref):
    c,r,l,h,o,k,t = (p[n] for n in ('C','R','L','H','O','K','T'))
    records = reply.reshape(c*r, -1)
    xy = np.arange(c*r)
    np.testing.assert_array_equal(records[:,0], np.full(c*r,trial))
    np.testing.assert_array_equal(records[:,1], xy % c)
    np.testing.assert_array_equal(records[:,2], xy // c)
    clk = records[:,3:9].astype(np.int64)
    if np.any(clk > 65535):
        raise ValueError('invalid 48-bit clock chunk')
    cycles = clk @ np.array([-1,-65536,-(1<<32),1,65536,1<<32]) % (1<<48)
    if not np.all((0 < cycles) & (cycles < 8_000_000)):
        raise ValueError('invalid inference interval')
    stats = records[:,9:17]
    np.testing.assert_array_equal(stats,ref['stats_history'])
    pos = 17
    for name,width in [('pre_history',l*h),('logits_history',l*o),('panel_history',k*t)]:
        got = records[:,pos:pos+width]
        want = ref[name]*np.float32(0.5 if trial == 2 else 1)
        np.testing.assert_array_equal(got, want.view(np.uint32), err_msg=name)
        pos += width
    if pos != records.shape[1]:
        raise ValueError('unexpected payload length')
    return dict(trial=trial, per_pe_cycles=cycles.tolist(), max_pe_cycles=int(cycles.max()),
                reply_words=reply.size, reply_sha256=hashlib.sha256(reply.tobytes()).hexdigest())


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
    result = dict(backend=role,case=record['case'],build_sha256=sha(root/'build.json'),trials=[],stop_success=False)
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
        n=c*r*(17+p['L']*(p['H']+p['O'])+p['K']*p['T'])
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
