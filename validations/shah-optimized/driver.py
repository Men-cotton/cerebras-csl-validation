#!/usr/bin/env python3
"""Compile/run the independent paper streaming reconstruction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import oracle

SOURCE = Path(__file__).resolve().parent

def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [SOURCE/n for n in ('driver.py','oracle.py','layout.csl','pe.csl')]}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['compile', 'run'])
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--result-out', type=Path)
    for k, v in [('R',2),('F',2),('V',3),('Q',2),('B',3),('channels',2)]:
        p.add_argument('--'+k, type=int, default=v)
    p.add_argument('--empty', action='store_true')
    p.add_argument('--workload', choices=['adversary','random'], default='adversary')
    p.add_argument('--degree', type=int, default=8)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--output-mode', choices=['buffered','per-chunk'], default='buffered')
    a=p.parse_args(); out=a.out.resolve()
    if a.action == 'compile':
        ROOT = SOURCE.parents[3]
        out.mkdir(parents=True,exist_ok=True)
        if (out/'build').exists(): raise ValueError('fresh output required')
        par=dict(R=a.R,F=a.F,V=a.V,Q=a.Q,B=a.B)
        case=make_case(dict(params=par,empty=a.empty,workload=a.workload,degree=a.degree,seed=a.seed))
        if max(a.R+a.B, a.F+1, a.V, a.Q) > 32760 or a.channels not in (1,2,4):
            raise ValueError('prototype i16 dimensions or channel limit exceeded')
        params={k:case['params'][k] for k in ['R','F','V','Q','B']}
        cmd=[str(ROOT/'cslc'),'--arch=wse3',str(SOURCE/'layout.csl'),
             '--fabric-dims=%d,%d'%(a.F+8,a.R+a.B+2),'--fabric-offsets=4,1',
             '--params='+','.join('%s:%d'%i for i in params.items())+',MEMCPYH2D_DATA_1_ID:0,MEMCPYD2H_DATA_1_ID:1','--memcpy',
             '--channels='+str(a.channels),'--max-parallelism=1','-o',str(out/'build')]
        record=dict(params=params,channels=a.channels,empty=a.empty,workload=a.workload,degree=a.degree,seed=a.seed,source_sha256=hashes(),command=cmd,metrics=case['metrics'])
        (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        start=time.monotonic(); subprocess.run(cmd,cwd=out,check=True,env={**os.environ,'CSL_IMPORT_PATH':str(SOURCE)})
        record['compile_seconds']=time.monotonic()-start
        if hashes()!=record['source_sha256']: raise ValueError('source changed during compile')
        (out/'host').mkdir(exist_ok=True)
        for name in record['source_sha256']: shutil.copy2(SOURCE/name,out/'host'/name)
        record['artifact_sha256']={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (out/'build').rglob('*') if f.is_file()}
        record['io_layout']={side:json.loads((out/'build'/side/'out.json').read_text())['params'] for side in ('west','east')}
        record['artifact_bytes']=sum(f.stat().st_size for f in (out/'build').rglob('*') if f.is_file())
        (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        return
    record=json.loads((out/'record.json').read_text())
    if hashes()!=record['source_sha256']: raise ValueError('source drift')
    for f,h in record['artifact_sha256'].items():
        if hashlib.sha256((out/f).read_bytes()).hexdigest()!=h: raise ValueError('artifact drift')
    run_case(record,out/'build',a.result_out or out,output_mode=a.output_mode)

def make_case(record):
    if record.get('workload','adversary') == 'random':
        return oracle.random_fixture(**record['params'],seed=record.get('seed',7),degree=record.get('degree',8))
    return oracle.fixture(**record['params'],empty=record.get('empty',False))

def run_case(record,build,out,cmaddr=None,output_mode='buffered'):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    if output_mode not in ('buffered','per-chunk'): raise ValueError('unknown output mode')
    if hashes()!=record['source_sha256']: raise ValueError('source drift')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if any((out/n).exists() for n in ('capture.npz','result.json')):
        raise ValueError('fresh result directory required')
    par=record['params']; R,F,V,Q,B=[par[k] for k in ['R','F','V','Q','B']]
    case=make_case(record)
    rt=SdkRuntime(str(build),cmaddr=cmaddr,**({} if cmaddr else dict(suppress_simfab_trace=True,simfab_numthreads=1)))
    kw=dict(data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR)
    calls=[]
    def invoke(name,*args,**kwargs):
        start=time.monotonic(); getattr(rt,name)(*args,**kwargs)
        calls.append(dict(api=name,seconds=time.monotonic()-start,streaming=kwargs.get('streaming'),nonblock=kwargs.get('nonblock')))
    rt.load(); rt.run()
    try:
        h=np.zeros((R,F,V),np.float32)
        for r in range(R):
            for f in range(F):
                for v in range(V):
                    if r*V+v < case['h'].shape[0]: h[r,f,v]=case['h'][r*V+v,f]
        invoke('memcpy_h2d',rt.get_id('h'),h.ravel(),1,0,F,R,V,streaming=False,nonblock=False,**kw)
        start=time.monotonic()
        result=np.zeros((B,F,Q),np.float32)
        for b,stream in enumerate(case['streams']):
            invoke('memcpy_h2d',0,stream.ravel(),0,0,1,R,stream.shape[1],streaming=True,nonblock=True,**kw)
            if output_mode=='per-chunk':
                invoke('memcpy_d2h',result[b].ravel(),1,1,R+b,F,1,Q,streaming=True,nonblock=False,**kw)
        if output_mode=='buffered':
            invoke('memcpy_d2h',result.ravel(),1,1,R,F,B,Q,streaming=True,nonblock=False,**kw)
        elapsed=time.monotonic()-start
        stats=np.zeros((R+B,F+1,4),np.uint32)
        invoke('memcpy_d2h',stats.ravel(),rt.get_id('stats'),0,0,F+1,R+B,4,streaming=False,nonblock=False,**kw)
        actual=result.transpose(0,2,1).reshape(B*Q,F)
        np.savez(out/'capture.npz',actual=actual,reference=case['reference'],stats=stats,h=case['h'],edges=np.array(case['edges']))
        np.testing.assert_allclose(actual,case['reference'],rtol=2e-5,atol=2e-6)
        for r in range(R):
            nz=sum(1 for _,c,_ in case['edges'] if r*V<=c<(r+1)*V)
            np.testing.assert_array_equal(stats[r,1:,:3],np.tile([nz,B*Q,B],(F,1)))
            if not np.all((stats[r,1:,3]>=B-1)&(stats[r,1:,3]<=B)):
                raise AssertionError('worker reuse completion counter')
            pairs=sum(len(stream[r])//2 for stream in case['streams'])
            padding=sum(np.count_nonzero(stream[r,::2]==oracle.EMPTY) for stream in case['streams'])
            np.testing.assert_array_equal(stats[r,0],[pairs,nz,B*Q,padding])
        for b in range(B):
            np.testing.assert_array_equal(stats[R+b,1:,:],np.tile([(B-b)*Q,(B-b-1)*Q,Q,1],(F,1)))
        report=dict(status='PASS',params=par,channels=record['channels'],output_mode=output_mode,workload=record.get('workload','adversary'),metrics={**case['metrics'],'d2h_output_calls':1 if output_mode=='buffered' else B},host_stream_completion_seconds=elapsed,
                    max_abs_error=float(np.max(np.abs(actual-case['reference']))),calls=calls,source_sha256=hashes())
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        return report
    finally: rt.stop()

if __name__=='__main__': main()
