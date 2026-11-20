"""Alternate replicated unicast and row broadcast on a single loaded native program."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
if not (HERE/'source_table.py').exists():sys.path.insert(0,str(HERE.parent))
from inventory import actual, read, require, save, sha
from source_table import TRIALS, assign, audit, device, source_tiles


def sequence():
    for i,(method,scale) in enumerate(TRIALS):
        for delivery in (('row','replicated') if i%2==0 else ('replicated','row')):
            yield i,method,scale,delivery


def original_reply(reply,tag,trial):
    require(np.all(reply[:,0]==tag),'unique physical transaction tag')
    logical=reply.copy()
    logical[:,0]=trial+1
    return logical


def execute(build,fixture,out,cmaddr,artifact):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder, SdkTarget, SimfabConfig, get_platform
    desc=read(build/'build.json')
    role='hardware' if cmaddr else 'simulator'
    require(role==desc['role'] and sha(fixture)==desc['fixture_sha256'],'input and target binding')
    inventory=actual(artifact)
    elf={Path(n).name:h for n,h in inventory.items() if n.startswith('bin/') and n.endswith('.elf')}
    require(elf==desc['elf_sha256'] and inventory==desc['inventory'],'loaded application or memcpy helper mismatch')
    require({k:int(v) for k,v in read(artifact/'out.json')['params'].items()}==desc['csl_params'],'loaded parameters')
    worker={key:sha(HERE/key if (HERE/key).is_file() else HERE.parent/key) for key in desc['worker_sha256']}
    require(worker==desc['worker_sha256'],'staged worker family')
    target=get_platform(cmaddr,SimfabConfig(num_threads=1,suppress_trace=True),SdkTarget.WSE3)
    platform=dict(system=bool(target.is_system()),simulation=bool(target.is_simulation()))
    require(platform==dict(system=bool(cmaddr),simulation=not cmaddr),'bound SDK target')
    z=device.load(fixture);cfg,groups,owner,slot=assign(z,desc['params']['P'],desc['params']['E'])
    require(cfg==desc['params'],'original graph geometry')
    p,r,f,q=(cfg[k] for k in ('P','R','F','Q'))
    tables={method:source_tiles(z,cfg,owner,slot,method) for method,_ in TRIALS[:2]}
    weights=np.zeros((2*f+1,q),np.float32);classes=z['wr'].shape[1]
    weights[:f,:classes]=z['wr'];weights[f:2*f,:classes]=z['wn'];weights[-1,:classes]=z['b']
    nrows=np.array(list(map(len,groups)),np.uint32)
    ids=np.full((p,r),0xffffffff,np.uint32);inv=np.zeros((p,r),np.float32);roots=np.zeros((p,r,f),np.float32)
    for pe,group in enumerate(groups):
        n=len(group)
        ids[pe,:n]=z['seed'][group];inv[pe,:n]=1/np.maximum(np.diff(z['ptr'])[group],1)
        roots[pe,:n]=z['x'][z['root'][group]]
    out.mkdir(parents=True,exist_ok=False)
    record=dict(role=role,platform=platform,build_sha256=sha(build/'build.json'),fixture_sha256=sha(fixture),
                inventory=inventory,workers=worker,frames=[],stopped=False)
    save(out/'result.json',record)
    rt=SdkRuntime(str(artifact),cmaddr=cmaddr,**({} if cmaddr else dict(simfab_numthreads=1,suppress_simfab_trace=True)))
    opts=dict(nonblock=False,streaming=False,order=MemcpyOrder.ROW_MAJOR,data_type=MemcpyDataType.MEMCPY_32BIT)
    symbols={key:rt.get_id(key) for key in ('weights','features','control','indices','inverse','roots','targets','reply','edge_ticks')}
    def upload(name,a):
        a=np.ascontiguousarray(a).reshape(p,-1)
        rt.memcpy_h2d(symbols[name],a.ravel(),0,0,p,1,a.shape[1],**opts)
        return a.size
    def receive(name,n):
        a=np.empty((p,n),np.uint32)
        rt.memcpy_d2h(a.ravel(),symbols[name],0,0,p,1,n,**opts)
        return a
    def deliver(name,values,delivery,events):
        a=np.ascontiguousarray(values).ravel()
        block=a if delivery=='row' else np.tile(a,p)
        call=rt.memcpy_h2d_rowbcast if delivery=='row' else rt.memcpy_h2d
        # Allocation, np.tile and symbol lookup are outside this host interval.
        t0=time.perf_counter_ns()
        call(symbols[name],block,0,0,p,1,a.size,**opts)
        t1=time.perf_counter_ns()
        events.append(dict(symbol=name,source_words=int(block.size),delivered_words=int(p*a.size),start_ns=t0,end_ns=t1))
        return int(block.size)
    previous={}
    rt.load();rt.run()
    try:
        for tag,(i,method,scale,delivery) in enumerate(sequence(),1):
            calls=[]
            ctrl=np.zeros((p,3),np.uint32);ctrl[:,0]=tag;ctrl[:,1]=nrows
            start=time.perf_counter_ns()
            setup=deliver('weights',weights,delivery,calls) if i==0 else 0
            setup+=upload('control',ctrl)+upload('indices',ids)+upload('inverse',inv)+upload('roots',roots*scale)
            rt.launch('begin_frame',nonblock=False)
            selectors=table=0
            for sources,owners in tables[method]:
                table+=deliver('features',z['x'][sources]*scale,delivery,calls)
                count=np.array(list(map(len,owners)),np.uint32)
                extent=int(count.max());require(extent<=desc['csl_params']['M'],'selector bank capacity')
                refs=np.zeros((p,extent),np.uint32)
                for pe,w in enumerate(owners):refs[pe,:len(w)]=w
                selectors+=upload('targets',refs);ctrl[:,2]=count;setup+=upload('control',ctrl)
                rt.launch('accumulate',nonblock=False)
            rt.launch('finish',nonblock=False)
            reply=receive('reply',16+r*(1+q));ticks=receive('edge_ticks',1)
            last=time.perf_counter_ns()
            y,measure=audit(original_reply(reply,tag,i),z,cfg,groups,i)
            if scale in previous:np.testing.assert_array_equal(y.view(np.uint32),previous[scale])
            else:previous[scale]=y.view(np.uint32).copy()
            path=out/f'frame-{tag:02d}.npz'
            np.savez_compressed(path,reply=reply,logits=y,node=z['seed'],active_edge_cycles=ticks)
            record['frames'].append(dict(tag=tag,trial=i+1,method=method,scale=scale,delivery=delivery,calls=calls,
                    start_ns=start,end_ns=last,source_words=table,selector_words=selectors,input_words=table+selectors+setup,
                    measured=measure,peak_active_edge_cycles=int(ticks.max()),raw_sha256=sha(path)))
            save(out/'result.json',record)
    finally:rt.stop()
    record['stopped']=True;save(out/'result.json',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build',type=Path);parser.add_argument('fixture',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--cmaddr');parser.add_argument('--artifact',type=Path,required=True)
    a=parser.parse_args();execute(a.build.resolve(),a.fixture.resolve(),a.output.resolve(),a.cmaddr,a.artifact.resolve())
