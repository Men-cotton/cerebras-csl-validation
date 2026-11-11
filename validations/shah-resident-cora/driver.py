"""Prepare and validate resident Shah-path GCN; SDK imports occur only at capture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import numpy as np
try:
    from . import oracle, proof
except ImportError:
    import oracle
    import proof

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ATOL, RTOL = 2e-5, 2e-4
FILES = ('driver.py', 'oracle.py', 'proof.py', 'layout.csl', 'pe.csl')
ROUTERS = ('graph_words', 'graph_len')
WORKERS = ('x_ptr', 'x_col', 'x_val', 'w0')
ACCUMULATORS = ('w1',)


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 << 20:
        raise ValueError('bounded regular file required: ' + str(path))
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x,np.generic) else x.tolist())+'\n')


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def load_npz(path):
    sha(path)
    with np.load(path, allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def identities():
    return {name:sha(HERE/name) for name in FILES}


def changed(original):
    return {**original, 'x':np.asarray(original['x'] * np.float32(-.75),np.float32)}


def role(name,p):
    r,f=p['R'],p['F']
    if name in ROUTERS: return 0,0,1,r
    if name in (*WORKERS,'h'): return 1,0,f,r
    if name in (*ACCUMULATORS,'pre','projected','logits'): return 1,r,f,r
    if name in ('stats','clocks'): return 0,0,f+1,2*r
    raise ValueError(name)


def region(array,name,p):
    x,y,w,h=role(name,p)
    return np.ascontiguousarray(array[y:y+h,x:x+w,:])


def prepare(out, kind, nodes, rows):
    out.mkdir(parents=True,exist_ok=False)
    original=oracle.load_original(kind=kind,nodes=nodes)
    packed=oracle.build_case(original,rows)
    np.savez_compressed(out/'original.npz',**{k:v for k,v in original.items() if isinstance(v,np.ndarray)})
    np.savez_compressed(out/'inputs.npz',**packed['arrays'])
    metadata={k:v for k,v in original.items() if not isinstance(v,np.ndarray)}
    write(out/'original.json',metadata)
    (out/'source').mkdir()
    for name in FILES: shutil.copyfile(HERE/name,out/'source'/name)
    raw_paths=[]
    if kind=='cora':
        for name in [*oracle.PINS,'research/experiments/data/E087/captures/checkpoint-original.json']:
            target=out/'raw'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target);raw_paths.append('raw/'+name)
    manifest=dict(schema=1,kind=kind,nodes=int(original['x'].shape[0]),params=packed['params'],
        metadata=packed['metadata'],source_sha256=identities(),
        files={n:sha(out/n) for n in ['original.npz','original.json','inputs.npz',*raw_paths]})
    write(out/'case.json',manifest)
    print(json.dumps(dict(status='PREPARED',params=manifest['params'],metadata=manifest['metadata'])))
    return manifest


def load_case(case):
    m=read(case/'case.json')
    for n,h in m['files'].items():
        if sha(case/n)!=h: raise ValueError('case drift '+n)
    for n,h in m['source_sha256'].items():
        if sha(case/'source'/n)!=h or sha(HERE/n)!=h: raise ValueError('source drift '+n)
    original={**read(case/'original.json'),**load_npz(case/'original.npz')}
    return m,original,load_npz(case/'inputs.npz')


def compiler_flags(p,hardware=False,channels=1):
    dims='762,1172' if hardware else f"{p['F']+8},{2*p['R']+2}"
    return ['--arch=wse3','--fabric-dims='+dims,'--fabric-offsets=4,1',
            '--params='+','.join(f'{k}:{v}' for k,v in p.items()),'--memcpy',
            '--channels='+str(channels),'--max-parallelism=1','-o','out']


def inventory(build):
    result=proof.compiled_inventory(build);proof.admit_compiled_inventory(result)
    return result


def compile_case(case):
    m,_,_=load_case(case)
    if (case/'build').exists(): raise ValueError('fresh build required')
    flags=compiler_flags(m['params']);flags[-1]=str(case/'build')
    command=[str(ROOT/'cslc'),str(case/'source/layout.csl'),*flags]
    record=dict(status='FAILED',command=command,case_sha256=sha(case/'case.json'))
    write(case/'compile-request.json',record);t=time.monotonic()
    try:
        subprocess.run(command,cwd=ROOT,check=True)
        record.update(inventory=inventory(case/'build'),status='PASS')
    finally:
        record['seconds']=time.monotonic()-t;write(case/'compile.json',record)
    print(json.dumps(dict(status=record['status'],seconds=record['seconds'],max_static_sram_end=record['inventory']['max_static_sram_end'])))


def unpack(raw,p,n):
    return raw.transpose(0,2,1).reshape(p['R']*p['V'],p['F'])[:n]


def upload_h(values,p):
    a=np.zeros((p['R'],p['F'],p['V']),np.float32)
    padded=np.zeros((p['R']*p['V'],p['F']),np.float32);padded[:len(values),:values.shape[1]]=values
    a[:]=padded.reshape(p['R'],p['V'],p['F']).transpose(0,2,1)
    return a


def capture(case,build,out,cmaddr=None,mode='all',run_id=None):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder
    m,original,arrays=load_case(case);p=m['params'];out.mkdir(parents=True,exist_ok=False)
    run_id=run_id or str(uuid.uuid4())
    if str(uuid.UUID(run_id))!=run_id:raise ValueError('run UUID')
    raw={};record=dict(run_id=run_id,clock_domain=f'pid:{os.getpid()}:monotonic:{run_id}',status='RUNNING',case_sha256=sha(case/'case.json'),mode=mode,backend='hardware' if cmaddr else 'simulator',api=[])
    rt=None
    compiled=read(case/'compile.json')
    actual_inventory=inventory(build)
    if compiled['status']!='PASS' or compiled['case_sha256']!=sha(case/'case.json') or actual_inventory!=compiled['inventory']:raise ValueError('actual runtime program differs from compilation')
    record['compiled_inventory']=actual_inventory
    write(out/'execution.json',record)
    def call(label,fn,*args,**kwargs):
        record['pending']=label;write(out/'execution.json',record);t=time.monotonic()
        value=fn(*args,**kwargs);record['api'].append(dict(api=label,seconds=time.monotonic()-t));record.pop('pending');write(out/'execution.json',record);return value
    try:
        rt=SdkRuntime(str(build),cmaddr=cmaddr,**({} if cmaddr else dict(suppress_simfab_trace=True,simfab_numthreads=1)))
        call('load',rt.load);call('run',rt.run)
        names=(*ROUTERS,*WORKERS,*ACCUMULATORS,'h','pre','projected','logits','stats','clocks')
        ids={n:rt.get_id(n) for n in names}
        def opts(a):
            return dict(streaming=False,nonblock=False,order=MemcpyOrder.ROW_MAJOR,data_type=MemcpyDataType.MEMCPY_16BIT if a.dtype==np.uint16 else MemcpyDataType.MEMCPY_32BIT)
        def upload(name,a):
            x,y,w,h=role(name,p);carrier=a.astype(np.uint32) if a.dtype==np.uint16 else a
            call('h2d:'+name,rt.memcpy_h2d,ids[name],carrier.ravel(),x,y,w,h,a.shape[-1],**opts(a))
        def download(name,extent,dtype):
            x,y,w,h=role(name,p);a=np.empty((h,w,extent),np.uint32 if dtype==np.uint16 else dtype)
            template=np.empty(0,dtype);call('d2h:'+name,rt.memcpy_d2h,a.ravel(),ids[name],x,y,w,h,extent,**opts(template))
            if dtype==np.uint16:
                if np.any(a>65535): raise AssertionError('nonzero upper u16 bits')
                a=a.astype(np.uint16)
            return a
        expected={n:region(arrays[n],n,p) for n in (*ROUTERS,*WORKERS,*ACCUMULATORS)}
        for name,a in expected.items(): upload(name,a)
        for name,a in expected.items():
            b=download(name,a.shape[-1],a.dtype);np.testing.assert_array_equal(b,a);raw['before_'+name]=b
        schedule=['gcn','ah16','ah7'] if mode=='all' else [mode]
        projections=oracle.project_inputs(original,R=p['R']) if mode!='gcn' else {}
        for kind in schedule:
            for repeat in range(2):
                key=kind+str(repeat)
                if kind=='gcn':
                    current=original if repeat==0 else changed(original)
                    a=oracle.build_case(current,p['R'])['arrays']['x_val'];expected['x_val']=region(a,'x_val',p);upload('x_val',expected['x_val'])
                else:
                    values=projections[kind]['h']
                    if repeat:values=(values*np.float32(-.5)).astype(np.float32)
                    upload('h',upload_h(values,p))
                call('launch:'+key,rt.launch,'f_run' if kind=='gcn' else 'f_ah',nonblock=False)
                for name in ('pre','projected','logits'):
                    raw[key+'_'+name]=download(name,p['V'],np.float32)
                raw[key+'_h']=download('h',p['V'],np.float32)
                raw[key+'_stats']=download('stats',8,np.uint32)
                raw[key+'_clocks']=download('clocks',6,np.uint16)
        for name,a in expected.items():
            b=download(name,a.shape[-1],a.dtype);np.testing.assert_array_equal(b,a);raw['after_'+name]=b
        np.savez_compressed(out/'raw.npz',**raw);record['status']='CAPTURED'
    except BaseException as error:
        record.update(status='FAILED',error=repr(error));raise
    finally:
        primary_error=sys.exc_info()[0] is not None
        if raw:np.savez_compressed(out/'raw.npz',**raw)
        try:
            if rt is not None:call('stop',rt.stop)
        except BaseException as error:
            record['stop_error']=repr(error)
            if not primary_error:raise
        finally:write(out/'execution.json',record)
    return audit(case,out)


def audit(case,out):
    m,o,arrays=load_case(case);p=m['params'];raw=load_npz(out/'raw.npz');r=read(out/'execution.json')
    if r['status']!='CAPTURED' or 'stop_error' in r or r.get('pending'): raise AssertionError('incomplete execution')
    lifecycle=[a['api'] for a in r['api'] if a['api'] in ('load','run','stop')]
    if lifecycle!=['load','run','stop'] or r['api'][-1]['api']!='stop':raise AssertionError('runtime lifecycle')
    if str(uuid.UUID(r['run_id']))!=r['run_id']:raise AssertionError('run UUID')
    compiled=read(case/'compile.json')
    if r['case_sha256']!=sha(case/'case.json') or compiled['case_sha256']!=r['case_sha256'] or r['compiled_inventory']!=compiled['inventory']:raise AssertionError('capture/compile/case identity')
    canonical=oracle.load_original(kind=m['kind'],nodes=m['nodes'],root=case/'raw')
    for name in ('x','src','dst','coeff','w0','w1','labels','valid'):
        if canonical.get(name) is None:
            if o.get(name) is not None:raise AssertionError('canonical optional input '+name)
        else:np.testing.assert_array_equal(o[name],canonical[name])
    if o.get('full')!=canonical.get('full'):raise AssertionError('full input scope')
    rebuilt=oracle.build_case(canonical,p['R'])
    if rebuilt['params']!=p or rebuilt['metadata']!=m['metadata']:raise AssertionError('packed metadata/source identity')
    for name in arrays:np.testing.assert_array_equal(arrays[name],rebuilt['arrays'][name])
    report={};n=m['nodes']
    projections=oracle.project_inputs(o,R=p['R']) if r['mode']!='gcn' else {}
    epoch=0
    for kind in (['gcn','ah16','ah7'] if r['mode']=='all' else [r['mode']]):
        for repeat in range(2):
            key=kind+str(repeat)
            epoch+=1
            if kind=='gcn':
                current=o if repeat==0 else changed(o);ref=oracle.independent_reference(current);checks={'pre':ref['pre'],'projected':ref['projected1'],'logits':ref['logits']}
            else:
                values=projections[kind]['h']
                if repeat:values=(values*np.float32(-.5)).astype(np.float32)
                ref=oracle.ah_reference(o,values);checks={'pre':ref}
            errors={}
            for name,wanted in checks.items():
                got=unpack(raw[key+'_'+name],p,n)
                np.testing.assert_allclose(got[:,:wanted.shape[1]],wanted,rtol=RTOL,atol=ATOL)
                np.testing.assert_array_equal(got[:,wanted.shape[1]:],0)
                padding=raw[key+'_'+name].transpose(0,2,1).reshape(p['R']*p['V'],p['F'])[n:]
                np.testing.assert_array_equal(padding,0)
                errors[name]=float(np.max(np.abs(got[:,:wanted.shape[1]].astype(np.float64)-wanted)))
            stats=raw[key+'_stats']
            expected_stats=np.zeros_like(stats);expected_stats[:,:,0]=0x600d;expected_stats[:,:,7]=epoch
            phases=2 if kind=='gcn' else 1
            for shard in range(p['R']):
                expected_stats[shard,0,1]=int(arrays['graph_len'][shard,0,0])//2*phases
                expected_stats[shard,1:,1]=m['metadata']['graph_nnz_per_source_shard'][shard]*phases
                expected_stats[shard,1:,2]=p['R']*p['V']*phases
                expected_stats[shard,1:,3]=p['R']*phases
                expected_stats[p['R']+shard,1:,4]=(p['R']-shard)*p['V']*phases
                expected_stats[p['R']+shard,1:,5]=(p['R']-shard-1)*p['V']*phases
                if kind=='gcn':
                    expected_stats[shard,1:,6]=m['metadata']['sparse_x_nnz_per_shard'][shard]
                    expected_stats[p['R']+shard,1:p['O']+1,6]=p['V']*p['F']
            np.testing.assert_array_equal(stats,expected_stats)
            if kind=='gcn':np.testing.assert_array_equal(raw[key+'_h'],raw[key+'_projected'])
            else:
                np.testing.assert_array_equal(raw[key+'_projected'],0);np.testing.assert_array_equal(raw[key+'_logits'],0)
                np.testing.assert_array_equal(raw[key+'_h'],upload_h(values,p))
            clocks=raw[key+'_clocks'].astype(np.uint64);start=clocks[:,:,0]+(clocks[:,:,1]<<16)+(clocks[:,:,2]<<32);end=clocks[:,:,3]+(clocks[:,:,4]<<16)+(clocks[:,:,5]<<32)
            spans=(end-start)&((1<<48)-1)
            if np.any(spans==0):raise AssertionError('empty PE clock interval')
            item=dict(max_abs_errors=errors,max_pe_cycles=int(spans.max()),stats_unique=np.unique(stats.reshape(-1,8),axis=0).tolist())
            if kind=='gcn' and o.get('full',False):
                pred=unpack(raw[key+'_logits'],p,n)[:,:p['O']].argmax(1);expected_pred=ref['logits'].argmax(1)
                np.testing.assert_array_equal(pred,expected_pred)
                item.update(input_scope='original' if repeat==0 else 'changed-X reset discriminator',prediction_disagreements=0,validation_correct=int(np.sum(pred[o['valid']]==o['labels'][o['valid']])) )
            report[key]=item
    expected={name:region(arrays[name],name,p) for name in (*ROUTERS,*WORKERS,*ACCUMULATORS)}
    for name,wanted in expected.items():
        np.testing.assert_array_equal(raw['before_'+name],wanted)
        if name=='x_val' and r['mode'] in ('all','gcn'):
            wanted=region(oracle.build_case(changed(o),p['R'])['arrays'][name],name,p)
        np.testing.assert_array_equal(raw['after_'+name],wanted)
    launches=[a['api'] for a in r['api'] if a['api'].startswith('launch:')]
    expected_launches=['launch:'+kind+str(i) for kind in (['gcn','ah16','ah7'] if r['mode']=='all' else [r['mode']]) for i in range(2)]
    if launches!=expected_launches:raise AssertionError('complete invocation schedule')
    result=dict(status='PASS',run_id=r['run_id'],case_sha256=sha(case/'case.json'),backend=r['backend'],observations=report,
        timing_scope='same-PE cycles and host blocking launch windows; no synchronized makespan or application E2E',raw_sha256=sha(out/'raw.npz'))
    write(out/'audit.json',result);print(json.dumps(result));return result


def simulate(case,out,mode):
    m,_,_=load_case(case);c=read(case/'compile.json')
    if c['status']!='PASS' or c['case_sha256']!=sha(case/'case.json') or inventory(case/'build')!=c['inventory']:raise ValueError('build identity')
    out.mkdir(parents=True,exist_ok=False);shutil.copytree(case/'build',out/'build')
    command=[str(ROOT/'cs_python'),'-B',str(HERE/'driver.py'),'capture','--case',str(case),'--build',str(out/'build'),'--out',str(out/'capture'),'--mode',mode]
    write(out/'request.json',dict(command=command,mode=mode))
    subprocess.run(command,cwd=ROOT,check=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=('prepare','compile','simulate','capture','audit'));a.add_argument('--case',type=lambda p:Path(p).resolve(),required=True);a.add_argument('--out',type=lambda p:Path(p).resolve());a.add_argument('--build',type=lambda p:Path(p).resolve());a.add_argument('--kind',choices=('tiny','cora'),default='tiny');a.add_argument('--nodes',type=int,default=2708);a.add_argument('--rows',type=int,default=2);a.add_argument('--mode',choices=('all','gcn','ah16','ah7'),default='all');a.add_argument('--cmaddr');args=a.parse_args()
    if args.action=='prepare':prepare(args.case,args.kind,args.nodes,args.rows)
    elif args.action=='compile':compile_case(args.case)
    elif args.action=='simulate':simulate(args.case,args.out,args.mode)
    elif args.action=='capture':capture(args.case,args.build,args.out,args.cmaddr,args.mode)
    else:audit(args.case,args.out)
if __name__=='__main__':main()
