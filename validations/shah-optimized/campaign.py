#!/usr/bin/env python3
"""Generate the predeclared E231 parameter batch; this never submits SDK jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import appliance
import driver


def configurations():
    anchor = dict(R=8, F=16, V=32, Q=16, B=8, channels=4, workload='balanced', degree=2, seed=7)
    groups = {}
    def add(axis, label, both=False, **overrides):
        groups.setdefault(axis, []).append(dict(label=label, both=both, config={**anchor, **overrides}))
    for c in (1, 2, 4):
        for w in ('balanced', 'paired-skew'):
            add('channels', f'c{c}-{w}', both=c==4 and w=='balanced', channels=c, workload=w)
    for r in (16, 32, 64):
        for w in ('balanced', 'paired-skew'):
            add('rows', f'r{r}-{w}', R=r, workload=w)
    add('rows', 'r64-c1-balanced', R=64, channels=1)
    for f in (8, 32, 64, 128):
        add('features', f'f{f}', F=f)
    for q,b in ((4,32),(8,16),(32,4),(64,2)):
        add('chunks', f'q{q}-b{b}', both=q in (4,64), Q=q, B=b)
    for d in (1,4,16):
        for w in ('balanced','paired-skew'):
            add('degree', f'd{d}-{w}', degree=d, workload=w)
    for r,c in ((8,1),(8,4),(16,4)):
        add('legacy', f'r{r}-c{c}', both=r==8 and c==4, R=r,F=r,V=2*r,Q=r,B=r,
            channels=c,workload='random',degree=8)
    for c,w in ((1,'balanced'),(4,'balanced'),(4,'paired-skew')):
        add('frontier',f'c{c}-{w}', R=4,F=256,V=512,Q=1024,B=2,channels=c,workload=w)
    return groups


def manifest(axis, cases):
    request_id = f'shah-optimized-{axis}-20261110-r001'
    steps=[]
    def step(name,action,args,kind):
        steps.append(dict(name=name,script='validations/shah-optimized/appliance.py',
                          args=[action,*args],kind=kind))
    def root(c): return '{output}/'+c['label']
    for c in cases:
        args=[]
        for key,value in c['config'].items(): args += ['--'+key,str(value)]
        step(c['label']+'-compile','compile',args+['--out',root(c)+'/compile'],'appliance')
    # Same ELF/input for each repetition; reverse config and schedule order in round2.
    for rep in range(1,4):
        ordered=cases if rep%2 else list(reversed(cases))
        for c in ordered:
            modes=['buffered','per-chunk'] if c['both'] else ['buffered']
            if rep%2==0: modes=list(reversed(modes))
            for mode in modes:
                name=f"{c['label']}-{mode}-r{rep}"
                dest=root(c)+f'/{mode}-r{rep}'
                binding=['--artifact-record',root(c)+'/compile/build.json']
                step(name,'run',binding+['--output-mode',mode,'--host-timeout','86400','--out',dest],'appliance')
                step(name+'-audit','audit',binding+['--capture',dest,'--out',dest+'-audit.json'],'host')
    result=dict(schema=1,id=request_id,description=(
        f'E231 predeclared {axis} sweep, three fresh runtimes per arm, fixed compile/input; '
        'buffered with selected per-chunk controls. Host stream windows, not PE or GCN latency. '
        'No automatic retry; failure and actual I/O mapping retained.'),steps=steps)
    assert len(json.dumps(result,indent=2).encode()) <= 1 << 20
    return result


def plan():
    result={}
    for axis,cases in configurations().items():
        rows=[]
        for c in cases:
            a=argparse.Namespace(**c['config'],empty=False)
            record=appliance.admitted_config(a)
            case=driver.make_case(record)
            p=case['params'];m=case['metrics']
            identity=hashlib.sha256(case['h'].tobytes())
            import numpy as np
            identity.update(np.array(case['edges']).tobytes())
            rows.append({**c,'input_sha256':identity.hexdigest(),'params':p,'metrics':m,
                         'rectangle_pes':(p['R']+p['B'])*(p['F']+1),
                         'accumulator_forward_word_hops':p['F']*p['Q']*p['B']*(p['B']-1)//2,
                         'useful_fmacs':m['nnz_occurrences']*p['F']})
        result[axis]=rows
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    for axis,cases in configurations().items():
        item=manifest(axis,cases)
        (a.out/(item['id']+'.json')).write_text(json.dumps(item,indent=2)+'\n')
    (a.out/'plan.json').write_text(json.dumps(plan(),indent=2)+'\n')
    print('PASS configs=33 compiles=33 runs=111 audits=111 requests=7')

if __name__=='__main__': main()
