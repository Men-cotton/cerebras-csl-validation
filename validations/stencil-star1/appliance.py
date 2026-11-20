"""Compile one native stencil or execute one frozen native/Direct Link stencil."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil

import numpy as np

import link_worker
import native_worker

HERE = Path(__file__).resolve().parent
PORT = 'e4a311bc7f0f12ff52503703f4e233182c840339f765d01d49650022fe951d3e'
GAINS = (1., -.5, 1.)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    tmp.replace(path)


def verify_sources(case, pin):
    if sha(case/'case.json') != pin:
        raise ValueError('frozen case digest differs')
    doc = read(case/'case.json')
    for file, expected in doc['files'].items():
        if sha(case/file) != expected:
            raise ValueError(f'changed case member {file}')
    for name, expected in doc['workers'].items():
        if sha(HERE/name) != expected:
            raise ValueError(f'worker {name} has changed')
    return doc


def verify_frames(case, folder, role):
    doc = read(case/'case.json')
    kind = doc['kind']
    params = doc['params']
    c,r,tx,ty,e = (params[k] for k in ('C','R','TX','TY','E'))
    report = read(folder/('result.json' if kind=='csl' else 'capture.json'))
    actual = ('physical' if role=='hardware' else 'SDK simulator') if kind=='csl' else role
    if report['backend'] != actual or report['stopped'] is not True or len(report['trials']) != 3:
        raise ValueError('target/completion/frame count mismatch')
    if kind == 'tungsten':
        if report['params'] != params or report['build_sha256'] != sha(case/'build.json') or report['stage'] != 'complete':
            raise ValueError('unexpected Direct Link build or report')
    else:
        values = [int(report['params'][k]) for k in ('KERNEL_X_DIM','KERNEL_Y_DIM','TILE_X','TILE_Y','ITERATIONS','RADIUS','PATTERN_ID')]
        if values != [c,r,tx,ty,e,1,1]:
            raise ValueError('native shape does not match the request')
    summaries = []
    for frame,(gain,trial) in enumerate(zip(GAINS,report['trials']),1):
        initial,prev,nxt,fp64 = native_worker.reference(c,r,tx,ty,e,gain)
        path = folder/f'frame-{frame}.npz'
        if sha(path) != trial['raw_sha256'] or trial['frame'] != frame or trial['gain'] != gain:
            raise ValueError('changed frame data or order')
        with np.load(path,allow_pickle=False) as raw:
            if kind == 'tungsten':
                np.testing.assert_array_equal(raw['upload'],initial.view(np.uint32).ravel())
                result = link_worker.audit(raw['reply'],params,frame,prev,nxt,fp64)
                if any(trial[k] != v for k,v in result.items()):
                    raise ValueError('Direct Link trial differs from its raw reply')
            else:
                words = native_worker.check_transport(raw,c,r,tx,ty,e)
                for key,expected in (('grid_prev',prev),('grid_next',nxt)):
                    np.testing.assert_array_equal(raw[key].view(np.uint32).ravel(),expected.view(np.uint32).ravel())
                start,end = raw['tsc_start'],raw['tsc_end']
                if np.any(start>65535) or np.any(end>65535):
                    raise ValueError('invalid native clock limbs')
                clk = ((end.astype(np.int64)-start.astype(np.int64)) @ np.array([1,65536,1<<32])) % (1<<48)
                actual = raw['grid_prev'].reshape(r,c,ty+2,tx+2)[:,:,1:-1,1:-1]
                error = float(np.abs(actual-native_worker.tile(fp64,c,r,tx,ty)).max())
                if (np.any(clk<=0) or int(clk.max())!=trial['max_cycles'] or int(clk.min())!=trial['min_cycles']
                    or words!=trial['sent_words'] or error!=trial['max_fp64_abs']):
                    raise ValueError('native clocks, transport or oracle differ')
        summaries.append({'frame':frame,'max_cycles':trial['max_cycles'],'max_fp64_abs':trial['max_fp64_abs']})
    return summaries


def compile_native(case, out, pin, namespace):
    from cerebras.sdk.client import SdkCompiler
    doc = verify_sources(case,pin)
    if doc['kind'] != 'csl':
        raise ValueError('native compilation needs a CSL case')
    verify_frames(case,case/'sim','simulator')
    out.mkdir(parents=True,exist_ok=False)
    with SdkCompiler(mgmt_namespace=namespace,resource_cpu=1000,resource_mem=8<<30,disable_version_check=False) as compiler:
        artifact = Path(compiler.compile(str(case/'src'),'layout.csl',' '.join(doc['flags']),str(out))).resolve()
    if artifact.parent != out or not artifact.is_file():
        raise ValueError('unexpected compiler output')
    save(out/'build.json',dict(case_sha256=pin,namespace=namespace,artifact=artifact.name,sha256=sha(artifact)))


def run_case(case,out,pin,compile_record,namespace):
    from cerebras.sdk.client import SdkLauncher
    doc = verify_sources(case,pin)
    verify_frames(case,case/'sim','simulator')
    kind = doc['kind']
    out.mkdir(parents=True,exist_ok=False)
    os.chdir(out)
    staged = [HERE/n for n in doc['workers']]
    if kind == 'csl':
        rec = read(compile_record)
        artifact = compile_record.parent/rec['artifact']
        if rec['case_sha256']!=pin or rec['namespace']!=namespace or sha(artifact)!=rec['sha256']:
            raise ValueError('native artifact differs')
        worker_args = 'native_worker.py out capture'
        report = 'result.json'
    else:
        build = read(case/'build.json')
        if sha(case/'hardware/out/out_port_map.json')!=PORT or build['ports_sha256']!=PORT:
            raise ValueError('unexpected Direct Link ports')
        if sha(case/'hardware/out/program.elf')!=build['roles']['hardware']['elf_sha256']:
            raise ValueError('unexpected hardware ELF')
        artifact = out/'bundle'
        shutil.copytree(case/'hardware'/'out',artifact/'out')
        staged.append(case/'build.json')
        worker_args = 'link_worker.py build.json out capture'
        report = 'capture.json'
    command = ('sha256sum '+ ' '.join(doc['workers']) + ' > workers.sha256 && '
               'timeout --signal=INT --kill-after=15s 1200s cs_python '+worker_args+
               ' --cmaddr %CMADDR% >worker.log 2>&1; code=$?; echo "$code" >worker-exit.txt; exit "$code"')
    info = dict(case_sha256=pin,kind=kind,namespace=namespace,downloads={},numerical_status='pending')
    (out/'capture').mkdir()
    try:
        with SdkLauncher(str(artifact),simulator=False,mgmt_namespace=namespace,
                         resource_cpu=1000,resource_mem=8<<30,disable_version_check=False) as launcher:
            for src in staged:
                launcher.stage(str(src))
            try:
                (out/'launcher-response.txt').write_text(str(launcher.run(command)))
            finally:
                files=['worker.log','worker-exit.txt','workers.sha256','capture/'+report]
                for path in files+[f'capture/frame-{i}.npz' for i in (1,2,3)]:
                    try:
                        launcher.download_artifact(path,str(out/path))
                        info['downloads'][path]={'sha256':sha(out/path)}
                    except Exception as exc:
                        info['downloads'][path]={'error':repr(exc)}
                    save(out/'collection.json',info)
        if any('error' in d for d in info['downloads'].values()) or (out/'worker-exit.txt').read_text().strip()!='0':
            raise ValueError('worker or collection failed')
        hashes=dict((name,h) for h,name in (line.split() for line in (out/'workers.sha256').read_text().splitlines()))
        if hashes!=doc['workers']:
            raise ValueError('staged worker digests differ')
        info['trials']=verify_frames(case,out/'capture','hardware')
        info['numerical_status']='passed'
    except BaseException as exc:
        info['error']=repr(exc)
        raise
    finally:
        save(out/'collection.json',info)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('compile','run'))
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--case-sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--build',type=Path)
    p.add_argument('--namespace',default='job-operator')
    a=p.parse_args()
    logging.basicConfig(level=logging.INFO)
    from cerebras.appliance import logger
    case,out=a.case.resolve(),a.out.resolve()
    if a.action=='compile':
        compile_native(case,out,a.case_sha256,a.namespace)
    else:
        run_case(case,out,a.case_sha256,a.build.resolve() if a.build else None,a.namespace)
