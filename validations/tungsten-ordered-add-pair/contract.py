"""Wire format and independent FP32/reference checks for the ordered-add pair."""
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np


TAG=0x514F4932
SEEDS=(88457,27)
BLOCKS=((0,0),(1,1),(0,1),(1,0))
PORT='e4a311bc7f0f12ff52503703f4e233182c840339f765d01d49650022fe951d3e'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def require(condition,message):
    if not condition:raise ValueError(message)


def inputs(fixture,n):
    with np.load(fixture,allow_pickle=False) as loaded:
        arrays=np.asarray(loaded['abc'])
        require(arrays.dtype==np.dtype('<f4') and arrays.shape==(2,3,n),'input tensor type or shape')
        np.testing.assert_array_equal(loaded['seeds'],SEEDS)
    words=[]
    for block,(index,first) in enumerate(BLOCKS):
        words.append(np.concatenate((np.array([TAG,block,first],dtype='uint32'),arrays[index].view('uint32').ravel())))
    return arrays,np.stack(words)


def verify(build,fixture,capture,role):
    manifest=read(build);n=manifest['elements']
    arrays,upload=inputs(fixture,n)
    record=read(capture/'capture.json')
    require(sha(fixture)==manifest['fixture_sha256'] and record['build_sha256']==sha(build),'actual build or input differs')
    require(record['role']==role and record['stopped'] and record['stage']=='complete' and
            record['elf_sha256']==manifest['roles'][role]['elf_sha256'] and record['ports_sha256']==PORT,
            'runtime completion or target ELF identity')
    require(record['coordinates']==[[0,0],[0,1]] and record['system_target']==(role=='hardware'),'application PE placement')
    require(sha(capture/'raw.npz')==record['raw_sha256'] and record['worker_sha256']==manifest['worker_sha256'],
            'worker or returned raw bytes differ')
    trials=[];cycle_values={0:[],1:[]}
    with np.load(capture/'raw.npz',allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw['uploads'],upload)
        np.testing.assert_array_equal(raw['guard'],np.full(4,0xdeadc0de,dtype='uint32'))
        reply=raw['replies']
        require(reply.dtype==np.dtype('<u4') and reply.shape==(4,4,n+10),'reply type or dimensions')
        for block,(seed,first) in enumerate(BLOCKS):
            a,b,c=arrays[seed]
            ordered=np.add(np.add(a,b,dtype=np.float32),c,dtype=np.float32)
            other=np.add(a,np.add(b,c,dtype=np.float32),dtype=np.float32)
            discriminators=ordered.view('uint32')!=other.view('uint32')
            require(discriminators[::7].all(),'input no longer distinguishes evaluation order')
            for turn,row in enumerate(reply[block]):
                mode=first^(turn%2)
                np.testing.assert_array_equal(row[:4],[TAG,block,turn,mode])
                limbs=row[4:10].astype(np.int64).reshape(2,3)
                require(np.all(limbs<=65535),'invalid 48-bit clock limbs')
                start,end=map(int,limbs@np.array([1,65536,1<<32],dtype=np.int64))
                elapsed=(end-start)%(1<<48)
                require(0<elapsed<1000000,'invalid elapsed PE cycles')
                np.testing.assert_array_equal(row[10:],ordered.view('uint32'))
                cycle_values[mode].append(elapsed)
                trials.append(dict(block=block,turn=turn,seed=SEEDS[seed],mode=mode,cycles=elapsed,
                                   start=start,end=end,discriminating_lanes=int(discriminators.sum()),
                                   output_sha256=hashlib.sha256(row[10:].tobytes()).hexdigest()))
    return dict(backend=role,elements=n,compute_pes=1,relay_pes=1,build_sha256=sha(build),raw_sha256=record['raw_sha256'],
                expression_cycles=cycle_values[0],staged_cycles=cycle_values[1],
                median_expression=median(cycle_values[0]),median_staged=median(cycle_values[1]),trials=trials)
