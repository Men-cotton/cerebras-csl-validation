"""Independent FP64 mean GraphSAGE inference from original COO occurrences, weights and biases."""
from pathlib import Path
import hashlib
import json

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(checkpoint,n,scale):
    with np.load(checkpoint,allow_pickle=False) as archive:
        x=archive['x'][:n].astype(np.float64)*scale
        all_src,all_dst=archive['src'],archive['dst']
        keep=(all_src<n)&(all_dst<n)
        src,dst=all_src[keep],all_dst[keep]
        degree=np.bincount(dst,minlength=n)
        def mean(values):
            summed=np.zeros_like(values)
            np.add.at(summed,dst,values[src])
            return summed/np.maximum(degree,1)[:,None]
        pre=x@archive['ws0'].astype('f8')+mean(x)@archive['wn0'].astype('f8')+archive['b0']
        positive=np.maximum(pre,0)
        norm=np.linalg.norm(positive,axis=1,keepdims=True)
        hidden=np.divide(positive,norm,out=np.zeros_like(pre),where=norm!=0)
        logits=hidden@archive['ws1'].astype('f8')+mean(hidden)@archive['wn1'].astype('f8')+archive['b1']
        typical={k:archive[k][archive[k]<n] for k in ('train_idx','val_idx','test_idx')}
        labels=archive['labels'][:n]
    return dict(pre=pre,hidden=hidden,logits=logits),labels,typical,len(src)


def generate(checkpoint,nodes,source,output):
    valid=nodes>=0
    n=int(valid.sum())
    np.testing.assert_array_equal(np.sort(nodes[valid]),np.arange(n))
    bank={'nodes':nodes}
    for frame,factor in enumerate((1.,.5),1):
        layers,labels,splits,edges=reference(checkpoint,n,factor)
        for name,value in layers.items():
            expected=np.zeros((*nodes.shape,value.shape[-1]),dtype='<f8')
            expected[valid]=value[nodes[valid]]
            bank[f'{name}_{frame}']=expected.reshape(len(nodes),-1)
            if frame==1:
                bare=np.fromfile(source/('out-'+name+'.bin'),dtype='<f4').reshape(len(nodes),-1)
                np.testing.assert_allclose(bare,bank[f'{name}_{frame}'],rtol=2e-4,atol=2e-5)
                bank['bare_'+name]=bare
    bank.update(labels=labels,**splits)
    np.savez_compressed(output,**bank)
    return n,edges


def validate_reference(case):
    build=json.loads((case/'build.json').read_text())
    f=case/'fixture'
    manifest=json.loads((f/'case.json').read_text())
    if (sha(f/'case.json')!=build['original_fixture_sha256'] or
            sha(f/'input_model.npz')!=manifest['files']['input_model.npz'] or
            sha(f/'node_ids.npy')!=manifest['files']['node_ids.npy']):
        raise ValueError('fixture inventory disagrees with the original CSL case')
    nodes=np.load(f/'node_ids.npy',allow_pickle=False)
    valid=nodes>=0
    np.testing.assert_array_equal(np.sort(nodes[valid]),np.arange(build['nodes']))
    with np.load(case/'reference.npz',allow_pickle=False) as frozen:
        np.testing.assert_array_equal(nodes,frozen['nodes'])
        for index,factor in enumerate((1.,.5),1):
            expected,labels,splits,edges=reference(f/'input_model.npz',build['nodes'],factor)
            if edges!=build['edges']:raise ValueError('edge occurrence count differs')
            np.testing.assert_array_equal(frozen['labels'],labels)
            for k,ids in splits.items():np.testing.assert_array_equal(frozen[k],ids)
            for k,value in expected.items():
                stored=frozen[f'{k}_{index}'].reshape(*nodes.shape,-1)
                np.testing.assert_allclose(stored[valid],value[nodes[valid]],rtol=1e-12,atol=1e-13)
                np.testing.assert_array_equal(stored[~valid],0)
