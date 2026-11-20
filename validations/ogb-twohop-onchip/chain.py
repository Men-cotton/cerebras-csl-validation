"""Contract for on-chip two-hop inference and a same-image host-echo control."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from source_table import assign, device, source_tiles, upload

TRIALS = ((False, 1.), (True, 1.), (True, .5), (False, .5), (False, 1.))


def prepare(first, last, owners, entries):
    a, b = device.load(first), device.load(last)
    if not a['relu'] or b['relu'] or a['wr'].shape[1] != b['wr'].shape[0]:
        raise ValueError('incompatible GraphSAGE layers')
    np.testing.assert_array_equal(a['seed'], b['node'])
    np.testing.assert_array_equal(np.sort(a['seed']), a['seed'])
    cfg, groups, owner, slot = assign(a, owners, entries)
    z = len(b['seed']); n = len(a['seed']); h = cfg['Q']; c = b['wr'].shape[1]
    if h != a['wr'].shape[1]:
        raise ValueError('hidden dimension must have no padded lanes')
    q = (c+3)//4*4
    cfg.update(FRAMES=len(TRIALS))
    tiles = source_tiles(a,cfg,owner,slot,'dictionary')
    final = dict(P=owners,R=cfg['R'],H=h,N=n,Z=z,A=len(b['src']),Q=q,FRAMES=len(TRIALS))
    matrix = np.zeros((2*h+1,q), np.float32)
    matrix[:h,:c],matrix[h:2*h,:c],matrix[-1,:c] = b['wr'],b['wn'],b['b']
    inverse = (1/np.maximum(np.diff(b['ptr']),1)).astype(np.float32)
    setup = np.concatenate([b['node'].astype(np.uint32), b['seed'].astype(np.uint32), b['root'].astype(np.uint32),
                            b['ptr'][1:].astype(np.uint32),b['src'].astype(np.uint32),inverse.view(np.uint32),
                            matrix.ravel().view(np.uint32)])
    final['SETUP'] = len(setup)
    packet = 16+cfg['R']*(1+h)
    if max((owners-1)*packet, len(setup), n*h, q*z, int(np.diff(b['ptr']).max()))>32767:
        raise ValueError('a local signed 16-bit counter overflows')
    return a,b,cfg,groups,owner,slot,tiles,final,setup


def first_upload(data, trial):
    a,b,cfg,groups,owner,slot,tiles,final,setup=data
    echo, scale=TRIALS[trial]
    index=0 if trial==0 else 1 if scale==1 else 2
    stream = upload(a,cfg,groups,tiles,index)
    prefix = [setup] if trial==0 else []
    pad = (sum(len(v) for v in prefix) + len(stream) + 1) % 2
    header = [trial+1, int(echo)+2*pad] + ([0] if pad else [])
    return np.concatenate([*prefix,np.array(header,np.uint32),stream[1:]])


def digest(words):
    value=5381
    for word in words.ravel():
        value=(value*33+int(word))&0xffffffff
    return value


def clock(fields):
    if np.any(fields>65535):raise ValueError('invalid clock limbs')
    ts=fields.astype(np.int64).reshape(-1,3)@np.array([1,65536,1<<32])
    periods=np.diff(ts)%(1<<48)
    if not np.all((periods>0)&(periods<1<<38)):raise ValueError('invalid clock intervals')
    return periods.tolist()


def audit(data, trial, result, echo_words=None, expected_hidden=None):
    a,b,cfg,groups,owner,slot,tiles,end,setup=data
    echo, scale=TRIALS[trial];p,n,h,z,q=(end[k] for k in ('P','N','H','Z','Q'))
    np.testing.assert_array_equal(result[[0,1,2,4]],[trial+1,echo,0,end['A']]); stamps=clock(result[5:17]);at=17
    stats=result[at:at+16*p].reshape(p,16);at+=16*p
    np.testing.assert_array_equal(stats[:,0],trial+1);np.testing.assert_array_equal(stats[:,1],np.arange(p))
    np.testing.assert_array_equal(stats[:,2],list(map(len,groups)));np.testing.assert_array_equal(stats[:,3],[np.diff(a['ptr'])[g].sum() for g in groups]);pe_cycles=[sum(clock(row[4:])) for row in stats]
    np.testing.assert_array_equal(result[at:at+z],b['seed']);at+=z
    if len(result)>at+z*q:
        np.testing.assert_array_equal(result[at+z*q:],[trial+1])
    logits=result[at:at+z*q].view(np.float32).reshape(z,q);c=b['wr'].shape[1];np.testing.assert_array_equal(logits[:,c:],0);logits=logits[:,:c].copy()
    truth=device.oracle(a,scale);means=np.zeros((z,h),np.float64)
    for row in range(z):
        ids=b['src'][b['ptr'][row]:b['ptr'][row+1]]
        if len(ids):means[row]=truth[ids].mean(0)
    target=truth[b['root']]@b['wr'].astype(np.float64)+means@b['wn'].astype(np.float64)+b['b'].astype(np.float64)
    np.testing.assert_allclose(logits,target,rtol=3e-4,atol=2e-4)
    np.testing.assert_array_equal(logits.argmax(1),target.argmax(1))
    if scale==1:
        np.testing.assert_allclose(logits,b['fp64'],rtol=3e-4,atol=2e-4)
        np.testing.assert_array_equal(logits.argmax(1),b['gpu'].argmax(1))
    if (echo_words is not None)!=echo:raise ValueError('unexpected intermediate return')
    hidden_error=None
    if echo:
        states=echo_words.view(np.float32).reshape(n,h)
        np.testing.assert_allclose(states,truth,rtol=3e-4,atol=2e-4)
        if digest(echo_words)!=result[3]:raise ValueError('host echo does not match the device hidden digest')
        hidden_error=float(np.abs(states-truth).max())
    if expected_hidden is not None:
        if digest(expected_hidden)!=result[3]:raise ValueError('retained hidden values differ from the echoed frame')
    return logits,dict(first_stage_peak_cycles=max(pe_cycles),collector_first_to_last_cycles=sum(stamps),
       collector_gather_cycles=stamps[0],hidden_echo_interval_cycles=stamps[1],final_layer_cycles=stamps[2],
       hidden_digest=int(result[3]),hidden_fp64_max_abs=hidden_error,logit_fp64_max_abs=float(np.abs(logits-target).max()),
       argmax_disagreements=0)
