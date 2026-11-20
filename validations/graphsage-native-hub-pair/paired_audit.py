"""Rebuild original-COO GraphSAGE outputs and counters for both modes of a paired capture."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from paired_worker import digest, sha, save


def require(ok, message):
    if not ok:
        raise ValueError(message)


def packets(p, arrays, raw, weight):
    before = arrays['packet_template'].reshape(p['C']*p['R'], p['NTX'], p['T']+2)
    after = raw['packet_template'].reshape(before.shape)
    np.testing.assert_array_equal(after[:, :, :2], before[:, :, :2])
    hidden = raw['hidden'].reshape(-1, p['L'], p['H']).astype(np.float64)
    projected = hidden @ weight.astype(np.float64)
    begin = (p['O']-1)//p['T']*p['T']
    width = p['O'] - begin
    for pe, (count, _) in enumerate(arrays['counts']):
        slots = arrays['tx_slot'][pe, :count]
        np.testing.assert_allclose(after[pe, :count, 2:2+width].view('<f4'),
                                   projected[pe, slots, begin:], atol=2e-5, rtol=2e-4)
        np.testing.assert_array_equal(after[pe, :count, 2+width:], 0)
        np.testing.assert_array_equal(after[pe, count:], before[pe, count:])


def entry_intervals(words):
    require(words.ndim == 2 and words.shape[1] == 9 and words.dtype == np.uint16, 'nine entry clock words')
    full = words.astype(np.int64).reshape(-1,3,3) @ np.array([1,65536,1<<32],np.int64)
    work = (full[:,1]-full[:,0]) % (1<<48)
    join = (full[:,0]-full[:,2]) % (1<<48)
    inclusive = (full[:,1]-full[:,2]) % (1<<48)
    require(np.all(join > 0) and np.all(work > 0) and np.all(inclusive == work+join)
            and np.all(inclusive < (1<<47)), 'invalid pre-entry, post-entry, completion clock order')
    return dict(per_pe_barrier_cycles=join.tolist(),per_pe_including_barrier=inclusive.tolist(),
                max_including_barrier=int(inclusive.max()))


def check_device_frame(host, base, doc, mode, arrays, raw, reference):
    p = doc['params']
    required = {**doc['outputs'], 'packet_template': doc['inputs']['packet_template']}
    require(set(raw) == set(required), 'incomplete returned outputs')
    for name, (length, dtype) in required.items():
        require(raw[name].shape == (p['C']*p['R'], length) and str(raw[name].dtype) == dtype, 'wrong data '+name)
    original_expected, ctx = reference
    expected = {k:a.copy() for k,a in original_expected.items()}
    ntiles = (p['H']+p['T']-1)//p['T']+(p['O']+p['T']-1)//p['T']
    routes = arrays['counts'].astype(np.uint32)
    expected['traffic'] = np.column_stack((routes, routes*(p['T']+2)))*ntiles
    expected['stats'][:,3] = expected['traffic'][:,1]
    words = np.zeros((p['C']*p['R'],2),np.uint32)
    if mode == 'split':
        moved = doc['modes'][mode]['plan']['moved']
        lanes = p['H']+p['O']
        expected['stats'][0,5] -= np.uint32(moved*lanes)
        expected['stats'][2*p['C'],5] += np.uint32(moved*lanes)
        words[0,0] = lanes
        words[p['C'],:] = lanes
        words[2*p['C'],1] = lanes
    np.testing.assert_array_equal(raw['split_words'], words)
    packets(p, arrays, raw, ctx['original']['wn1'])
    if doc.get('entry_barrier_clocks'):
        entry=entry_intervals(raw['clocks'])
        frame=host.check_frame(base,dict(raw,clocks=np.ascontiguousarray(raw['clocks'][:,:6])),expected,ctx)
        frame['entry_barrier']=entry
    else:
        frame=host.check_frame(base,raw,expected,ctx)
    return frame,int(words[:,1].sum())


def audit(case, folder, original, backend, *, sdk=None):
    sys.path.insert(0, str(original.parents[1]))
    import saved_audit as host
    doc = json.loads((case/'case.json').read_text())
    base = host.load(original)
    for name, key in [('case.json', 'original_case_sha256'), ('input_model.npz', 'original_checkpoint_sha256'),
                      ('node_ids.npy', 'original_node_ids_sha256'), ('inputs.npz', 'original_inputs_sha256')]:
        require(sha(original/name) == doc[key], 'different original ' + name)
    for name, wanted in doc['files'].items():
        require(sha(case/name) == wanted, 'changed paired case ' + name)
    p = doc['params']
    require(base['params'] == p, 'geometry differs')
    execution = json.loads((folder/'capture.json').read_text())
    require(execution['status'] == 'completed' and execution['backend'] == backend and
            execution['case_sha256'] == sha(case/'case.json') and execution['startup_count'] == 1 and
            set(execution['modes']) == {'control', 'split'}, 'incomplete one-image paired execution')
    original_inputs = host.load_npz(original/'inputs.npz')
    report = dict(backend=backend, case_sha256=sha(case/'case.json'),
                  original_checkpoint_sha256=doc['original_checkpoint_sha256'], nodes=base['nodes'],
                  pes=p['C']*p['R'], modes={})
    ctx_by_scale = {}
    for scale in (1., .5):
        feature = original_inputs['x_val']*np.float32(scale)
        ctx_by_scale[scale] = host.independent_expected(original, base, dict(original_inputs,x_val=feature), scale)
    for mode in ('control', 'split'):
        arrays = host.load_npz(case/(mode+'.npz'))
        rec = execution['modes'][mode]
        initial = {k: digest(a) for k, a in arrays.items()}
        require(rec['initial'] == initial, 'partial initial device-input witness')
        require([f['scale'] for f in rec['frames']] == [1., .5, 1.], 'A/half-A/restored-A order')
        summary = []
        first = None
        for i, meta in enumerate(rec['frames'], 1):
            scale = meta['scale']
            file = folder/f'{mode}-{i}.npz'
            require(meta['frame'] == i and sha(file) == meta['raw_sha256'], 'partial frame')
            raw = host.load_npz(file)
            hashes = dict(initial, x_val=digest(arrays['x_val']*np.float32(scale)),
                          packet_template=digest(raw['packet_template']))
            require(meta['final'] == hashes, 'partial final immutable input witness')
            frame, moved_words = check_device_frame(host, base, doc, mode, arrays, raw, ctx_by_scale[scale])
            if i == 1:
                first = raw
            if i == 3:
                compare_fields(raw, first)
            if sdk:
                compare_fields(raw, host.load_npz(sdk/file.name))
            summary.append(dict(frame=i, scale=scale, raw_sha256=sha(file), moved_partial_word_hops=moved_words,
                                **frame))
        report['modes'][mode] = summary
    return report


def compare_fields(actual, witness):
    require(set(actual) == set(witness), 'mismatched comparison schema')
    for key in set(actual) - {'clocks'}:
        np.testing.assert_array_equal(actual[key].view('u1'), witness[key].view('u1'), err_msg=key)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('case','folder','original','output'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--backend',required=True,choices=['hardware','simulator'])
    parser.add_argument('--sdk',type=Path)
    a=parser.parse_args()
    result=audit(a.case.resolve(),a.folder.resolve(),a.original.resolve(),a.backend,sdk=a.sdk)
    save(a.output,result)
    print({k:[r['local_interval_min_max'] for r in v] for k,v in result['modes'].items()})
