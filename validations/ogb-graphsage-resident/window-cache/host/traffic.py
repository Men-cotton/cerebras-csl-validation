"""Count trained two-layer GraphSAGE H2D payloads using the resident worker's assignments."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OGB = HERE.parent.parent / "ogb-graphsage"
sys.path.insert(0, str(OGB / "resident"))
sys.path.insert(0, str(OGB / "transfer"))
from graph import assign, load_case, read, require, sha, weights
from rectangles import rectangles


@lru_cache(maxsize=32768)
def regions(counts, lanes):
    return tuple(rectangles(counts, lanes))


def region_words(regions, lanes):
    return sum((end-begin)*length*lanes for begin, end, length in regions)


def prefixes(counts, lanes):
    return region_words(regions(tuple(map(int, counts)), lanes), lanes)


def feature_regions(sequences, f):
    counts = tuple(map(len, sequences))
    plans = regions(counts, f)
    volume = region_words(plans, f)
    # If two adjacent PE payloads contain the same source IDs in the same
    # positions, all FP32 words match (including the zero-padded suffix).
    duplicates = 0
    for begin, end, length in plans:
        for pe in range(begin+1, end):
            left, right = sequences[pe-1], sequences[pe]
            if len(left) == len(right) and np.array_equal(left, right):
                duplicates += length
    return volume, duplicates*f


def sharing(sequences, p):
    # One code for each (source ID, PE). The sorted encoding also determines
    # contiguous multicast intervals on the actual single physical compute row.
    owner = np.repeat(np.arange(p, dtype=np.int64), list(map(len, sequences)))
    sources = np.concatenate(sequences)
    if not len(sources):
        return Counter(), np.zeros(p+1, np.int64)
    codes = np.unique(sources.astype(np.int64)*p+owner)
    ids = codes//p
    first = np.r_[True, ids[1:] != ids[:-1]]
    fanout = np.diff(np.r_[np.flatnonzero(first), len(ids)])
    spans = int(1 + np.count_nonzero(first[1:] | (np.diff(codes) != 1)))
    owners = np.bincount(codes%p, minlength=p)
    return (Counter(occurrences=len(sources), owner_vectors=len(codes),
                    union_vectors=len(fanout), contiguous_vectors=spans,
                    peak_owner_vectors=int(owners.max()), peak_union_vectors=len(fanout)),
            np.bincount(fanout, minlength=p+1))


def add_counts(total, more):
    for key, value in more.items():
        if key.startswith('peak_'):
            total[key] = max(total[key], int(value))
        else:
            total[key] += int(value)


def window(mode, rows, ptr, indices, mapped, degree, cfg):
    p, r, d, f, e, q = (cfg[k] for k in ('P','R','D','F','E','Q'))
    groups = assign(rows, degree, p, r, mode, f, q)
    ns = tuple(map(len, groups))
    dims = d if mode == 0 else f
    row_count = len(rows)
    csr = mode != 0
    live_root = row_count*dims
    raw_params = prefixes(ns, 1)
    root_words = prefixes(ns, dims)
    counted = Counter(rows=row_count, windows=1, root_live_words=live_root,
                      csl_root_words=root_words, csl_row_metadata_words=raw_params*(1+csr),
                      direct_root_words=live_root, direct_row_metadata_words=p + row_count*(1+csr)+3,
                      csl_control_words=p*5)
    hist = np.zeros(p+1, np.int64)
    if csr:
        seq = []
        for group in groups:
            pieces = [indices[ptr[i]:ptr[i+1]] for i in group]
            seq.append(np.concatenate(pieces) if pieces else np.empty(0, np.int32))
        if mapped is not None:
            seq = [mapped[s] for s in seq]
        lengths = list(map(len, seq))
        parts = (max(lengths)+e-1)//e
        require(parts <= cfg['K'], 'existing image does not admit window edge bound')
        share, hist = sharing(seq,p)
        add_counts(counted,share)
        live = sum(lengths)*f
        counted.update(edge_live_words=live, csl_control_words=parts*p*5,
                       csl_end_words=parts*raw_params, direct_edge_words=live,
                       direct_end_words=parts*(row_count+2*p), parts=parts, peak_parts=parts)
        for j in range(parts):
            bank = [s[j*e:(j+1)*e] for s in seq]
            cost, same = feature_regions(bank, f)
            counted['csl_edge_words'] += cost
            counted['adjacent_identical_bank_words'] += same
    counted['csl_words'] = sum(counted[k] for k in ('csl_root_words','csl_row_metadata_words',
                                                        'csl_control_words','csl_end_words','csl_edge_words'))
    counted['direct_words'] = sum(counted[k] for k in ('direct_root_words','direct_row_metadata_words',
                                                       'direct_edge_words','direct_end_words'))
    return counted, hist


def full_csr(cache, graph, n):
    for file, key in [('ptr.npy', 'ptr_sha256'),('index.npy','index_sha256')]:
        require(sha(cache/file) == graph[key], 'original CSR changed')
    ptr, idx = (np.load(cache/f, mmap_mode='r') for f in ('ptr.npy','index.npy'))
    require(len(ptr)==n+1 and ptr[0]==0 and ptr[-1]==len(idx)==graph['undirected_unique_arcs'] and
            np.all(np.diff(ptr)>=0) and idx.min()>=0 and idx.max()<n, 'original CSR extents')
    return ptr, idx


def main(args):
    data, scope, base = args.dataset, args.scope, args.cache
    cfgfile = base/('resident-'+data+'-csl-p32')/'build.json'
    cfg = read(cfgfile)['params']
    require(cfg==read(base/('resident-'+data+'-tungsten-p32')/'build.json')['params'] and cfg['P']==32,
            'matched deployed geometry')
    evidence = OGB/'evidence'/data
    model = read(evidence/'model.json')
    case = base/('resident-'+data+'-'+('four' if scope=='whole' else scope))
    info, pilot = load_case(case)
    require(info['original_graph']==model['graph'] and info['original_checkpoint_sha256']==
            sha(evidence/'weights.npz')==model['weights_sha256'], 'original checkpoint and graph')
    comp = read(evidence/'compaction.json')
    live = np.asarray(comp['live_channels'], np.int32)
    with np.load(evidence/'weights.npz', allow_pickle=False) as z:
        for field, original in [('wr0','root0.weight'),('wn0','neigh0.weight')]:
            np.testing.assert_array_equal(pilot[field],z[original][live].T)
        for field, original in [('wr1','root1.weight'),('wn1','neigh1.weight')]:
            np.testing.assert_array_equal(pilot[field],z[original][:,live].T)
        np.testing.assert_array_equal(pilot['b0'],z['neigh0.bias'][live])
        np.testing.assert_array_equal(pilot['b1'],z['neigh1.bias'])
    mat = weights(pilot,cfg)
    if scope=='whole':
        n = model['nodes']
        ptr, sources = full_csr(base/data,model['graph'],n)
        roots = (n,n,n)
        topologies = [(None,None,None),(ptr,sources,None),(ptr,sources,None)]
    else:
        roots = [len(pilot[k]) for k in ('raw_ids','a_ids','b_ids')]
        topologies = [(None,None,None),(pilot['a_ptr'],pilot['a_src'],pilot['raw_ids']),
                      (pilot['b_ptr'],pilot['b_src'],pilot['a_ids'])]
    boot = len(mat)
    p,r = cfg['P'], cfg['R']
    provenance = dict(model_sha256=sha(evidence/'model.json'), checkpoint_sha256=model['weights_sha256'],
                      graph=model['graph'],case_sha256=sha(case/'case.json'), pilot_graph_sha256=info['graph_sha256'],
                      packed_weights_sha256=hashlib.sha256(mat.tobytes()).hexdigest(),
                      graph_py_sha256=sha(OGB/'resident/graph.py'), dma_py_sha256=sha(OGB/'transfer/rectangles.py'))
    result=dict(dataset=data,scope=scope,params=cfg,provenance=provenance,
                boot=dict(words=boot, unicast_words=p*boot, row_broadcast_words=boot),
                observed_sdk={},phases={})
    # Keep observed SDK journals optional, allowing the census to run on CPU
    # before the SDK captures complete. Every compared row is bound to its raw packet.
    journal={}
    if scope!='whole':
        for lang in ('csl','tungsten'):
            cap=base/('resident-'+data+'-'+scope+'-'+lang+'-p32-capture')
            if not (cap/'windows.jsonl').is_file():
                continue
            control=read(cap/'result.json')
            require(control['case_sha256']==provenance['case_sha256'] and control['params']==cfg and
                    control['platform']=={'system':False,'simulation':True},'observed simulator identity')
            with (cap/'windows.jsonl').open() as log:
                raw = [json.loads(line) for line in log if line.endswith('\n')]
            entries = {}
            for row in raw:
                if row['trial']!=1: continue
                name=row['name']
                require(name==Path(name).name and sha(cap/name)==row['raw_sha256'] and name not in entries,
                        'observed window/return identity')
                entries[name] = row
            journal[lang]=entries
            result['observed_sdk'][lang]=dict(compared_full_input_windows=len(entries),planned_all_trials=control['planned_frames'],
                                               loaded_image=control['loaded_image'])
    for mode,(size,(ptr,idx,mapped)) in enumerate(zip(roots,topologies)):
        total=Counter()
        hist=np.zeros(p+1,np.int64)
        degree=np.diff(ptr).astype(np.int64,copy=False) if ptr is not None else None
        for i,start in enumerate(range(0,size,p*r)):
            rows=np.arange(start,min(start+p*r,size),dtype=np.int32)
            w,h=window(mode,rows,ptr,idx,mapped,degree,cfg)
            if mode==0 and i==0:
                w['csl_words']+=boot;w['direct_words']+=boot+1
            name=f't1-s{mode}-w{i:04d}.npz'
            for lang,entries in journal.items():
                if name not in entries:continue
                row=entries[name]
                require(row['rows']==len(rows) and row['edges']==w['occurrences'] and row['parts']==w['parts']
                        and row['input_words']==w['csl_words' if lang=='csl' else 'direct_words'],
                        'SDK H2D count discrepancy '+lang+' '+name)
            add_counts(total,w)
            hist+=h
            if i and i%1000==0: print(data,scope,mode,i,flush=True)
        if mode:
            require(int(hist.sum())==total['union_vectors'] and int(hist@np.arange(p+1))==total['owner_vectors'] and
                    total['union_vectors']<=total['contiguous_vectors']<=total['owner_vectors']<=total['occurrences'],
                    'multicast source/owner accounting')
        result['phases'][('project','aggregate_hidden','aggregate_logits')[mode]]=dict(
                **dict(sorted(total.items())), sources_by_owner_fanout=hist.tolist())
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(data,scope,'checked',args.out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=('arxiv','products'));parser.add_argument('scope',choices=('four','128','whole'))
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
