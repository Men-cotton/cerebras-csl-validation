"""Audit two original-ID placements of a trained Cora GCN on one native executable."""
import json
from pathlib import Path
import sys

import numpy as np

from paired_worker import digest, sha


def read(path):
    return json.loads(path.read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def loaded(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def original_order(tensor, nodes, width):
    valid = nodes >= 0
    data = np.empty((valid.sum(), width), tensor.dtype)
    data[nodes[valid]] = tensor.reshape(*nodes.shape, width)[valid]
    return data


def references(case, family):
    sys.path.insert(0, str(family.resolve()))
    import saved_audit as native
    import grouped_contract
    import hub_contract
    from multi_upload import audit_uploads
    doc = read(case/'case.json')
    p = doc['params']
    base = family/'cases'/doc['baseline_case']
    manifest = native.load(base)
    require(sha(base/'case.json') == doc['baseline_case_sha256'], 'original case identity')
    for key in ('params','inputs','outputs','launch'):
        require(doc[key] == manifest[key], 'original ABI '+key)
    before, context = native.independent_expected(base, manifest)
    old = loaded(base/'inputs.npz')
    new = loaded(case/'swap.npz')
    modes = doc['modes']
    require(sha(base/'inputs.npz') == modes['original']['inputs_sha256']
            and sha(case/'swap.npz') == modes['swap']['inputs_sha256'], 'packed upload identity')
    require(doc['mode_order'] in (['original','swap'],['original','swap','secondary']) and modes['original']['file']=='inputs.npz'
            and modes['swap']['file']=='swap.npz' and doc['retain_input_words'] is True, 'return contract')
    require(set(old) == set(new) == set(doc['inputs']), 'complete input inventory')
    for key in new:
        require(new[key].dtype == old[key].dtype and new[key].shape == old[key].shape, 'new input ABI '+key)
    for key in ['x_ptr','x_col','x_val','w0','w1','split_control']:
        np.testing.assert_array_equal(old[key].view('u1'), new[key].view('u1'), err_msg=key)
    nodes = np.load(case/'swap_nodes.npy', allow_pickle=False)
    plan = read(case/'plan.json')
    exchanged = plan['swapped_original_ids']
    def ownership(ids):
        pes,slots=np.nonzero(ids>=0)
        result=np.empty(doc['nodes'],np.int64)
        np.testing.assert_array_equal(np.sort(ids[pes,slots]),np.arange(len(result)))
        result[ids[pes,slots]]=pes
        return result
    owners=ownership(nodes)
    previous=ownership(context['nodes'])
    np.testing.assert_array_equal(np.flatnonzero(owners!=previous),sorted(exchanged))
    np.testing.assert_array_equal(owners[exchanged],previous[exchanged[::-1]])
    np.testing.assert_array_equal(owners//p['C'],previous//p['C'])
    plans=doc['split_plans']
    require(plans[0]['hub']==163 and len(plans)==len(modes)-1,'explicit primary/secondary split order')
    _, old_stats, old_traffic=audit_uploads(grouped_contract,p,old,context['nodes'],context['original'],plans[:1])
    np.testing.assert_array_equal(old_stats,before['stats'])
    np.testing.assert_array_equal(old_traffic,before['traffic'])
    groups, stats, traffic = audit_uploads(grouped_contract,p,new,nodes,context['original'],plans[:1])
    for left, right in zip(groups, context['groups'], strict=True):
        np.testing.assert_array_equal(left, right)
    after = dict(before, stats=stats, traffic=traffic)
    for key, width in [('pre', p['H']), ('logits',p['O'])]:
        ref = original_order(before[key], context['nodes'], width)
        padded = np.zeros((*nodes.shape,width),ref.dtype)
        padded[nodes >= 0] = ref[nodes[nodes >= 0]]
        after[key] = padded.reshape(len(nodes),-1)
    require(sha(case/'plan.json') == doc['link_plan_sha256'] and sha(case/'swap_nodes.npy') == doc['new_nodes_sha256'],
            'physical placement plan identity')
    bindings={'original':(old,before,context['nodes']),'swap':(new,after,nodes)}
    if 'secondary' in modes:
        require(modes['secondary']['file']=='secondary.npz' and sha(case/'secondary.npz')==modes['secondary']['inputs_sha256'],
                'secondary hub upload identity')
        arrays=loaded(case/'secondary.npz')
        require(set(arrays)==set(old),'complete secondary hub inputs')
        for key in old:
            require(arrays[key].shape==old[key].shape and arrays[key].dtype==old[key].dtype,'secondary input ABI '+key)
        for key in ('x_ptr','x_col','x_val','w0','w1','slots'):
            np.testing.assert_array_equal(arrays[key].view('u1'),old[key].view('u1'),err_msg=key)
        groups,stats,traffic=audit_uploads(grouped_contract,p,arrays,context['nodes'],context['original'],plans)
        for left,right in zip(groups,context['groups'],strict=True):
            np.testing.assert_array_equal(left,right)
        bindings['secondary']=(arrays,dict(before,stats=stats,traffic=traffic,split_words=hub_contract.split_words(p,arrays)),
                               context['nodes'])
    return doc, context, bindings


def frame(raw, expected, nodes, context, scale):
    require(set(raw) == set(expected) | {'clocks'}, 'full GCN output inventory')
    errors = {}
    for key, ref in expected.items():
        got = raw[key]
        require(got.shape == ref.shape, 'layer shape '+key)
        if np.issubdtype(ref.dtype,np.integer):
            np.testing.assert_array_equal(got,ref,err_msg=key)
        else:
            require(np.isfinite(got).all() and got.dtype==np.float32,'finite FP32 '+key)
            np.testing.assert_allclose(got,ref*scale,atol=3e-5,rtol=2e-4,err_msg=key)
            np.testing.assert_array_equal(got[ref==0],0.0,err_msg='structural zeros '+key)
            errors[key]=float(np.abs(got.astype(np.float64)-ref*scale).max())
    logits = original_order(raw['logits'],nodes,7)
    prediction = logits.argmax(1)
    np.testing.assert_array_equal(prediction,context['logits'].argmax(1))
    valid = context['original']['valid']
    correct = int((prediction[valid] == context['original']['labels'][valid]).sum())
    require(len(prediction)==2708 and len(valid)==500 and correct==402,'canonical original-ID heldout')
    clock=raw['clocks'].astype(np.uint64)
    require(clock.shape==(len(nodes),6) and (clock<=65535).all(),'six 16-bit PE clock words')
    ticks=clock[:,0::3] | (clock[:,1::3]<<np.uint64(16)) | (clock[:,2::3]<<np.uint64(32))
    span=(ticks[:,1]-ticks[:,0])&np.uint64((1<<48)-1)
    require(((span>0)&(span<1<<47)).all(),'completed 48-bit PE intervals')
    return dict(per_pe_cycles=span.tolist(),max_cycles=int(span.max()),max_abs_fp64=errors,
                original_pre=digest(original_order(raw['pre'],nodes,16)),
                original_logits=digest(logits),original_predictions=digest(prediction),
                original_ids=len(nodes[nodes>=0]),heldout_correct=correct,heldout_total=len(valid))


def audit(case, family, capture, backend):
    doc, ctx, bindings=references(case, family)
    rec=read(capture/'capture.json')
    require(rec['status']=='completed' and rec['backend']==backend and rec['startup_count']==1 and
            rec['case_sha256']==sha(case/'case.json') and set(rec['modes'])==set(bindings),'one completed loaded program')
    result={}
    for mode, (arrays, expected, nodes) in bindings.items():
        log=rec['modes'][mode]
        initial=capture/f'{mode}-initial.npz'
        require(sha(initial)==log['initial_file_sha256'], 'initial D2H file identity')
        before=loaded(initial)
        require(set(before)==set(arrays) and set(log['initial'])==set(arrays),'complete initial readback')
        for key,values in arrays.items():
            np.testing.assert_array_equal(before[key].view('u1'), values.view('u1'),err_msg=key)
            require(digest(before[key])==log['initial'][key],'initial readback '+key)
        result[mode]=[]
        saved={}
        for i,(item,scale) in enumerate(zip(log['frames'],(1.,0.5,1.),strict=True),1):
            file=capture/f'{mode}-{i}.npz'
            require(item['frame']==i and item['scale']==scale and sha(file)==item['raw_sha256'],'frame identity')
            words=loaded(file)
            require(set(item['final'])==set(arrays),'complete final readback hashes')
            for key, values in arrays.items():
                got=words.pop('after_'+key)
                if key=='x_val':
                    values=values*np.float32(scale)
                np.testing.assert_array_equal(got.view('u1'), values.view('u1'),err_msg='final '+key)
                require(digest(got)==item['final'][key],'final readback '+key)
            stats=frame(words,expected,nodes,ctx,scale)
            if i==1:
                saved={k:digest(v) for k,v in words.items() if k!='clocks'}
            elif i==3:
                require(saved=={k:digest(v) for k,v in words.items() if k!='clocks'},'restored A output')
            if i in (1,3):
                local=loaded(case/(mode+'-sdk.npz'))
                for key in saved:
                    np.testing.assert_array_equal(local[key].view('u1'),words[key].view('u1'),err_msg='SDK–physical '+key)
            result[mode].append(dict(frame=i,scale=scale,raw_sha256=sha(file),**stats))
    for a,b in zip(result['original'],result['swap'],strict=True):
        require(all(a[k]==b[k] for k in ('original_pre','original_logits','original_predictions')),'original-ID equality across owners')
    if 'secondary' in result:
        for a,b in zip(result['original'],result['secondary'],strict=True):
            require(a['original_predictions']==b['original_predictions'],'stable original-ID secondary hub predictions')
    return result
