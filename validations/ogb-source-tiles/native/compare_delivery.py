"""Reconstruct all twelve original-ID frames, shared transfers and paired elapsed times."""
from hashlib import sha256
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from inventory import read, require, sha
from paired_worker import original_reply, sequence
from source_table import assign, audit, device, source_tiles


def checked(build,fixture,capture,expected,role):
    meta=read(build/'build.json');got=read(capture/'result.json')
    require(got['role']==role==meta['role'] and got['platform']=={'system':role=='hardware','simulation':role=='simulator'}
            and got['stopped'] and len(got['frames'])==12,'complete ordered SDK transactions')
    require(got['build_sha256']==sha(build/'build.json') and got['fixture_sha256']==meta['fixture_sha256']==sha(fixture)
            and got['inventory']==meta['inventory'] and got['workers']==meta['worker_sha256'],'runtime inventory binding')
    z=device.load(fixture);cfg,groups,owner,slot=assign(z,meta['params']['P'],meta['params']['E'])
    require(cfg==meta['params'],'work geometry')
    p,r,f,q=(cfg[k] for k in ('P','R','F','Q'))
    routes={k:source_tiles(z,cfg,owner,slot,k) for k in ('edges','dictionary')}
    table=[];last=0;stored={}
    for tag,((i,method,scale,delivery),frame) in enumerate(zip(sequence(),got['frames']),1):
        require((frame['tag'],frame['trial'],frame['method'],frame['scale'],frame['delivery'])==
                (tag,i+1,method,scale,delivery),'alternating original-input transaction')
        raw=capture/f'frame-{tag:02d}.npz'
        require(frame['raw_sha256']==sha(raw),'raw PE words')
        with np.load(raw,allow_pickle=False) as zraw:
            require(set(zraw.files)=={'reply','logits','node','active_edge_cycles'} and
                    zraw['reply'].dtype==np.uint32 and zraw['reply'].shape==(p,16+r*(1+q)) and
                    zraw['active_edge_cycles'].dtype==np.uint32 and zraw['active_edge_cycles'].shape==(p,1),
                    'all PE replies, clocks, and original IDs')
            y,measure=audit(original_reply(zraw['reply'],tag,i),z,cfg,groups,i)
            np.testing.assert_array_equal(y.view(np.uint32),zraw['logits'].view(np.uint32))
            np.testing.assert_array_equal(zraw['node'],z['seed'])
            np.testing.assert_array_equal(y.view(np.uint32),expected[scale])
            if scale==1:
                np.testing.assert_allclose(y,z['fp64'],rtol=3e-4,atol=2e-4)
                np.testing.assert_array_equal(y.argmax(1),z['gpu'].argmax(1))
            require(frame['peak_active_edge_cycles']==int(zraw['active_edge_cycles'].max()) and
                    np.all(zraw['active_edge_cycles'].ravel()<=measure['peak_edge_interval']),'active accumulation clocks')
        for key,value in measure.items():
            if key.startswith('fp64_'):np.testing.assert_allclose(frame['measured'][key],value,rtol=1e-8,atol=1e-13)
            else:require(frame['measured'][key]==value,'clock and output reconstruction')
        tiles=routes[method]
        factor=1 if delivery=='row' else p
        desired=[('features',len(ids)*f) for ids,_ in tiles]
        if i==0:desired.insert(0,('weights',(2*f+1)*q))
        calls=frame['calls'];require(len(calls)==len(desired),'shared DMA call count')
        start,end=frame['start_ns'],frame['end_ns']
        require(last<=start<end,'blocking transaction ordering');last=end
        tail=start;source_ns=weight_ns=source_words=weight_words=0
        for call,(symbol,count) in zip(calls,desired):
            a,b=call['start_ns'],call['end_ns']
            require(tail<=a<b<=end and call['symbol']==symbol and call['source_words']==count*factor and
                    call['delivered_words']==count*p,'target bytes and blocking SDK time')
            tail=b
            if symbol=='features':source_ns+=b-a;source_words+=count*factor
            else:weight_ns+=b-a;weight_words+=count*factor
        controls=p*(3+r*(2+f)+3*len(tiles))
        refs=p*sum(max(map(len,words)) for _,words in tiles)
        require(frame['selector_words']==refs and frame['source_words']==source_words and
                frame['input_words']==refs+controls+weight_words+source_words,'full byte accounting')
        digest=sha256(y.tobytes()).hexdigest()
        if scale in stored:require(stored[scale]==digest,'repeated/scaled output identity')
        stored[scale]=digest
        table.append(dict(tag=tag,trial=i+1,delivery=delivery,method=method,scale=scale,logits_sha256=digest,
            total_host_ns=end-start,feature_h2d_ns=source_ns,weight_h2d_ns=weight_ns,feature_calls=len(tiles),
            feature_h2d_bytes=4*source_words,feature_delivered_bytes=p*4*sum(len(x)*f for x,_ in tiles),
            weight_h2d_bytes=4*weight_words,input_bytes=frame['input_words']*4,peak_interval=measure['peak_first_to_last'],
            peak_active_edge_cycles=frame['peak_active_edge_cycles'],fp64_max_abs=measure['fp64_max_abs'],
            argmax_mismatches=measure['argmax_differences'],raw_sha256=frame['raw_sha256']))
    pairs=[]
    for i in range(6):
        ds={s['delivery']:s for s in table[2*i:2*i+2]}
        a,b=ds['row'],ds['replicated']
        require(a['logits_sha256']==b['logits_sha256'] and a['feature_delivered_bytes']==b['feature_delivered_bytes'] and
                a['feature_h2d_bytes']*p==b['feature_h2d_bytes'],'same numerical image, P recipients')
        pairs.append(dict(trial=i+1,scale=a['scale'],method=a['method'],row_feature_h2d_ns=a['feature_h2d_ns'],
                          replicated_feature_h2d_ns=b['feature_h2d_ns'],row_total_ns=a['total_host_ns'],
                          replicated_total_ns=b['total_host_ns']))
    return dict(role=role,build_sha256=got['build_sha256'],fixture_sha256=got['fixture_sha256'],image=got['inventory'],
                windows=1,nodes=len(z['seed']),arcs=len(z['src']),classes=z['wr'].shape[1],transactions=table,pairs=pairs)
