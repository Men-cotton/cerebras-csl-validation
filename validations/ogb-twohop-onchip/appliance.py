"""Launch the physical five-frame matched on-chip and host-echo GraphSAGE experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import numpy as np

from chain import TRIALS, device, first_upload, prepare
from image import equivalence
from report_chain import replay

HERE=Path(__file__).resolve().parent
PORT='e4a311bc7f0f12ff52503703f4e233182c840339f765d01d49650022fe951d3e'
WORKER=('worker_chain.py','chain.py','source_table.py','ogb_contract.py')
HOST=(*WORKER,'originals.py','report_chain.py','image.py','appliance.py')
FILES=('capture/result.json',*(f'capture/frame-{i}.npz' for i in range(1,6)),'worker.log','worker-exit.txt')


def read(p):return json.loads(Path(p).read_text())


def require(x,why):
    if not x:raise ValueError(why)


def save(p,data):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def checked_data(case):
    rel=read(case/'release.json');model=read(case/'model.json');graph=read(case/'original.json')
    require(graph['normalized_csr']==model['graph'] and model['weights_sha256']==device.sha(case/'weights.npz')
            and graph['weights_sha256']==model['weights_sha256'],'original checkpoint and graph identity')
    records=[]
    with np.load(case/'weights.npz',allow_pickle=False) as weights:
        for stage,name in enumerate(('first.npz','last.npz')):
            z=device.load(case/name)
            for field,matrix in (('wn',f'neigh{stage}.weight'),('wr',f'root{stage}.weight'),('b',f'neigh{stage}.bias')):
                np.testing.assert_array_equal(z[field],weights[matrix].T)
            orig=graph['layers'][stage]
            require(orig['rows_sha256']==digest(z['seed']) and orig['original_arc_sources_sha256']==digest(z['node'][z['src']])
                    and orig['row_pointers_sha256']==digest(z['ptr']) and orig['input_feature_sha256']==digest(z['x'])
                    and orig['gpu_output_sha256']==digest(z['gpu']) and orig['fp64_output_sha256']==digest(z['fp64']),
                    'original IDs, adjacency, features and independent model references')
            records.append(z)
    first,last=records
    np.testing.assert_array_equal(first['seed'],last['node'])
    require(len(first['node'])==rel['raw_nodes'] and len(first['seed'])==rel['hidden_nodes']
            and len(last['seed'])==rel['destinations'],'complete true two-hop closure')
    return first,last


def images(case):
    desc=read(case/'build.json');pairs=[]
    for name,sha in desc['source_sha256'].items():
        require(Path(name).name==name and device.sha(case/'src'/name)==sha,'device source')
    require(device.sha(case/'chain.map')==desc['map_sha256'],'original machine map')
    for role,extent in [('simulator',(desc['first']['P'],2)),('hardware',(762,1172))]:
        root=case/role/'out';elf=root/'program.elf'
        target=desc['roles'][role]
        require(target['fabric']==list(extent) and target['elf_sha256']==device.sha(elf) and
                device.sha(root/'out_port_map.json')==desc['port_sha256']==PORT,'physical/SDK image and Direct Link wiring')
        pairs.append(equivalence(elf,extent,(desc['first']['P'],2),(case/role/'ranges.txt').read_text()))
    require(pairs[0]==pairs[1] and pairs[0]['canonical_sha256']==desc['canonical_sha256'],'identical application loads at both geometries')
    return desc


def observed(case,cap,expected,role,attested):
    desc=read(case/'build.json');meta=read(cap/'result.json')
    require(meta['role']==role,'actual target role')
    first=case/'first.npz';last=case/'last.npz';data=prepare(first,last,desc['first']['P'],desc['first']['E'])
    metrics,words=replay(first,last,case,cap,expected)
    if attested:
        hashes={n:device.sha(HERE/n) for n in WORKER}
        require(meta['actual_elf_sha256']==desc['roles'][role]['elf_sha256'] and meta['ports_sha256']==PORT and
                meta['host_sha256']==hashes and meta['platform']=={'system':role=='hardware','simulation':role=='simulator'},
                'actually staged workload, ELF, port and SDK platform')
    for i,((echo,scale),frame) in enumerate(zip(TRIALS,meta['frames'])):
        with np.load(cap/f'frame-{i+1}.npz',allow_pickle=False) as raw:
            expected_keys={'reply','logits','node'}|({'echo'} if echo else set())
            require(set(raw.files)==expected_keys and raw['reply'].dtype==np.uint32,'complete raw return inventory')
            p=desc['first']['P'];n=desc['final']['N']*desc['final']['H'];width=17+16*p+desc['final']['Z']*(1+desc['final']['Q'])
            require(raw['reply'].shape==(width+width%2,) and raw['reply'][2]==0,'returned shape, original-node coverage and echo comparison')
            if echo:require(raw['echo'].dtype==np.uint32 and raw['echo'].shape==(n,),'full exact intermediate')
            if attested:
                require(frame['first_packet_sha256']==digest(first_upload(data,i)) and
                        frame['echo_h2d_sha256']==(digest(raw['echo']) if echo else None),'actual initial input and re-entered hidden words')
            frame_metrics=metrics['frames'][i]
            frame_metrics['synchronized_host_ns']=frame['synchronized_host_ns']
            frame_metrics['final_logits_sha256']=digest(raw['logits'])
            frame_metrics['first_input_sha256']=digest(first_upload(data,i))
            frame_metrics['full_echo_sha256']=digest(raw['echo']) if echo else None
            require(frame['synchronized_host_ns']>0,'host interval')
    return metrics,words


def checked_case(name,pin):
    require(name==Path(name).name and name in ('small-arxiv-p2','arxiv-p16-e24','products-p16-e24'),'registered closure')
    case=HERE/'cases'/name
    require(device.sha(case/'release.json')==pin,'frozen release SHA')
    rel=read(case/'release.json')
    references={n:device.sha(HERE/'reference-source'/n) for n in ('worker_chain.py','build_chain.py')}
    require(rel['reference_sources_at_export']==references,'reference build and worker sources')
    require(rel['host']=={n:device.sha(HERE/n) for n in HOST},'registered driver or independent replay changed')
    for n,h in rel['files'].items():
        p=(case/n).resolve()
        require(p.is_relative_to(case.resolve()) and device.sha(p)==h,'frozen case file '+n)
    desc=images(case);checked_data(case)
    require(desc['first_sha256']==device.sha(case/'first.npz') and desc['last_sha256']==device.sha(case/'last.npz'),
            'compiled original inputs')
    sdk,expectations=observed(case,case/'sdk',None,'simulator',False)
    require(sdk['raw_vertices']==rel['raw_nodes'] and sdk['hidden_vertices']==rel['hidden_nodes'] and
            sdk['final_vertices']==rel['destinations'],'SDK covers the frozen raw/hidden/final graph')
    if (case/'bound-sdk').is_dir():observed(case,case/'bound-sdk',expectations,'simulator',True)
    return case,rel,desc,expectations


def run(case,pin,out,namespace):
    from cerebras.sdk.client import SdkLauncher
    case,release,desc,expected=checked_case(case,pin)
    out.mkdir(parents=True,exist_ok=False)
    artifact=out/'artifact';shutil.copytree(case/'hardware/out',artifact/'out')
    cap=out/'capture';cap.mkdir()
    receipt=dict(case=case.name,release_sha256=pin,role='hardware',namespace=namespace,actual_expected_elf=desc['roles']['hardware']['elf_sha256'],
                 build_sha256=device.sha(case/'build.json'),downloads={},numerical_status='pending')
    save(out/'collection.json',receipt)
    command=shlex.join(['timeout','--signal=INT','--kill-after=30s','2700s','cs_python','-B','worker_chain.py',
             '.','first.npz','last.npz','capture','--artifact','out','--cmaddr','%CMADDR%'])
    try:
        with SdkLauncher(str(artifact),simulator=False,mgmt_namespace=namespace,resource_cpu=1000,
                         resource_mem=8<<30,disable_version_check=False) as launcher:
            try:
                for p in (*[HERE/n for n in WORKER],case/'build.json',case/'first.npz',case/'last.npz'):
                    launcher.stage(str(p))
                response=launcher.run(command+' >worker.log 2>&1; rc=$?; echo "$rc" >worker-exit.txt; exit "$rc"')
                (out/'launcher-response.txt').write_text(str(response))
            finally:
                for n in FILES:
                    try:
                        launcher.download_artifact(n,str(out/n));receipt['downloads'][n]={'sha256':device.sha(out/n)}
                    except Exception as exc:receipt['downloads'][n]={'error':repr(exc)}
                    save(out/'collection.json',receipt)
        require(all('sha256' in v for v in receipt['downloads'].values()) and (out/'worker-exit.txt').read_text().strip()=='0',
                'complete physical return')
        measurements,_=observed(case,cap,expected,'hardware',True)
        save(out/'replay.json',measurements);receipt['numerical_status']='passed'
    finally:save(out/'collection.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('run','check'));p.add_argument('--case',required=True)
    p.add_argument('--release-sha256',required=True);p.add_argument('--out',type=Path);p.add_argument('--namespace',default='job-operator')
    a=p.parse_args()
    if a.action=='check':checked_case(a.case,a.release_sha256)
    elif not a.out:p.error('--out required for system run')
    else:run(a.case,a.release_sha256,a.out.resolve(),a.namespace)
