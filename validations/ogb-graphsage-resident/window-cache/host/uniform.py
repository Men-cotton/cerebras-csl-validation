"""Check row-broadcast replacements against the exact native DMA calls and returned SDK values."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
RESIDENT=HERE.parent.parent/'ogb-graphsage/resident'
sys.path.insert(0,str(RESIDENT))
from graph import decode, load_case, packed, read, require, schedule, sha, weights
from run import Device


@lru_cache(maxsize=4096)
def replacement(equal, words, penalty):
    width=len(equal)+1
    if penalty is None:  # At most the original single blocking call.
        return ((0,width,all(equal) and width>1),)
    # Choose the lowest input words + penalty per call within this DMA.
    # Fewer calls break ties. An original unicast is always an option.
    best=[(0,0)]*(width+1)
    plan=[None]*width
    for first in range(width-1,-1,-1):
        choice=None
        uniform=True
        for end in range(first+1,width+1):
            if end>first+1:
                uniform=uniform and equal[end-2]
            bcast=uniform and end>first+1
            sent=words*(1 if bcast else end-first)
            cost=(sent+penalty+best[end][0],1+best[end][1])
            if choice is None or cost<choice:
                choice=cost
                plan[first]=(first,end,bcast)
        best[first]=choice
    result=[]
    i=0
    while i<width:
        item=plan[i];result.append(item);i=item[1]
    return tuple(result)


class CounterRuntime:
    def __init__(self):
        self.totals=defaultdict(Counter)
        self.raw=None

    def get_id(self,name):return name

    def _count(self,name,sums):
        for k,v in sums.items():
            self.totals[name][k]+=v

    def memcpy_h2d(self,name,src,x,y,width,height,n,**flags):
        require(y==0 and height==1 and not flags['streaming'] and not flags['nonblock'],'blocking single compute row')
        values=np.ascontiguousarray(src).view(np.uint32).reshape(width,n)
        same=tuple(bool(x) for x in np.all(values[1:]==values[:-1],axis=1))
        result=dict(base_words=values.size,base_calls=1)
        for label,penalty in (('one_call',None),('penalty256',256),('penalty4096',4096),('min_words',0)):
            plan=replacement(same,n,penalty)
            pos=0
            result[label+'_words']=0
            result[label+'_calls']=len(plan)
            for begin,end,bcast in plan:
                require(begin==pos and begin<end,'region coverage');pos=end
                if bcast:
                    require(np.all(values[begin:end]==values[begin]),'different original payload in broadcast')
                result[label+'_words']+=n*(1 if bcast else end-begin)
                result[label+'_broadcasts']=result.get(label+'_broadcasts',0)+int(bcast)
            require(pos==width and result[label+'_words']<=values.size,'full original region')
        self._count(name,result)

    def memcpy_h2d_rowbcast(self,name,src,x,y,width,height,n,**flags):
        require(name=='weights' and x==y==0 and height==1 and src.size==n,'resident model bootstrap')
        sums={'base_words':n,'base_calls':1}
        for label in ('one_call','penalty256','penalty4096','min_words'):
            sums.update({label+'_words':n,label+'_calls':1,label+'_broadcasts':1})
        self._count(name,sums)

    def launch(self,name,**flags):
        require(name in ('begin_frame','accumulate','finish') and not flags['nonblock'],'native resident launch')

    def memcpy_d2h(self,host,symbol,*args,**flags):
        require(symbol=='reply' and host.size==self.raw.size,'raw SDK replay')
        host[:]=self.raw.ravel()


def main(a):
    dataset,scope,cache=a.dataset,a.scope,a.cache
    case=cache/f'resident-{dataset}-{scope}'
    build=cache/f'resident-{dataset}-csl-p32'
    cap=cache/f'resident-{dataset}-{scope}-csl-p32-capture'
    state=read(cap/'result.json');cfg=read(build/'build.json')['params']
    require(state['params']==cfg and state['case_sha256']==sha(case/'case.json') and state['build_sha256']==sha(build/'build.json')
            and state['platform']=={'simulation':True,'system':False} and state['loaded_image']==read(build/'build.json')['elf'],
            'original SDK case, image and platform')
    require(state['host_sha256']=={p.name:sha(p) for p in [RESIDENT/'run.py',RESIDENT/'graph.py',RESIDENT.parent/'transfer/rectangles.py']},
            'matching SDK transfer and data packing code')
    info,g=load_case(case);plan=schedule(g,cfg)
    reference={}
    for line in (cap/'windows.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['trial']==1:reference[row['name']]=row
    outputs=[np.full((len(g[k]),v),np.nan,np.float32) for k,v in zip(('raw_ids','a_ids','b_ids'),(2*cfg['F'],cfg['F'],len(g['b1'])))]
    counter=CounterRuntime()
    dev=Device.__new__(Device);dev.direct=False;dev.dev=counter;dev.cfg=cfg;dev.order=dev.dtype=None
    modes={};raw={}
    for mode in range(3):
        counter.totals.clear()
        for step in (s for s in plan if s['mode']==mode and s['trial']==1):
            name=f't1-s{mode}-w{step["index"]:04d}.npz'
            row=reference[name]
            require(sha(cap/name)==row['raw_sha256'] and row['mode']==mode,'original returned SDK words')
            raw[name]=row['raw_sha256']
            with np.load(cap/name,allow_pickle=False) as z:counter.raw=z['reply']
            previous=outputs[mode-1] if mode else None
            batch=packed(g,step,cfg,previous,1.)
            boot=(len(plan),weights(g,cfg)) if step['tag']==1 else None
            returned,words,_=dev.exchange(batch,step,boot)
            require(words==row['input_words'] and np.array_equal(returned,counter.raw),'actual SDK interface word count')
            outputs[mode][step['rows']],_,_=decode(returned,g,step,batch,cfg)
        modes[str(mode)]={name:dict(val) for name,val in sorted(counter.totals.items())}
    require(all(np.isfinite(s).all() for s in outputs),'complete original-input two-layer rollout')
    out=dict(dataset=dataset,scope=scope,trial=1,raw=raw,phases=modes,case_sha256=state['case_sha256'],
             build_sha256=state['build_sha256'],loaded_image=state['loaded_image'],worker=state['host_sha256'],
             class_count=info['classes'],destinations=info['destinations'])
    a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=('arxiv','products'));parser.add_argument('scope',choices=('four','128','whole'))
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
