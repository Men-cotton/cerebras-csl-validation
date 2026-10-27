"""E176 parameterized resident database and independent region-sort oracle."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec); sys.modules[name]=value
    spec.loader.exec_module(value); return value

host=module("e176_original_topk",HERE.parent/"model.py")
reference=module("e176_independent_topk",HERE.parent/"reference.py")
storage=module("e176_bounded_storage",ROOT/"kernels/sparse/resident-2d-spmm/physical_schedule/armed_clock/storage.py")
# Same checked flat-NPZ format, with explicit larger complete-mesh bounds.
storage.LIMIT=64*1024**2
storage.JSON_LIMIT=64*1024**2
need=storage.need
INVALID=0xffffffff
QNAN=0x7fc00176
SEQUENCE=("A","B","A")
INPUT_NAMES=("database","identities","meta","query_seed")
OUTPUT_NAMES=("query","distances","best","stats")

def parameters(width=2,height=2,capacity=16,dim=8,k=4):
    p=dict(Pw=width,Ph=height,CAP=capacity,D=dim,K=k)
    need(all(type(v) is int for v in p.values()),"integer parameters")
    need(2<=width<=512 and 2<=height<=512,"mesh bounds")
    need(1<=capacity<=4096 and 1<=dim<=4096 and capacity*dim<=4096 and 1<=k<=256,"local capacity bounds")
    need(k<=width*height*capacity,"K no greater than database")
    need(declared_bytes(p)<28*1024,"declared numerical SRAM bound")
    need(width*height*(4*capacity*dim+4*capacity+8+4*dim)<=32*1024**2,"full input snapshot allocation bound")
    need(width*height*8*k<=32*1024**2,"full candidate snapshot allocation bound")
    return p

def declared_bytes(p):
    return 4*p["CAP"]*(p["D"]+2)+8*p["D"]+24*p["K"]+92

def spec(p):
    c,d,k=p["CAP"],p["D"],p["K"]
    return dict(database=(c*d,"float32"),identities=(c,"uint32"),meta=(2,"uint32"),query_seed=(d,"float32"),
                query=(d,"float32"),distances=(c,"float32"),best=(2*k,"uint32"),stats=(16,"uint32"))

def schema(p,names=None):
    return {n:((p["Pw"]*p["Ph"],size),np.dtype(dtype).str) for n,(size,dtype) in spec(p).items()
            if names is None or n in names}

def witness(actual,expected,label):
    need(actual.dtype==expected.dtype and actual.shape==expected.shape and actual.tobytes()==expected.tobytes(),"exact bytes: "+label)

def make_case(p,profile="balanced"):
    need(profile in ("balanced","uneven"),"known profile")
    c,d,P=p["CAP"],p["D"],p["Pw"]*p["Ph"]
    counts=[c]*P
    if profile=="uneven": counts[-1]=0; counts[0]=max(0,c-1)
    n=sum(counts); need(n>=p["K"] and n>=2,"profile has enough rows")
    rows=np.arange(n,dtype=np.int64)
    database=np.empty((n,d),np.float32)
    database[:,0]=rows.astype(np.float32)/np.float32(64)
    for j in range(1,d): database[:,j]=((rows*17+j*13)%257-128).astype(np.float32)/np.float32(16)
    ids=(0x01000001+n-1-rows).astype(np.uint32)
    owners=[]; at=0
    for count in counts: owners.append(list(range(at,at+count))); at+=count
    return dict(params=dict(p),profile=profile,database=database,ids=ids,owners=owners,
                queries=[database[-1].copy(),database[0].copy(),database[-1].copy()])

def pack(case,check_scores=True):
    p=case["params"]; parameters(p["Pw"],p["Ph"],p["CAP"],p["D"],p["K"])
    data=case["database"]; queries=case["queries"]; n=len(data); P=p["Pw"]*p["Ph"]
    need(len(queries)==3,"A/B/A query count")
    for q in queries:
        checked,_=host.admit_inputs(data,q)
        need(checked.shape[1]==p["D"],"query/database width")
        if check_scores: host.direct_scores(data,q)
    witness(queries[0],queries[2],"query restoration")
    ids=host.admit_ids(case["ids"],n); owners=case["owners"]
    need(p["K"]<=n<=P*p["CAP"],"row count")
    need(len(owners)==P and all(len(r)<=p["CAP"] for r in owners) and
         all(type(i) is int for r in owners for i in r) and sorted(i for r in owners for i in r)==list(range(n)),"exact ownership")
    out=dict(database=np.zeros((P,p["CAP"]*p["D"]),np.float32),identities=np.full((P,p["CAP"]),INVALID,np.uint32),
             meta=np.zeros((P,2),np.uint32),query_seed=np.full((P,p["D"]),QNAN,np.uint32).view(np.float32))
    for pe,rows in enumerate(owners):
        out["database"][pe,:len(rows)*p["D"]]=data[rows].ravel()
        out["identities"][pe,:len(rows)]=ids[rows]
        out["meta"][pe]=(len(rows),0x176)
    return out

def selected(scores,ids,k):
    out=np.tile(np.asarray([0x7f800000,INVALID],np.uint32),k)
    if len(scores):
        chosen,values=reference.global_sort(scores,ids,min(k,len(scores)))
        out[:2*len(chosen):2]=values.view(np.uint32); out[1:2*len(chosen):2]=chosen
    return out

def expected(case,generation):
    p=case["params"]; w,h,c,d,k=(p[n] for n in ("Pw","Ph","CAP","D","K")); P=w*h
    q=case["queries"][generation-1]
    # Score each database row only once per query. Regions use independent global sorting.
    scores=reference.scalar_fp32_scores(case["database"],q)
    out=dict(query=np.tile(q,(P,1)),distances=np.full((P,c),np.inf,np.float32),
             best=np.empty((P,2*k),np.uint32),stats=np.zeros((P,16),np.uint32))
    for pe,rows in enumerate(case["owners"]):
        x,y=pe%w,pe//w
        out["distances"][pe,:len(rows)]=scores[rows]
        # Nonleft PE retains the suffix of its row; a left PE retains all rows below it too.
        region=range(y*w+x,(y+1)*w) if x else range(y*w,P)
        selected_rows=[i for owner in region for i in case["owners"][owner]]
        out["best"][pe]=selected(scores[selected_rows],case["ids"][selected_rows],k)
        rr,rt,cr,ct=int(x<w-1),int(x>0),int(x==0 and y<h-1),int(x==0 and y>0)
        out["stats"][pe]=[0x600d,generation,1,generation,len(rows),1,len(rows)*d,len(rows),rr,rt,cr,ct,0,6,rr+cr,d]
    return out

def loaded(case):
    p=case["params"]; out=np.zeros((p["Pw"]*p["Ph"],16),np.uint32)
    out[:,0]=0xad01; out[:,2]=1; out[:,4]=list(map(len,case["owners"]))
    return out

def query_seed(case,generation):
    p=case["params"]; out=np.full((p["Pw"]*p["Ph"],p["D"]),QNAN,np.uint32).view(np.float32)
    out[0]=case["queries"][generation-1]; return out

def audit_arrays(case,generation,arrays,expected_arrays=None):
    wanted=expected(case,generation) if expected_arrays is None else expected_arrays
    need(set(arrays)==set(wanted),"complete output inventory")
    for n in wanted: witness(arrays[n],wanted[n],n)
    return dict(generation=generation,ids=arrays["best"][0,1::2].tolist(),score_bits=arrays["best"][0,::2].tolist())

def cost(case):
    p=case["params"]; P=p["Pw"]*p["Ph"]; counts=list(map(len,case["owners"]))
    return dict(rows=sum(counts),rows_per_pe=counts,dimensions=p["D"],k=p["K"],
                distance_operations_per_pe=[3*n*p["D"] for n in counts],
                candidate_bytes_per_pe=24*p["K"],declared_numerical_bytes_per_pe=declared_bytes(p),
                candidate_byte_hops_per_query=8*p["K"]*(P-1),query_byte_hops_per_query=4*p["D"]*(P-1),
                longest_merge_chain=p["Pw"]+p["Ph"]-2,
                max_to_mean_rows=max(counts)*P/sum(counts),
                scope="analytic application arrays and payload byte-hops; excludes code, stack, SDK, control and stall")
