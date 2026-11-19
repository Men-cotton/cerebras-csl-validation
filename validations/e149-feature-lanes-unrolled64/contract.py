"""Numerical and frame contract for E149 matched F8."""
import hashlib
import numpy as np

PIN = "b44fac8b662d84afb3cf7fe9a055f928e4f95fdfe878c75833153e29594b6338"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(test,message):
    if not test:
        raise ValueError(message)


def verify_graph(graph):
    require(graph["features"].shape==(64,8) and len(graph["values"])==256,"E149 shape")
    require(np.array_equal(np.bincount(graph["sources"]//16, minlength=4),[64]*4),"balanced source shards")


def panel(graph, trial):
    require(trial in (1,2,3),"A/B/A frame")
    return graph["features"]*np.float32(-.5 if trial==2 else 1.)


def input_words(graph, trial):
    data=panel(graph,trial).reshape(4,16,4,2).transpose(0,3,1,2).copy()
    return np.concatenate((np.array([trial,0x53454c4c],np.uint32),data.ravel().view(np.uint32)))


def check(graph, trial, reply):
    reply=np.asarray(reply,np.uint32)
    require(reply.shape==(584,),"full reply")
    out=np.empty((64,8),np.float32)
    intervals=[]
    for x in range(2):
        column=reply[x*292:(x+1)*292]
        for y in range(4):
            chunk=column[y*9:(y+1)*9].astype(np.int64)
            require(chunk[:3].tolist()==[trial,y*2+x,64] and np.all(chunk[3:]<=65535),"PE identity and completion")
            begin,end=chunk[3:].reshape(2,3)@np.array([1,65536,1<<32],np.int64)
            delta=int((end-begin)%(1<<48))
            require(0<delta<1000000,"PE interval")
            intervals.append(delta)
        out[:,x::2]=column[36:].view(np.float32).reshape(64,4)
    expected=np.zeros((64,8),np.float64)
    np.add.at(expected,graph["destinations"],graph["values"][:,None].astype(np.float64)*panel(graph,trial)[graph["sources"]])
    require(np.array_equal(out.astype(np.float64),expected),"all E149 owner values")
    return dict(max_pe_cycles=max(intervals),pe_cycles_x_then_y=intervals,owner_sha256=hashlib.sha256(out.tobytes()).hexdigest())


def packed_graph(graph):
    dest,src,w=(graph[k] for k in ("destinations","sources","values"))
    outputs={k:np.empty((4,2,64),dtype) for k,dtype in (("sources",np.uint32),("destinations",np.uint16),("weights",np.float32))}
    for y in range(4):
        indices=np.flatnonzero(src//16==y)
        order=indices[np.argsort(dest[indices],kind="stable")]
        outputs["sources"][y]=(src[order]-16*y)*4
        outputs["destinations"][y]=dest[order]*4
        outputs["weights"][y]=w[order]
    return outputs
