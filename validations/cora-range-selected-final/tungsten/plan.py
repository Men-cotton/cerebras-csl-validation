"""Check recovered outgoing records and their order independently of the selected host packing logic."""

import numpy as np


def expected_plan(graph, sl, index, shards, method):
    src,dst,w=(graph[field] for field in ("src","dst","weights"))
    lo,hi=map(int,graph["bounds"][index:index+2])
    capacity=(len(graph["features"])+shards-1)//shards
    result=[]
    for y in range(shards):
        indices=np.flatnonzero((dst>=lo)&(dst<hi)&(src//capacity==y))
        by_row={}
        for e in indices:
            by_row.setdefault(int(dst[e]),[]).append(int(e))
        eligible=[]
        for row,edges in sorted(by_row.items()):
            last=edges[-1]
            if int(src[last])==row and (method=="tail" or len(edges)==1):
                eligible.append(row)
        if method=="tail":
            runs=[]
            for row in eligible:
                if not runs or runs[-1][-1]+1!=row:
                    runs.append([])
                runs[-1].append(row)
            eligible=max(runs,key=len) if runs else []
        selected={by_row[row][-1] for row in eligible}
        ordered=[e for row in sorted(by_row) for e in by_row[row]]
        wire_order=[e for e in ordered if e not in selected]+[e for e in ordered if e in selected]
        n=len(ordered);k=n-len(selected)
        assert int(sl["encoded_counts"][y])==k<<16|n
        pair=sl["pairs"][y,:n]
        assert np.all(((pair>>16)%8==0)&((pair&65535)%8==0))
        np.testing.assert_array_equal((pair>>16)//8+y*capacity,src[wire_order])
        np.testing.assert_array_equal((pair&65535)//8+lo,dst[wire_order])
        np.testing.assert_array_equal(sl["values"][y,:n].view(np.uint32),w[wire_order].view(np.uint32))
        # A stable merge by destination restores all original per-row FMA inputs, even duplicate edges.
        merged=sorted(wire_order,key=lambda e:int(dst[e]))
        assert merged==ordered
        width=2*(eligible[-1]-eligible[0]+1) if eligible else 0
        assert method!="tail" or width==2*(n-k)
        result.append([k,n-k,width])
    return np.asarray(result,np.uint32)


def check_original_plan(graph, sl, index, shards, actual, method):
    wanted=expected_plan(graph,sl,index,shards,method)
    assert actual.shape==(shards,4,3)
    np.testing.assert_array_equal(actual,np.broadcast_to(wanted[:,None,:],actual.shape))
