"""Reconstruct grouped Cora packet routes, sparse features, and edge occurrences from original inputs."""
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def schema(p, output=False):
    k,l,t,h,o=p["K"],p["L"],p["T"],p["H"],p["O"]
    if output:
        return dict(pre=[l*h,"float32"], logits=[l*o,"float32"], panel_sum=[k*t,"float32"],
                    stats=[8,"uint32"], clocks=[6,"uint16"], traffic=[4,"uint32"])
    dims=dict(slots=l, x_ptr=k+1, x_col=p["MX"], x_val=p["MX"], w0=p["FW"]*h, w1=h*o,
              edge_ptr=l+1, edge_src=p["MG"], edge_weight=p["MG"], send_slot=p["NSEND"],
              packet_header=2*p["NPLAN"], packet_tag=2*p["NPLAN"], packet_count=2*p["NPLAN"],
              packet_offset=2*p["NPLAN"], counts=6)
    return {name:[n, "float32" if name in ("x_val","w0","w1","edge_weight")
                    else "uint32" if name in ("packet_header","packet_tag") else "uint16"]
            for name,n in dims.items()}


def named_bytes(p):
    arrays = {**schema(p), **schema(p, True)}
    public = sum(n*np.dtype(dtype).itemsize for n,dtype in arrays.values())
    scratch_words = (p["K"]+p["NRX"]+p["L"])*p["T"] + 32+2+2
    return public + 4*scratch_words + 16  # eight u16 state variables, conservative


def audit_uploads(p, a, nodes, raw):
    c,r,k,l,t = (p[name] for name in ("C","R","K","L","T"))
    x=raw["x"]
    pes=c*r
    require(nodes.shape==(pes,l) and np.array_equal(np.sort(nodes[nodes>=0]),np.arange(len(x))), "node ownership")
    owner=np.empty(len(x),dtype=np.int64)
    for pe, owned in enumerate(nodes):
        owner[owned[owned>=0]]=pe
    groups=[np.flatnonzero(owner//c==row) for row in range(r)]
    require(max(map(len,groups))==k and np.all(nodes>=-1), "row ownership capacity")
    for pe in range(pes):
        row,col=divmod(pe,c)
        group=groups[row]
        slot=np.full(l,65535,np.uint16)
        valid=nodes[pe]>=0
        slot[valid]=np.searchsorted(group,nodes[pe,valid])
        np.testing.assert_array_equal(a["slots"][pe],slot)
        shard=np.zeros((p["FW"],p["H"]),dtype=np.float32)
        weight=raw["w0"][col::c]
        shard[:len(weight)]=weight
        np.testing.assert_array_equal(a["w0"][pe],shard.ravel())
        np.testing.assert_array_equal(a["w1"][pe],raw["w1"].ravel())
        ii,jj=np.nonzero(x[group,col::c])
        count=np.bincount(ii,minlength=k)
        np.testing.assert_array_equal(a["x_ptr"][pe],np.r_[0,count.cumsum()])
        for key,val in (("x_col",jj),("x_val",x[group[ii],col+c*jj])):
            np.testing.assert_array_equal(a[key][pe,:len(ii)],val)
            require(not np.any(a[key][pe,len(ii):]),"sparse feature padding")
    ghost_maps=[]
    totals=np.zeros((pes,4),np.uint32)
    for stage,width in enumerate((t,(p["O"]-1)%t+1)):
        maps=[{} for _ in range(pes)]
        received=np.zeros(pes,dtype=np.int64)
        factor=((p["H"]+t-1)//t+(p["O"]+t-1)//t-1) if stage==0 else 1
        for pe in range(pes):
            row,col=divmod(pe,c)
            sent=int(a["counts"][pe,stage])
            require(sent<=p["NPLAN"],"send plan length")
            words=0
            offset=0
            for j in range(sent):
                index=j+stage*p["NPLAN"]
                header=int(a["packet_header"][pe,index])
                dest_col=(header>>20)-4
                dest_row=((header>>8)&0xfff)-1
                require(0<=dest_row<r and dest_col==col and dest_row!=row,"packet destination")
                dp=dest_row*c+col
                count=int(a["packet_count"][pe,index])
                tag=int(a["packet_tag"][pe,index])
                first=tag&65535
                size=1+count*width
                require(0<count and first+count<=p["NRX"] and size<=31 and tag>>16==count,"packet extent")
                require((header&0xff)==((0x80 if row>dest_row else 0x40)|size),"route direction and length")
                require(int(a["packet_offset"][pe,index])==offset and offset+count<=p["NSEND"],"source offsets")
                for i in range(count):
                    slot=int(a["send_slot"][pe,offset+i])
                    require(slot<len(groups[row]) and first+i not in maps[dp],"unique valid ghost placement")
                    maps[dp][first+i]=int(groups[row][slot])
                offset+=count
                words+=size+1
                received[dp]+=1
                totals[dp,1]+=factor
                totals[dp,3]+=factor*(size+1)
            require(words==int(a["counts"][pe,4+stage]), "planned word count")
            totals[pe,0]+=factor*sent
            totals[pe,2]+=factor*words
        np.testing.assert_array_equal(received,a["counts"][:,2+stage])
        ghost_maps.append(maps)
    require(ghost_maps[0]==ghost_maps[1],"same ghost mapping across tile widths")
    order=[{} for _ in range(pes)]
    remote=[set() for _ in range(pes)]
    for s,d,coeff in zip(raw["src"],raw["dst"],raw["coeff"]):
        pe=int(owner[d])
        order[pe].setdefault(int(d),[]).append((int(s),coeff))
        if owner[s]//c!=pe//c:
            remote[pe].add(int(s))
    for pe,rows in enumerate(order):
        maps=ghost_maps[0][pe]
        require(set(maps.values())==remote[pe] and len(maps)==len(remote[pe]),"exact remote graph demand")
        ptr=a["edge_ptr"][pe].astype(int)
        require(ptr[0]==0 and np.all(np.diff(ptr)>=0) and ptr[-1]<=p["MG"],"bounded row pointers")
        for i,d in enumerate(nodes[pe]):
            original=sorted(rows.get(d,[]),key=lambda item:owner[item[0]]//c)
            require(len(original)==ptr[i+1]-ptr[i],"edge count")
            for e,(src,val) in enumerate(original,start=ptr[i]):
                loc=int(a["edge_src"][pe,e])
                source=int(groups[pe//c][loc]) if loc<k else maps[loc-k]
                require(source==src and a["edge_weight"][pe,e]==val,"ordered edge identity")
    tiles=(p["H"]+t-1)//t+(p["O"]+t-1)//t
    stats=np.zeros((pes,8),np.uint32)
    stats[:,[0,1,2,7]]=[0x600d,tiles,tiles,tiles]
    stats[:,3]=totals[:,1]
    stats[:,4]=a["x_ptr"][:,-1].astype(np.uint32)*p["H"]
    stats[:,5]=np.bincount(owner[raw["dst"]],minlength=pes)*(p["H"]+p["O"])
    stats[:,6]=np.count_nonzero(nodes>=0,axis=1)*p["H"]*p["O"]
    return groups,stats,totals
