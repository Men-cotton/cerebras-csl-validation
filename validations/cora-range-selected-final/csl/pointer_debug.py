"""Read the programmed source pointers and the actual residual/scalar-index graph from each stopped PE."""

import hashlib
import json
from pathlib import Path

import numpy as np


def capture(sl,r,artifact,out,index):
    from elftools.elf.elffile import ELFFile
    from cerebras.elf.cself import ELFMemory
    from cerebras.sdk.debug.debug_util import debug_util
    coordinates={}
    full_names={}
    optional=("diagonal_weights","tail_weights")
    for elf in sorted((artifact/"bin").glob("*.elf")):
        with elf.open("rb") as f:
            section=ELFFile(f).get_section_by_name(".symtab")
            syms={s.name.rsplit("$$",1)[-1]:(s.name,int(s["st_value"])) for s in section.iter_symbols()}
        for x,y in ELFMemory(str(elf)).iter_coordinates():
            if 4<=x<8 and 1<=y<=r:
                at=(y-1,x-4)
                assert at not in coordinates
                coordinates[at]=syms["features"][1]
                for field in ("inputs","graph")+tuple(f for f in optional if f in syms):
                    full_names.setdefault(field,set()).add(syms[field][0])
    assert len(coordinates)==r*4 and all(len(v)==1 for v in full_names.values())
    debug=debug_util(str(artifact),cmaddr=None)
    dtype={"inputs":np.uint16,"graph":np.uint32,"diagonal_weights":np.float32,"tail_weights":np.float32}
    arrays={k:debug.get_symbol_rect(((4,1),(4,r)),next(iter(v)),dtype[k]).transpose(1,0,2).copy() for k,v in full_names.items()}
    arrays["feature_base"]=np.array([[coordinates[y,x] for x in range(4)] for y in range(r)],np.int64)
    ecap=sl["pairs"].shape[1]
    records=0
    for y,n0 in enumerate(sl["counts"]):
        n=int(n0);k=int(sl["encoded_counts"][y]>>16)
        pair=sl["pairs"][y,:n];vals=sl["values"][y,:n].view(np.uint32)
        graph=np.concatenate((np.array((index,sl["encoded_counts"][y]),np.uint32),sl["pairs"][y],sl["values"][y].view(np.uint32)))
        for x in range(4):
            np.testing.assert_array_equal(arrays["graph"][y,x],graph)
            np.testing.assert_array_equal(arrays["inputs"][y,x,:k],coordinates[y,x]+(pair[:k]>>16))
            if "diagonal_weights" in arrays:
                expected=np.zeros(512,np.uint32);positions=(pair[k:]&65535)//4
                expected[positions]=vals[k:];expected[positions+1]=vals[k:]
                np.testing.assert_array_equal(arrays["diagonal_weights"][y,x].view(np.uint32),expected)
            if "tail_weights" in arrays:
                np.testing.assert_array_equal(arrays["tail_weights"][y,x,:n-k].view(np.uint32),vals[k:])
            records+=k
    np.savez_compressed(out/"pointer-sram.npz",**arrays)
    digest=hashlib.sha256((out/"pointer-sram.npz").read_bytes()).hexdigest()
    s=dict(source="stopped 64-PE SDK simulator",window=index,pe_count=r*4,checked_residual_source_pointers=records,
           complete_graph_words=r*4*(2+2*ecap),symbols=sorted(full_names),raw_sha256=digest)
    (out/"pointer-sram.json").write_text(json.dumps(s,indent=2,sort_keys=True)+"\n")
    return s
