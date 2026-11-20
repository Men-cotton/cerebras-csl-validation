"""Compare actual per-stage physical or simulated values for every active original row."""
import argparse
import json
from pathlib import Path

import numpy as np
from graph import read, require, save, sha


def compare(left,right,exact=False):
    a,b=read(left/"result.json"),read(right/"result.json")
    require(all(a[k]==b[k] for k in ("case_sha256","params","planned_frames","scope","dataset")),"different model case or layout")
    require(a["stage"]==b["stage"]=="PASS" and a["stopped"] and b["stopped"],"incomplete captures")
    for source,state in ((left,a),(right,b)):
        require(sha(source/"windows.jsonl")==state["journal_sha256"],"changed capture journal")
    with (left/"windows.jsonl").open() as ls,(right/"windows.jsonl").open() as rs:
        ranges=list(zip((json.loads(line) for line in ls),(json.loads(line) for line in rs),strict=True))
    require(len(ranges)==a["planned_frames"],"complete device frame coverage")
    cfg=a["params"];r=cfg["R"]
    phases={}
    for x,y in ranges:
        require(all(x[k]==y[k] for k in ("name","trial","mode","rows","edges","parts","tag")),"different row windows")
        name=x["name"]
        require(sha(left/name)==x["raw_sha256"] and sha(right/name)==y["raw_sha256"],"changed returned data")
        key=f't{x["trial"]}-s{x["mode"]}'
        score=phases.setdefault(key,{"rows":0,"edges":0,"different_f32_words":0,"max_abs":0.,
                                      "left_sum_window_peak_active":0,"right_sum_window_peak_active":0})
        score["rows"]+=x["rows"];score["edges"]+=x["edges"]
        score["left_sum_window_peak_active"]+=x["peak_active_cycles"]
        score["right_sum_window_peak_active"]+=y["peak_active_cycles"]
        fields=(2*cfg["F"],cfg["F"],cfg["Q"])[x["mode"]]
        offset=4+6*(2+x["parts"])
        with np.load(left/name,allow_pickle=False) as lz,np.load(right/name,allow_pickle=False) as rz:
            lhs,rhs=lz["reply"],rz["reply"]
            np.testing.assert_array_equal(lhs[:,:4],rhs[:,:4])
            for pe,n0 in enumerate(lhs[:,2]):
                n=int(n0)
                if not n:continue
                np.testing.assert_array_equal(lhs[pe,offset:offset+n],rhs[pe,offset:offset+n])
                lo=offset+r;hi=lo+n*fields
                lv,rv=lhs[pe,lo:hi],rhs[pe,lo:hi]
                xf,yf=lv.view(np.float32),rv.view(np.float32)
                score["different_f32_words"]+=int(np.count_nonzero(lv!=rv))
                score["max_abs"]=max(score["max_abs"],float(np.abs(xf-yf).max()))
                require(np.allclose(xf,yf,rtol=4e-4,atol=3e-4),"different active logits or intermediate activations")
                if exact:np.testing.assert_array_equal(lv,rv)
    return {"left_role":a["role"],"right_role":b["role"],"left_language":a["language"],"right_language":b["language"],
            "case_sha256":a["case_sha256"],"left_journal_sha256":a["journal_sha256"],"right_journal_sha256":b["journal_sha256"],
            "phases":phases}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("left",type=Path);p.add_argument("right",type=Path)
    p.add_argument("--exact",action="store_true");p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();save(a.out,compare(a.left,a.right,a.exact))
