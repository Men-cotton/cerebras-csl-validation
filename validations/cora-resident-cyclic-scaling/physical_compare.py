"""Compare verified physical reports for contiguous and cyclic source placement, including loaded ELF bytes."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
import tarfile

import numpy as np
from contract import require, save, sha


def read(path):
    return json.loads(path.read_text())


def load_digest(data):
    require(data[:6] == b"\x7fELF\x02\x01", "64-bit little-endian CSL application ELF")
    pos = struct.unpack_from("<Q",data,32)[0]
    stride, count = struct.unpack_from("<HH",data,54)
    require(stride>=56 and pos+count*stride<=len(data),"program header inventory")
    entries=[]
    for i in range(count):
        kind,flags,off,va,pa,length,span,align=struct.unpack_from("<IIQQQQQQ",data,pos+i*stride)
        if kind!=1:
            continue
        require(off+length<=len(data) and length<=span,"ELF LOAD bounds")
        entries.append([flags,va,pa,length,span,align,hashlib.sha256(data[off:off+length]).hexdigest()])
    require(entries,"executable LOAD segments")
    return hashlib.sha256(json.dumps(entries,separators=(",",":")).encode()).hexdigest()


def loads(archive,record):
    require(archive.is_file() and sha(archive)==record["sha256"],"returned physical compiler archive")
    payload={}
    with tarfile.open(archive,"r:*") as tar:
        for entry in tar:
            path=PurePosixPath(entry.name)
            if not (entry.isfile() and path.parent.name=="bin" and path.name in record["application_elfs"]):
                continue
            require(entry.size<=32<<20 and path.name not in payload, "unique application ELF")
            data=tar.extractfile(entry).read()
            require(hashlib.sha256(data).hexdigest()==record["application_elfs"][path.name],"compiled ELF identity")
            payload[path.name]=load_digest(data)
    require(set(payload)==set(record["application_elfs"]),"complete ELF slots")
    return payload


def compare(range_report, cyclic_report, range_build, cyclic_build, root):
    release=read(root/"release.json")
    expected=read(root/"order.json")["cases"]
    records={"range":read(range_report),"cyclic":read(cyclic_report)}
    expected_rel={"range":release["parent_range_release_sha256"],"cyclic":sha(root/"release.json")}
    for mode,data in records.items():
        require(data["backend"]=="hardware" and data["release_sha256"]==expected_rel[mode],"independently audited physical report")
    out={}
    for idx,r in enumerate((4,16)):
        key=str(r)
        original,balanced=(records[m]["cases"] for m in ("range","cyclic"))
        builds=[]
        paths=[]
        for mode,file in (("range",range_build[idx]),("cyclic",cyclic_build[idx])):
            b=read(file)
            require(records[mode]["physical_compile_archives"][key]["build_record_sha256"]==sha(file)
                    and b["shards"]==r and b["release_sha256"]==expected_rel[mode],
                    "compiled physical program in its own audited run")
            builds.append(b)
            paths.append(file.parent/b["artifact"])
        require(builds[0]["flags"]==builds[1]["flags"],"same compile geometry and flags")
        exact = builds[0]["application_elfs"] == builds[1]["application_elfs"]
        equivalent = exact or loads(paths[0],builds[0])==loads(paths[1],builds[1])
        require(equivalent, "identical CSL loaded program across source placements")
        gains={}
        for lang in ("csl","tungsten"):
            case=f"{lang}-r{r}"
            a,b=original[case],balanced[case]
            for mode,x in (("range",a),("cyclic",b)):
                require(x["owner_sha256"]==expected[key][mode]["bits"] and x["validation"]==[402,500]
                        and x["original_class_mismatches"]==[0,0,0],"same original-ID FMA oracle")
            gains[lang]={m:(np.asarray(a[m])/np.asarray(b[m])).tolist() for m in
                         ("setup_sum","compute_sum","max_owner_cumulative_sum","phase_max_cumulative_sum")}
        out[key]=dict(exact_csl_elf_inventory=exact,identical_csl_loads=equivalent,
                      tungsten_hardware_elf=release["direct"][key]["roles"]["hardware"]["elf_sha256"],
                      placement_speedup=gains,
                      range={lang:original[f"{lang}-r{r}"] for lang in ("csl","tungsten")},
                      cyclic={lang:balanced[f"{lang}-r{r}"] for lang in ("csl","tungsten")})
    return dict(backend="physical; both physical reports replayed independently",range_report_sha256=sha(range_report),
                cyclic_report_sha256=sha(cyclic_report),cases=out)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--range-report",type=Path,required=True)
    p.add_argument("--cyclic-report",type=Path,required=True)
    p.add_argument("--range-build",type=Path,nargs=2,required=True,metavar=("R4","R16"))
    p.add_argument("--cyclic-build",type=Path,nargs=2,required=True,metavar=("R4","R16"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    save(a.output,compare(a.range_report,a.cyclic_report,a.range_build,a.cyclic_build,Path(__file__).resolve().parent))
