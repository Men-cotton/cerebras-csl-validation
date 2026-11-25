"""Bind the paired worker to every loaded physical CSL ELF, its ABI and the frozen inputs."""
import argparse
import json
from pathlib import Path

from paired_worker import execute, save, sha
import portable
import saved_audit


def after_run(before, after):
    """Keep every compiled ELF byte-identical and record SDK-generated route metadata."""
    portable.admit_compiled_inventory(after)
    original, final = before['files'], after['files']
    if any(final.get(name) != entry for name,entry in original.items()):
        raise ValueError('linked firmware changed during execution')
    added = {name: row for name,row in final.items() if name not in original}
    known = {'generated/coord.elf','generated/MEMCPY_XY_ROUTES.elf','generated/default.elf'}
    if (set(added)-known or any(row['static_sram_end'] or row['sram_ranges'] or row['outside_sram_ranges']
                               for row in added.values())):
        raise ValueError('unexpected additional runtime ELF or allocated compute section')
    return added


def run(build, case, out, cmaddr):
    doc = json.loads(case.read_text())
    inv = portable.compiled_inventory(build)
    portable.admit_compiled_inventory(inv)
    saved_audit.check_abi(build, doc)
    proof = dict(elf=inv, rpc_sha256=sha(build/'bin/out_rpc.json'), params_sha256=sha(build/'out.json'),
                 case_sha256=sha(case), worker_sha256=sha(Path(__file__).with_name('paired_worker.py')))
    save(out.parent/'linked.json', proof)
    execute(build, case, out, cmaddr)
    proof['runtime_created']=after_run(inv,portable.compiled_inventory(build))
    for path in ('out.json','bin/out_rpc.json'):
        expected=proof['params_sha256' if path=='out.json' else 'rpc_sha256']
        if sha(build/path)!=expected:
            raise ValueError('linked runtime metadata changed: '+path)
    save(out.parent/'linked.json', proof)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('build','case','out'):
        p.add_argument(n,type=Path)
    p.add_argument('--cmaddr')
    a=p.parse_args()
    run(a.build.resolve(),a.case.resolve(),a.out.resolve(),a.cmaddr)
