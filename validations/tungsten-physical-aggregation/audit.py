"""Recheck a returned capture and reconstruct its maximum per-owner compute interval."""
import hashlib
import json
from pathlib import Path

from contract import TRIALS, WIDTHS, digest, input_words, verify

HERE = Path(__file__).resolve().parent
PORT_SHA = 'e4a311bc7f0f12ff52503703f4e233182c840339f765d01d49650022fe951d3e'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_capture(build, capture, role):
    width = build['width']
    if width not in WIDTHS or capture['owners'] != width or capture['backend'] != role:
        raise ValueError('wrong captured geometry or backend')
    if (capture['passed'] is not True or capture['stop_success'] is not True or capture['stage'] != 'complete'
            or capture['system_target'] is not (role == 'hardware') or capture['simulator_target'] is not (role == 'simulator')):
        raise ValueError('runtime completion or target unverified')
    if (capture['elf_sha256'] != build['roles'][role]['elf_sha256']
            or capture['source'] != build['source'] or capture['port_map_sha256'] != PORT_SHA):
        raise ValueError('capture came from different sources, port mapping, or ELF')
    for name in ('worker.py', 'contract.py'):
        if capture['worker_source'][name] != sha(HERE/name):
            raise ValueError('worker changed since simulation or execution')
    logical = [[x,y] for y in range(2) for x in range(width)]
    if [v['logical'] for v in capture['coordinates']] != logical:
        raise ValueError('incomplete logical coordinates')
    if [v['physical'] for v in capture['coordinates']] != logical:
        raise ValueError('physical geometry differs from edge placement')
    if [r['trial'] for r in capture['trials']] != list(TRIALS):
        raise ValueError('incomplete input sequence')
    for r in capture['trials']:
        verified = verify(width, r['trial'], r['reply'])
        if (r['input_sha256'] != digest(input_words(r['trial']))
                or any(r[k] != v for k, v in verified.items()) or r['send_receive_wait_ns'] <= 0):
            raise ValueError('raw input, output, clock, or timing audit failed')
    return [r['max_compute_cycles'] for r in capture['trials']]


def load_verified(root, width):
    root = Path(root)
    build = json.loads((root/'build.json').read_text())
    if build['width'] != width or build['roles']['hardware']['fabric'] != [762,1172]:
        raise ValueError('wrong physical dimensions')
    if build['roles']['simulator']['fabric'] != [width,2]:
        raise ValueError('wrong simulation dimensions')
    for role in ('simulator','hardware'):
        out = root/role/'out'
        if (sha(out/'program.elf') != build['roles'][role]['elf_sha256']
                or sha(out/'out_port_map.json') != PORT_SHA or build['ports_sha256'] != PORT_SHA):
            raise ValueError('paired ELF or SDK map changed')
    # The packaging step copies exactly the two hardware inputs into a fresh
    # directory. The simulator can add generated default/coord ELFs in its tree.
    if {f.name for f in (root/'hardware/out').iterdir()} != {'out_port_map.json','program.elf'}:
        raise ValueError('unexpected hardware artifact member')
    if sha(root/'program.img') != build['machine_image_sha256']:
        raise ValueError('machine image changed')
    for name in ('owner.w', 'collect.w', 'program.paint'):
        if sha(root/'source'/name) != build['source'][name]:
            raise ValueError('input sources changed')
        if name == 'collect.w' and sha(HERE/name) != build['source'][name]:
            raise ValueError('collector changed since compilation')
        if name == 'owner.w':
            sources = [HERE/'owner.w', HERE/'owner-degree4.w']
            if not any(path.is_file() and sha(path) == build['source'][name] for path in sources):
                raise ValueError('owner compute source changed since compilation')
    capture = json.loads((root/'verification/result.json').read_text())
    check_capture(build, capture, 'simulator')
    return build


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('build_directory', type=Path)
    p.add_argument('capture', type=Path)
    p.add_argument('--owners', choices=WIDTHS, type=int, required=True)
    p.add_argument('--role', choices=('simulator','hardware'), required=True)
    a = p.parse_args()
    print(check_capture(load_verified(a.build_directory, a.owners), json.loads(a.capture.read_text()), a.role))
