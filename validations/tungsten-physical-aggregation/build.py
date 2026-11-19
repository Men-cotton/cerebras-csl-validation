"""Build both fabric ELFs from one machine image and check their ordered loads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

from contract import WIDTHS

HERE = Path(__file__).resolve().parent
SDK = HERE.parents[1]/'sdr-toolset'
FABRIC = (762, 1172)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layout(width):
    parts = ['let host_in = color(0)', 'let host_out = color(1)', 'let down = color(23)',
             'let odd = color(21)', 'let even = color(20)']
    tiles = []
    for x in range(width):
        owner, relay = f'owner{x}', f'gather{x}'
        tiles.extend([owner, relay])
        rows = 64 // width
        packet = 8 + 4*rows
        ahead, behind = ('odd', 'even') if x % 2 else ('even', 'odd')
        outgoing = behind if x else 'host_out'
        parts.extend([f'let {owner} = tile({x} 0)',
                      f"{owner} : code('owner [X:{x} ROWS:{rows} rx:host_in tx:down])",
                      f"{owner} : paint(host_in [L>{'CR' if x+1 < width else 'C'}])",
                      f'{owner} : paint(down [C>U])',
                      f'let {relay} = tile({x} 1)',
                      f"{relay} : code('collect [OWN:{packet} TRAILING:{packet*(width-x-1)} local_rx:down right_rx:{ahead} tx:{outgoing}])",
                      f'{relay} : paint(down [D>C])',
                      f'{relay} : paint({outgoing} [C>L])'])
        if x + 1 < width:
            parts.append(f'{relay} : paint({ahead} [R>C])')
    parts.append('union('+' '.join(tiles)+')')
    return '\n'.join(parts)+'\n'


def equivalence(elf, fabric, application, readelf):
    # cs-readelf is authoritative for the fabric-dependent rectangle encoding.
    raw = elf.read_bytes()
    if raw[:6] != b'\x7fELF\x02\x01':
        raise ValueError('expected an ELF64 little-endian artifact')
    eh = struct.unpack_from('<16sHHIQQQIHHHHHH', raw)
    decoded = subprocess.check_output([readelf, '--arch', '--fs', '--rect', str(elf)], text=True)
    (elf.parent/'ranges.txt').write_text(decoded)
    if f'fabric is {fabric[0]}x{fabric[1]}' not in decoded.splitlines() or 'WSE-3' not in decoded:
        raise ValueError('ELF fabric mismatch')
    matches = re.findall(r'(Block|Scatter)\s+#(\d+) (\d+)x(\d+) @\((\d+),(\d+)\) \+(0x[0-9a-f]+).*? (\d+) bytes', decoded)
    ranges = {int(idx): (kind, int(w), int(h), int(x), int(y), int(address, 16), int(n))
              for kind, idx, w, h, x, y, address, n in matches}
    canonical = bytearray(raw)
    owned = {(x, y): [] for x in range(application[0]) for y in range(application[1])}
    loads, notes = [], 0
    for i in range(eh[10]):
        off = eh[5]+i*eh[9]
        ty, _, start, va, _, n, mem, _ = struct.unpack_from('<IIQQQQQQ', raw, off)
        if start + n > len(raw):
            raise ValueError('truncated ELF')
        if ty == 1:
            kind, w, h, x, y, address, size = ranges.pop(i)
            if va != address or mem != size or n != size*(w*h if kind == 'Scatter' else 1):
                raise ValueError('ELF and cs-readelf load differ')
            for cy in range(y, y+h):
                for cx in range(x, x+w):
                    if (cx, cy) not in owned or (va < 0xe400 and va+mem > 0xc000):
                        raise ValueError('load outside the application or in reserved SRAM')
                    if va < 0xc000:
                        owned[cx, cy].append((va, va+mem))
            canonical[off+24:off+32] = bytes(8)
            loads.append((i, kind, x, y, w, h, address, size))
        elif ty == 4:
            if (n != 32 or raw[start:start+21] != struct.pack('<III', 9, 8, 1) + b'Cerebras\0'
                    or raw[start+24:start+32] != struct.pack('<II', *fabric)):
                raise ValueError('unexpected ELF note')
            # Standard ELF note names are padded to four bytes; mielf may leave
            # the name padding unspecified between the two converter processes.
            canonical[start+21:start+32] = bytes(11)
            notes += 1
    if ranges or notes != 1 or not loads:
        raise ValueError('incomplete ELF inspection')
    for segments in owned.values():
        segments.sort()
        if not segments or any(b > c for (_, b), (c, _) in zip(segments, segments[1:])):
            raise ValueError('missing or overlapping SRAM loads')
    return dict(loads=loads, canonical_sha256=hashlib.sha256(canonical).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('width', type=int, choices=WIDTHS)
    p.add_argument('directory', type=Path)
    p.add_argument('--sdk', type=Path, default=SDK)
    p.add_argument('--readelf', required=True, help='official cs-readelf executable or SDK wrapper')
    p.add_argument('--port-map', type=Path, required=True, help='out_port_map.json from the professor\'s SDK layout')
    a = p.parse_args()
    root = a.directory.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root/'source'
    source.mkdir()
    for name in ('owner.w', 'collect.w'):
        shutil.copyfile(HERE/name, source/name)
    (source/'program.paint').write_text(layout(a.width))
    sdk = a.sdk.resolve()
    env = {**os.environ, 'PATH': str(sdk/'bin')+':'+os.environ['PATH'],
           'LD_LIBRARY_PATH':str(sdk/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),
           'SDK_INSTALL':str(sdk), 'CARBIDE_INSTALL':str(sdk)}
    commands = []
    def run(argv):
        commands.append(argv)
        with (root/'compile.log').open('ab') as log:
            log.write((' '.join(argv)+'\n').encode())
            log.flush()
            subprocess.run(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    version = subprocess.check_output(['carbide','version'], env=env, text=True).strip()
    run(['paint','arch=sdr','toolname','w','carbide','toolname','yeti','carbide',
         'lib='+str(source),'source='+str(source/'program.paint'),'image=program.obj',
         'map=program.map','note=program.note','entrymap=program.entry','regiondump=regions.json'])
    run(['milink','image=program.obj','out=program.img','map=program.map','note=program.note'])
    run(['weight','image=program.img','map=program.map','note=program.note'])
    ports = json.loads(a.port_map.read_text())
    expected = dict(buses=[dict(io_type='IN', lvds_id=0, port_name='rx_port'),dict(io_type='OUT', lvds_id=1, port_name='tx_port')],
                    lvds_ports=[dict(color=0,loc=0,side='WEST'),dict(color=1,loc=1,side='WEST')])
    if ports != expected:
        raise ValueError('SDK-generated port map does not match ingress/egress endpoints')
    manifest = dict(width=a.width, compiler_version=version, source={f.name:sha(f) for f in source.iterdir()},
                    machine_image_sha256=sha(root/'program.img'), ports_sha256=sha(a.port_map), roles={})
    eq = {}
    for role, dims in [('simulator', (a.width, 2)), ('hardware', FABRIC)]:
        out = root/role/'out'
        out.mkdir(parents=True)
        elf = out/'program.elf'
        run(['mielf','image=program.img','elf='+str(elf),f'w={dims[0]}',f'h={dims[1]}'])
        eq[role] = equivalence(elf, dims, (a.width, 2), a.readelf)
        (out/'ranges.txt').rename(root/role/'ranges.txt')
        shutil.copyfile(a.port_map, out/'out_port_map.json')
        manifest['roles'][role] = dict(fabric=list(dims), elf_sha256=sha(elf))
    if eq['simulator'] != eq['hardware']:
        raise ValueError('paired ELF loads or non-fabric bytes differ')
    manifest.update(commands=commands, equivalence=eq['hardware'])
    (root/'build.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
