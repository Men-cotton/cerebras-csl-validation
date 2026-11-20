"""Verify the segment layout using the frozen cs-readelf rectangle and architecture report."""
import hashlib
import re
import struct


def equivalence(elf, fabric, application, decoded, *, active_rows):
    # cs-readelf is authoritative for the fabric-dependent rectangle encoding.
    raw = elf.read_bytes()
    if raw[:6] != b'\x7fELF\x02\x01':
        raise ValueError('expected an ELF64 little-endian artifact')
    eh = struct.unpack_from('<16sHHIQQQIHHHHHH', raw)
    decoded = decoded.read_text()
    if f'fabric is {fabric[0]}x{fabric[1]}' not in decoded.splitlines() or 'WSE-3' not in decoded:
        raise ValueError('ELF fabric mismatch')
    matches = re.findall(r'(Block|Scatter)\s+#(\d+) (\d+)x(\d+) @\((\d+),(\d+)\) \+(0x[0-9a-f]+).*? (\d+) bytes', decoded)
    ranges = {int(idx): (kind, int(w), int(h), int(x), int(y), int(address, 16), int(n))
              for kind, idx, w, h, x, y, address, n in matches}
    canonical = bytearray(raw)
    rows = tuple(active_rows)
    if not rows or any(y < 0 or y >= application[1] for y in rows):
        raise ValueError('invalid application row')
    owned = {(x, y): [] for x in range(application[0]) for y in rows}
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
                    if not (0 <= cx < application[0] and 0 <= cy < application[1]) or (va < 0xe400 and va+mem > 0xc000):
                        raise ValueError('load outside the application or in reserved SRAM')
                    if va < 0xc000:
                        if (cx, cy) not in owned:
                            raise ValueError('SRAM load on a row without a kernel')
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
