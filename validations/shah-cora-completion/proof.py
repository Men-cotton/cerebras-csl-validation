"""Generated-kernel identity/SRAM gate, derived from native Cora portable.py."""
import hashlib
import os
from pathlib import Path
import re
import struct
import time
ELF_TOTAL_LIMIT = 256 << 20
def require(ok,message):
    if not ok: raise ValueError(message)
def bounded(path,limit):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=limit,'bounded generated file')
    return path.read_bytes()

def _elf_storage(payload):
    require(len(payload) >= 16 and payload[:4] == b"\x7fELF" and payload[4] in (1, 2)
            and payload[5] in (1, 2), "generated ELF header")
    wide, endian = payload[4] == 2, "<" if payload[5] == 1 else ">"
    header_fmt = endian + ("HHIQQQIHHHHHH" if wide else "HHIIIIIHHHHHH")
    require(len(payload) >= 16 + struct.calcsize(header_fmt), "generated ELF header extent")
    header = struct.unpack_from(header_fmt, payload, 16)
    offset, size, count = header[5], header[10], header[11]
    section_fmt = endian + ("IIQQQQIIQQ" if wide else "IIIIIIIIII")
    require(0 < count <= 4096 and size >= struct.calcsize(section_fmt)
            and offset >= 16 + struct.calcsize(header_fmt) and offset + size * count <= len(payload), "generated ELF section extent")
    ranges, outside = [], []
    for index in range(count):
        section = struct.unpack_from(section_fmt, payload, offset + index * size)
        require(section[1] == 8 or section[5] == 0 or section[4] + section[5] <= len(payload), "ELF section payload")
        if section[2] & 2 and section[5]:
            target = ranges if section[3] < 48 * 1024 else outside
            target.append([section[3], section[3] + section[5]])
    return dict(static_sram_end=max((end for _, end in ranges), default=0),
                sram_ranges=ranges, outside_sram_ranges=outside)


def compiled_inventory(build):
    """Read only generated kernels, never compiler/runtime binary internals."""
    build = Path(build).absolute()
    require(build == build.resolve(), "compiled directory symlink")
    start, total, entries = time.monotonic(), 0, 0
    inventory = {}
    def scan_error(error):
        raise error
    for directory, subdirs, files in os.walk(build, followlinks=False, onerror=scan_error):
        entries += len(subdirs) + len(files)
        require(entries <= 20000 and time.monotonic() - start <= 30, "compiled inventory scan bound")
        for name in subdirs:
            require(not (Path(directory) / name).is_symlink(), "compiled symlink directory")
        for name in sorted(files):
            require(time.monotonic() - start <= 30, "compiled inventory deadline")
            path = Path(directory) / name
            require(not path.is_symlink(), "compiled symlink file")
            if not name.endswith(".elf"):
                continue
            payload = bounded(path, 32 << 20)
            total += len(payload)
            require(total <= ELF_TOTAL_LIMIT, "compiled ELF total byte bound")
            relative = str(path.relative_to(build))
            inventory[relative] = dict(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), **_elf_storage(payload))
    require(inventory and len(inventory) <= 16384, "generated ELF inventory required")
    require(time.monotonic() - start <= 30, "compiled inventory deadline")
    return dict(files=inventory, artifact_bytes=total,
                max_static_sram_end=max(row["static_sram_end"] for row in inventory.values()),
                scope="worker-observed generated ELF hashes and allocated low-address sections before launch; stack/runtime high-water unmeasured")


def admit_compiled_inventory(inventory):
    files = inventory.get("files")
    require(isinstance(files, dict) and 0 < len(files) <= 16384, "compiled ELF inventory")
    require(all(isinstance(row, dict) and type(row.get("bytes")) is int and 0 < row["bytes"] <= 32 << 20
                and isinstance(row.get("sha256"), str) and re.fullmatch("[0-9a-f]{64}", row["sha256"])
                for row in files.values()), "compiled ELF entry bounds")
    total = sum(row["bytes"] for row in files.values())
    require(total <= ELF_TOTAL_LIMIT and inventory.get("artifact_bytes") == total, "compiled ELF total byte bound")
    ends = [row.get("static_sram_end") for row in files.values()]
    require(all(type(end) is int and end >= 0 for end in ends), "compiled SRAM endpoints")
    require(inventory.get("max_static_sram_end") == max(ends), "compiled SRAM maximum")
    require(max(ends) < 44 * 1024, "compiled static SRAM exceeds 44 KiB admission")


