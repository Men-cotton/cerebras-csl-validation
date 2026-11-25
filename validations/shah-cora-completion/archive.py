"""Bounded case/capture archives and identity of generated program artifacts."""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tarfile
import time
import sys
import zipfile
try:
    import proof
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import proof

MAX_BYTES = 256 << 20
MAX_MEMBERS = 20000
MAX_SECONDS = 45


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compare_audits(fresh, remote):
    """Require identical PASS reports except FP64 error-diagnostic roundoff.

    Numeric max_abs_error values and direct entries of max_abs_errors allow
    abs(a-b) <= 1e-12 + 1e-8 * max(abs(a), abs(b)). All schema, identity,
    hashes, counters, clocks and other values must match exactly.
    """
    require(type(fresh) is dict and type(remote) is dict
            and fresh.get('status') == remote.get('status') == 'PASS', 'both audits must PASS')

    def compare(a, b, path=()):
        location = '/'.join(map(str, path)) or '<root>'
        require(type(a) is type(b), 'audit value type differs: ' + location)
        if path and path[-1] == 'max_abs_errors':
            require(type(a) is dict, 'error diagnostic mapping required: ' + location)
        error_value = path and (path[-1] == 'max_abs_error'
                                or len(path) >= 2 and path[-2] == 'max_abs_errors')
        if error_value:
            require(type(a) in (int, float) and math.isfinite(a) and math.isfinite(b)
                    and a >= 0 and b >= 0, 'finite nonnegative error diagnostic: ' + location)
            require(abs(a-b) <= 1e-12 + 1e-8 * max(abs(a), abs(b)),
                    'audit error diagnostic differs: ' + location)
        elif type(a) is dict:
            require(a.keys() == b.keys(), 'audit keys differ: ' + location)
            for key in a:
                compare(a[key], b[key], (*path, key))
        elif type(a) is list:
            require(len(a) == len(b), 'audit list length differs: ' + location)
            for index, (left, right) in enumerate(zip(a, b)):
                compare(left, right, (*path, index))
        else:
            if type(a) is float:
                require(math.isfinite(a) and math.isfinite(b), 'finite audit value: ' + location)
            require(a == b, 'audit value differs: ' + location)

    compare(fresh, remote)


def digest(path, limit=MAX_BYTES):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
            "bounded regular file: " + str(path))
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    digest(path, 16 << 20)
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def safe_name(value):
    name = PurePosixPath(value)
    require(name.parts and not name.is_absolute() and '..' not in name.parts
            and str(name) == value.rstrip('/'), 'unsafe archive member')
    return name


def unpack(path, output, max_bytes=MAX_BYTES):
    digest(path, max_bytes)
    output = Path(output)
    require(not output.exists() and output.absolute() == output.resolve(), 'fresh nonsymlink extraction root')
    output.mkdir(parents=True)
    began, total, seen = time.monotonic(), 0, set()
    with tarfile.open(path, 'r:*') as archive:
        for count, member in enumerate(archive):
            name = safe_name(member.name)
            require(str(name) not in seen and (member.isdir() or member.isfile()), 'duplicate/special archive member')
            seen.add(str(name)); total += member.size
            require(count < MAX_MEMBERS and total <= max_bytes and time.monotonic() - began <= MAX_SECONDS,
                    'archive extraction budget')
            target = output / name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream, target.open('xb') as destination:
                    copied = 0
                    while block := stream.read(1 << 20):
                        copied += len(block)
                        require(copied <= member.size and time.monotonic() - began <= MAX_SECONDS, 'archive member extent/time')
                        destination.write(block)
                    require(copied == member.size, 'short archive member')
    return output


def program_tree(build):
    build = Path(build).absolute()
    inventory = proof.compiled_inventory(build)
    proof.admit_compiled_inventory(inventory)
    began, count, total, metadata = time.monotonic(), 0, 0, {}
    def failed(error):
        raise error
    for folder, dirs, files in os.walk(build, followlinks=False, onerror=failed):
        for name in dirs + files:
            count += 1
            require(count <= MAX_MEMBERS and not (Path(folder) / name).is_symlink()
                    and time.monotonic() - began <= MAX_SECONDS, 'program metadata traversal')
        for name in files:
            if name not in ('out.json', 'out_rpc.json'):
                continue
            path = Path(folder) / name
            total += path.stat().st_size
            require(total <= 16 << 20, 'generated JSON total bound')
            metadata[path.relative_to(build).as_posix()] = digest(path, 2 << 20)
    require({'out.json', 'west/out.json', 'east/out.json'} <= set(metadata), 'generated I/O metadata missing')
    return dict(elf=inventory, metadata_sha256=metadata)


def program_archive(path, sources):
    """Compute the same ELF/SRAM inventory as proof.compiled_inventory(tree)."""
    digest(path)
    began, total, seen, elfs, metadata, csl = time.monotonic(), 0, set(), {}, {}, {}
    with tarfile.open(path, 'r:*') as archive:
        for count, member in enumerate(archive):
            name = safe_name(member.name)
            require(str(name) not in seen and (member.isdir() or member.isfile()), 'duplicate/special compiler member')
            seen.add(str(name)); total += member.size
            require(count < MAX_MEMBERS and total <= MAX_BYTES and time.monotonic() - began <= MAX_SECONDS,
                    'compiler archive budget')
            if not member.isfile():
                continue
            selected = name.suffix == '.elf' or name.name in ('out.json', 'out_rpc.json')
            source = len(name.parts) >= 2 and name.parts[-2] == 'csl' and name.name in sources
            if not (selected or source):
                continue
            require(member.size <= (32 << 20 if name.suffix == '.elf' else 2 << 20), 'compiler member byte bound')
            payload = archive.extractfile(member).read()
            require(len(payload) == member.size, 'short compiler member')
            sha = hashlib.sha256(payload).hexdigest()
            if name.suffix == '.elf':
                elfs[str(name)] = dict(sha256=sha, bytes=len(payload), **proof._elf_storage(payload))
            elif selected:
                metadata[str(name)] = sha
            if source:
                require(name.name not in csl and sha == sources[name.name], 'compiled CSL differs from pinned source')
                csl[name.name] = sha
    roots = {str(PurePosixPath(name).parent.parent) for name in metadata
             if PurePosixPath(name).parts[-2:] == ('west', 'out.json')}
    roots = {r for r in roots if r + '/east/out.json' in metadata and r + '/out.json' in metadata}
    require(len(roots) == 1 and set(csl) == set(sources) and elfs, 'unique generated tree and CSL required')
    prefix = next(iter(roots)) + '/'
    require(all(name.startswith(prefix) for name in (*elfs, *metadata)), 'multiple generated program trees')
    files = {name[len(prefix):]: entry for name, entry in elfs.items()}
    inventory = dict(files=files, artifact_bytes=sum(v['bytes'] for v in files.values()),
                     max_static_sram_end=max(v['static_sram_end'] for v in files.values()),
                     scope='worker-observed generated ELF hashes and allocated low-address sections before launch; stack/runtime high-water unmeasured')
    proof.admit_compiled_inventory(inventory)
    return dict(elf=inventory, metadata_sha256={name[len(prefix):]: sha for name, sha in metadata.items()})


def check_npz(path, max_bytes=MAX_BYTES):
    """Bound decompressed numeric containers before the shared driver loads them."""
    import numpy as np
    digest(path, max_bytes)
    began, total, seen = time.monotonic(), 0, set()
    with zipfile.ZipFile(path) as data:
        members = data.infolist()
        require(0 < len(members) <= 1024, 'NPZ member count')
        for member in members:
            name = safe_name(member.filename)
            require(len(name.parts) == 1 and name.suffix == '.npy' and str(name) not in seen,
                    'NPZ member identity')
            seen.add(str(name)); total += member.file_size
            require(total <= max_bytes and time.monotonic() - began <= MAX_SECONDS, 'NPZ expanded byte/time bound')
            with data.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), 'NPY header version')
                header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = header(stream)
                require(not dtype.hasobject and len(shape) <= 4 and all(type(n) is int and n >= 0 for n in shape), 'numeric NPY shape')
                count = 1
                for n in shape:
                    count *= n
                    require(count * dtype.itemsize <= max_bytes, 'NPY allocation bound')
                require(stream.tell() + count * dtype.itemsize == member.file_size, 'NPY shape/file extent')
