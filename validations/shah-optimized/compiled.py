"""Bounded identity reads of generated program files; no executable decoding."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time

MAX_BYTES = 256 << 20
MAX_FILES = 4096
MAX_SECONDS = 10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected(path):
    return path.suffix == ".elf" or path.name in ("out.json", "out_rpc.json")


def io_layout(metadata):
    result = {}
    for side, mux in (("west", "demux"), ("east", "mux")):
        data = metadata[side + "/out.json"]
        params = json.loads(data)["params"]
        count = int(params["num_" + mux])
        require(1 <= count <= 4, "unexpected compiled I/O partition count")
        result[side] = dict(source_sha256=hashlib.sha256(data).hexdigest(),
            height=int(params["height"]),
            partitions=[int(params[f"height_{mux}_{i}"]) for i in range(count)])
    return result


def read_identity(stream, size, keep, began):
    require(0 <= size <= MAX_BYTES and (not keep or size <= 2 << 20), "compiled member size bound")
    digest = hashlib.sha256()
    payload = bytearray()
    count = 0
    while True:
        require(time.monotonic() - began <= MAX_SECONDS, "compiled inventory time bound")
        block = stream.read(1 << 20)
        if not block:
            break
        count += len(block)
        require(count <= size, "compiled member grew")
        digest.update(block)
        if keep:
            payload.extend(block)
    require(count == size, "short compiled member")
    return digest.hexdigest(), bytes(payload)


def archive_inventory(path):
    """Find the unique generated tree by its west/east metadata paths.

    Returned E219 SDK 2.10.0 artifacts use cs_<hash>/out/{bin,west,east};
    the wrapper prefix is deliberately not assumed here.
    """
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_BYTES,
            "invalid compiled archive")
    began = time.monotonic()
    hashes, metadata, names = {}, {}, set()
    total = 0
    with tarfile.open(path, "r:*") as archive:
        for number, member in enumerate(archive):
            name = PurePosixPath(member.name)
            total += member.size
            require(number < MAX_FILES and total <= MAX_BYTES and
                    time.monotonic() - began <= MAX_SECONDS, "compiled archive scan bound")
            require(name.parts and not name.is_absolute() and ".." not in name.parts and
                    str(name) not in names and (member.isdir() or member.isfile()),
                    "compiled archive path/type/duplicate")
            names.add(str(name))
            if member.isdir() or not selected(name):
                continue
            keep = name.name == "out.json"
            with archive.extractfile(member) as stream:
                sha, data = read_identity(stream, member.size, keep, began)
            hashes[str(name)] = sha
            if keep:
                metadata[str(name)] = data
    roots = {str(PurePosixPath(name).parent.parent) for name in metadata
             if PurePosixPath(name).parts[-2:] == ("west", "out.json")}
    roots = {root for root in roots if root + "/east/out.json" in metadata and root + "/out.json" in metadata}
    require(len(roots) == 1, "ambiguous or missing generated program tree")
    root = next(iter(roots)) + "/"
    files = {name[len(root):]: sha for name, sha in hashes.items() if name.startswith(root)}
    require(any(name.endswith(".elf") for name in files), "compiled artifact has no generated ELF")
    require(all(name.startswith(root) for name in hashes), "multiple generated trees in archive")
    kept = {name[len(root):]: data for name, data in metadata.items() if name.startswith(root)}
    return dict(files=files, io_layout=io_layout(kept))


def tree_inventory(build):
    build = Path(build).resolve()
    require(build.is_dir(), "missing unpacked program")
    began, total, count = time.monotonic(), 0, 0
    files, metadata = {}, {}
    def failed(error):
        raise error
    for current, directories, names in os.walk(build, followlinks=False, onerror=failed):
        for name in directories + names:
            count += 1
            path = Path(current) / name
            require(count <= MAX_FILES and not path.is_symlink() and
                    time.monotonic() - began <= MAX_SECONDS, "unpacked program scan bound/symlink")
        for name in names:
            path = Path(current) / name
            if not selected(path):
                continue
            require(path.is_file(), "generated program special file")
            size = path.stat().st_size
            total += size
            require(total <= MAX_BYTES, "unpacked program byte bound")
            relative = path.relative_to(build).as_posix()
            with path.open("rb") as stream:
                sha, data = read_identity(stream, size, path.name == "out.json", began)
            files[relative] = sha
            if path.name == "out.json":
                metadata[relative] = data
    require(any(name.endswith(".elf") for name in files), "unpacked program has no generated ELF")
    return dict(files=files, io_layout=io_layout(metadata))
