"""Bounded archive, ABI and generated-ELF helpers derived from native_appliance/saved_audit.py.
The family-specific contract is defined only in saved_audit.py."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile
import tempfile
import time
import zipfile
import numpy as np

JSON_LIMIT = 4 << 20
INVENTORY_JSON_LIMIT = 16 << 20
ELF_TOTAL_LIMIT = 1 << 30
CASE_LIMIT = 128 << 20
CAPTURE_LIMIT = 256 << 20
CASE_REQUIRED = {"provenance.json", "inputs.npz", "expected.npz", "node_ids.npz", "source/layout.csl", "source/pe.csl"}
HOST_FILES = ("appliance.py", "appliance_worker.py", "saved_audit.py", "sdk_runner.py", "portable.py")

def require(ok, message):
    if not ok:
        raise ValueError(message)


def bounded(path, limit):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular file required: " + str(path))
    with path.open("rb") as stream:
        payload = stream.read(limit+1)
    require(len(payload) <= limit, "file exceeds bound: " + str(path))
    return payload


def sha(path, limit=CASE_LIMIT):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular file required: " + str(path))
    result, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            size += len(chunk)
            require(size <= limit, "file exceeds hash bound")
            result.update(chunk)
    return result.hexdigest()


def read_json(path, *, limit=JSON_LIMIT):
    require(type(limit) is int and 0 < limit <= INVENTORY_JSON_LIMIT, "JSON read limit")
    return json.loads(bounded(path, limit))


def write_json(path, value, *, limit=JSON_LIMIT):
    require(type(limit) is int and 0 < limit <= INVENTORY_JSON_LIMIT, "JSON write limit")
    path = Path(path)
    require(not path.is_symlink() and (not path.exists() or path.is_file()), "regular JSON destination")
    content = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    require(len(content) <= limit, "JSON output bound")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="."+path.name, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def local_file(folder, name):
    require(isinstance(name, str) and name, "nonempty relative path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == name,
            "canonical relative path")
    folder = Path(folder).absolute()
    path = folder/name
    require(path.resolve() == path and path.is_relative_to(folder), "symlink or path escape")
    return path


def load_npz(path, limit=CASE_LIMIT):
    """Bound ZIP members and validate NPY extents before NumPy allocation."""
    bounded(path, limit)
    result, total = {}, 0
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) <= 256, "NPZ entry bound")
        for item in entries:
            name = item.filename
            require(name.endswith(".npy") and "/" not in name and "\\" not in name
                    and name[:-4] not in result and not item.is_dir(), "NPZ unique flat members")
            total += item.file_size
            require(0 <= item.file_size <= limit and total <= limit, "NPZ expanded byte bound")
            payload = archive.read(item)
            stream = io.BytesIO(payload)
            version = np.lib.format.read_magic(stream)
            require(version in ((1, 0), (2, 0)), "NPY version")
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(stream)
            require(not fortran and not dtype.hasobject and dtype.kind in "fiu"
                    and dtype.itemsize in (2, 4, 8), "NPY type/order")
            require(len(payload)-stream.tell() == math.prod(shape)*dtype.itemsize, "exact NPY extent")
            result[name[:-4]] = np.load(io.BytesIO(payload), allow_pickle=False)
    return result


def compile_workers(value):
    require(type(value) is int and value in (1, 4, 8), "compile workers must be 1, 4 or 8")
    return value


def compiler_arguments(params, backend, workers):
    workers = compile_workers(workers)
    require(backend in ("hardware", "simulator"), "compiler backend")
    dimensions = "762,1172" if backend == "hardware" else f"{params['C']+7},{params['R']+2}"
    return (f"--arch=wse3 --fabric-dims={dimensions} --fabric-offsets=4,1 --params="
            + ",".join(f"{k}:{v}" for k, v in sorted(params.items()))
            + f" --max-parallelism={workers} --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 -o out")


def compile_binding(record):
    workers = compile_workers(record.get("compile_workers"))
    require(type(record.get("compile_resource_cpu")) is int
            and record["compile_resource_cpu"] == workers * 1000, "compiler CPU resource binding")
    require(record.get("compiler_args") == compiler_arguments(record["params"], record["backend"], workers),
            "compiler worker/argument binding")
    return workers


def unpack_archive(path, destination, prefix, limit, required=()):
    bounded(path, limit)
    destination = Path(destination).absolute()
    require(destination == destination.resolve(), "archive destination symlink")
    destination.mkdir(parents=True, exist_ok=False)
    names, total, count = set(), 0, 0
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            count += 1
            total += member.size
            name = PurePosixPath(member.name)
            require(count <= 256 and 0 <= member.size <= limit and total <= limit, "archive byte/entry bound")
            require(not name.is_absolute() and ".." not in name.parts and name.parts
                    and name.parts[0] == prefix and str(name) not in names, "archive path or duplicate")
            require(member.isdir() or member.isfile(), "archive link or special file")
            names.add(str(name))
            relative = name.relative_to(prefix)
            if not relative.parts:
                require(member.isdir(), "archive root directory")
                continue
            target = local_file(destination, str(relative))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream:
                    payload = stream.read(limit+1)
                require(len(payload) == member.size, "archive member extent")
                with target.open("xb") as stream:
                    stream.write(payload)
    for name in required:
        require(local_file(destination, name).is_file(), "missing archive member: " + name)
    return destination


def _schema(value, label):
    require(isinstance(value, dict) and 1 <= len(value) <= 64, label + " schema size")
    for name, field in value.items():
        require(isinstance(name, str) and re.fullmatch("[a-z][a-z0-9_]{0,63}", name), "symbol name")
        require(isinstance(field, list) and len(field) == 2 and type(field[0]) is int
                and 1 <= field[0] <= 32767 and field[1] in ("float32", "uint32", "uint16"), label + " field schema")


def tolerance(value):
    require(isinstance(value, dict) and set(value) == {"rtol", "atol"}
            and all(type(value[n]) in (int, float) and math.isfinite(value[n])
                    and 0 <= value[n] <= 1e-3 for n in value), "bounded numerical tolerance")
    return {name: float(value[name]) for name in ("rtol", "atol")}


def mask_arrays(masks, expected, label, *, float_outputs=False):
    require(isinstance(masks, dict) and set(masks) <= set(expected), label + " field inventory")
    for name, value in masks.items():
        require(isinstance(value, np.ndarray) and value.dtype == np.dtype("uint32")
                and value.shape == expected[name].shape and np.all((value == 0) | (value == 1)), label + " ABI")
        if float_outputs:
            require(expected[name].dtype == np.dtype("float32"), label + " float output")
    return masks


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


