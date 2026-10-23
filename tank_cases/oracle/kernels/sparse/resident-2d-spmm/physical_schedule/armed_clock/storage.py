"""Bounded immutable NPZ admission and atomic incremental JSON receipts."""
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import zipfile
import numpy as np

LIMIT = 1024**2
JSON_LIMIT = 4*LIMIT
DTYPES = ("<f4", "<u4", "<u2")


def need(ok, message):
    if not ok: raise ValueError(message)


def descriptor(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def schema(arrays): return {name: (tuple(value.shape), value.dtype.str) for name, value in arrays.items()}

def read_json(path):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= JSON_LIMIT, "bounded regular JSON")
    with path.open("rb") as stream: data = stream.read(JSON_LIMIT+1)
    need(len(data) <= JSON_LIMIT, "JSON stored cap")
    return json.loads(data)


def write_json_new(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    need(len(data) <= JSON_LIMIT, "fresh JSON cap")
    with path.open("xb") as stream: stream.write(data)


def write_npz(path, arrays):
    need(not path.exists(), "fresh NPZ snapshot")
    need(sum(a.nbytes for a in arrays.values()) <= LIMIT//2, "snapshot allocation cap")
    with path.open("xb") as stream: np.savez(stream, **arrays)
    need(path.stat().st_size <= LIMIT, "stored snapshot cap")


def load_npz(path, expected):
    need(type(expected) is dict and len(expected) <= 32, "expected inventory bound")
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= LIMIT, "bounded regular NPZ")
    with path.open("rb") as stream: data = stream.read(LIMIT+1)
    need(len(data) <= LIMIT, "NPZ stored cap")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist(); names = [e.filename for e in entries]
        need(len(entries) <= 32 and len(set(names)) == len(names) and set(names) == {n+".npy" for n in expected}, "exact flat NPZ inventory")
        need(all(n.isascii() and n.isidentifier() for n in expected), "flat NPZ names")
        need(sum(e.file_size for e in entries) <= LIMIT, "expanded NPZ cap")
        for entry in entries:
            need(not entry.is_dir() and not entry.flag_bits & 1 and entry.compress_type in (0, 8), "regular NPZ member")
            with archive.open(entry) as stream:
                magic = stream.read(8)
                need(magic in (b"\x93NUMPY\x01\x00", b"\x93NUMPY\x02\x00"), "NPY version")
                size = 2 if magic[-2:] == b"\x01\x00" else 4
                word = stream.read(size); need(len(word) == size, "NPY length word")
                length = int.from_bytes(word, "little"); need(0 < length <= 4096, "NPY header cap")
                raw = stream.read(length); need(len(raw) == length, "NPY complete header")
                try: header = ast.literal_eval(raw.decode("latin1").strip())
                except (SyntaxError, ValueError, RecursionError) as error: raise ValueError("invalid NPY header") from error
                need(type(header) is dict and set(header) == {"descr", "fortran_order", "shape"}, "NPY fields")
                dtype, shape = header["descr"], header["shape"]
                need(type(dtype) is str and dtype in DTYPES and header["fortran_order"] is False, "NPY scalar dtype/order")
                need(type(shape) is tuple and len(shape) <= 2 and all(type(v) is int and 0 <= v <= LIMIT for v in shape), "NPY dimensions")
                payload = math.prod(shape)*np.dtype(dtype).itemsize
                need(payload <= LIMIT and 8+size+length+payload == entry.file_size, "NPY exact payload")
                need((shape, dtype) == expected[entry.filename[:-4]], "NPY exact expected schema")
                need(len(stream.read(payload+1)) == payload, "NPY payload and CRC")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive: result = {n: archive[n].copy() for n in expected}
    need(schema(result) == expected, "loaded schema")
    return result


def journal(path, value):
    need(path.name == "runtime.json" and not path.is_symlink(), "owned incremental journal")
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    need(len(data) <= JSON_LIMIT, "journal cap")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix="runtime-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name); stream.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)
