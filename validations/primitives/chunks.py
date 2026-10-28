"""Write and verify complete captures one canonical rectangular chunk at a time.

The caller owns SDK execution, numerical comparisons and the outer archive
quota. This module binds storage to a frozen plan and runtime identity. It does
not retain payload arrays or replace an execution/API-sequence audit.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

import numpy as np

FORMAT = "csl-capture-chunks-v1"
METADATA_LIMIT = 4 << 20
MAX_FIELDS = 256
MAX_CHUNKS = 8192
MAX_CHUNK_BYTES = 16 << 20
MAX_TOTAL_BYTES = 1 << 40
IDENTITY_KEYS = {"run_id", "case_sha256", "artifact_sha256", "request_sha256", "worker_sha256"}
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}")
SHA = re.compile(r"[0-9a-f]{64}")
DTYPES = {"float32": np.dtype("<f4"), "uint32": np.dtype("<u4")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def integer(value, low, high, label):
    require(type(value) is int and low <= value <= high, label)


def make_plan(identity, grid, fields, *, chunk_bytes=4 << 20,
              total_bytes=128 << 20, max_chunks=MAX_CHUNKS):
    """Freeze ordered field rectangles and explicit payload/metadata bounds.

    uint16 SDK symbols use a uint32 host carrier and must be declared as such.
    Full PE coverage means coverage of each caller-supplied field rectangle;
    callers requiring all PEs must supply the full grid rectangle.
    """
    value = dict(format=FORMAT, identity=identity, grid=grid, fields=fields,
                 limits=dict(chunk_bytes=chunk_bytes, total_bytes=total_bytes, max_chunks=max_chunks))
    validate_plan(value)
    return json.loads(encoded(value))


def validate_plan(plan):
    require(isinstance(plan, dict) and set(plan) == {"format", "identity", "grid", "fields", "limits"}
            and plan["format"] == FORMAT, "plan schema")
    identity = plan["identity"]
    require(isinstance(identity, dict) and set(identity) == IDENTITY_KEYS, "identity inventory")
    require(isinstance(identity["run_id"], str), "run UUID")
    try:
        require(str(uuid.UUID(identity["run_id"])) == identity["run_id"], "canonical run UUID")
    except (AttributeError, ValueError) as error:
        raise ValueError("canonical run UUID") from error
    require(all(isinstance(identity[key], str) and SHA.fullmatch(identity[key])
                for key in IDENTITY_KEYS-{"run_id"}), "identity hashes")
    grid = plan["grid"]
    require(isinstance(grid, list) and len(grid) == 2, "grid schema")
    for size in grid:
        integer(size, 1, 65535, "grid dimension")
    limits = plan["limits"]
    require(isinstance(limits, dict) and set(limits) == {"chunk_bytes", "total_bytes", "max_chunks"}, "limit inventory")
    integer(limits["chunk_bytes"], 4, MAX_CHUNK_BYTES, "chunk byte bound")
    require(limits["chunk_bytes"] % 4 == 0, "word-aligned chunk bound")
    integer(limits["total_bytes"], 4, MAX_TOTAL_BYTES, "total byte bound")
    integer(limits["max_chunks"], 1, MAX_CHUNKS, "chunk count bound")
    fields = plan["fields"]
    require(isinstance(fields, list) and 1 <= len(fields) <= MAX_FIELDS, "field count")
    names, total, chunks = set(), 0, 0
    for field in fields:
        require(isinstance(field, dict) and set(field) == {"name", "dtype", "count", "rect"}, "field schema")
        name = field["name"]
        require(isinstance(name, str) and NAME.fullmatch(name) and name not in names, "unique safe field name")
        names.add(name)
        require(isinstance(field["dtype"], str) and field["dtype"] in DTYPES, "host carrier dtype")
        integer(field["count"], 1, 65535, "words per PE")
        rect = field["rect"]
        require(isinstance(rect, list) and len(rect) == 4 and all(type(n) is int for n in rect), "rectangle schema")
        x, y, width, height = rect
        require(x >= 0 and y >= 0 and width > 0 and height > 0
                and x+width <= grid[0] and y+height <= grid[1], "rectangle exceeds grid")
        cell_bytes = 4*field["count"]
        require(cell_bytes <= limits["chunk_bytes"], "one PE exceeds chunk bound")
        cells = limits["chunk_bytes"]//cell_bytes
        chunks += ((height+(cells//width)-1)//(cells//width) if cells >= width
                   else height*((width+cells-1)//cells))
        total += width*height*cell_bytes
        require(chunks <= limits["max_chunks"] and total <= limits["total_bytes"], "capture admission bound")
    require(len(encoded(plan)) <= METADATA_LIMIT, "plan metadata bound")
    require(_manifest_size(plan, total) <= METADATA_LIMIT, "completed manifest metadata bound")
    return dict(payload_bytes=total, chunks=chunks)


def iter_plan(plan):
    """Yield canonical row-major strips without an all-PE coverage bitmap."""
    validate_plan(plan)
    yield from _iter_plan(plan)


def _iter_plan(plan):
    """Internal descriptor iterator for an already structurally checked plan."""
    plan_sha256 = digest(encoded(plan))
    ordinal = 0
    for index, field in enumerate(plan["fields"]):
        x, y, width, height = field["rect"]
        cell_bytes = 4*field["count"]
        cells = plan["limits"]["chunk_bytes"]//cell_bytes
        rows = max(1, cells//width)
        offset = 0
        for row in range(0, height, rows):
            chunk_height = min(rows, height-row)
            for col in range(0, width, min(width, cells)):
                chunk_width = min(width-col, cells)
                size = chunk_width*chunk_height*cell_bytes
                yield dict(index=ordinal, field=field["name"], dtype=field["dtype"], count=field["count"],
                           rect=[x+col, y+row, chunk_width, chunk_height], offset=offset, bytes=size,
                           file=f"{index:03d}-{field['name']}.bin", plan_sha256=plan_sha256)
                offset += size
                ordinal += 1


def _manifest_size(plan, total):
    """Exact completed JSON length; hashes always contain 64 ASCII characters.

    Count individual encoded records without retaining the chunk inventory or
    touching payload files. This admission runs before creating a Writer root.
    """
    size = len(encoded(dict(format=FORMAT, plan=plan, files=[], chunks=[], payload_bytes=total)))
    files = {}
    for index, descriptor in enumerate(_iter_plan(plan)):
        # Empty-list brackets are already in the envelope; add record contents
        # and only the separating commas, excluding encoded()'s final newline.
        size += len(encoded(dict(descriptor, sha256="0"*64))) - 1 + int(index > 0)
        files[descriptor["file"]] = descriptor["offset"] + descriptor["bytes"]
    for index, (name, extent) in enumerate(files.items()):
        size += len(encoded(dict(file=name, bytes=extent, sha256="0"*64))) - 1 + int(index > 0)
    return size


def regular_root(path):
    path = Path(path).absolute()
    require(path == path.resolve(), "capture directory symlink")
    return path


class Writer:
    """Consume exactly one current descriptor and payload; never retain arrays."""
    def __init__(self, folder, plan):
        self._admission = validate_plan(plan)
        self._plan = json.loads(encoded(plan))
        self.root = regular_root(folder)
        self.root.mkdir(parents=True, exist_ok=False)
        self._iterator = iter(iter_plan(self._plan))
        self._current = next(self._iterator)
        self._records, self._files = [], []
        self.stream, self.file_hash = None, None
        self.file_name, self.file_size = None, 0
        self.closed, self.failed = False, False
        self.peak_chunk_bytes = 0

    @property
    def current(self):
        return (None if self._current is None else
                dict(self._current, rect=list(self._current["rect"])))

    def _close_field(self):
        if self.stream is not None:
            self.stream.close()
            self._files.append(dict(file=self.file_name, bytes=self.file_size, sha256=self.file_hash.hexdigest()))
            self.stream = None

    def write(self, descriptor, values):
        require(not self.closed and not self.failed and self._current is not None, "writer not accepting chunks")
        require(isinstance(descriptor, dict) and encoded(descriptor) == encoded(self._current), "chunk identity/order/coverage")
        _, _, width, height = descriptor["rect"]
        require(isinstance(values, np.ndarray) and values.dtype == DTYPES[descriptor["dtype"]]
                and values.shape == (width*height, descriptor["count"])
                and values.flags.c_contiguous, "chunk payload shape/dtype/contiguity")
        require(values.nbytes == descriptor["bytes"], "chunk payload extent")
        try:
            if descriptor["file"] != self.file_name:
                self._close_field()
                self.file_name, self.file_size = descriptor["file"], 0
                self.stream = (self.root/self.file_name).open("xb")
                self.file_hash = hashlib.sha256()
            require(self.file_size == descriptor["offset"], "append offset")
            payload = memoryview(values).cast("B")
            require(self.stream.write(payload) == len(payload), "short payload write")
            self.file_hash.update(payload)
            self.file_size += len(payload)
            self._records.append(dict(self._current, rect=list(self._current["rect"]), sha256=digest(payload)))
            self.peak_chunk_bytes = max(self.peak_chunk_bytes, len(payload))
            self._current = next(self._iterator, None)
        except BaseException:
            self.failed = True
            self.close()
            raise

    def finish(self):
        require(not self.closed and not self.failed and self._current is None, "incomplete capture")
        self._close_field()
        manifest = dict(format=FORMAT, plan=self._plan, files=self._files, chunks=self._records,
                        payload_bytes=self._admission["payload_bytes"])
        payload = encoded(manifest)
        require(len(payload) <= METADATA_LIMIT, "capture metadata bound")
        with (self.root/"capture.json").open("xb") as stream:
            require(stream.write(payload) == len(payload), "short manifest write")
        self.closed = True
        return dict(manifest_sha256=digest(payload), payload_bytes=self._admission["payload_bytes"],
                    chunks=len(self._records), peak_chunk_bytes=self.peak_chunk_bytes,
                    scope="complete stored payload; execution and numerical checks belong to caller")

    def close(self):
        self._close_field()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _manifest(folder, expected_plan, manifest_sha256):
    admission = validate_plan(expected_plan)
    require(isinstance(manifest_sha256, str) and SHA.fullmatch(manifest_sha256), "manifest digest schema")
    root = regular_root(folder)
    path = root/"capture.json"
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= METADATA_LIMIT, "complete bounded manifest")
    with path.open("rb") as stream:
        payload = stream.read(METADATA_LIMIT+1)
    require(len(payload) <= METADATA_LIMIT and digest(payload) == manifest_sha256, "manifest identity")
    manifest = json.loads(payload, object_pairs_hook=unique_object)
    require(isinstance(manifest, dict) and set(manifest) == {"format", "plan", "files", "chunks", "payload_bytes"}
            and manifest["format"] == FORMAT and encoded(manifest["plan"]) == encoded(expected_plan), "frozen capture plan")
    require(type(manifest["payload_bytes"]) is int and manifest["payload_bytes"] == admission["payload_bytes"], "total payload extent")
    records = manifest["chunks"]
    require(isinstance(records, list) and len(records) == admission["chunks"], "complete chunk count")
    files = {}
    for descriptor, observed in zip(iter_plan(expected_plan), records):
        require(isinstance(observed, dict) and set(observed) == set(descriptor) | {"sha256"}
                and encoded({key: observed[key] for key in descriptor}) == encoded(descriptor), "canonical chunk identity/order/coverage")
        require(isinstance(observed["sha256"], str) and SHA.fullmatch(observed["sha256"]), "chunk digest")
        files[descriptor["file"]] = descriptor["offset"]+descriptor["bytes"]
    observed_files = manifest["files"]
    require(isinstance(observed_files, list) and len(observed_files) == len(files), "complete field files")
    for observed, (name, size) in zip(observed_files, files.items()):
        require(isinstance(observed, dict) and set(observed) == {"file", "bytes", "sha256"}
                and observed["file"] == name and type(observed["bytes"]) is int and observed["bytes"] == size
                and isinstance(observed["sha256"], str) and SHA.fullmatch(observed["sha256"]), "field identity/extent")
        path = root/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == size, "field file extent")
    # Bound the directory scan even when an unexpected file is present.
    expected_names = set(files) | {"capture.json"}
    seen = set()
    with os.scandir(root) as entries:
        for entry in entries:
            require(len(seen) < len(expected_names) and entry.name in expected_names
                    and entry.name not in seen, "capture file inventory")
            seen.add(entry.name)
    require(seen == expected_names, "capture file inventory")
    return root, manifest


def iter_verified(folder, expected_plan, manifest_sha256):
    """Yield checked read-only arrays; full verification requires exhaustion.

    Only one chunk payload is kept internally. A caller retaining yielded arrays
    owns that extra memory; audit() releases each array before requesting another.
    """
    root, manifest = _manifest(folder, expected_plan, manifest_sha256)
    records = iter(manifest["chunks"])
    current = next(records, None)
    for field in manifest["files"]:
        field_hash = hashlib.sha256()
        with (root/field["file"]).open("rb") as stream:
            while current is not None and current["file"] == field["file"]:
                payload = stream.read(current["bytes"])
                require(len(payload) == current["bytes"] and digest(payload) == current["sha256"], "chunk payload integrity")
                field_hash.update(payload)
                array = np.frombuffer(payload, dtype=DTYPES[current["dtype"]]).reshape(-1, current["count"])
                yield dict(current, rect=list(current["rect"])), array
                del array, payload
                current = next(records, None)
            require(stream.read(1) == b"" and field_hash.hexdigest() == field["sha256"], "field payload integrity")
    require(current is None, "capture exhaustion")


def audit(folder, expected_plan, manifest_sha256, check=None):
    """Exhaust verified chunks; a numerical callback must raise on failure.

    Callback return values are ignored. Structural success alone is not a claim
    of numerical correctness or successful device execution.
    """
    iterator = iter_verified(folder, expected_plan, manifest_sha256)
    count, total, peak = 0, 0, 0
    while True:
        try:
            descriptor, array = next(iterator)
        except StopIteration:
            break
        if check is not None:
            check(descriptor, array)
        count += 1
        total += array.nbytes
        peak = max(peak, array.nbytes)
        del descriptor, array
    return dict(structure_status="PASS", chunks=count, payload_bytes=total, peak_chunk_bytes=peak,
                numerical_callback="completed for every chunk" if check is not None else "not supplied",
                identity=expected_plan["identity"])
