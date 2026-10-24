"""Derive a fresh native-full case by lossless NPZ encoding only.

Parent compile/run artifacts are neither copied nor promoted. Run this host
operation through the budget runner; export remains the capture-admission gate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from models.meshgraphnet.native_appliance import saved_audit as common
from models.meshgraphnet.native_full import driver

RUN_ROOT = ROOT / "research/runs/E200"
NPZ_LIMIT = 64 << 20
FILE_COUNT = 120
RESERVED = {"case.json", "parent-case.json", "compressor-source.py"}


def require(value, message):
    if not value:
        raise ValueError(message)


def strict_json(payload):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(value):
        raise ValueError("nonfinite JSON value: " + value)

    return json.loads(payload, object_pairs_hook=unique, parse_constant=reject)


def directory(path):
    path = Path(path).absolute()
    require(path == path.resolve() and path.is_dir() and not path.is_symlink(),
            "regular canonical source directory")
    return path


def file_inventory(folder, manifest, *, derived=False):
    files = manifest.get("files")
    require(isinstance(files, dict) and 1 <= len(files) <= FILE_COUNT - (0 if derived else 2),
            "bounded frozen file inventory")
    total = common.local_file(folder, "case.json").stat().st_size
    result = {}
    for name, digest in sorted(files.items()):
        path = common.local_file(folder, name)
        require(name != "case.json" and (derived or name not in RESERVED), "reserved frozen filename")
        require(Path(name).parts[0] not in {"build", "runtime"}
                and name not in {"build.json", "raw.npz", "execution.json", "result.json"},
                "execution artifacts cannot be frozen inputs")
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "SHA256 encoding")
        require(common.sha(path, common.CASE_LIMIT) == digest, "parent/frozen file changed: " + name)
        size = path.stat().st_size
        total += size
        require(total <= common.CASE_LIMIT, "128 MiB frozen case bound")
        result[name] = dict(bytes=size, sha256=digest)
    return result, total


def array_inventory(arrays):
    return {name: dict(shape=list(value.shape), dtype=value.dtype.str, bytes=value.nbytes,
                       sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())
            for name, value in sorted(arrays.items())}


def same_arrays(before, after):
    require(set(before) == set(after), "NPZ array key inventory changed")
    for name, expected in before.items():
        actual = after[name]
        require(actual.shape == expected.shape and actual.dtype == expected.dtype,
                "NPZ array shape/dtype changed: " + name)
        require(actual.tobytes(order="C") == expected.tobytes(order="C"),
                "NPZ array bits changed: " + name)


class BoundedWriter:
    """Support ZIP header seeks while bounding the written file extent."""
    def __init__(self, stream, limit):
        self.stream, self.limit = stream, limit

    def write(self, value):
        require(self.stream.tell() + len(value) <= self.limit, "compressed NPZ byte bound")
        return self.stream.write(value)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def derive(case, output):
    case = directory(case)
    output = Path(output).absolute()
    require(output == output.resolve() and RUN_ROOT in output.parents
            and not output.exists() and not output.is_symlink(), "fresh canonical output below research/runs/E200")
    require(case not in output.parents, "derived case must not be inside its parent case")
    source = Path(__file__).absolute()
    require(source == source.resolve(), "regular compressor source path")
    source_bytes = common.bounded(source, common.JSON_LIMIT)
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    dependencies = {str(Path(module.__file__).relative_to(ROOT)): common.sha(Path(module.__file__))
                    for module in (common, driver)}
    parent_bytes = common.bounded(common.local_file(case, "case.json"), common.JSON_LIMIT)
    parent_hash = hashlib.sha256(parent_bytes).hexdigest()
    parent = strict_json(parent_bytes)
    require(parent.get("experiment") == "native-full-physical-v1" and "encoding_provenance" not in parent,
            "original native-full case required")
    original_files, original_bytes = file_inventory(case, parent)
    # Admit NPY extents before the original driver's uncompressed array reads.
    for name in original_files:
        if name.endswith(".npz"):
            arrays = common.load_npz(case / name, limit=NPZ_LIMIT)
            del arrays
    require(driver.load(case) == parent, "parent native contract changed during preflight")
    require(common.bounded(case / "case.json", common.JSON_LIMIT) == parent_bytes, "parent manifest changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "parent-case.json").write_bytes(parent_bytes)
    (output / "compressor-source.py").write_bytes(source_bytes)
    encoded_files, changes = {}, {}
    extent = len(parent_bytes) + len(source_bytes)
    for name, original in original_files.items():
        before = common.local_file(case, name)
        after = common.local_file(output, name)
        after.parent.mkdir(parents=True, exist_ok=True)
        require(common.sha(before, common.CASE_LIMIT) == original["sha256"], "parent changed before copy")
        row = dict(parent=original)
        if name.endswith(".npz"):
            arrays = common.load_npz(before, limit=NPZ_LIMIT)
            with after.open("xb") as stream:
                np.savez_compressed(BoundedWriter(stream, NPZ_LIMIT), **arrays)
            restored = common.load_npz(after, limit=NPZ_LIMIT)
            same_arrays(arrays, restored)
            row.update(encoding="NPZ ZIP_DEFLATED", arrays=array_inventory(arrays))
            del arrays, restored
        else:
            payload = common.bounded(before, common.CASE_LIMIT)
            with after.open("xb") as stream:
                stream.write(payload)
            require(common.sha(after, common.CASE_LIMIT) == original["sha256"], "byte copy changed")
            row["encoding"] = "byte-identical copy"
        require(common.sha(before, common.CASE_LIMIT) == original["sha256"], "parent changed during copy")
        digest, size = common.sha(after, common.CASE_LIMIT), after.stat().st_size
        encoded_files[name] = digest
        row["derived"] = dict(bytes=size, sha256=digest)
        changes[name] = row
        extent += size
        require(extent <= common.CASE_LIMIT, "derived frozen file byte bound")
    encoded_files.update({"parent-case.json": parent_hash, "compressor-source.py": source_hash})
    result = copy.deepcopy(parent)
    result["files"] = encoded_files
    result["encoding_provenance"] = dict(
        schema="e200-lossless-npz-case-v1", parent_case_sha256=parent_hash,
        parent_manifest_file="parent-case.json", compressor_source_sha256=source_hash,
        compressor_source_file="compressor-source.py", dependency_sha256=dependencies,
        parent_frozen_bytes=original_bytes, derived_files_bytes_excluding_manifest=extent, files=changes,
        equivalence="identical array keys, shapes, dtypes and C-order bits, including signed zero",
        scope="NPZ encoding only; no parent compile/run artifacts copied; no new compile or execution claim")
    common.write_json(output / "case.json", result)
    _, derived_bytes = file_inventory(output, result, derived=True)
    # Check the saved provenance and every emitted NPZ again after all writes.
    require(common.bounded(output / "parent-case.json", common.JSON_LIMIT) == parent_bytes
            and common.bounded(output / "compressor-source.py", common.JSON_LIMIT) == source_bytes,
            "retained parent/compressor bytes")
    for name, row in changes.items():
        if name.endswith(".npz"):
            restored = common.load_npz(output / name, limit=NPZ_LIMIT)
            require(array_inventory(restored) == row["arrays"], "derived array corruption: " + name)
            del restored
    require(driver.load(output) == result, "derived native numerical contract")
    require(file_inventory(case, parent) == (original_files, original_bytes)
            and common.bounded(case / "case.json", common.JSON_LIMIT) == parent_bytes,
            "parent frozen case changed")
    require(common.sha(source) == source_hash and all(common.sha(ROOT / name) == digest
                for name, digest in dependencies.items()), "compressor dependencies changed")
    require(not (output / "build").exists() and not (output / "build.json").exists(), "no derived compilation")
    return dict(schema="e200-lossless-npz-encoding-result-v1", status="PASS_ENCODING_ONLY",
                output=str(output), parent_case_sha256=parent_hash,
                derived_case_sha256=common.sha(output / "case.json"),
                parent_frozen_bytes=original_bytes, derived_frozen_bytes=derived_bytes,
                bytes_saved=original_bytes - derived_bytes, numerical_contract="native_full.driver.load PASS",
                capture_admission="NOT_RUN; requires portable exporter with unchanged 128/256 MiB bounds",
                compile_status="NOT_RUN", execution_status="NOT_RUN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.case, args.output), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
