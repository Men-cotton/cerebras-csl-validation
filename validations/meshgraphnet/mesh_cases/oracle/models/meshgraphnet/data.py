"""Bounded decoding of one original CylinderFlow TFRecord trajectory.

The field contract follows google-deepmind/deepmind-research/meshgraphnets/
dataset.py and the public cylinder_flow/meta.json. No TensorFlow dependency is
needed: the accepted protobuf subset is Example -> Features -> bytes_list.
The input is one complete framed record, including both masked CRC32C values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


MAX_RECORD_BYTES = 16 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024
MAX_NODES = 16_384
MAX_CELLS = 32_768
STEPS = 600
FIELDS = {
    "cells": {"type": "static", "shape": [1, -1, 3], "dtype": "int32"},
    "mesh_pos": {"type": "static", "shape": [1, -1, 2], "dtype": "float32"},
    "node_type": {"type": "static", "shape": [1, -1, 1], "dtype": "int32"},
    "velocity": {"type": "dynamic", "shape": [600, -1, 2], "dtype": "float32"},
    "pressure": {"type": "dynamic", "shape": [600, -1, 1], "dtype": "float32"},
}
SOURCE_URL = "https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/"


def _crc_table():
    result = []
    for number in range(256):
        for _ in range(8):
            number = (number >> 1) ^ (0x82F63B78 if number & 1 else 0)
        result.append(number)
    return tuple(result)


CRC_TABLE = _crc_table()


def crc32c(data):
    """Castagnoli CRC, with the standard initial and final complements."""
    crc = 0xFFFFFFFF
    for byte in data:
        crc = CRC_TABLE[(crc ^ byte) & 255] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc32c(data):
    crc = crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _read_bounded(path, maximum):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError(f"not a regular bounded input: {path}")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("input grew beyond its byte bound")
    return data


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _metadata(blob):
    try:
        meta = json.loads(blob, object_pairs_hook=_unique_json)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid metadata JSON") from error
    if not isinstance(meta, dict):
        raise ValueError("metadata must be an object")
    if meta.get("features") != FIELDS or meta.get("field_names") != list(FIELDS):
        raise ValueError("metadata field names, shapes or dtypes differ from CylinderFlow")
    if (type(meta.get("trajectory_length")) is not int
            or meta["trajectory_length"] != STEPS
            or meta.get("simulator") != "comsol"
            or type(meta.get("dt")) not in (float, int) or meta["dt"] != 0.01
            or meta.get("collision_radius", "missing") is not None):
        raise ValueError("metadata trajectory/simulator contract differs from CylinderFlow")
    # JSON equality alone equates booleans and integers inside a shape.
    for field in meta["features"].values():
        if any(type(dimension) is not int for dimension in field["shape"]):
            raise ValueError("metadata dimensions must be integers")
    return meta


def _varint(data, offset):
    value = 0
    start = offset
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        if shift == 63 and byte > 1:
            raise ValueError("protobuf varint exceeds uint64")
        value |= (byte & 127) << shift
        if not byte & 128:
            if offset - start > 1 and byte == 0:
                raise ValueError("noncanonical protobuf varint")
            return value, offset
    raise ValueError("unterminated protobuf varint")


def _message(data, *, limit):
    """Decode only bounded length-delimited fields; reject unknown wire types."""
    fields = []
    offset = 0
    while offset < len(data):
        if len(fields) == limit:
            raise ValueError("protobuf field count exceeds bound")
        tag, offset = _varint(data, offset)
        if tag & 7 != 2 or not 0 < tag >> 3 < 2**29:
            raise ValueError("unexpected protobuf field or wire type")
        length, offset = _varint(data, offset)
        end = offset + length
        if end > len(data):
            raise ValueError("truncated protobuf field")
        fields.append((tag >> 3, data[offset:end]))
        offset = end
    return fields


def _one(data, number):
    fields = _message(data, limit=1)
    if len(fields) != 1 or fields[0][0] != number:
        raise ValueError("unexpected singleton protobuf field")
    return fields[0][1]


def _example(payload):
    result = {}
    features = _one(memoryview(payload), 1)
    for number, entry in _message(features, limit=len(FIELDS)):
        if number != 1:
            raise ValueError("unexpected Features field")
        members = _message(entry, limit=2)
        if len(members) != 2 or {field for field, _ in members} != {1, 2}:
            raise ValueError("feature map entry requires one key and one value")
        members = dict(members)
        if len(members[1]) > 32:
            raise ValueError("feature key exceeds bound")
        try:
            key = bytes(members[1]).decode("ascii")
        except UnicodeError as error:
            raise ValueError("non-ASCII feature key") from error
        if key not in FIELDS or key in result:
            raise ValueError("unknown or duplicate feature key")
        # Feature.bytes_list is field 1; BytesList.value is repeated field 1.
        result[key] = _one(_one(members[2], 1), 1)
    if set(result) != set(FIELDS):
        raise ValueError("Example is missing required CylinderFlow fields")
    return result


def _decode(record, metadata):
    _metadata(metadata)
    if len(record) < 16:
        raise ValueError("truncated TFRecord framing")
    length, length_crc = struct.unpack_from("<QI", record)
    if length > MAX_RECORD_BYTES - 16 or len(record) != length + 16:
        raise ValueError("expected exactly one bounded complete TFRecord")
    if masked_crc32c(memoryview(record)[:8]) != length_crc:
        raise ValueError("TFRecord length CRC32C mismatch")
    payload = memoryview(record)[12:12 + length]
    data_crc, = struct.unpack_from("<I", record, 12 + length)
    if masked_crc32c(payload) != data_crc:
        raise ValueError("TFRecord payload CRC32C mismatch")
    raw = _example(payload)
    if len(raw["mesh_pos"]) % 8 or len(raw["cells"]) % 12:
        raise ValueError("invalid static array byte lengths")
    nodes = len(raw["mesh_pos"]) // 8
    cells = len(raw["cells"]) // 12
    if not 1 <= nodes <= MAX_NODES or not 1 <= cells <= MAX_CELLS:
        raise ValueError("mesh shape exceeds explicit bounds")
    shapes = {
        "cells": (cells, 3), "mesh_pos": (nodes, 2), "node_type": (nodes, 1),
        "velocity": (STEPS, nodes, 2), "pressure": (STEPS, nodes, 1),
    }
    arrays = {}
    for name, shape in shapes.items():
        size = int(np.prod(shape)) * 4
        if len(raw[name]) != size:
            raise ValueError(f"inconsistent {name} byte length/shape")
        dtype = "<i4" if name in ("cells", "node_type") else "<f4"
        arrays[name] = np.frombuffer(raw[name], dtype=dtype).reshape(shape).copy()
    if np.any(arrays["cells"] < 0) or np.any(arrays["cells"] >= nodes):
        raise ValueError("triangle index outside mesh")
    face = arrays["cells"]
    if np.any((face[:, 0] == face[:, 1]) | (face[:, 1] == face[:, 2])
              | (face[:, 2] == face[:, 0])):
        raise ValueError("triangle repeats a vertex")
    if np.any(arrays["node_type"] < 0) or np.any(arrays["node_type"] >= 9):
        raise ValueError("node type outside official enum range")
    for name in ("mesh_pos", "velocity", "pressure"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"nonfinite {name}")
    return arrays


def load_first_record(record_path, metadata_path):
    """Read exactly one bounded framed trajectory; return independent arrays.

    Static arrays have no leading singleton time axis. Dynamic arrays retain
    all 600 frames. This function does not tile the static mesh in memory.
    """
    return _decode(_read_bounded(record_path, MAX_RECORD_BYTES),
                   _read_bounded(metadata_path, MAX_METADATA_BYTES))


def triangles_to_edges(cells):
    """Match common.py: all 01, then 12, then 20; first unique max/min pairs."""
    cells = np.asarray(cells)
    if (cells.ndim != 2 or cells.shape[1] != 3 or cells.dtype.kind not in "iu"
            or len(cells) > MAX_CELLS or np.any(cells < 0)
            or np.any(cells > np.iinfo(np.int32).max)):
        raise ValueError("expected bounded nonnegative integer triangles")
    unique = {}
    for first, second in ((0, 1), (1, 2), (2, 0)):
        for face in cells:
            a, b = int(face[first]), int(face[second])
            if a == b:
                raise ValueError("triangle repeats a vertex")
            unique.setdefault((max(a, b), min(a, b)), None)
    high = np.asarray([pair[0] for pair in unique], dtype=np.int32)
    low = np.asarray([pair[1] for pair in unique], dtype=np.int32)
    return np.concatenate((high, low)), np.concatenate((low, high))


def edge_features(mesh_pos, senders, receivers):
    """Official CFD geometry: sender minus receiver, then Euclidean length."""
    pos = np.asarray(mesh_pos)
    senders, receivers = np.asarray(senders), np.asarray(receivers)
    if (pos.ndim != 2 or pos.shape[1] != 2 or not 1 <= len(pos) <= MAX_NODES
            or pos.dtype.kind != "f" or not np.isfinite(pos).all()
            or senders.ndim != 1 or receivers.shape != senders.shape
            or senders.dtype.kind not in "iu" or receivers.dtype.kind not in "iu"
            or senders.size > 6 * MAX_CELLS):
        raise ValueError("invalid bounded geometry or edge vectors")
    if any(np.any(index < 0) or np.any(index >= len(pos))
           for index in (senders, receivers)):
        raise ValueError("edge endpoint outside mesh")
    delta = pos.astype(np.float32)[senders] - pos.astype(np.float32)[receivers]
    norm = np.sqrt(np.sum(delta * delta, axis=1, keepdims=True, dtype=np.float32))
    result = np.concatenate((delta, norm), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("geometry overflow")
    return result


def fixture():
    """Synthetic two-triangle oracle with duplicate cell and isolated boundary.

    This is an adversarial arithmetic fixture, not a CylinderFlow simulation.
    """
    arrays = {
        "cells": np.array([[0, 1, 2], [1, 3, 2], [0, 1, 2]], dtype=np.int32),
        "mesh_pos": np.array([[0, 0], [1, 0], [0, 1], [1, 1], [3, 2]], dtype=np.float32),
        "node_type": np.array([[0], [5], [4], [6], [6]], dtype=np.int32),
        "velocity": np.array([[0.125, -0.25], [0.5, 0.125], [1, 0], [0, 0], [0, 0]], dtype=np.float32),
        "pressure": np.array([[0.25], [-0.5], [0.125], [0], [0]], dtype=np.float32),
    }
    arrays["senders"], arrays["receivers"] = triangles_to_edges(arrays["cells"])
    arrays["edge_features"] = edge_features(arrays["mesh_pos"], arrays["senders"], arrays["receivers"])
    return arrays


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args(argv)
    if not 0 <= args.frame < STEPS or args.output.suffix != ".npz":
        parser.error("frame must be 0..599 and output must end with .npz")
    sidecar = args.output.with_suffix(".json")
    if args.output.exists() or sidecar.exists():
        parser.error("output and provenance paths must be fresh")
    record = _read_bounded(args.record, MAX_RECORD_BYTES)
    metadata = _read_bounded(args.metadata, MAX_METADATA_BYTES)
    arrays = _decode(record, metadata)
    arrays["velocity"] = arrays["velocity"][args.frame].copy()
    arrays["pressure"] = arrays["pressure"][args.frame].copy()
    arrays["senders"], arrays["receivers"] = triangles_to_edges(arrays["cells"])
    arrays["edge_features"] = edge_features(arrays["mesh_pos"], arrays["senders"], arrays["receivers"])
    provenance = {
        "schema": 1, "dataset": "DeepMind MeshGraphNets cylinder_flow",
        "source_base_url": SOURCE_URL, "input_record_sha256": hashlib.sha256(record).hexdigest(),
        "input_record_bytes": len(record), "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "framing_crc32c_verified": True, "record_index": 0, "frame": args.frame,
        "trajectory_length": STEPS, "dt": 0.01,
        "nodes": len(arrays["mesh_pos"]), "cells": len(arrays["cells"]),
        "directed_edges": len(arrays["senders"]),
        "source_identity_limit": "Input hashes bind local bytes; URL generation is recorded by the downloader.",
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                           "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                   for name, value in arrays.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    provenance["npz_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    with sidecar.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "provenance": str(sidecar),
                      "nodes": provenance["nodes"], "directed_edges": provenance["directed_edges"]}))


if __name__ == "__main__":
    _main()
