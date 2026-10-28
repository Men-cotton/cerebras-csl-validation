"""Input-free halo schema 2 and immutable identities for streamed execution.

The aggregate byte limit is independent of the per-D2H chunk limit. Appliance
admission currently permits only the E210 3x3 pilot; this is not a scale result.
"""
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

if __package__:
    from . import chunks, halo
else:
    import chunks
    import halo

require = chunks.require
ROOT = Path(__file__).resolve().parent
WORKER_FILES = ("stream_worker.py", "stream_contract.py", "halo.py", "chunks.py")
APP_FILES = WORKER_FILES + ("stream_appliance.py", "export_stream.py")
ORACLE = "independent-halo-integer-window-v1"
CASE_NAME = "halo-stream-3x3-c128"
PILOT_SEQUENCE = [[17, 0], [32, 1], [32, 0]]
SOURCE_NAMES = {"layout.csl", "pe.csl", "vendor_layout.csl", "vendor_pe.csl"}
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
TREE_LIMIT = 256 << 20
JSON_LIMIT = chunks.METADATA_LIMIT
LOG_LIMIT = 16 << 20


def read_json(path):
    return json.loads(halo.bounded(path), object_pairs_hook=chunks.unique_object)


def write_json(path, value):
    payload = chunks.encoded(value)
    require(len(payload) <= JSON_LIMIT, "metadata byte bound")
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    require(not path.is_symlink() and not temporary.exists() and not temporary.is_symlink(), "metadata output path")
    with temporary.open("xb") as stream:
        stream.write(payload)
    temporary.replace(path)


def new_directory(path):
    path = chunks.regular_root(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def inventory(folder, byte_limit=TREE_LIMIT, entry_limit=4096):
    root = chunks.regular_root(folder)
    require(root.is_dir(), "inventory directory")
    start, pending, total, entries, result = time.monotonic(), [root], 0, 0, {}
    while pending:
        with os.scandir(pending.pop()) as children:
            for child in children:
                entries += 1
                require(entries <= entry_limit and time.monotonic()-start <= 10, "inventory scan bound")
                require(not child.is_symlink(), "inventory symlink")
                if child.is_dir(follow_symlinks=False):
                    pending.append(Path(child.path))
                else:
                    require(child.is_file(follow_symlinks=False), "inventory special file")
                    size = child.stat(follow_symlinks=False).st_size
                    total += size
                    require(total <= byte_limit, "inventory byte bound")
                    result[Path(child.path).relative_to(root).as_posix()] = size
    require(time.monotonic()-start <= 10, "inventory scan time bound")
    return result


def source_identity(names=WORKER_FILES):
    return {name: halo.file_hash(ROOT/name, JSON_LIMIT) for name in names}


def worker_digest():
    return chunks.digest(chunks.encoded(source_identity()))


def validate_manifest(case):
    """Schema-2 structural gate; compatible with halo.contract's validator API."""
    require(isinstance(case, dict) and set(case) == {"schema", "kind", "grid", "params", "symbols", "functions",
            "operations", "validations", "census", "capture", "oracle", "source_files"}, "stream case inventory")
    require(type(case["schema"]) is int and case["schema"] == 2 and case["kind"] == "halo", "stream halo schema")
    grid = case["grid"]
    require(isinstance(grid, list) and len(grid) == 2 and all(type(n) is int for n in grid), "stream grid")
    width, height = grid
    require(2 <= width <= 755 and 2 <= height <= 1170 and 17*(width-1)+31*(height-1) <= 32767,
            "runtime coordinate arithmetic/fabric bounds")
    require(case["symbols"] == halo.SYMBOLS and case["functions"] == halo.FUNCTIONS, "stream ABI")
    capture = case["capture"]
    require(isinstance(capture, dict) and set(capture) == {"chunk_bytes", "total_bytes", "max_chunks"}, "capture bounds")
    oracle = case["oracle"]
    require(isinstance(oracle, dict) and set(oracle) == {"name", "source_sha256", "reference_sha256"}
            and oracle["name"] == ORACLE, "procedural oracle")
    require(all(isinstance(oracle[k], str) and chunks.SHA.fullmatch(oracle[k])
                for k in ("source_sha256", "reference_sha256")), "oracle hashes")
    source = case["source_files"]
    require(isinstance(source, dict) and set(source) == SOURCE_NAMES and
            all(isinstance(h, str) and chunks.SHA.fullmatch(h) for h in source.values()), "CSL source identity")
    require(oracle["source_sha256"] == halo.file_hash(ROOT/"halo.py", JSON_LIMIT), "procedural oracle source differs")
    return {}, {}


def fields(case):
    return halo.contract(case, sys.modules[__name__])


def make_plan(case, identity):
    definitions, _ = fields(case)
    return chunks.make_plan(identity, case["grid"], definitions, **case["capture"])


def pilot(case):
    fields(case)
    require(case["grid"] == [3, 3] and case["census"]["sequence"] == PILOT_SEQUENCE and
            case["capture"] == dict(chunk_bytes=128, total_bytes=7668, max_chunks=66),
            "only the 3x3 / 128-byte E210 pilot is admitted; larger campaigns require a separate rung")


def package_identity(package, name):
    root = chunks.regular_root(package)
    require(isinstance(name, str) and NAME.fullmatch(name), "case name")
    manifest = read_json(root/"manifest.json")
    require(set(manifest) == {"schema", "source_revision", "cases", "files"} and manifest["schema"] == 2,
            "stream package schema")
    require(isinstance(manifest["source_revision"], str) and re.fullmatch(r"[0-9a-f]{40}", manifest["source_revision"]),
            "full source revision")
    require(set(manifest["cases"]) == {CASE_NAME} and name == CASE_NAME, "pilot package case inventory")
    folder = root/"cases"/name
    case_bytes = halo.bounded(folder/"case.json")
    case = json.loads(case_bytes, object_pairs_hook=chunks.unique_object)
    pilot(case)
    expected = {f"cases/{name}/case.json", f"cases/{name}/oracle.py"} | {
        f"cases/{name}/source/{n}" for n in SOURCE_NAMES}
    require(set(manifest["files"]) == expected and set(inventory(root, 16 << 20)) == expected | {"manifest.json"},
            "exact stream package file inventory")
    for relative, sha in manifest["files"].items():
        require(halo.file_hash(root/relative, JSON_LIMIT) == sha, "package source hash: "+relative)
    require(case["oracle"]["source_sha256"] == halo.file_hash(ROOT/"halo.py", JSON_LIMIT) and
            case["oracle"]["reference_sha256"] == halo.file_hash(folder/"oracle.py", JSON_LIMIT), "frozen oracle identity")
    require(all(halo.file_hash(folder/"source"/n, JSON_LIMIT) == sha for n, sha in case["source_files"].items()),
            "case CSL source hashes")
    case_hash = chunks.digest(case_bytes)
    require(manifest["cases"][name] == dict(case_sha256=case_hash, params=case["params"]), "package case identity")
    return root, folder, case, dict(case_name=name, case_sha256=case_hash,
                                   package_manifest_sha256=halo.file_hash(root/"manifest.json", JSON_LIMIT))


def check_abi(case, out, rpc):
    fields(case)
    parameters = out["params"]
    require(isinstance(parameters, dict) and set(parameters) == set(case["params"]) and
            all(type(v) in (str, int) for v in parameters.values()) and
            {k: int(v) for k, v in parameters.items()} == case["params"], "compiled parameters")
    types = {"float32": "[*]f32", "uint32": "[*]u32", "uint16": "[*]u16"}
    wanted = {n: ("Var", types[s[1]]) for n, s in case["symbols"].items()}
    wanted.update({n: ("Func", "void") for n in case["functions"]})
    symbols = rpc["rpc_symbols"]
    require(isinstance(symbols, list) and len(symbols) == len(wanted) and
            len({s["name"] for s in symbols}) == len(symbols) and
            {s["name"]: (s["kind"], s["type"]) for s in symbols} == wanted, "compiled symbol inventory/types")
    variables = [s for s in symbols if s["kind"] == "Var"]
    require(all(type(s.get("id")) is int and s["id"] >= 0 for s in variables) and
            len({s["id"] for s in variables}) == len(variables), "compiled variable IDs")
    for symbol in symbols:
        if symbol["kind"] == "Func":
            require(symbol.get("inputs", []) == case["functions"][symbol["name"]], "compiled RPC arguments")
    return {s["name"]: s["id"] for s in variables}


def request(case_bytes, identity, backend, artifact_sha256, build_sha256):
    value = dict(schema=2, run_id=str(uuid.uuid4()), backend=backend, **identity,
                 artifact_sha256=artifact_sha256, build_record_sha256=build_sha256,
                 worker_sha256=worker_digest(), worker_sources=source_identity())
    validate_request(value, case_bytes, backend)
    return value


def validate_request(value, case_bytes, backend):
    require(set(value) == {"schema", "run_id", "backend", "case_name", "case_sha256", "package_manifest_sha256",
            "artifact_sha256", "build_record_sha256", "worker_sha256", "worker_sources"} and value["schema"] == 2,
            "stream request schema")
    require(str(uuid.UUID(value["run_id"])) == value["run_id"] and value["backend"] == backend
            and backend in ("local-sim", "hardware", "appliance-sim"), "request UUID/backend")
    require(value["case_name"] == CASE_NAME, "request case name")
    require(all(isinstance(value[k], str) and chunks.SHA.fullmatch(value[k]) for k in
                ("case_sha256", "package_manifest_sha256", "artifact_sha256", "build_record_sha256", "worker_sha256")),
            "request digest schema")
    require(value["case_sha256"] == chunks.digest(case_bytes) and value["worker_sources"] == source_identity()
            and value["worker_sha256"] == worker_digest(), "request case/worker/helper/oracle binding")
    pilot(json.loads(case_bytes))


def plan_identity(request_bytes):
    value = json.loads(request_bytes)
    result = {k: value[k] for k in chunks.IDENTITY_KEYS-{"request_sha256"}}
    return dict(result, request_sha256=chunks.digest(request_bytes))
