"""Bounded readback for the frozen independent-halo primitive; local SDK CLI."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
import zipfile

import numpy as np

if __package__:
    from . import chunks
else:
    import chunks

require = chunks.require
PAYLOAD_LIMIT = 92 << 20
BUILD_LIMIT = 256 << 20
SYMBOLS = {"y": [32, "float32"], "Q": [32, "float32"],
           "count": [1, "uint32"], "clocks": [6, "uint16"]}
FUNCTIONS = {"apply_epochs": [{"name": "length", "type": "i16"},
                               {"name": "input_phase", "type": "i16"}]}
# W, E, S, N, bottom, top, center, as in the frozen independent oracle.
NEIGHBORS = ((1, 0, -1, 0), (2, 0, 1, 0), (3, 1, 0, 0),
             (4, -1, 0, 0), (5, 0, 0, -1), (6, 0, 0, 1), (7, 0, 0, 0))


def contract(case, worker):
    """Accept the existing launch/readback program exactly; never rewrite H2D."""
    inputs, _ = worker.validate_manifest(case)
    require(case["kind"] == "halo" and not inputs and case["symbols"] == SYMBOLS
            and case["functions"] == FUNCTIONS and case["validations"] == {"restore": []},
            "frozen input-free halo ABI")
    width, height = case["grid"]
    require(case["params"] == dict(width=width, height=height), "halo compile parameters")
    census = case["census"]
    require(all(type(census[key]) is int and census[key] == value for key, value in
                (("epochs_per_launch", 64), ("max_n", 32), ("block_size", 16))), "frozen halo constants")
    sequence = census["sequence"]
    require(isinstance(sequence, list) and 1 <= len(sequence) <= 51, "halo launch count")
    expected, fields, generations = [], [], {}
    for generation, pair in enumerate(sequence):
        require(isinstance(pair, list) and len(pair) == 2 and all(type(v) is int for v in pair)
                and 2 <= pair[0] <= 32 and pair[1] in (0, 1), "halo length/phase")
        expected.append(dict(op="launch", symbol="apply_epochs", args=pair))
        for name, (count, dtype) in SYMBOLS.items():
            key = f"s{generation}_{name}"
            operation = dict(op="d2h", symbol=name, array=key, rect=[0, 0, width, height])
            if name == "clocks":
                operation["clock"] = True
            expected.append(operation)
            fields.append(dict(name=key, dtype="float32" if dtype == "float32" else "uint32",
                               count=count, rect=operation["rect"]))
            generations[key] = (generation, name, pair)
    require(chunks.encoded(case["operations"]) == chunks.encoded(expected), "exact halo generation/operation sequence")
    return fields, generations


def make_plan(case, identity, worker, chunk_bytes=4 << 20, *, total_bytes=PAYLOAD_LIMIT, max_chunks=chunks.MAX_CHUNKS):
    fields, _ = contract(case, worker)
    return chunks.make_plan(identity, case["grid"], fields, chunk_bytes=chunk_bytes,
                            total_bytes=total_bytes, max_chunks=max_chunks)


def window(case, symbol, pair, rectangle):
    """Generate only this rectangle, testing neighbors against the full grid.

    Integer temporaries have the chunk's PE count, never the full grid's. At
    64 epochs the exact integer y/Q controls are below 2**24, including tails.
    """
    width, height = case["grid"]
    x, y, columns, rows = rectangle
    n, phase = pair
    require(symbol in ("y", "Q", "count") and 0 <= x < width and 0 <= y < height
            and columns > 0 and rows > 0 and x+columns <= width and y+rows <= height,
            "oracle window")
    if symbol == "count":
        return np.full((rows*columns, 1), 64, dtype=np.uint32)
    points = np.arange(rows*columns, dtype=np.int32)
    r, c = (y+points//columns)[:, None], (x+points % columns)[:, None]
    k = np.arange(n, dtype=np.int32)[None, :]
    initial = np.zeros((len(points), n), dtype=np.int32)
    weights = np.zeros_like(initial)
    for coefficient, dr, dc, dk in NEIGHBORS:
        sr, sc, sk = r+dr, c+dc, k+dk
        valid = (sr >= 0) & (sr < height) & (sc >= 0) & (sc < width) & (sk >= 0) & (sk < n)
        initial += np.where(valid, coefficient*(1+(17*sc+31*sr+7*sk+11*phase) % 64), 0)
        weights += coefficient*valid
    result = np.zeros((len(points), 32), dtype=np.float32)
    result[:, :n] = initial+4032*weights if symbol == "y" else 64*initial+129024*weights
    return result


def clock_summary(values):
    require(values.dtype == np.uint32 and values.ndim == 2 and values.shape[1] == 6
            and bool((values <= 65535).all()), "complete uint16 clock carriers")
    wide = values.astype(np.uint64)
    start = wide[:, 0]+(wide[:, 1] << 16)+(wide[:, 2] << 32)
    end = wide[:, 3]+(wide[:, 4] << 16)+(wide[:, 5] << 32)
    elapsed = (end-start) & np.uint64((1 << 48)-1)
    require(bool(((elapsed > 0) & (elapsed < 1 << 47)).all()), "positive unambiguous same-PE clocks")
    # A Python integer accumulator cannot overflow across the complete grid.
    return dict(pe_count=len(elapsed), local_cycles_min=int(elapsed.min()),
                local_cycles_max=int(elapsed.max()), local_cycles_sum=sum(map(int, elapsed)))


def audit(folder, case_bytes, plan, manifest_sha256, worker, *, total_bytes=PAYLOAD_LIMIT, max_chunks=chunks.MAX_CHUNKS):
    require(chunks.digest(case_bytes) == plan["identity"]["case_sha256"], "audit case identity")
    case = json.loads(case_bytes)
    require(chunks.encoded(make_plan(case, plan["identity"], worker, plan["limits"]["chunk_bytes"],
                                           total_bytes=total_bytes, max_chunks=max_chunks))
            == chunks.encoded(plan), "halo capture plan")
    _, generations = contract(case, worker)
    clocks, words = {}, 0

    def check(descriptor, values):
        nonlocal words
        _, symbol, pair = generations[descriptor["field"]]
        if symbol != "clocks":
            reference = window(case, symbol, pair, descriptor["rect"])
            require(np.array_equal(values.view(np.uint32), reference.view(np.uint32)),
                    "exact halo numerical mismatch: "+descriptor["field"])
            words += values.size
            return
        summary = clock_summary(values)
        previous = clocks.get(descriptor["field"])
        if previous is None:
            clocks[descriptor["field"]] = summary
        else:
            previous["pe_count"] += summary["pe_count"]
            previous["local_cycles_sum"] += summary["local_cycles_sum"]
            previous["local_cycles_min"] = min(previous["local_cycles_min"], summary["local_cycles_min"])
            previous["local_cycles_max"] = max(previous["local_cycles_max"], summary["local_cycles_max"])

    result = chunks.audit(folder, plan, manifest_sha256, check)
    require(len(clocks) == len(case["census"]["sequence"]) and
            all(v["pe_count"] == case["grid"][0]*case["grid"][1] for v in clocks.values()),
            "all-PE clock coverage")
    for value in clocks.values():
        value["max_local_cycles_per_epoch"] = value["local_cycles_max"]/64
    return dict(result, numerical_status="PASS", numerical_words=words, clocks=clocks,
                scope="All y/Q/count bits and all same-PE clock pairs; no global makespan or execution proof.")


def program(case, plan, runner, ids, datatype, order, folder, events):
    """Read each Writer rectangle directly from SDK before the next launch."""
    with chunks.Writer(folder, plan) as writer:
        for index, operation in enumerate(case["operations"]):
            name = operation["symbol"]
            if operation["op"] == "launch":
                runner.launch(name, *map(np.int16, operation["args"]), nonblock=False)
                events.append(dict(operation=index, op="launch", symbol=name, args=operation["args"]))
                continue
            require(operation["op"] == "d2h", "unsupported halo input operation")
            require(writer.current is not None and writer.current["field"] == operation["array"], "readback generation order")
            count, dtype = case["symbols"][name]
            while writer.current is not None and writer.current["field"] == operation["array"]:
                descriptor = writer.current
                x, y, width, height = descriptor["rect"]
                carrier = np.full(width*height*count, 0xdeadbeef, dtype=np.uint32)
                values = carrier.view(np.float32) if dtype == "float32" else carrier
                runner.memcpy_d2h(values, ids[name], x, y, width, height, count,
                                  streaming=False, nonblock=False, order=order.ROW_MAJOR,
                                  data_type=datatype.MEMCPY_16BIT if dtype == "uint16" else datatype.MEMCPY_32BIT)
                events.append(dict(operation=index, op="d2h", chunk=descriptor["index"], symbol=name,
                                   symbol_id=ids[name], rect=descriptor["rect"], words_per_pe=count,
                                   memcpy_bits=16 if dtype == "uint16" else 32))
                writer.write(descriptor, values.reshape(width*height, count))
                del values, carrier, descriptor
        return writer.finish()


def expected_events(case, plan, ids):
    descriptors = iter(chunks.iter_plan(plan))
    current = next(descriptors, None)
    for index, operation in enumerate(case["operations"]):
        name = operation["symbol"]
        if operation["op"] == "launch":
            yield dict(operation=index, op="launch", symbol=name, args=operation["args"])
        else:
            while current is not None and current["field"] == operation["array"]:
                yield dict(operation=index, op="d2h", chunk=current["index"], symbol=name,
                           symbol_id=ids[name], rect=current["rect"], words_per_pe=current["count"],
                           memcpy_bits=16 if case["symbols"][name][1] == "uint16" else 32)
                current = next(descriptors, None)
    require(current is None, "complete operation chunks")


def bounded(path, limit=chunks.METADATA_LIMIT):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, "bounded regular file")
    with path.open("rb") as stream:
        payload = stream.read(limit+1)
    require(len(payload) <= limit, "file grew beyond bound")
    return payload


def file_hash(path, limit=BUILD_LIMIT):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, "bounded hash input")
    sha, total = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while True:
            payload = stream.read(1 << 20)
            if not payload:
                break
            total += len(payload)
            require(total <= limit, "hash byte bound")
            sha.update(payload)
    return sha.hexdigest()


def verify_build(build, inventory):
    files = inventory["files"]
    require(isinstance(files, dict) and 1 <= len(files) <= 512, "small existing build inventory")
    seen, total, entries, began = set(), 0, 0, time.monotonic()

    def unreadable(error):
        raise error

    for root, directories, filenames in os.walk(build, followlinks=False, onerror=unreadable):
        require(time.monotonic()-began <= 10, "build inventory time bound")
        for name in directories+filenames:
            entries += 1
            require(entries <= 10000 and not (Path(root)/name).is_symlink(), "bounded regular build tree")
        for name in filenames:
            path = Path(root)/name
            total += path.stat().st_size
            require(total <= BUILD_LIMIT and path.is_file(), "build visible byte bound")
            if path.suffix != ".elf" and name not in ("out.json", "out_rpc.json"):
                continue
            relative = path.relative_to(build).as_posix()
            require(relative in files, "unlisted compiled artifact")
            require(file_hash(path) == files[relative], "compiled file identity")
            seen.add(relative)
    require(time.monotonic()-began <= 10 and seen == set(files) and {"out.json", "bin/out_rpc.json"} <= seen
            and any(n.endswith(".elf") for n in seen), "complete compiled file inventory")


def load_worker(path):
    spec = importlib.util.spec_from_file_location("halo_reference_worker", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def run_local(args, sdk_factory=None):
    """New local request, unchanged frozen case/build; never opens expected.npz."""
    worker = load_worker(args.worker_source)
    case_dir, build = Path(args.case).resolve(), Path(args.build).resolve()
    case_bytes = bounded(case_dir/"case.json")
    case = json.loads(case_bytes)
    contract(case, worker)
    inputs = case_dir/"inputs.npz"
    require(file_hash(inputs, 1 << 20) == case["files"]["inputs.npz"], "empty input identity")
    with zipfile.ZipFile(inputs) as archive:
        require(not archive.infolist(), "halo must have no immutable input arrays")
    inventory_bytes = bounded(args.build_inventory)
    inventory = json.loads(inventory_bytes)
    verify_build(build, inventory)
    out_bytes, rpc_bytes = bounded(build/"out.json"), bounded(build/"bin/out_rpc.json")
    ids = worker.check_abi(case, json.loads(out_bytes), json.loads(rpc_bytes))
    sources = {"halo.py": file_hash(Path(__file__)), "chunks.py": file_hash(Path(chunks.__file__)),
               "tank_worker.py": file_hash(Path(args.worker_source))}
    request = dict(schema=1, backend="local-sim", run_id=str(uuid.uuid4()),
                   case_sha256=chunks.digest(case_bytes), artifact_sha256=chunks.digest(inventory_bytes),
                   worker_sha256=chunks.digest(chunks.encoded(sources)), chunk_bytes=args.chunk_bytes,
                   scope="new local streaming request using an existing frozen case and compiled inventory")
    request_bytes = chunks.encoded(request)
    identity = {key: request[key] for key in chunks.IDENTITY_KEYS-{"request_sha256"}}
    identity["request_sha256"] = chunks.digest(request_bytes)
    plan = make_plan(case, identity, worker, args.chunk_bytes)
    destination = chunks.regular_root(args.out)
    runtime = chunks.regular_root(destination.with_name(destination.name+"-runtime"))
    require(not destination.exists() and not runtime.exists(), "fresh capture and runtime directories")
    destination.mkdir(parents=True)
    runtime.mkdir()
    for name, payload in {"request.json": request_bytes, "plan.json": chunks.encoded(plan),
                          "sources.json": chunks.encoded(sources), "case.json": case_bytes,
                          "local-build.json": inventory_bytes, "out.json": out_bytes,
                          "out_rpc.json": rpc_bytes}.items():
        (destination/name).write_bytes(payload)
    result = dict(execution_status="FAILED", numerical_status="NOT_RUN", identity=identity, events=[], lifecycle=[])
    runner, previous = None, Path.cwd()
    try:
        sdk_build = runtime/"build"
        shutil.copytree(build, sdk_build)
        verify_build(sdk_build, inventory)
        os.chdir(runtime)
        runtime_type, datatype, order = (sdk_factory or worker.sdk_classes)()
        runner = runtime_type(str(sdk_build), cmaddr=None, simfab_numthreads=1, suppress_simfab_trace=True)
        result["lifecycle"].append("construct")
        observed = {}
        for name in case["symbols"]:
            observed[name] = runner.get_id(name)
            result["lifecycle"].append("get_id:"+name)
        require(observed == ids, "runtime/compiled symbol IDs")
        runner.load()
        result["lifecycle"].append("load")
        runner.run()
        result["lifecycle"].append("run")
        result["capture"] = program(case, plan, runner, ids, datatype, order, destination/"chunks", result["events"])
    except BaseException as error:
        result["error"] = type(error).__name__+": "+str(error)[:1024]
    finally:
        try:
            if runner is not None:
                runner.stop()
                result["lifecycle"].append("stop")
        except BaseException as error:
            result["stop_error"] = type(error).__name__+": "+str(error)[:1024]
        finally:
            os.chdir(previous)
    try:
        verify_build(build, inventory)
        require(bounded(case_dir/"case.json") == case_bytes and bounded(args.build_inventory) == inventory_bytes
                and file_hash(inputs, 1 << 20) == case["files"]["inputs.npz"], "frozen inputs/build inventory changed")
        require(bounded(destination/"request.json") == request_bytes and sources ==
                {"halo.py": file_hash(Path(__file__)), "chunks.py": file_hash(Path(chunks.__file__)),
                 "tank_worker.py": file_hash(Path(args.worker_source))}, "request/worker changed during execution")
        require(result["events"] == list(expected_events(case, plan, ids)), "complete SDK operation/generation order")
        require(result["lifecycle"] == ["construct"]+["get_id:"+n for n in case["symbols"]]+["load", "run", "stop"],
                "complete SDK lifecycle")
        if not any(key in result for key in ("error", "stop_error")):
            result["execution_status"] = "PASS"
            result["numerical_status"] = "FAIL"
            result["audit"] = audit(destination/"chunks", case_bytes, plan, result["capture"]["manifest_sha256"], worker)
            result["numerical_status"] = "PASS"
    except BaseException as error:
        result["audit_error"] = type(error).__name__+": "+str(error)[:1024]
    payload = chunks.encoded(result)
    require(len(payload) <= chunks.METADATA_LIMIT, "execution metadata bound")
    (destination/"execution.json").write_bytes(payload)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--build-inventory", type=Path, required=True)
    parser.add_argument("--worker-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=4 << 20)
    args = parser.parse_args()
    result = run_local(args)
    print(json.dumps({key: result[key] for key in ("execution_status", "numerical_status", "identity")}))
    return 0 if result["execution_status"] == result["numerical_status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
