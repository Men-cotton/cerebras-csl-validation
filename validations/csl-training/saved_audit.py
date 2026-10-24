"""Portable frozen-case and full SGD-capture checks; NumPy only, no PyTorch/SDK."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
import zipfile

import numpy as np

JSON_LIMIT = 2 << 20
CASE_LIMIT = 128 << 20
CAPTURE_LIMIT = 256 << 20
CASE_FILES = {"inputs.npz", "samples.npz", "expected.npz", "graph.npz",
              "original_samples.npz", "mapping.npz", "sdk_runner.py",
              "source/pe.csl", "source/layout.csl"}
HOST_FILES = ("appliance.py", "appliance_worker.py", "saved_audit.py")


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


def read_json(path):
    return json.loads(bounded(path, JSON_LIMIT))


def write_json(path, value):
    path = Path(path)
    require(not path.is_symlink() and (not path.exists() or path.is_file()), "regular JSON destination")
    content = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    require(len(content) <= JSON_LIMIT, "JSON output bound")
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


def schemas(p):
    l, e, w = p["L"], p["E"], p["W"]
    inputs = dict(counts=(2, "uint32"), src_owner=(e, "uint32"), src_slot=(e, "uint32"),
                  dst_slot=(e, "uint32"), src_begin=(p["C"]*p["R"]+1, "uint32"),
                  edge_input=(3*e, "float32"), weights=(w, "float32"),
                  node_input=(11*l, "float32"), target=(2*l, "float32"),
                  loss_mask=(l, "float32"), scalars=(2, "float32"))
    outputs = dict(prediction=(2*l, "float32"), loss=(1, "float32"), gradients=(w, "float32"),
                   stats=(16, "uint32"), clocks=(6, "uint16"), lifecycle=(2, "uint32"))
    return inputs, outputs


def load(case):
    case = Path(case).absolute()
    manifest = read_json(local_file(case, "case.json"))
    require(manifest.get("schema") == 1 and manifest.get("experiment") == "E198", "case identity")
    p = manifest.get("params")
    require(isinstance(p, dict) and set(p) == {"C", "R", "H", "S", "L", "E", "W"}
            and all(type(v) is int and 1 <= v <= 32767 for v in p.values()), "bounded case parameters")
    require(p["H"] == 4 and p["S"] == 2 and p["W"] == 578 and min(p["C"], p["R"]) >= 2
            and p["C"]*p["R"] <= 1024, "supported H4/S2 training ABI")
    require(type(manifest.get("steps")) is int and 1 <= manifest["steps"] <= 16, "bounded update count")
    ins, outs = schemas(p)
    require(manifest.get("inputs") == {k: list(v) for k, v in ins.items()} and
            manifest.get("outputs") == {k: list(v) for k, v in outs.items()}, "case array ABI")
    pins = manifest.get("files")
    require(isinstance(pins, dict) and set(pins) == CASE_FILES, "exact frozen case inventory")
    total = 0
    for name, digest in pins.items():
        require(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest), "SHA256 encoding")
        path = local_file(case, name)
        require(sha(path) == digest, "frozen file changed: " + name)
        total += path.stat().st_size
        require(total <= CASE_LIMIT, "case byte bound")
    return manifest


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


def pack_values(values, ids):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    result = np.zeros((*ids.shape, values.shape[-1]), np.float32)
    valid = ids >= 0
    result[valid] = values[ids[valid]]
    return result.reshape(len(ids), -1)


def check_abi(build, manifest):
    build = Path(build)
    actual = read_json(build/"out.json")["params"]
    require({k: int(v) for k, v in actual.items()} == manifest["params"], "compiled parameter ABI")
    symbols = read_json(build/"bin/out_rpc.json")["rpc_symbols"]
    expected = {name: ("Var", {"float32": "[*]f32", "uint32": "[*]u32", "uint16": "[*]u16"}[dtype], ())
                for name, (_, dtype) in dict(manifest["inputs"], **manifest["outputs"]).items()}
    expected["run"] = ("Func", "void", ())
    require(len(symbols) == len(expected) and len({s["name"] for s in symbols}) == len(symbols)
            and len({s["id"] for s in symbols}) == len(symbols), "unique complete compiled symbols")
    require({s["name"]: (s["kind"], s["type"], tuple(arg["type"] for arg in s.get("inputs", [])))
             for s in symbols} == expected, "compiled symbol ABI")


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


def admit_capture(folder, *, allow_incomplete_worker=False):
    folder = Path(folder).absolute()
    record, build, request = (read_json(folder/name) for name in ("capture.json", "manifest.json", "request.json"))
    statuses = {"PASS", "COMPLETED"}
    if allow_incomplete_worker:
        statuses.add("incomplete")  # Older workers ran the numerical audit remotely.
    require(record.get("status") in statuses and record.get("backend") == build.get("backend")
            and build.get("backend") in ("hardware", "simulator"), "complete capture backend")
    require(record.get("run_id") == request.get("run_id") and
            record.get("manifest_sha256") == request.get("manifest_sha256") == sha(folder/"manifest.json"),
            "capture manifest/request binding")
    require(record.get("request_sha256") == sha(folder/"request.json"), "capture request bytes")
    require(sha(folder/"case/case.json") == build["case_sha256"], "captured frozen case")
    require(set(build["host_sha256"]) == set(HOST_FILES), "captured host inventory")
    for name, expected in build["host_sha256"].items():
        require(sha(local_file(folder/"host", name)) == expected, "captured host source: " + name)
    check_abi(folder/"build", load(folder/"case"))
    for name, expected in record["compiled_metadata_sha256"].items():
        require(name in ("out.json", "bin/out_rpc.json") and sha(folder/"build"/name) == expected,
                "captured compiled metadata")
    require(set(record["compiled_metadata_sha256"]) == {"out.json", "bin/out_rpc.json"}, "compiled metadata inventory")
    result = admit_execution(folder/"case", folder/"runtime")
    require(result["backend"] == build["backend"], "runtime backend binding")
    result.update(run_id=request["run_id"], manifest_sha256=sha(folder/"manifest.json"),
                  request_sha256=sha(folder/"request.json"), host_sha256=build["host_sha256"],
                  worker_status=record["status"], worker_error=record.get("error"))
    return result


def _load_execution(case, output):
    case, output = Path(case), Path(output)
    manifest = load(case)
    execution = read_json(output / "execution.json")
    require(execution["status"] == "COMPLETED" and
            "error" not in execution and "stop_error" not in execution and
            execution["case_sha256"] == sha(case / "case.json") and
            execution["raw_sha256"] == sha(output / "raw.npz"), "completed bound capture")
    raw = load_npz(output / "raw.npz")
    fixed = load_npz(case / "inputs.npz")
    samples = load_npz(case / "samples.npz")
    ids = load_npz(case / "mapping.npz")["node_ids"]
    schema = dict(manifest["inputs"], **manifest["outputs"])
    expected_calls = ["construct"] + ["get_id:"+name for name in schema] + ["load", "run"]
    expected_calls += ["h2d:"+name for name in fixed]
    expected_calls += ["d2h:initial_lifecycle", "d2h:initial_weights"]
    expected_raw = {"initial_lifecycle": schema["lifecycle"], "initial_weights": schema["weights"]}
    for step in range(manifest["steps"]):
        for name in samples:
            label = f"step{step}_input_{name}"
            expected_calls += ["h2d:"+name, "d2h:"+label]
            expected_raw[label] = schema[name]
        expected_calls += ["launch:run"]
        for name in list(manifest["outputs"]) + ["weights"]:
            label = f"step{step}_{name}"
            expected_calls += ["d2h:"+label]
            expected_raw[label] = schema[name]
    for name in fixed:
        if name != "weights":
            expected_calls += ["d2h:after_"+name]
            expected_raw["after_"+name] = schema[name]
    expected_calls += ["stop"]
    require([row["api"] for row in execution["calls"]] == expected_calls,
            "exact SDK call order; no host gradient or optimizer")
    require(set(raw) == set(expected_raw), "complete exact capture inventory")
    for name, (count, dtype) in expected_raw.items():
        carrier = "uint32" if dtype == "uint16" else dtype
        require(raw[name].shape == (len(ids), count) and raw[name].dtype == np.dtype(carrier),
                "capture shape/dtype: " + name)
    return manifest, execution, raw, fixed, samples, ids


def admit_execution(case, output):
    """Verify completed SDK calls and complete, bound arrays without value checks."""
    manifest, execution, raw, fixed, samples, ids = _load_execution(case, output)
    return dict(status="PASS", execution_status="PASS", numerical_status="NOT_RUN",
                backend=execution["backend"], case_sha256=execution["case_sha256"],
                raw_sha256=execution["raw_sha256"], steps=manifest["steps"], arrays=len(raw),
                claim="SDK calls and capture complete; numerical correctness not checked")


def audit_capture(folder):
    """Replay numerical checks locally, including complete legacy failed workers."""
    folder = Path(folder)
    admission = admit_capture(folder, allow_incomplete_worker=True)
    result = audit(folder/"case", folder/"runtime")
    return dict(admission, **result)


def audit(case, output):
    case, output = Path(case), Path(output)
    manifest, execution, raw, fixed, samples, ids = _load_execution(case, output)
    expected = load_npz(case/"expected.npz")
    p, reports = manifest["params"], []
    np.testing.assert_array_equal(raw["initial_lifecycle"], 0)
    np.testing.assert_array_equal(raw["initial_weights"], fixed["weights"])
    require(sorted(ids[ids >= 0].tolist()) == list(range(int(fixed["counts"][:, 0].sum()))), "node ownership coverage")
    for step in range(manifest["steps"]):
        for name, sample in samples.items():
            np.testing.assert_array_equal(raw[f"step{step}_input_{name}"], sample[step], err_msg=name)
        errors = {}
        for name in ("prediction", "loss", "gradients", "weights"):
            observed = raw[f"step{step}_{name}"]
            require(np.isfinite(observed).all(), "finite " + name)
            wanted = expected[f"step{step}_{name}"]
            if name == "prediction":
                wanted = pack_values(wanted, ids)
                np.testing.assert_array_equal(observed.reshape(*ids.shape, 2)[ids < 0], 0)
            else:
                wanted = np.tile(wanted.reshape(1, -1), (len(ids), 1))
                np.testing.assert_array_equal(observed, np.tile(observed[0], (len(ids), 1)),
                                               err_msg="replicated " + name)
            np.testing.assert_allclose(observed, wanted, rtol=8e-4, atol=8e-5, err_msg=f"step{step}_{name}")
            errors[name] = float(np.max(np.abs(observed.astype(np.float64)-wanted)))
        lifecycle = np.tile(np.array([1, step+1], np.uint32), (len(ids), 1))
        np.testing.assert_array_equal(raw[f"step{step}_lifecycle"], lifecycle)
        stats = raw[f"step{step}_stats"]
        expected_stats = []
        owners, blocks = len(ids), p["S"]
        for nodes, edges in fixed["counts"]:
            nodes, edges = int(nodes), int(edges)
            expected_stats.append([0x600D, 1, blocks, edges, nodes, blocks*edges, blocks*nodes,
                nodes, 2*blocks*owners, 3*blocks*(owners+1)+1,
                (blocks+2)*nodes+(blocks+1)*edges, blocks*owners, 1, nodes, edges, 0])
        np.testing.assert_array_equal(stats, np.asarray(expected_stats, np.uint32))
        clocks = raw[f"step{step}_clocks"].astype(np.uint64)
        require(np.all(clocks < 65536), "u16 clock carrier")
        begin = clocks[:, 0] + (clocks[:, 1] << 16) + (clocks[:, 2] << 32)
        end = clocks[:, 3] + (clocks[:, 4] << 16) + (clocks[:, 5] << 32)
        spans = (end-begin) & ((1 << 48)-1)
        require(np.all((spans > 0) & (spans < 1 << 47)), "valid local clock span")
        reports.append(dict(step=step, errors=errors, max_pe_local_cycles=int(spans.max())))
    for name, value in fixed.items():
        if name != "weights":
            np.testing.assert_array_equal(raw["after_" + name], value, err_msg="immutable " + name)
    calls = [call["api"] for call in execution["calls"]]
    require(calls.count("launch:run") == manifest["steps"] and calls.count("h2d:weights") == 1,
            "exact update count, no host weight update")
    return dict(status="PASS", numerical_status="PASS", reports=reports, backend=execution["backend"],
        case_sha256=sha(case / "case.json"), raw_sha256=sha(output / "raw.npz"),
        claim=("fabricated host fixture accepted; no device execution" if execution["backend"] == "host-test"
               else "full CSL forward/backward/SGD numerical agreement; PE-local cycles only"),
        scope=manifest["scope"], tolerance=dict(rtol=8e-4, atol=8e-5))


def audit_report(folder):
    """A numerical failure is a separate result; incomplete capture still raises."""
    admission = admit_capture(folder, allow_incomplete_worker=True)
    try:
        result = audit(Path(folder)/"case", Path(folder)/"runtime")
    except (AssertionError, ValueError) as error:
        return dict(admission, status="FAIL", numerical_status="FAIL",
                    numerical_error=str(error)[:65536],
                    claim="SDK calls and capture complete; local numerical checks failed")
    return dict(admission, **result)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit a saved CSL training capture locally; no SDK calls.")
    parser.add_argument("capture", type=Path, help="capture-data directory or downloaded capture tar.gz")
    parser.add_argument("--out", type=Path, required=True, help="fresh numerical report path")
    args = parser.parse_args(argv)
    require(not args.out.exists() and not args.out.is_symlink(), "report output must be fresh")
    try:
        if args.capture.is_dir():
            report = audit_report(args.capture)
        else:
            with tempfile.TemporaryDirectory(prefix="csl-training-audit-") as temporary:
                folder = unpack_archive(args.capture, Path(temporary)/"capture", "capture", CAPTURE_LIMIT)
                report = audit_report(folder)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile, tarfile.TarError) as error:
        report = dict(status="FAIL", execution_status="FAIL", numerical_status="NOT_RUN",
                      error=str(error)[:65536], claim="capture completeness or identity check failed")
    write_json(args.out, report)
    print(json.dumps(dict(status=report["status"], execution_status=report["execution_status"],
                          numerical_status=report["numerical_status"], report=str(args.out))))
    return 0 if report["numerical_status"] == "PASS" else 1 if report["execution_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
