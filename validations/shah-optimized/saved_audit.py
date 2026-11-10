"""Offline audit from original edge occurrences, frozen input and artifact identity."""
import importlib.util
import io
import json
import math
from pathlib import Path
import tarfile
import zipfile
import uuid

import numpy as np

from appliance import digest, load_record, read_json, require

MAX_CAPTURE = 64 << 20


def read_capture(path):
    digest(path, MAX_CAPTURE)
    files = {}
    total = 0
    with tarfile.open(path, "r:*") as archive:
        for index, member in enumerate(archive):
            require(index < 16, "too many capture members")
            name = Path(member.name)
            require(not name.is_absolute() and ".." not in name.parts, "unsafe capture path")
            if member.isdir():
                require(member.name.rstrip("/") in ("capture", "."), "unexpected capture directory")
                continue
            require(member.isfile() and len(name.parts) <= 2 and
                    (len(name.parts) == 1 or name.parts[0] == "capture"), "invalid capture member")
            require(name.name in ("status.json", "result.json", "capture.npz", "io_layout.json", "compiled_inventory.json") and name.name not in files,
                    "unknown or duplicate capture file")
            total += member.size
            require(0 <= member.size <= MAX_CAPTURE and total <= MAX_CAPTURE, "capture exceeds size budget")
            if name.suffix == ".json":
                require(member.size <= 2 << 20, "oversize JSON capture")
            files[name.name] = archive.extractfile(member).read()
    require(set(files) == {"status.json", "result.json", "capture.npz", "io_layout.json", "compiled_inventory.json"}, "incomplete capture")
    # Bound decompressed NPZ members before NumPy allocation; object arrays are forbidden below.
    with zipfile.ZipFile(io.BytesIO(files["capture.npz"])) as archive:
        members = archive.infolist()
        require(len(members) == 5 and len({m.filename for m in members}) == 5 and
                sum(m.file_size for m in members) <= MAX_CAPTURE, "invalid NPZ inventory or size")
    with np.load(io.BytesIO(files["capture.npz"]), allow_pickle=False) as arrays:
        require(set(arrays.files) == {"actual", "reference", "stats", "h", "edges"}, "invalid captured arrays")
        raw = {key: arrays[key].copy() for key in arrays.files}
    return (json.loads(files["status.json"]), json.loads(files["result.json"]), raw,
            json.loads(files["io_layout.json"]), json.loads(files["compiled_inventory.json"]))


def frozen_oracle(root):
    spec = importlib.util.spec_from_file_location("shah_frozen_oracle", root / "oracle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_io_layout(record, layout):
    R, B = record["params"]["R"], record["params"]["B"]
    result = {}
    require(set(layout) == {"west", "east"}, "missing compiled I/O side")
    for side, begin, end in (("west", 0, R), ("east", R, R + B)):
        item = layout[side]
        heights = item["partitions"]
        require(item["height"] == R + B and len(heights) == record["channels"] and
                all(isinstance(h, int) and h > 0 for h in heights) and sum(heights) == R + B,
                "compiled partition geometry differs from request")
        require(isinstance(item["source_sha256"], str) and len(item["source_sha256"]) == 64,
                "missing compiled partition identity")
        offset = 0
        active = []
        for channel, height in enumerate(heights):
            if max(begin, offset) < min(end, offset + height):
                active.append(channel)
            offset += height
        result[side] = dict(partitions=heights, active_channels=active,
                            source_sha256=item["source_sha256"])
    return result


def verify_capture_binding(request, collection, status, inventory, io_layout):
    run_id = request["run_id"]
    require(str(uuid.UUID(run_id)) == run_id and collection.get("run_id") == run_id and
            status.get("run_id") == run_id, "capture belongs to another run")
    # request's compiled_inventory is independently reconstructed by load_record.
    require(inventory == request["compile_record"]["compiled_inventory"],
            "captured program files differ from pinned compiler archive")
    require(io_layout == inventory["io_layout"], "captured I/O map differs from compiled metadata")


def verify_numerics(record, mode, report, raw, oracle):
    params = record["params"]
    R, F, V, Q, B = [params[k] for k in ("R", "F", "V", "Q", "B")]
    if record["workload"] in ("balanced", "paired-skew"):
        case = oracle.controlled_fixture(**params, seed=record["seed"], degree=record["degree"],
                                         skew=record["workload"] == "paired-skew")
    elif record["workload"] == "random":
        case = oracle.random_fixture(**params, seed=record["seed"], degree=record["degree"])
    else:
        case = oracle.fixture(**params, empty=record["empty"])
    # Input reconstruction is separate from arithmetic reconstruction using captured occurrences.
    np.testing.assert_array_equal(raw["h"], case["h"])
    np.testing.assert_array_equal(raw["edges"], np.array(case["edges"]))
    require(raw["h"].dtype == np.float32 and raw["actual"].dtype == np.float32 and
            raw["actual"].shape == (B * Q, F), "wrong numerical dtype/shape")
    edges = raw["edges"].reshape(-1, 3)
    expected = np.zeros((B * Q, F), np.float64)
    for row, col, value in edges:
        require(row == int(row) and col == int(col) and 0 <= row < B * Q and
                0 <= col < raw["h"].shape[0] and math.isfinite(float(value)), "invalid raw edge")
        expected[int(row)] += float(value) * raw["h"][int(col)].astype(np.float64)
    require(np.all(np.isfinite(raw["actual"])), "nonfinite output")
    np.testing.assert_array_equal(raw["reference"], expected)
    np.testing.assert_allclose(raw["actual"], expected, rtol=2e-5, atol=2e-6)
    stats = raw["stats"]
    require(stats.dtype == np.uint32 and stats.shape == (R + B, F + 1, 4), "invalid counter ABI")
    for r in range(R):
        nz = sum(1 for _, c, _ in edges if r * V <= c < (r + 1) * V)
        pairs = sum(len(stream[r]) // 2 for stream in case["streams"])
        padding = sum(np.count_nonzero(stream[r, ::2] == oracle.EMPTY) for stream in case["streams"])
        np.testing.assert_array_equal(stats[r, 0], [pairs, nz, B * Q, padding])
        np.testing.assert_array_equal(stats[r, 1:, :3], np.tile([nz, B * Q, B], (F, 1)))
        require(np.all((stats[r, 1:, 3] >= B - 1) & (stats[r, 1:, 3] <= B)), "worker reuse count")
    for b in range(B):
        np.testing.assert_array_equal(stats[R + b, 1:],
                                      np.tile([(B - b) * Q, (B - b - 1) * Q, Q, 1], (F, 1)))
    require(mode in ("buffered", "per-chunk"), "invalid output mode")
    expected_calls = [("memcpy_h2d", False, False)]
    for _ in range(B):
        expected_calls.append(("memcpy_h2d", True, True))
        if mode == "per-chunk":
            expected_calls.append(("memcpy_d2h", True, False))
    if mode == "buffered":
        expected_calls.append(("memcpy_d2h", True, False))
    expected_calls.append(("memcpy_d2h", False, False))
    calls = report["calls"]
    require([(c["api"], c["streaming"], c["nonblock"]) for c in calls] == expected_calls,
            "host transfer sequence differs from requested output schedule")
    require(all(math.isfinite(c["seconds"]) and c["seconds"] >= 0 for c in calls), "invalid API duration")
    duration = report["host_stream_completion_seconds"]
    require(math.isfinite(duration) and duration >= 0, "invalid completion duration")
    error = float(np.max(np.abs(raw["actual"] - expected)))
    require(report["max_abs_error"] == error, "reported error does not match capture")
    metrics = {**case["metrics"], "d2h_output_calls": 1 if mode == "buffered" else B}
    require(report["metrics"] == metrics, "reported workload metrics mismatch")
    return dict(max_abs_error=error, host_stream_completion_seconds=duration,
                input_calls=B, output_calls=1 if mode == "buffered" else B,
                metrics=metrics)


def audit(capture, artifact_record, root):
    capture, root = Path(capture), Path(root)
    request = read_json(capture / "request.json")
    compiled = request["compile_record"]
    record, _ = load_record(artifact_record, compiled["backend"], root)
    require(record == compiled, "run refers to a different compile record")
    mode = request["output_mode"]
    collection = read_json(capture / "collection.json")
    require(collection["status"] == "complete" and not collection["errors"], "incomplete collection")
    downloaded = {}
    for name in ("capture", "host.log", "host-exit.txt"):
        item = collection["downloaded"][name]
        require(isinstance(item["path"], str) and Path(item["path"]).name == item["path"], "invalid collection path")
        path = capture / item["path"]
        require(item["sha256"] == digest(path, MAX_CAPTURE), "collection hash mismatch")
        downloaded[name] = path
    require(len(set(downloaded.values())) == 3, "collection aliases")
    require(downloaded["host-exit.txt"].read_text().strip() == "0", "nonzero worker exit")
    status, report, raw, io_layout, inventory = read_capture(downloaded["capture"])
    verify_capture_binding(request, collection, status, inventory, io_layout)
    require(status["status"] == "PASS" and status["compile_record"] == record and
            status["output_mode"] == mode, "worker admission identity mismatch")
    expected_source = {**record["source_sha256"],
        **{name: record["host_sha256"][name] for name in ("driver.py", "oracle.py")}}
    require(report["status"] == "PASS" and report["source_sha256"] == expected_source and
            report["params"] == record["params"] and report["channels"] == record["channels"] and
            report["output_mode"] == mode and report["workload"] == record["workload"], "report identity mismatch")
    result = verify_numerics(record, mode, report, raw, frozen_oracle(root))
    io_layout = verify_io_layout(record, io_layout)
    return dict(status="PASS", scope="saved numerical/counter/call audit; remote terminal status is separate",
                backend=record["backend"], params=record["params"], channels=record["channels"],
                output_mode=mode, workload=record["workload"], artifact_sha256=record["artifact_sha256"],
                capture_sha256=digest(downloaded["capture"], MAX_CAPTURE), run_id=request["run_id"],
                compiled_io_layout=io_layout, compiled_program_files=len(inventory["files"]), **result)
