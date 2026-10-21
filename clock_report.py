"""Offline admission and reports for fixed command-clock campaigns.

No SDK imports or job submission. A fresh runtime is the independent unit;
its A/B/A generations and its PEs remain nested observations.
"""
import csv
import io
from itertools import combinations
import json
from pathlib import Path, PurePosixPath
import tarfile
import uuid

import clock_appliance as adapter
import clock_common as common

STATUSES = ("pass", "numeric_fail", "incomplete", "unknown", "skipped_dependency", "started", "pending")
CONTEXT = ("id", "cohort", "block", "position", "arm", "build")
ROOT = Path(__file__).resolve().parent
LOCAL_RUNTIME_AUXILIARIES = frozenset((
    "generated/coord.elf", "generated/default.elf", "generated/MEMCPY_XY_ROUTES.elf",
))


def _context(entry):
    return {key: entry[key] for key in CONTEXT}


def _directory(path):
    path = Path(path).absolute()
    common.require(path == path.resolve() and path.is_dir(), "regular directory without symlink traversal: " + str(path))
    return path


def _source_root(run_dir, source_root=None):
    frozen = run_dir.parent.parent / "driver"
    root = _directory(source_root if source_root is not None else frozen if frozen.exists() else ROOT)
    _directory(root / "clock_source")
    return root


def _canonical_uuid(value, label):
    common.require(isinstance(value, str), label + " must be a UUID string")
    common.require(str(uuid.UUID(value)) == value, "canonical " + label)


def _safe_child(root, name):
    common.require(isinstance(name, str) and name not in ("", ".", "..") and
                   Path(name).name == name, "artifact filename")
    path = Path(root) / name
    common.require(not path.is_symlink() and path.resolve().parent == Path(root).resolve(),
                   "artifact path escapes output")
    return path


def _local_run_inventory(path, original):
    """The simulator adds three auxiliary ELFs to its private build copy."""
    from clock_local import inventory
    actual = inventory(_directory(path))
    common.require(all(actual.get(name) == digest for name, digest in original.items()),
                   "local run original compiled artifact hashes")
    extras = set(actual) - set(original)
    common.require(extras <= LOCAL_RUNTIME_AUXILIARIES, "unexpected local runtime artifact")
    return {name: actual[name] for name in sorted(extras)}


def _optional_fabric(capture):
    """Read one optional regular file; never extract archive paths."""
    if capture.is_dir():
        path = capture / "fabric.json"
        return adapter.bounded_bytes(path, 1 << 20) if path.exists() else None
    # capture_payloads has already checked the complete archive inventory.
    with tarfile.open(capture, "r:gz") as archive:
        for member in archive:
            if str(PurePosixPath(member.name)) == "capture/fabric.json":
                common.require(member.isfile() and member.size <= 1 << 20, "bounded fabric metadata")
                with archive.extractfile(member) as stream:
                    data = stream.read((1 << 20) + 1)
                common.require(len(data) == member.size, "fabric metadata size")
                return data
    return None


def _host_calls(calls):
    result = []
    for index, call in enumerate(calls):
        record = dict(index=index, name=call["name"], status=call["status"],
                      wall_ns=call["wall_end_ns"] - call["wall_start_ns"],
                      process_cpu_ns=call["cpu_end_ns"] - call["cpu_start_ns"],
                      pmu=call.get("pmu", {"status": "not_requested"}))
        for field in ("thread_before", "thread_after"):
            if field in call:
                record[field] = call[field]
        result.append(record)
    return result


def analyze_run(run_dir, build_dir, entry, plan, *, source_root=None):
    """Validate original artifacts, retain compact raw members, save analysis."""
    run_dir, build_dir = _directory(run_dir), _directory(build_dir)
    sources = _source_root(run_dir, source_root)
    build_bytes = adapter.bounded_bytes(build_dir / "clock-build.json", adapter.JSON_LIMIT)
    common.require(adapter.bounded_bytes(run_dir / "clock-build.json", adapter.JSON_LIMIT) == build_bytes,
                   "run and build manifest bytes differ")
    manifest = json.loads(build_bytes)
    common.require(manifest.get("schema") == 1 and manifest.get("action") == "compile", "compiled manifest schema")
    _canonical_uuid(manifest["compile_id"], "compile UUID")
    expected_build = next(b for b in plan["builds"] if b["id"] == entry["build"])
    common.require(manifest["params"] == expected_build["params"], "planned build parameters differ")
    backend = "local-sim" if plan["backend"] == "local-sim" else adapter.worker_backend(plan["backend"])
    common.require(manifest["backend"] == backend, "planned backend differs")
    request = adapter.read_json(run_dir / "clock-request.json")
    common.require(set(request) == {"config", "run_id", "manifest_sha256"}, "run request fields")
    _canonical_uuid(request["run_id"], "runtime UUID")
    common.require(request["manifest_sha256"] == common.digest_bytes(build_bytes), "request manifest hash")
    common.require(request["config"] == common.validate_config(entry["config"]), "planned run configuration differs")

    if backend == "local-sim":
        from clock_local import inventory
        common.require({k: manifest[k] for k in ("source_sha256", "host_sha256")} ==
                       common.source_identity(sources), "local compiled source identity")
        original = adapter.read_json(build_dir / "compile-request.json")
        common.require(original == {k: v for k, v in manifest.items() if k != "artifacts_sha256"},
                       "local original compile request")
        common.require(adapter.read_json(build_dir / "compile-status.json") ==
                       dict(status="complete", manifest_sha256=request["manifest_sha256"]),
                       "local compile completion")
        common.require(inventory(_directory(build_dir / "build")) == manifest["artifacts_sha256"],
                       "local original compiled artifact hashes")
        runtime_auxiliaries = _local_run_inventory(run_dir / "build", manifest["artifacts_sha256"])
        launch = adapter.read_json(run_dir / "launch.json")
        common.require(all(launch.get(k) == v for k, v in request.items()) and
                       launch.get("run_launcher_sha256") == manifest["launcher_sha256"], "local launch request differs")
        completion = adapter.read_json(run_dir / "local-status.json")
        common.require(completion.get("status") in ("pass", "observed_numeric_fail"), "local run incomplete")
        capture = run_dir / "capture"
        worker_status, exit_code = completion["status"], completion.get("exit_code")
        jobs = []
    else:
        admitted, _ = adapter.load_manifest(build_dir / "clock-build.json", plan["backend"], sources)
        common.require(admitted == manifest, "manifest changed during admission")
        original = adapter.read_json(run_dir / "request.json")
        common.require(original == dict(action="run", backend=backend,
                       command=adapter.host_command(plan["backend"], plan["host_timeout"]),
                       artifact_sha256=manifest["artifact_sha256"], **request), "original launch request differs")
        completion = adapter.read_json(run_dir / "collection.json")
        common.require(completion.get("status") == "complete" and not completion.get("errors") and
                       completion.get("accounting", {}).get("status") == "complete", "collection incomplete")
        common.require(completion.get("run_id") == request["run_id"] and
                       completion.get("manifest_sha256") == request["manifest_sha256"], "collection identity differs")
        downloads = completion.get("downloaded", {})
        common.require(set(downloads) == {"capture", "host.log", "host-exit.txt"}, "complete downloaded inventory")
        paths = {}
        for name, record in downloads.items():
            path = _safe_child(run_dir, record["path"])
            limit = {"capture": 64 << 20, "host.log": adapter.LOG_LIMIT, "host-exit.txt": 128}[name]
            common.require(path.stat().st_size == record["bytes"] and
                           adapter.digest(path, limit) == record["sha256"], "download hash/size: " + name)
            paths[name] = path
        common.require(len(set(paths.values())) == 3, "distinct downloaded files")
        capture = paths["capture"]
        text = adapter.bounded_bytes(paths["host-exit.txt"], 128).decode().strip()
        common.require(text in ("0", "1") and int(text) == completion.get("host_exit"), "completed worker exit")
        worker_status, exit_code = completion.get("worker_status"), int(text)
        jobs = completion.get("jobs", [])

    meta, validation = adapter.validate_capture(capture, manifest, request)
    common.require(meta["status"] == worker_status and exit_code == {"pass": 0, "observed_numeric_fail": 1}.get(worker_status),
                   "worker completion status differs")
    if backend != "local-sim":
        common.require(completion.get("capture_validation") == validation, "stored capture validation differs")
    payloads = adapter.capture_payloads(capture)
    common.require(json.loads(payloads["capture.json"]) == meta and
                   common.digest_bytes(payloads["calls.json"]) == meta["calls_sha256"] and
                   common.digest_bytes(payloads["raw.npz"]) == meta["raw_sha256"],
                   "capture changed during admission")
    fabric_bytes = _optional_fabric(capture)
    fabric = None
    if fabric_bytes is not None:
        value = json.loads(fabric_bytes)
        common.require(isinstance(value, dict), "fabric metadata object")
        fabric = {key: value.get(key) for key in ("core_frequency", "fabric_dimensions")}
        payloads["fabric.json"] = fabric_bytes
    observed = run_dir / "observed"
    common.require(not observed.is_symlink(), "observed directory must not be a symlink")
    if observed.exists():
        common.require(observed.is_dir() and {p.name for p in observed.iterdir()} == set(payloads),
                       "existing observed inventory differs")
        for name, data in payloads.items():
            common.require(adapter.bounded_bytes(observed / name, len(data)) == data,
                           "existing observed bytes differ: " + name)
    else:
        observed.mkdir()
        for name, data in payloads.items():
            (observed / name).write_bytes(data)
    calls = json.loads(payloads["calls.json"])
    analysis = dict(schema=1, **_context(entry), status="pass" if worker_status == "pass" else "numeric_fail",
                    run_id=request["run_id"], manifest_sha256=request["manifest_sha256"],
                    backend=backend, params=manifest["params"], config=request["config"],
                    validation=validation, steps=validation.get("steps", []),
                    host=meta.get("host", {}), runtime_identity=meta.get("host", {}).get("runtime_modules", []),
                    calls=_host_calls(calls), jobs=jobs, fabric=fabric,
                    observed_sha256={name: common.digest_bytes(data) for name, data in payloads.items()},
                    clock_scope="Same-PE intervals; no cross-PE timestamp subtraction.",
                    pmu_scope="Submitting thread user mode, including instrumentation; no empty-control subtraction.")
    if backend == "local-sim":
        analysis["runtime_auxiliary_sha256"] = runtime_auxiliaries
    path = run_dir / "analysis.json"
    if path.exists() or path.is_symlink():
        common.require(common.read_json(path) == analysis, "existing analysis differs from original evidence")
    else:
        common.write_json(path, analysis)
    return analysis


def _csv(path, fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue())


def _read_analysis(directory, build_dir, entry, status, plan):
    """Recheck compact raw evidence when regenerating a report."""
    directory, build_dir = _directory(directory), _directory(build_dir)
    analysis = common.read_json(directory / "analysis.json")
    common.require(all(analysis.get(k) == v for k, v in _context(entry).items()) and
                   analysis["status"] == status and analysis["config"] == entry["config"],
                   "analysis differs from scheduled run")
    manifest = adapter.read_json(directory / "clock-build.json")
    common.require(adapter.bounded_bytes(directory / "clock-build.json", adapter.JSON_LIMIT) ==
                   adapter.bounded_bytes(build_dir / "clock-build.json", adapter.JSON_LIMIT),
                   "analysis build manifest differs")
    expected_params = next(b["params"] for b in plan["builds"] if b["id"] == entry["build"])
    expected_backend = "local-sim" if plan["backend"] == "local-sim" else adapter.worker_backend(plan["backend"])
    common.require(manifest["params"] == expected_params and manifest["backend"] == expected_backend,
                   "analysis planned build differs")
    request = adapter.read_json(directory / "clock-request.json")
    common.require(set(request) == {"config", "run_id", "manifest_sha256"}, "analysis request fields")
    _canonical_uuid(request["run_id"], "runtime UUID")
    common.require(common.digest(directory / "clock-build.json") == request["manifest_sha256"] and
                   analysis["manifest_sha256"] == request["manifest_sha256"] and
                   request["config"] == entry["config"] and analysis["run_id"] == request["run_id"],
                   "analysis request binding")
    observed = _directory(directory / "observed")
    pins = analysis["observed_sha256"]
    common.require(set(pins) in (set(adapter.CAPTURE_LIMITS), set(adapter.CAPTURE_LIMITS) | {"fabric.json"}),
                   "retained capture inventory")
    common.require({p.name for p in observed.iterdir()} == set(pins), "retained directory inventory")
    for name, digest in pins.items():
        common.require(common.digest(observed / name) == digest, "retained capture hash: " + name)
    meta, validation = adapter.validate_capture(observed, manifest, request)
    common.require(analysis["status"] == {"pass": "pass", "observed_numeric_fail": "numeric_fail"}.get(meta["status"]),
                   "analysis numerical status differs")
    common.require(analysis["validation"] == validation and analysis["steps"] == validation.get("steps", []) and
                   analysis["backend"] == manifest["backend"] and analysis["params"] == manifest["params"],
                   "analysis differs from raw evidence")
    calls = json.loads(adapter.bounded_bytes(observed / "calls.json", adapter.JSON_LIMIT))
    common.require(analysis["calls"] == _host_calls(calls) and analysis["host"] == meta.get("host", {}),
                   "analysis host observations differ")
    common.require(analysis["runtime_identity"] == meta.get("host", {}).get("runtime_modules", []),
                   "analysis runtime identity differs")
    fabric = None
    if "fabric.json" in pins:
        value = adapter.read_json(observed / "fabric.json")
        fabric = {key: value.get(key) for key in ("core_frequency", "fabric_dimensions")}
    common.require(analysis["fabric"] == fabric, "analysis fabric observations differ")
    return analysis


def _compact_build(build_dir, entry, plan, sources, validated_builds):
    """Validate the original compiled artifact once per replay, never a run copy."""
    data = adapter.bounded_bytes(build_dir / "clock-build.json", adapter.JSON_LIMIT)
    key = str(build_dir)
    digest = common.digest_bytes(data)
    cache = validated_builds if validated_builds is not None else {}
    if key in cache:
        common.require(cache[key]["manifest_sha256"] == digest and cache[key]["source_root"] == str(sources),
                       "compiled build changed during replay")
        manifest = cache[key]["manifest"]
    else:
        manifest = json.loads(data)
        common.require(manifest.get("schema") == 1 and manifest.get("action") == "compile", "compiled manifest schema")
        _canonical_uuid(manifest["compile_id"], "compile UUID")
        if plan["backend"] == "local-sim":
            from clock_local import inventory
            common.require(manifest["backend"] == "local-sim" and
                           {k: manifest[k] for k in ("source_sha256", "host_sha256")} == common.source_identity(sources),
                           "local replay source identity")
            common.require(manifest["launcher_sha256"] == common.digest(sources / "clock_local.py"), "local launcher identity")
            common.require(adapter.read_json(build_dir / "compile-request.json") ==
                           {k: v for k, v in manifest.items() if k != "artifacts_sha256"}, "local original compile request")
            common.require(adapter.read_json(build_dir / "compile-status.json") ==
                           dict(status="complete", manifest_sha256=digest), "local compile completion")
            common.require(inventory(_directory(build_dir / "build")) == manifest["artifacts_sha256"],
                           "local original compiled artifact hashes")
        else:
            admitted, _ = adapter.load_manifest(build_dir / "clock-build.json", plan["backend"], sources)
            common.require(admitted == manifest, "replay manifest changed")
        cache[key] = dict(manifest_sha256=digest, source_root=str(sources), manifest=manifest)
    expected = next(b["params"] for b in plan["builds"] if b["id"] == entry["build"])
    common.require(manifest["params"] == expected and common.validate_params(expected) == expected,
                   "replay planned build parameters")
    return manifest


def read_compact_run(run_dir, build_dir, entry, plan, *, omitted_paths, validated_builds=None, source_root=None):
    """Read-only replay after the caller verifies the compact bundle inventory.

    Missing originals require exact campaign-relative declarations in
    omitted_paths. The absent archive's historical hash is only a recorded
    identity; it cannot be recomputed. Retained raw/API records and build
    artifacts are independently checked. Use a fresh validated_builds dict for
    each replay to avoid rehashing the same original build for every run.
    """
    run_dir, build_dir = _directory(run_dir), _directory(build_dir)
    sources = _source_root(run_dir, source_root)
    root = run_dir.parent.parent
    common.require(run_dir == _safe_child(root / "runs", entry["id"]) and
                   build_dir == _safe_child(root / "builds", entry["build"]), "planned replay paths")
    omissions = set(omitted_paths)
    for name in omissions:
        common.require(isinstance(name, str) and name and
                       PurePosixPath(name).parts and str(PurePosixPath(name)) == name.rstrip("/") and
                       not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts,
                       "canonical omitted path")
    manifest = _compact_build(build_dir, entry, plan, sources, validated_builds)
    request = adapter.read_json(run_dir / "clock-request.json")
    expected_status = common.read_json(run_dir / "analysis.json")["status"]
    common.require(expected_status in ("pass", "numeric_fail"), "completed compact analysis")
    analysis = _read_analysis(run_dir, build_dir, entry, expected_status, plan)
    original_status = "pass" if expected_status == "pass" else "observed_numeric_fail"
    expected_exit = 0 if expected_status == "pass" else 1
    if plan["backend"] == "local-sim":
        launch = adapter.read_json(run_dir / "launch.json")
        common.require(all(launch.get(k) == v for k, v in request.items()) and
                       launch.get("run_launcher_sha256") == manifest["launcher_sha256"], "compact local launch request")
        completion = adapter.read_json(run_dir / "local-status.json")
        common.require(completion.get("status") == original_status and completion.get("exit_code") == expected_exit,
                       "compact local completion")
        common.require(analysis["jobs"] == [], "compact local job identity")
        run_build = run_dir / "build"
        common.require(not run_build.is_symlink(), "compact local run build symlink")
        if run_build.exists():
            extras = _local_run_inventory(run_build, manifest["artifacts_sha256"])
            common.require(analysis.get("runtime_auxiliary_sha256", {}) == extras,
                           "compact local runtime auxiliary hashes")
        else:
            common.require(str(run_build.relative_to(root)) + "/" in omissions, "undeclared missing local build copy")
    else:
        original = adapter.read_json(run_dir / "request.json")
        common.require(original == dict(action="run", backend=manifest["backend"],
                       command=adapter.host_command(plan["backend"], plan["host_timeout"]),
                       artifact_sha256=manifest["artifact_sha256"], **request), "compact original launch request")
        completion = adapter.read_json(run_dir / "collection.json")
        common.require(completion.get("status") == "complete" and not completion.get("errors") and
                       completion.get("accounting", {}).get("status") == "complete" and
                       completion.get("run_id") == request["run_id"] and
                       completion.get("manifest_sha256") == request["manifest_sha256"] and
                       completion.get("worker_status") == original_status and completion.get("host_exit") == expected_exit and
                       completion.get("capture_validation") == analysis["validation"] and
                       completion.get("jobs", []) == analysis["jobs"], "compact collection binding")
        downloads = completion.get("downloaded", {})
        common.require(set(downloads) == {"capture", "host.log", "host-exit.txt"}, "compact download inventory")
        names = set()
        for name, record in downloads.items():
            path = _safe_child(run_dir, record["path"])
            common.require(path.name not in names, "distinct compact downloads")
            names.add(path.name)
            limit = {"capture": 64 << 20, "host.log": adapter.LOG_LIMIT, "host-exit.txt": 128}[name]
            common.require(type(record["bytes"]) is int and 0 <= record["bytes"] <= limit and
                           isinstance(record["sha256"], str) and len(record["sha256"]) == 64 and
                           all(c in "0123456789abcdef" for c in record["sha256"]), "recorded download identity")
            if path.exists():
                common.require(path.stat().st_size == record["bytes"] and adapter.digest(path, limit) == record["sha256"],
                               "compact remaining download hash: " + name)
            else:
                common.require(name == "capture" and str(path.relative_to(root)) in omissions,
                               "undeclared missing original download: " + name)
            if name == "host-exit.txt":
                common.require(adapter.bounded_bytes(path, 128).decode().strip() == str(expected_exit), "compact worker exit")
    return analysis


def _contrasts(plan, analyses):
    """Pair within scheduled blocks, separately for each nested generation."""
    groups = {}
    for entry in plan["runs"]:
        groups.setdefault((entry["cohort"], entry["block"]), []).append(entry)
    records = []
    for (cohort, block), entries in groups.items():
        # Smoke contains unrelated plumbing checks; one-arm diagnostics have no pair.
        if cohort == "smoke" or len(entries) < 2:
            continue
        by_arm = sorted(entries, key=lambda e: e["arm"])
        for left, right in combinations(by_arm, 2):
            common.require(left["arm"] != right["arm"], "duplicate comparison arm in a block")
            record = dict(cohort=cohort, block=block,
                          left_id=left["id"], right_id=right["id"],
                          left_arm=left["arm"], right_arm=right["arm"],
                          runtime_order=[e["id"] for e in sorted(entries, key=lambda e: e["position"])],
                          status="unavailable", generations=[])
            a, b = analyses.get(left["id"]), analyses.get(right["id"])
            if a and b and a["status"] == b["status"] == "pass":
                record["status"] = "paired"
                for sa, sb in zip(a["steps"], b["steps"]):
                    common.require(sa["step"] == sb["step"], "paired generation index")
                    x, y = sa["max_span_cycles"], sb["max_span_cycles"]
                    record["generations"].append(dict(step=sa["step"], left_max_span_cycles=x,
                        right_max_span_cycles=y, right_minus_left_cycles=y-x,
                        right_over_left=y/x))
            records.append(record)
    return dict(independent_unit="Scheduled block of distinct fresh runtimes; generations are nested, not independent repeats.",
                metric="Maximum of same-PE span cycles within one generation; not synchronized global makespan.",
                pair_records=records,
                paired_records=sum(r["status"] == "paired" for r in records),
                unavailable_records=sum(r["status"] != "paired" for r in records),
                note="The three comparisons within a granularity block share runtimes. No winner or frequency fit is computed.")


def write_report(root, plan, state):
    """Write partial or complete reports from admitted per-run analyses."""
    root = _directory(root)
    for name in ("pe.csv", "markers.csv", "host.csv", "summary.json", "summary.md", "contrasts.json"):
        common.require(not (root / name).is_symlink(), "report output symlink: " + name)
    runs, analyses, seen_ids = [], {}, set()
    pe_rows, mark_rows, host_rows = [], [], []
    counts = dict.fromkeys(STATUSES, 0)
    for entry in plan["runs"]:
        status = state.get("runs", {}).get(entry["id"], {}).get("status", "pending")
        common.require(status in STATUSES, "unknown campaign run status")
        counts[status] += 1
        record = dict(_context(entry), status=status)
        current = state.get("runs", {}).get(entry["id"], {})
        for key in ("error", "remote_stop_unconfirmed", "jobs"):
            if key in current:
                record[key] = current[key]
        if status in ("pass", "numeric_fail"):
            path = root / "runs" / entry["id"] / "analysis.json"
            try:
                analysis = _read_analysis(path.parent, root / "builds" / entry["build"], entry, status, plan)
                runtime_id = analysis["run_id"]
                common.require(str(uuid.UUID(runtime_id)) == runtime_id and runtime_id not in seen_ids,
                               "duplicate or invalid runtime UUID")
                common.require(status != "pass" or [s["step"] for s in analysis["steps"]] == [0, 1, 2],
                               "complete nested generations")
                seen_ids.add(runtime_id)
                analyses[entry["id"]] = analysis
                record.update(run_id=runtime_id, analysis="runs/" + entry["id"] + "/analysis.json",
                              max_span_cycles=[s["max_span_cycles"] for s in analysis["steps"]])
            except (OSError, ValueError, KeyError, TypeError) as error:
                record["analysis_error"] = str(error)
                runs.append(record)
                continue
            base = dict(_context(entry), run_id=runtime_id, backend=analysis["backend"])
            for step in analysis["steps"]:
                for y, row in enumerate(step["pe"]):
                    for x, pe in enumerate(row):
                        pe_rows.append(dict(base, step=step["step"], x=x, y=y, **pe))
                for mark in step.get("marks", []):
                    for y, row in enumerate(mark["pe"]):
                        for x, pe in enumerate(row):
                            mark_rows.append(dict(base, step=step["step"], index=mark["index"], x=x, y=y,
                                piece_rectangle=json.dumps(mark["piece_rectangle"], separators=(",", ":")), **pe))
            for call in analysis["calls"]:
                pmu = call["pmu"]
                row = dict(base, index=call["index"], name=call["name"], status=call["status"],
                           wall_ns=call["wall_ns"], process_cpu_ns=call["process_cpu_ns"],
                           pmu_status=pmu["status"], pmu_scope=pmu.get("scope", ""),
                           pmu_error=json.dumps(pmu.get("error", {}), sort_keys=True),
                           pmu_scaled=pmu.get("scaled", ""))
                for field in ("cycles", "instructions", "time_enabled_ns", "time_running_ns"):
                    row["pmu_" + field] = pmu.get(field, "")
                before, after = call.get("thread_before", {}), call.get("thread_after", {})
                row.update(cpu_before=before.get("cpu", ""), cpu_after=after.get("cpu", ""))
                for field in ("thread_time_ns", "voluntary_context_switches", "involuntary_context_switches", "minor_faults", "major_faults"):
                    a, b = before.get(field), after.get(field)
                    row[field] = b-a if a is not None and b is not None else ""
                host_rows.append(row)
        runs.append(record)

    common_fields = list(CONTEXT) + ["run_id", "backend"]
    _csv(root / "pe.csv", common_fields + ["step", "x", "y", "start", "end", "work_start", "span_cycles", "work_cycles", "pre_work_cycles"], pe_rows)
    _csv(root / "markers.csv", common_fields + ["step", "index", "x", "y", "piece_rectangle", "timestamp", "from_start_cycles", "from_previous_cycles", "first_word", "last_word"], mark_rows)
    _csv(root / "host.csv", common_fields + ["index", "name", "status", "wall_ns", "process_cpu_ns",
         "pmu_status", "pmu_scope", "pmu_error", "pmu_scaled", "pmu_cycles", "pmu_instructions",
         "pmu_time_enabled_ns", "pmu_time_running_ns", "cpu_before", "cpu_after", "thread_time_ns",
         "voluntary_context_switches", "involuntary_context_switches", "minor_faults", "major_faults"], host_rows)
    contrasts = _contrasts(plan, analyses)
    common.write_json(root / "contrasts.json", contrasts)
    errors = sum("analysis_error" in r for r in runs)
    unrun = counts["pending"] + counts["started"] + counts["skipped_dependency"]
    final_status = state.get("status", "unknown")
    if errors or (final_status in ("pass", "complete", "completed") and (unrun or len(analyses) != len(plan["runs"]))):
        final_status = "incomplete"
    summary = dict(schema=1, backend=plan["backend"], profile=plan["profile"],
                   status=final_status, controller_status=state.get("status", "unknown"), counts=counts,
                   data_completeness="complete" if len(analyses) == len(plan["runs"]) and not errors else "partial",
                   planned_fresh_runtimes=len(plan["runs"]), admitted_fresh_runtimes=len(analyses),
                   passed_fresh_runtimes=sum(a["status"] == "pass" for a in analyses.values()),
                   nested_generations=sum(len(a["steps"]) for a in analyses.values()),
                   pe_intervals=len(pe_rows), marker_observations=len(mark_rows), host_regions=len(host_rows),
                   pmu_measured=sum(r["pmu_status"] == "ok" for r in host_rows),
                   pmu_unavailable=sum(r["pmu_status"] == "unavailable" for r in host_rows),
                   analysis_errors=errors, runs=runs,
                   independent_unit=plan["independent_unit"], evidence_scope=plan["evidence_scope"],
                   files=["pe.csv", "markers.csv", "host.csv", "contrasts.json"])
    common.write_json(root / "summary.json", summary)
    lines = ["# Command-clock campaign observations", "",
             f"Backend: {plan['backend']}. Campaign state: {summary['status']}.", "",
             f"{summary['passed_fresh_runtimes']} passing fresh runtimes out of {len(plan['runs'])} planned; "
             f"{summary['nested_generations']} nested generations retained.", "",
             "A/B/A generations are not independent repetitions. Every generation, including the first, is retained.", "",
             "| State | Planned entries |", "| --- | ---: |"]
    lines += [f"| {name} | {count} |" for name, count in counts.items()]
    lines += ["", "`pe.csv` preserves each PE interval; `markers.csv` records command markers; "
              "`host.csv` retains PMU availability and thread observations. `contrasts.json` pairs fresh "
              "runtimes within scheduled blocks and keeps the three generation comparisons separate.", "",
              "PE intervals use one PE's clock. Their maximum is not a synchronized global makespan. "
              "Marker RPCs and PMU instrumentation can change the measured path. CPU cycles and "
              "instructions include submitting-thread instrumentation; empty controls are not subtracted. "
              "Equal CPU endpoints do not rule out an intervening migration. No frequency or winner is inferred.", ""]
    if summary["analysis_errors"]:
        lines += [f"{summary['analysis_errors']} recorded outcomes lack an admissible analysis; see summary.json.", ""]
    (root / "summary.md").write_text("\n".join(lines))
    return summary
