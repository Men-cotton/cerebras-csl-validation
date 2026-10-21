#!/usr/bin/env python3
"""Retry unfinished entries in a fresh campaign, preserving the original evidence.

The default destination is OLD-retry and must not exist. This explicit retry
does not establish that an interrupted remote job has stopped. Original job IDs
and remote-stop uncertainty remain in plan.retry_of. Importing submits no jobs.
"""
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import stat
import sys
import tarfile
import tempfile
import time
import uuid

import clock_appliance as appliance
import clock_campaign as campaign
import clock_common as common
import clock_report as report

ROOT = Path(__file__).resolve().parent
RETRY_STATUSES = {"pending", "incomplete", "started", "unknown", "skipped_dependency"}
DONE_STATUSES = {"pass", "numeric_fail"}
COPY_LIMIT = 1 << 30
COPY_ENTRIES = 4096
COPY_SECONDS = 60


def regular_directory(path):
    path = Path(path).absolute()
    common.require(path == path.resolve() and path.is_dir(), "regular directory without symlink traversal")
    return path


def child(root, name):
    common.require(isinstance(name, str) and name not in ("", ".", "..") and
                   Path(name).name == name, "safe campaign entry or artifact name")
    path = root / name
    common.require(not path.is_symlink() and path == path.resolve(), "symlink in campaign path")
    return path


def copy_tree(source, destination, *, limit=COPY_LIMIT):
    """Copy regular files with bounded bytes, entries and elapsed time."""
    source = regular_directory(source)
    destination.mkdir(parents=True, exist_ok=False)
    pending = [(source, destination)]
    total = entries = 0
    started = time.monotonic()
    while pending:
        origin, target = pending.pop()
        with os.scandir(origin) as children:
            for item in children:
                entries += 1
                common.require(entries <= COPY_ENTRIES and time.monotonic()-started <= COPY_SECONDS,
                               "copy inventory/time bound exceeded")
                common.require(not item.is_symlink(), "copy rejects symlinks")
                dest = target / item.name
                if item.is_dir(follow_symlinks=False):
                    dest.mkdir()
                    pending.append((Path(item.path), dest))
                    continue
                common.require(item.is_file(follow_symlinks=False), "copy requires regular files")
                descriptor = os.open(item.path, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(descriptor, "rb") as src, dest.open("xb") as dst:
                    before = os.fstat(src.fileno())
                    common.require(stat.S_ISREG(before.st_mode) and total+before.st_size <= limit,
                                   "copy byte bound exceeded")
                    written = 0
                    while True:
                        common.require(time.monotonic()-started <= COPY_SECONDS, "copy time bound exceeded")
                        data = src.read(1 << 20)
                        if not data:
                            break
                        total += len(data)
                        written += len(data)
                        common.require(total <= limit, "copy grew beyond byte bound")
                        dst.write(data)
                    after = os.fstat(src.fileno())
                    common.require(written == before.st_size == after.st_size and
                                   before.st_mtime_ns == after.st_mtime_ns, "source changed during copy")
    return total


def check_old(old):
    plan = common.read_json(old / "plan.json")
    state = common.read_json(old / "state.json")
    common.require(isinstance(plan, dict) and isinstance(state, dict), "campaign plan/state objects")
    common.require(state["plan_sha256"] == common.digest(old / "plan.json"), "frozen plan changed")
    regular_directory(old / "driver")
    regular_directory(old / "driver/clock_source")
    campaign.check_sources(old, plan)
    trusted = {name: common.digest(ROOT / name) for name in campaign.SOURCE_FILES}
    common.require(trusted == plan["source_sha256"], "trusted repository sources differ from old frozen driver")
    common.require(plan["backend"] in ("hardware", "simulator", "local-sim"), "campaign backend")
    common.require(isinstance(plan["campaign_id"], str) and
                   str(uuid.UUID(plan["campaign_id"])) == plan["campaign_id"], "campaign UUID")
    for group in ("builds", "runs"):
        common.require(isinstance(plan[group], list) and len(plan[group]) <= 1024, "bounded campaign entries")
        ids = [entry["id"] for entry in plan[group]]
        common.require(len(set(ids)) == len(ids) and set(state[group]) == set(ids), "plan/state entry inventory")
        regular_directory(old / group)
        for entry in plan[group]:
            child(old / group, entry["id"])
            common.require(state[group][entry["id"]]["status"] in RETRY_STATUSES | DONE_STATUSES,
                           "unknown old status")
    build_ids = {build["id"] for build in plan["builds"]}
    for build in plan["builds"]:
        common.require(common.validate_params(build["params"]) == build["params"], "planned build parameters")
    for entry in plan["runs"]:
        common.require(entry["build"] in build_ids and
                       common.validate_config(entry["config"]) == entry["config"], "planned run binding")
    return plan, state


def raw_terminal(old, plan, entry):
    """Observe terminal raw evidence without writing or promoting collection."""
    common.require(plan["backend"] != "local-sim", "raw fallback requires an appliance capture")
    folder = regular_directory(old / "runs" / entry["id"])
    build = next(b for b in plan["builds"] if b["id"] == entry["build"])
    manifest = campaign.validate_build(old, plan, build)
    build_bytes = appliance.bounded_bytes(old / "builds" / entry["build"] / "clock-build.json", appliance.JSON_LIMIT)
    common.require(appliance.bounded_bytes(folder / "clock-build.json", appliance.JSON_LIMIT) == build_bytes,
                   "raw recovery manifest bytes differ")
    request = appliance.read_json(folder / "clock-request.json")
    common.require(set(request) == {"config", "run_id", "manifest_sha256"} and
                   request["config"] == entry["config"] and
                   request["manifest_sha256"] == hashlib.sha256(build_bytes).hexdigest(), "raw recovery request binding")
    common.require(isinstance(request["run_id"], str) and
                   str(uuid.UUID(request["run_id"])) == request["run_id"], "raw recovery runtime UUID")
    original = appliance.read_json(folder / "request.json")
    common.require(original == dict(action="run", backend=manifest["backend"],
                   command=appliance.host_command(plan["backend"], plan["host_timeout"]),
                   artifact_sha256=manifest["artifact_sha256"], **request), "raw recovery original launch binding")
    candidates = []
    collection = appliance.read_json(folder / "collection.json") if (folder / "collection.json").exists() else {}
    record = collection.get("downloaded", {}).get("capture")
    if record is not None:
        path = child(folder, record["path"])
        common.require(path.is_file() and path.stat().st_size == record["bytes"] and
                       appliance.digest(path, 64 << 20) == record["sha256"], "raw recovery download hash/size")
        candidates.append(path)
    else:
        # These are the adapter's requested name and the ordinary SDK suffix.
        # Other filenames require their original downloaded-inventory binding.
        for name in ("capture", "capture.tar.gz"):
            path = child(folder, name)
            if path.exists():
                candidates.append(path)
    common.require(candidates, "no saved capture available for raw recovery")
    outcomes = []
    for path in candidates:
        meta, validation = appliance.validate_capture(path, manifest, request)
        common.require(meta["status"] in ("pass", "observed_numeric_fail"), "capture is not terminal")
        outcomes.append((meta["status"], meta["raw_sha256"], meta["calls_sha256"]))
    common.require(len(set(outcomes)) == 1, "conflicting saved captures")
    return dict(recovery="terminal_raw_capture", status=outcomes[0][0], run_id=request["run_id"],
                collection_admitted=False,
                collection_state="unfinalized" if collection.get("status") != "complete" else "not_admitted",
                raw_sha256=outcomes[0][1], calls_sha256=outcomes[0][2])


def recover(old, plan, entry):
    folder = old / "runs" / entry["id"]
    if not folder.exists():
        return None, "no original run directory"
    errors = []
    try:
        with tempfile.TemporaryDirectory(prefix="clock-retry-recovery-") as temporary:
            copied = Path(temporary) / "run"
            copy_tree(folder, copied, limit=128 << 20)
            analysis = report.analyze_run(copied, old / "builds" / entry["build"], entry, plan,
                                          source_root=old / "driver")
            return dict(recovery="complete_original", status=analysis["status"],
                        run_id=analysis["run_id"], collection_admitted=True), None
    except (ValueError, OSError, KeyError, TypeError, tarfile.TarError, EOFError) as error:
        errors.append(type(error).__name__+": "+str(error))
    try:
        result = raw_terminal(old, plan, entry)
        result["collection_admission_error"] = errors[0]
        return result, None
    except (ValueError, OSError, KeyError, TypeError, tarfile.TarError, EOFError) as error:
        errors.append(type(error).__name__+": "+str(error))
    return None, "; ".join(errors)[:4096]


def old_record(old, entry, slot, *, group="runs"):
    record = dict(status=slot["status"], jobs=deepcopy(slot.get("jobs", [])),
                  remote_stop_unconfirmed=slot.get("remote_stop_unconfirmed", False))
    record["remote_liveness_unknown"] = bool(record["remote_stop_unconfirmed"] or
                                            slot["status"] in ("started", "unknown", "incomplete"))
    if slot.get("run_id") is not None:
        record["run_id"] = slot["run_id"]
    try:
        record["locally_recorded_jobs"] = appliance.local_jobs(old / group / entry["id"],
                                                               "compile" if group == "builds" else "launch")
    except (ValueError, OSError) as error:
        record["job_log_error"] = str(error)
    return record


def prepare_locked(old, out):
    plan, state = check_old(old)
    old_plan_hash, old_state_hash = common.digest(old / "plan.json"), common.digest(old / "state.json")
    common.require(not out.exists() and not out.is_symlink(),
                   "retry output already exists; resume its valid plan or choose another fresh --out")
    selected, decisions = [], {}
    for entry in plan["runs"]:
        slot = state["runs"][entry["id"]]
        decision = old_record(old, entry, slot)
        if slot["status"] in DONE_STATUSES:
            decision["decision"] = "exclude_recorded_terminal"
        else:
            recovered, error = recover(old, plan, entry)
            if recovered is not None:
                decision.update(decision="exclude_recovered_terminal", recovered=recovered)
            else:
                decision.update(decision="retry", recovery_error=error)
                selected.append(deepcopy(entry))
        decisions[entry["id"]] = decision
    if not selected:
        print("NO RETRY: all entries already have terminal results.")
        return None
    needed = {entry["build"] for entry in selected}
    builds = [deepcopy(build) for build in plan["builds"] if build["id"] in needed]
    reusable, build_errors = set(), {}
    build_records = {build["id"]: old_record(old, build, state["builds"][build["id"]], group="builds")
                     for build in builds}
    for build in builds:
        try:
            campaign.validate_build(old, plan, build)
            reusable.add(build["id"])
            build_records[build["id"]]["decision"] = "reuse_validated_build"
        except (ValueError, OSError, KeyError, TypeError) as error:
            build_errors[build["id"]] = str(error)
            build_records[build["id"]]["decision"] = "recompile"
    out.mkdir(parents=True, exist_ok=False)
    for name in campaign.SOURCE_FILES:
        destination = out / "driver" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(appliance.bounded_bytes(old / "driver" / name, 4 << 20))
    for name in ("builds", "runs", "conditions"):
        (out / name).mkdir()
    fresh = deepcopy(plan)
    fresh.update(created_at=campaign.now(), campaign_id=str(uuid.uuid4()), builds=builds, runs=selected,
                 expected_fresh_runtimes=len(selected),
                 selection="Explicit retry subset from live original state and retained terminal captures; original relative order retained.")
    fresh["retry_of"] = dict(path=str(old), campaign_id=plan["campaign_id"], plan_sha256=old_plan_hash,
                            state_sha256=old_state_hash, entries=decisions, builds=build_records,
                            build_validation_errors=build_errors,
                            analysis_scope="New attempts only. Never merge original and retry runtimes into paired blocks.",
                            remote_scope="Local controller locking does not establish remote job termination.")
    fresh["cohorts"] = []
    for cohort in plan["cohorts"]:
        members = [entry for entry in selected if entry["cohort"] == cohort["id"]]
        if members:
            value = deepcopy(cohort)
            value.update(original_fresh_runtimes=cohort["fresh_runtimes"],
                         original_blocks=cohort["blocks"], fresh_runtimes=len(members),
                         blocks=len({entry["block"] for entry in members}), schedule_scope="retry subset")
            fresh["cohorts"].append(value)
    estimates = fresh["estimates"]
    count, compiling = len(selected), len(builds)-len(reusable)
    estimates.update(hardware_wall_seconds_low=round(compiling*estimates["hardware_compile_seconds_per_build"]+
                                                     count*estimates["hardware_runtime_seconds_low"], 3),
                     hardware_wall_seconds_high=round(compiling*estimates["hardware_compile_seconds_per_build"]+
                                                      count*estimates["hardware_runtime_seconds_high"], 3),
                     retained_bytes_approx=len(builds)*51000+count*10990000,
                     builds_to_compile=compiling, reused_builds=len(reusable), selected_fresh_runtimes=count)
    campaign.check_sources(out, fresh)
    new_state = dict(schema=1, status="pending", builds={}, runs={entry["id"]: dict(status="pending") for entry in selected})
    for build in builds:
        identity = build["id"]
        if identity in reusable:
            copy_tree(old / "builds" / identity, out / "builds" / identity)
            campaign.validate_build(out, fresh, build)
            new_state["builds"][identity] = dict(status="pass", reused_from=str(old / "builds" / identity),
                                                manifest_sha256=common.digest(out / "builds" / identity / "clock-build.json"))
        else:
            new_state["builds"][identity] = dict(status="pending")
    for entry in selected:
        campaign.atomic_json(out / "conditions" / (entry["id"]+".json"), dict(config=entry["config"]))
    common.require(common.digest(old / "plan.json") == old_plan_hash and
                   common.digest(old / "state.json") == old_state_hash, "original plan/state changed during retry preparation")
    campaign.check_sources(old, plan)
    campaign.atomic_json(out / "plan.json", fresh)
    new_state["plan_sha256"] = common.digest(out / "plan.json")
    campaign.atomic_json(out / "state.json", new_state)
    print("RETRY", count, "fresh runtimes;", len(reusable), "reused builds;", compiling, "compiles")
    print("OUTPUT", out)
    uncertain = [entry["id"] for entry in selected if decisions[entry["id"]]["remote_liveness_unknown"]]
    uncertain.extend("build:"+build["id"] for build in builds if build["id"] not in reusable and
                     build_records[build["id"]]["remote_liveness_unknown"])
    if uncertain:
        print("REMOTE STOP UNCONFIRMED", ",".join(uncertain), "(original job records retained in plan.retry_of)")
    return out


def prepare_retry(old, out=None, *, launch=False):
    old = regular_directory(old)
    out = Path(out).absolute() if out is not None else old.with_name(old.name+"-retry")
    common.require(out == out.resolve() and old not in out.parents and out not in old.parents and out != old,
                   "fresh retry directory must be separate from original campaign")
    common.require(not (old / ".campaign.lock").is_symlink(), "original controller lock must not be a symlink")
    with campaign.output_lock(old):
        prepared = prepare_locked(old, out)
        if prepared is not None and launch:
            common.require(campaign.launch_tmux(prepared) == 0, "retry tmux launch failed")
        return prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path, help="live original campaign directory on the operator node")
    parser.add_argument("--out", type=Path, help="fresh output; defaults to OLD-retry")
    parser.add_argument("--plan-only", action="store_true", help="prepare only; never launch tmux or SDK jobs")
    args = parser.parse_args(argv)
    try:
        prepare_retry(args.old, args.out, launch=not args.plan_only)
        return 0
    except (ValueError, OSError, KeyError, TypeError, tarfile.TarError, EOFError) as error:
        print("error:", error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
