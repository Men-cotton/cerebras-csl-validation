#!/usr/bin/env python3
"""Serial MeshGraphNet hardware campaign; default launch uses private tmux."""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
import zipfile

import mesh_appliance as appliance
import mesh_audit

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ("small-colored", "crossing-colored", "full-broadcast", "full-selective", "full-colored")


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def output_lock(root, name=".controller.lock"):
    path = root / name
    appliance.require(not path.is_symlink(), "controller lock symlink")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another controller owns this output: " + str(root)) from error
        yield
    finally:
        os.close(descriptor)


def check_frozen(root, plan):
    appliance.require(set(plan["source_sha256"]) == set(appliance.SOURCES), "controller source inventory")
    appliance.require(all(appliance.digest(root / "driver" / name) == value
                          for name, value in plan["source_sha256"].items()), "frozen controller changed")
    manifest = mesh_audit.verify_package(root / "mesh_cases")
    appliance.require(appliance.digest(root / "mesh_cases/manifest.json") == plan["package_manifest_sha256"],
                      "frozen package changed")
    return manifest


def load(root):
    root = appliance.regular_directory(root)
    plan, state = appliance.read_json(root / "plan.json"), appliance.read_json(root / "state.json")
    appliance.require(plan.get("schema") == 1 and state.get("plan_sha256") == appliance.digest(root / "plan.json"),
                      "campaign plan identity")
    manifest = check_frozen(root, plan)
    appliance.require(plan["backend"] in ("hardware", "appliance-sim") and
                      1 <= plan["client_timeout"] <= 86400 and 1 <= plan["host_timeout"] <= 1800, "campaign bounds")
    for group in ("builds", "runs"):
        entries = plan[group]
        appliance.require(isinstance(entries, list) and 1 <= len(entries) <= 64, "campaign entry bound")
        names = [entry["id"] for entry in entries]
        appliance.require(all(isinstance(name, str) and appliance.NAME.fullmatch(name) for name in names) and
                          len(set(names)) == len(names) and set(state[group]) == set(names), "campaign inventory")
        appliance.require(all(entry["case"] in manifest["cases"] for entry in entries), "campaign case identity")
    builds = {entry["id"]: entry["case"] for entry in plan["builds"]}
    appliance.require(all(builds.get(entry["build"]) == entry["case"] for entry in plan["runs"]), "run/build binding")
    statuses = {"pending", "started", "pass", "incomplete", "unknown", "skipped_dependency"}
    appliance.require(all(slot["status"] in statuses for group in ("builds", "runs")
                          for slot in state[group].values()), "unknown attempt status")
    return plan, state


def prepare(args):
    if args.resume is not None:
        appliance.require(args.out is None, "--out and --resume are mutually exclusive")
        root = appliance.regular_directory(args.resume)
        load(root)
        return root
    package = appliance.regular_directory(args.package)
    manifest = mesh_audit.verify_package(package)
    names = args.cases.split(",")
    appliance.require(1 <= len(names) <= 16 and len(set(names)) == len(names) and
                      all(name in manifest["cases"] and appliance.NAME.fullmatch(name) for name in names), "selected cases")
    appliance.require(type(args.full_repeats) is int and 1 <= args.full_repeats <= 8, "full repeats must be 1..8")
    appliance.require(1 <= args.client_timeout <= 86400 and 1 <= args.host_timeout <= 1800, "timeout bound")
    appliance.sdk_options(args.namespace)
    appliance.pair(args.fabric_offsets)
    if args.fabric_dims:
        appliance.pair(args.fabric_dims)
    files = appliance.inventory(package, byte_limit=256 << 20)
    appliance.require(set(files) == set(manifest["files"]) | {"manifest.json"}, "package has unlisted files")
    root = appliance.new_directory(args.out or Path.cwd() / (
        "mesh-campaign-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]))
    (root / "driver").mkdir()
    pins = appliance.source_identity()
    for name in appliance.SOURCES:
        (root / "driver" / name).write_bytes(appliance.bounded(ROOT / name))
    for name, size in files.items():
        target = root / "mesh_cases" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(appliance.bounded(package / name, max(size, 1)))
    builds = [dict(id=name, case=name) for name in names]
    runs = []
    for repetition in range(args.full_repeats):
        order = names if repetition % 2 == 0 else list(reversed(names))
        for name in order:
            if repetition and not name.startswith("full-"):
                continue
            runs.append(dict(id=f"r{len(runs)+1:03d}", case=name, build=name, repetition=repetition + 1))
    appliance.require(len(runs) <= 64, "run count bound")
    plan = dict(schema=1, campaign_id=str(uuid.uuid4()), created_at=now(), backend=args.backend,
                namespace=args.namespace, fabric_dims=args.fabric_dims, fabric_offsets=args.fabric_offsets,
                host_timeout=args.host_timeout, client_timeout=args.client_timeout,
                builds=builds, runs=runs, source_sha256=pins,
                package_manifest_sha256=appliance.digest(root / "mesh_cases/manifest.json"),
                source_commit=manifest.get("source_commit"),
                order="Selected case order, reversed on alternating full-case repetitions; small cases once.",
                policy="One artifact per case; serial clients; failures continue; no automatic retry of attempted entries.",
                timing_scope="Client wall time includes service/queue costs; worker/API times are nested; PE clocks are local intervals.")
    for folder in ("builds", "runs", "client-logs"):
        (root / folder).mkdir()
    appliance.write_json(root / "plan.json", plan)
    appliance.write_json(root / "state.json", dict(schema=1, status="pending", plan_sha256=appliance.digest(root / "plan.json"),
        builds={entry["id"]: dict(status="pending") for entry in builds},
        runs={entry["id"]: dict(status="pending") for entry in runs}))
    check_frozen(root, plan)
    return root


def command(root, plan, entry, group):
    result = [sys.executable, "-B", str(root / "driver/mesh_appliance.py"), "compile" if group == "builds" else "run",
              "--backend", plan["backend"], "--package", str(root / "mesh_cases"), "--case", entry["case"],
              "--out", str(root / group / entry["id"])]
    if group == "builds":
        result += ["--fabric-offsets", plan["fabric_offsets"]]
        if plan["fabric_dims"]:
            result += ["--fabric-dims", plan["fabric_dims"]]
        if plan["namespace"]:
            result += ["--namespace", plan["namespace"]]
    else:
        result += ["--artifact-record", str(root / "builds" / entry["build"] / "build.json"),
                   "--host-timeout", str(plan["host_timeout"])]
    return result


def execute(argv, log, timeout):
    start = time.monotonic()
    with log.open("xb") as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                appliance.require(time.monotonic() - start <= timeout, "client timeout; remote stop unconfirmed")
                appliance.require(log.stat().st_size <= appliance.LOG_LIMIT, "client log bound; remote stop unconfirmed")
                time.sleep(0.2)
            appliance.require(log.stat().st_size <= appliance.LOG_LIMIT, "client log bound")
            return process.returncode
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def validate_build(root, plan, entry):
    return appliance.load_build(root / "builds" / entry["id"] / "build.json", root / "mesh_cases",
                                entry["case"], plan["backend"])[0]


def validate_run(root, plan, entry):
    folder = root / "runs" / entry["id"]
    build, _ = appliance.load_build(root / "builds" / entry["build"] / "build.json", root / "mesh_cases",
                                    entry["case"], plan["backend"])
    record, request = appliance.read_json(folder / "collection.json"), appliance.read_json(folder / "request.json")
    appliance.require(record.get("status") == "complete" and record.get("accounting", {}).get("status") == "complete" and
                      record.get("request") == request and record.get("host_exit") == 0, "collection incomplete")
    expected = dict(schema=1, run_id=request["run_id"], backend=plan["backend"], case_name=entry["case"],
                    case_sha256=build["case_sha256"], package_manifest_sha256=plan["package_manifest_sha256"],
                    artifact_sha256=build["artifact_sha256"], worker_sha256=plan["source_sha256"]["mesh_worker.py"])
    appliance.require(request == expected and str(uuid.UUID(request["run_id"])) == request["run_id"], "run request differs")
    appliance.require(appliance.digest(folder / "build.json") == appliance.digest(root / "builds" / entry["build"] / "build.json"),
                      "run build copy differs")
    launch = appliance.read_json(folder / "launch-request.json")
    appliance.require(launch == dict(command=appliance.host_command(plan["backend"], plan["host_timeout"]),
                      build_sha256=appliance.digest(folder / "build.json"), request_sha256=appliance.digest(folder / "request.json")),
                      "launch request differs")
    appliance.require(set(record["downloaded"]) == {"capture", "host.log", "host-exit.txt"}, "download inventory")
    for item in record["downloaded"].values():
        name = item["path"]
        appliance.require(isinstance(name, str) and name not in ("", ".", "..") and Path(name).name == name, "download filename")
        path = folder / name
        appliance.require(path.stat().st_size == item["bytes"] and appliance.digest(path) == item["sha256"], "download identity")
    appliance.require(appliance.bounded(folder / record["downloaded"]["host-exit.txt"]["path"], 128).decode().strip() == "0",
                      "worker exit differs")
    report = mesh_audit.audit_capture(root / "mesh_cases", entry["case"], folder / "capture", request)
    appliance.require(record["audit_sha256"] == appliance.digest(folder / "audit.json") and
                      report == appliance.read_json(folder / "audit.json") and report.get("status") == "PASS", "capture audit differs")
    return report


def bundle(root):
    files = appliance.inventory(root, byte_limit=1 << 30, entry_limit=4096)
    files.pop("bundle.json", None)
    pins = {name: dict(bytes=size, sha256=appliance.digest(root / name)) for name, size in files.items()}
    appliance.write_json(root / "bundle.json", dict(schema=1, files=pins,
        scope="Complete local package, source, artifacts and captures; failed attempts retained. No omitted originals."))
    files["bundle.json"] = (root / "bundle.json").stat().st_size
    destination = root.with_suffix(root.suffix + ".zip")
    temporary = destination.with_name(destination.name + ".tmp")
    appliance.require(not destination.is_symlink() and not temporary.exists() and not temporary.is_symlink(), "unsafe bundle path")
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name in sorted(files):
                archive.write(root / name, arcname=root.name + "/" + name)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def run_campaign(root, runner=execute):
    plan, state = load(root)
    state.update(status="running", started_at=now())
    appliance.write_json(root / "state.json", state)
    start, code = time.monotonic(), 2
    try:
        for group, validator in (("builds", validate_build), ("runs", validate_run)):
            for entry in plan[group]:
                slot = state[group][entry["id"]]
                if slot["status"] in ("pass", "started"):
                    previous = slot["status"]
                    try:
                        validator(root, plan, entry)
                        slot.update(status="pass", revalidated_at=now())
                    except Exception as error:
                        slot.update(status="unknown" if previous == "started" else "incomplete",
                                    error=str(error), remote_stop_unconfirmed=True)
                    appliance.write_json(root / "state.json", state)
                if slot["status"] != "pending":
                    continue
                if group == "runs" and state["builds"][entry["build"]]["status"] != "pass":
                    slot.update(status="skipped_dependency", finished_at=now())
                    appliance.write_json(root / "state.json", state)
                    continue
                check_frozen(root, plan)
                argv = command(root, plan, entry, group)
                slot.update(status="started", started_at=now(), command=argv)
                appliance.write_json(root / "state.json", state)
                print(group.upper(), entry["id"], entry["case"], flush=True)
                began = time.monotonic()
                try:
                    ret = runner(argv, root / "client-logs" / (group + "-" + entry["id"] + ".log"), plan["client_timeout"])
                    slot["client_exit"] = ret
                    appliance.require(ret == 0, "client exit " + str(ret))
                    validator(root, plan, entry)
                    slot["status"] = "pass"
                except KeyboardInterrupt:
                    slot.update(status="unknown", remote_stop_unconfirmed=True)
                    raise
                except Exception as error:
                    slot.update(status="incomplete", error=str(error), remote_stop_unconfirmed=True)
                    print("CONTINUE", entry["id"], str(error), flush=True)
                finally:
                    slot.update(finished_at=now(), client_wall_seconds=time.monotonic() - began)
                    status_file = root / group / entry["id"] / ("compile-status.json" if group == "builds" else "collection.json")
                    if status_file.exists():
                        try:
                            slot["job_ids"] = appliance.read_json(status_file).get("job_ids", [])
                        except Exception as error:
                            slot["job_record_error"] = str(error)
                    appliance.write_json(root / "state.json", state)
                print("RESULT", entry["id"], slot["status"], flush=True)
        code = 0 if all(slot["status"] == "pass" for group in ("builds", "runs") for slot in state[group].values()) else 2
        state["status"] = "complete" if code == 0 else "completed_with_failures"
    except KeyboardInterrupt:
        state["status"], code = "interrupted", 130
    except Exception as error:
        state.update(status="incomplete", error=str(error))
    finally:
        comparison = dict(status="NOT_COMPARABLE", comparisons=[])
        try:
            records = [dict(case_name=entry["case"], run_id=entry["id"],
                            capture_dir=str(root / "runs" / entry["id"] / "capture"),
                            report=appliance.read_json(root / "runs" / entry["id"] / "audit.json"))
                       for entry in plan["runs"] if entry["case"].startswith("full-") and
                       state["runs"][entry["id"]]["status"] == "pass"]
            comparison = mesh_audit.compare_runs(root / "mesh_cases", records)
        except Exception as error:
            comparison = dict(status="FAILED", error=str(error))
            if code != 130:
                code, state["status"] = 2, "completed_with_failures"
        state.update(finished_at=now(), exit_code=code,
                     controller_wall_seconds=state.get("controller_wall_seconds", 0) + time.monotonic() - start)
        appliance.write_json(root / "state.json", state)
        counts = {group: {status: sum(slot["status"] == status for slot in state[group].values())
                         for status in sorted({slot["status"] for slot in state[group].values()})} for group in ("builds", "runs")}
        appliance.write_json(root / "summary.json", dict(status=state["status"], counts=counts,
            backend=plan["backend"], timing_scope=plan["timing_scope"], comparison=comparison,
            remote_scope="Local timeout/lock release never proves remote job termination. Attempted failures are not resubmitted."))
    return code


def tmux_command(socket, *args):
    return ["tmux", "-L", socket, "-f", "/dev/null", *args]


def launch_tmux(root):
    appliance.require(shutil.which("tmux") is not None, "tmux is required on the operator node")
    with output_lock(root, ".launch.lock"):
        with output_lock(root):
            pass
        if (root / "launcher.json").exists():
            old = appliance.read_json(root / "launcher.json")
            if old.get("socket"):
                ret = subprocess.run(tmux_command(old["socket"], "has-session", "-t", "=mesh"),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                appliance.require(ret.returncode != 0, "existing tmux campaign is still present")
        socket = "mesh-" + uuid.uuid4().hex[:16]
        argv = [sys.executable, "-B", str(root / "driver/mesh_campaign.py"), "--foreground", "--resume", str(root)]
        record = dict(status="starting", socket=socket, session="mesh", command=argv, created_at=now())
        appliance.write_json(root / "launcher.json", record)
        ret = subprocess.run(tmux_command(socket, "new-session", "-d", "-s", "mesh", "-c", str(root / "driver"),
                                          "/bin/sh", "-c", "exec " + shlex.join(argv)), capture_output=True, text=True)
        if ret.returncode:
            appliance.write_json(root / "launcher.json", dict(record, status="failed", error=ret.stderr[:4096]))
            raise ValueError("tmux launch failed")
    print("OUTPUT", root)
    print("ATTACH", shlex.join(tmux_command(socket, "attach-session", "-t", "=mesh")))
    print("LOG", shlex.join(["tail", "-f", str(root / "driver.log")] ))
    print("STATUS", shlex.join(["cat", str(root / "state.json")] ))
    return 0


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, value):
        self.original.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.original.flush()
        self.log.flush()


def foreground(root):
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    with output_lock(root):
        previous = signal.signal(signal.SIGTERM, interrupted)
        record = appliance.read_json(root / "launcher.json") if (root / "launcher.json").exists() else {}
        record.update(status="running", pid=os.getpid(), started_at=now())
        appliance.write_json(root / "launcher.json", record)
        code = 2
        try:
            with (root / "driver.log").open("a", buffering=1) as log:
                with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                    code = run_campaign(root)
        finally:
            signal.signal(signal.SIGTERM, previous)
            record.update(status="complete" if code == 0 else "interrupted" if code == 130 else "completed_with_failures",
                          finished_at=now(), exit_code=code)
            appliance.write_json(root / "launcher.json", record)
            try:
                print("BUNDLE", bundle(root), flush=True)
            except Exception as error:
                code = 130 if code == 130 else 2
                record.update(status="interrupted" if code == 130 else "completed_with_failures",
                              exit_code=code, bundle_error=str(error))
                appliance.write_json(root / "launcher.json", record)
                state = appliance.read_json(root / "state.json")
                state.update(status=record["status"], exit_code=code, bundle_error=str(error))
                appliance.write_json(root / "state.json", state)
                print("BUNDLE ERROR", str(error), file=sys.stderr)
        return code


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--package", type=Path, default=ROOT / "mesh_cases")
    result.add_argument("--cases", default=",".join(DEFAULT_CASES))
    result.add_argument("--full-repeats", type=int, default=2)
    result.add_argument("--backend", choices=("hardware", "appliance-sim"), default="hardware")
    result.add_argument("--namespace")
    result.add_argument("--fabric-dims")
    result.add_argument("--fabric-offsets", default="4,1")
    result.add_argument("--host-timeout", type=int, default=300)
    result.add_argument("--client-timeout", type=int, default=1200)
    result.add_argument("--out", type=Path)
    result.add_argument("--resume", type=Path)
    result.add_argument("--plan-only", action="store_true")
    result.add_argument("--foreground", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        root = prepare(args)
        if args.plan_only:
            print("PLAN ONLY", root)
            return 0
        if args.foreground:
            if ROOT != root / "driver":
                os.execv(sys.executable, [sys.executable, "-B", str(root / "driver/mesh_campaign.py"),
                                         "--foreground", "--resume", str(root)])
            return foreground(root)
        return launch_tmux(root)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print("error:", str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
