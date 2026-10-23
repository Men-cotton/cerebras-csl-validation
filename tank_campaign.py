#!/usr/bin/env python3
"""Serial cross-tank hardware campaign; default launch uses private tmux."""
import argparse
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
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

import tank_appliance as appliance
import tank_audit

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ("topk-small", "halo-small", "topk24", "halo16", "topk32", "halo32", "topk64", "halo64")


def now():
    return datetime.now(timezone.utc).isoformat()


def predecessors(operator_root, explicit):
    """Find the existing mesh follow-up without changing its frozen controller."""
    roots = {appliance.regular_directory(p) for p in explicit}
    entries = list(operator_root.iterdir())
    appliance.require(len(entries) <= 4096, "operator directory entry bound")
    for candidate in entries:
        if candidate.name.startswith("mesh-followup-") and candidate.is_dir() and not candidate.is_symlink():
            if not (candidate / "plan.json").is_file() or not (candidate / "state.json").is_file():
                continue
            plan = appliance.read_json(candidate / "plan.json")
            state = appliance.read_json(candidate / "state.json")
            if plan.get("kind") == "mesh-followup" and state.get("status") != "complete":
                roots.add(appliance.regular_directory(candidate))
    return [dict(path=str(root), plan_sha256=appliance.digest(root / "plan.json"))
            for root in sorted(roots)]


def predecessor_ready(entry, reservations=None):
    root = appliance.regular_directory(entry["path"])
    appliance.require(appliance.digest(root / "plan.json") == entry["plan_sha256"], "predecessor plan changed")
    try:
        with ExitStack() as temporary:
            held = temporary if reservations is None else reservations
            held.enter_context(output_lock(root))
            plan, state = appliance.read_json(root / "plan.json"), appliance.read_json(root / "state.json")
            appliance.require(plan.get("kind") == "mesh-followup" and
                              state.get("plan_sha256") == entry["plan_sha256"], "predecessor identity")
            for name, digest in plan["source_sha256"].items():
                appliance.require(Path(name).name == name and
                                  appliance.digest(root / "driver" / name) == digest, "predecessor source changed")
            if state.get("status") in ("pending", "running"):
                return False
            appliance.require(state.get("status") == "complete", "predecessor did not finish successfully; inspect its jobs before submitting this campaign")
            appliance.require(set(state["cohorts"]) == {x["id"] for x in plan["cohorts"]}, "predecessor cohort inventory")
            for item in plan["cohorts"]:
                appliance.require(item["path"] == "cohorts/" + item["id"] and
                                  appliance.NAME.fullmatch(item["id"]), "predecessor cohort path")
                child = root / item["path"]
                appliance.require(appliance.digest(child / "plan.json") == item["plan_sha256"], "predecessor child plan changed")
                child_plan = appliance.read_json(child / "plan.json")
                child_state = appliance.read_json(child / "state.json")
                held.enter_context(output_lock(child))
                child_state = appliance.read_json(child / "state.json")
                appliance.require(child_state.get("status") == "complete" and
                                  child_state.get("plan_sha256") == item["plan_sha256"] and
                                  state["cohorts"][item["id"]]["status"] == "complete", "predecessor child incomplete")
                for name, digest in child_plan["source_sha256"].items():
                    appliance.require(Path(name).name == name and
                                      appliance.digest(child / "driver" / name) == digest, "predecessor child source changed")
                for group in ("builds", "runs"):
                    appliance.require(set(child_state[group]) == {x["id"] for x in child_plan[group]} and
                                      all(x["status"] == "pass" and not x.get("remote_stop_unconfirmed")
                                          for x in child_state[group].values()), "predecessor has unresolved jobs")
            return True
    except ValueError as error:
        if str(error).startswith("another controller owns this output:"):
            return False
        raise


@contextmanager
def hardware_queue(root, plan):
    """Serialize new suites on this operator host; existing mesh uses its own locks."""
    operator_root = appliance.regular_directory(plan["operator_root"])
    path = operator_root / ".tank-hardware.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                print("WAIT shared hardware campaign lock", flush=True)
                time.sleep(30)
        while True:
            with ExitStack() as reservations:
                if all(predecessor_ready(item, reservations) for item in plan["after"]):
                    yield
                    return
            print("WAIT mesh follow-up", ", ".join(x["path"] for x in plan["after"]), flush=True)
            time.sleep(30)
    finally:
        os.close(descriptor)


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
    manifest = tank_audit.verify_package(root / "tank_cases")
    appliance.require(appliance.digest(root / "tank_cases/manifest.json") == plan["package_manifest_sha256"],
                      "frozen package changed")
    return manifest


def load(root):
    root = appliance.regular_directory(root)
    plan, state = appliance.read_json(root / "plan.json"), appliance.read_json(root / "state.json")
    appliance.require(plan.get("schema") == 1 and state.get("plan_sha256") == appliance.digest(root / "plan.json"),
                      "campaign plan identity")
    appliance.require(plan.get("kind") == "tank-campaign" and isinstance(plan.get("after"), list) and
                      len(plan["after"]) <= 32, "campaign predecessor bound")
    appliance.regular_directory(plan["operator_root"])
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
    appliance.require(all(type(entry.get("client_timeout")) is int and 1 <= entry["client_timeout"] <= 86400
                          for entry in plan["builds"]), "per-build timeout bound")
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
    manifest = tank_audit.verify_package(package)
    names = args.cases.split(",")
    appliance.require(1 <= len(names) <= 16 and len(set(names)) == len(names) and
                      all(name in manifest["cases"] and appliance.NAME.fullmatch(name) for name in names), "selected cases")
    appliance.require(type(args.repeats) is int and 1 <= args.repeats <= 8, "repeats must be 1..8")
    appliance.require(1 <= args.client_timeout <= 86400 and 1 <= args.host_timeout <= 1800, "timeout bound")
    appliance.require(1 <= args.large_compile_timeout <= 86400, "large compile timeout bound")
    appliance.sdk_options(args.namespace)
    appliance.pair(args.fabric_offsets)
    if args.fabric_dims:
        appliance.pair(args.fabric_dims)
    files = appliance.inventory(package, byte_limit=256 << 20)
    appliance.require(set(files) == set(manifest["files"]) | {"manifest.json"}, "package has unlisted files")
    root = appliance.new_directory(args.out or Path.cwd() / (
        "tank-campaign-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]))
    (root / "driver").mkdir()
    pins = appliance.source_identity()
    for name in appliance.SOURCES:
        (root / "driver" / name).write_bytes(appliance.bounded(ROOT / name))
    for name, size in files.items():
        target = root / "tank_cases" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(appliance.bounded(package / name, max(size, 1)))
    builds = [dict(id=name, case=name, client_timeout=(args.large_compile_timeout if name == "halo64" else args.client_timeout))
              for name in names]
    runs = []
    for repetition in range(args.repeats):
        order = names if repetition % 2 == 0 else list(reversed(names))
        for name in order:
            runs.append(dict(id=f"r{len(runs)+1:03d}", case=name, build=name, repetition=repetition + 1))
    appliance.require(len(runs) <= 64, "run count bound")
    plan = dict(schema=1, kind="tank-campaign", campaign_id=str(uuid.uuid4()), created_at=now(), backend=args.backend,
                operator_root=str(ROOT), after=predecessors(ROOT, args.after),
                namespace=args.namespace, fabric_dims=args.fabric_dims, fabric_offsets=args.fabric_offsets,
                host_timeout=args.host_timeout, client_timeout=args.client_timeout,
                builds=builds, runs=runs, source_sha256=pins,
                package_manifest_sha256=appliance.digest(root / "tank_cases/manifest.json"),
                source_commit=manifest.get("source_commit"),
                order="Selected case order, reversed on alternating repetitions; each case is an independent cohort.",
                policy="One artifact per case; serial clients; failures continue; no automatic retry of attempted entries.",
                timing_scope="Client wall time includes service/queue costs; worker/API times are nested; PE clocks are local intervals.")
    plan["large_compile_estimate"] = dict(case="halo64", estimate_seconds=3439, client_timeout=args.large_compile_timeout,
        basis="E064 local halo32 compile 687.7s extrapolated to specialized halo64; hardware service/queue costs unknown")
    for folder in ("builds", "runs", "client-logs"):
        (root / folder).mkdir()
    appliance.write_json(root / "plan.json", plan)
    appliance.write_json(root / "state.json", dict(schema=1, status="pending", plan_sha256=appliance.digest(root / "plan.json"),
        builds={entry["id"]: dict(status="pending") for entry in builds},
        runs={entry["id"]: dict(status="pending") for entry in runs}))
    check_frozen(root, plan)
    return root


def command(root, plan, entry, group):
    result = [sys.executable, "-B", str(root / "driver/tank_appliance.py"), "compile" if group == "builds" else "run",
              "--backend", plan["backend"], "--package", str(root / "tank_cases"), "--case", entry["case"],
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
    return appliance.load_build(root / "builds" / entry["id"] / "build.json", root / "tank_cases",
                                entry["case"], plan["backend"])[0]


def validate_run(root, plan, entry):
    folder = root / "runs" / entry["id"]
    build, _ = appliance.load_build(root / "builds" / entry["build"] / "build.json", root / "tank_cases",
                                    entry["case"], plan["backend"])
    record, request = appliance.read_json(folder / "collection.json"), appliance.read_json(folder / "request.json")
    appliance.require(record.get("status") == "complete" and record.get("accounting", {}).get("status") == "complete" and
                      record.get("request") == request and record.get("host_exit") == 0, "collection incomplete")
    expected = dict(schema=1, run_id=request["run_id"], backend=plan["backend"], case_name=entry["case"],
                    case_sha256=build["case_sha256"], package_manifest_sha256=plan["package_manifest_sha256"],
                    artifact_sha256=build["artifact_sha256"], worker_sha256=plan["source_sha256"]["tank_worker.py"])
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
    report = tank_audit.audit_capture(root / "tank_cases", entry["case"], folder / "capture", request)
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
                    ret = runner(argv, root / "client-logs" / (group + "-" + entry["id"] + ".log"), entry.get("client_timeout", plan["client_timeout"]))
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
        comparison = dict(status="SEPARATE_CASES", reports=[
            dict(case_name=entry["case"], run_id=entry["id"], repetition=entry["repetition"],
                 report=appliance.read_json(root / "runs" / entry["id"] / "audit.json"))
            for entry in plan["runs"] if state["runs"][entry["id"]]["status"] == "pass"])
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
                ret = subprocess.run(tmux_command(old["socket"], "has-session", "-t", "=tank"),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                appliance.require(ret.returncode != 0, "existing tmux campaign is still present")
        socket = "tank-" + uuid.uuid4().hex[:16]
        argv = [sys.executable, "-B", str(root / "driver/tank_campaign.py"), "--foreground", "--resume", str(root)]
        record = dict(status="starting", socket=socket, session="tank", command=argv, created_at=now())
        appliance.write_json(root / "launcher.json", record)
        ret = subprocess.run(tmux_command(socket, "new-session", "-d", "-s", "tank", "-c", str(root / "driver"),
                                          "/bin/sh", "-c", "exec " + shlex.join(argv)), capture_output=True, text=True)
        if ret.returncode:
            appliance.write_json(root / "launcher.json", dict(record, status="failed", error=ret.stderr[:4096]))
            raise ValueError("tmux launch failed")
    print("OUTPUT", root)
    print("ATTACH", shlex.join(tmux_command(socket, "attach-session", "-t", "=tank")))
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
                    plan, _ = load(root)
                    with hardware_queue(root, plan):
                        code = run_campaign(root)
        except KeyboardInterrupt:
            code = 130
            state = appliance.read_json(root / "state.json")
            state.update(status="interrupted", exit_code=code, error="interrupted while waiting for hardware queue", finished_at=now())
            appliance.write_json(root / "state.json", state)
        except Exception as error:
            code = 2
            state = appliance.read_json(root / "state.json")
            state.update(status="blocked", exit_code=code, error=str(error), finished_at=now())
            appliance.write_json(root / "state.json", state)
            print("QUEUE ERROR", str(error), file=sys.stderr)
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
    result.add_argument("--package", type=Path, default=ROOT / "tank_cases")
    result.add_argument("--cases", default=",".join(DEFAULT_CASES))
    result.add_argument("--repeats", type=int, default=2)
    result.add_argument("--backend", choices=("hardware", "appliance-sim"), default="hardware")
    result.add_argument("--namespace")
    result.add_argument("--fabric-dims")
    result.add_argument("--fabric-offsets", default="4,1")
    result.add_argument("--host-timeout", type=int, default=300)
    result.add_argument("--client-timeout", type=int, default=1200)
    result.add_argument("--large-compile-timeout", type=int, default=7200, help="halo64 compiler client bound including queue time; worker remains separately bounded")
    result.add_argument("--out", type=Path)
    result.add_argument("--resume", type=Path)
    result.add_argument("--after", type=Path, action="append", default=[], help="Wait for this mesh-followup output; unfinished follow-ups in this checkout are also detected")
    result.add_argument("--plan-only", action="store_true")
    result.add_argument("--foreground", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.resume is not None:
            appliance.require(args.out is None, "--out and --resume are mutually exclusive")
            existing = appliance.regular_directory(args.resume)
            if ROOT != existing / "driver":
                # Select the frozen implementation before importing its package schema.
                plan = appliance.read_json(existing / "plan.json")
                pins = plan.get("source_sha256", {})
                appliance.require("tank_campaign.py" in pins and 1 <= len(pins) <= 32,
                                  "frozen resume source inventory")
                for name, digest in pins.items():
                    appliance.require(Path(name).name == name and
                                      appliance.digest(existing / "driver" / name) == digest,
                                      "frozen resume source changed")
                command = [sys.executable, "-B", str(existing / "driver/tank_campaign.py"), "--resume", str(existing)]
                if args.plan_only:
                    command.append("--plan-only")
                if args.foreground:
                    command.append("--foreground")
                os.execv(sys.executable, command)
        root = prepare(args)
        if args.plan_only:
            print("PLAN ONLY", root)
            return 0
        if args.foreground:
            if ROOT != root / "driver":
                os.execv(sys.executable, [sys.executable, "-B", str(root / "driver/tank_campaign.py"),
                                         "--foreground", "--resume", str(root)])
            return foreground(root)
        return launch_tmux(root)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print("error:", str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
