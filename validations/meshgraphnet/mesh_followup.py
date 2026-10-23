#!/usr/bin/env python3
"""Run two isolated CylinderFlow continuation campaigns in one private tmux."""
import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
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
import mesh_campaign as campaign

ROOT = Path(__file__).resolve().parent
SOURCES = ("mesh_followup.py",) + appliance.SOURCES
CASES = ("full-broadcast", "full-selective", "full-colored")
COHORTS = {"h4-s2": dict(H=4, S=2, C=8, R=8, K=6),
           "h8-s2": dict(H=8, S=2, C=16, R=8, K=3)}
MAX_INVOCATIONS = 16


def verify_cohort(package, name):
    """Bind the declared follow-up to a full known mesh, not a prefix fixture."""
    manifest = appliance.mesh_audit.verify_package(package)
    appliance.require(set(manifest["cases"]) == set(CASES), "cohort case inventory")
    expected = COHORTS[name]
    for mode, case_name in enumerate(CASES):
        case = appliance.read_json(package / "cases" / case_name / "case.json")
        p = case["params"]
        appliance.require(all(p[k] == expected[k] for k in ("H", "S", "C", "R")) and
                          p["STEPS"] == 4 and p["NORM"] == 0 and p["MODE"] == mode and
                          p["K"] == (1 if mode == 0 else expected["K"]), "cohort parameters")
        appliance.require(case["residency"] == dict(schema=1, chunks=[2, 2]), "cohort resident chunks")
        appliance.require(case.get("graph_scope") == "full" and case.get("graph_limit") is None and
                          case.get("original_graph_counts") == dict(nodes=1896, edges=10908) and
                          case.get("census", {}).get("nodes") == 1896 and
                          case["census"].get("directed_edges") == 10908 and
                          case.get("partition") == "spatial", "full CylinderFlow graph required")
    return manifest


def check_parent(root):
    root = appliance.regular_directory(root)
    plan, state = appliance.read_json(root / "plan.json"), appliance.read_json(root / "state.json")
    appliance.require(plan.get("schema") == 1 and plan.get("kind") == "mesh-followup" and
                      state.get("plan_sha256") == appliance.digest(root / "plan.json"), "follow-up plan identity")
    appliance.require(set(plan["source_sha256"]) == set(SOURCES) and
                      all(appliance.digest(root / "driver" / name) == value
                          for name, value in plan["source_sha256"].items()), "frozen follow-up source changed")
    return plan, state


def load(root):
    root = appliance.regular_directory(root)
    plan, state = check_parent(root)
    appliance.require([entry["id"] for entry in plan["cohorts"]] == list(COHORTS) and
                      set(state["cohorts"]) == set(COHORTS), "follow-up cohort inventory")
    for entry in plan["cohorts"]:
        name = entry["id"]
        appliance.require(entry["path"] == "cohorts/" + name, "cohort path")
        child = root / entry["path"]
        child_plan, _ = campaign.load(child)
        appliance.require(appliance.digest(child / "plan.json") == entry["plan_sha256"], "child plan identity")
        appliance.require(child_plan["source_sha256"] == {n: plan["source_sha256"][n] for n in appliance.SOURCES},
                          "child source differs from frozen parent")
        for key in ("backend", "namespace", "fabric_dims", "fabric_offsets", "host_timeout", "client_timeout"):
            appliance.require(child_plan[key] == plan[key], "child setting differs: " + key)
        appliance.require([b["case"] for b in child_plan["builds"]] == list(CASES), "child build cases")
        verify_cohort(child / "mesh_cases", name)
        slot = state["cohorts"][name]
        appliance.require(slot["status"] in ("pending", "started", "complete", "completed_with_failures", "unknown") and
                          type(slot["invocations"]) is int and 0 <= slot["invocations"] <= MAX_INVOCATIONS,
                          "cohort status or resume bound")
    return plan, state


def prepare(args):
    if args.resume is not None:
        appliance.require(args.out is None, "--out and --resume are mutually exclusive")
        root = appliance.regular_directory(args.resume)
        load(root)
        return root
    packages = appliance.regular_directory(args.package)
    for name in COHORTS:
        verify_cohort(appliance.regular_directory(packages / name), name)
    # Validate shared arguments before creating any output. Child prepare uses
    # the existing controller's package bounds, inventory and source freezing.
    appliance.require(1 <= args.full_repeats <= 8 and 1 <= args.host_timeout <= 1800 and
                      1 <= args.client_timeout <= 86400, "follow-up repetition/timeout bounds")
    appliance.sdk_options(args.namespace)
    appliance.pair(args.fabric_offsets)
    if args.fabric_dims:
        appliance.pair(args.fabric_dims)
    root = appliance.new_directory(args.out or Path.cwd() / (
        "mesh-followup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]))
    (root / "driver").mkdir()
    (root / "client-logs").mkdir()
    pins = {}
    for name in SOURCES:
        value = appliance.bounded(ROOT / name)
        (root / "driver" / name).write_bytes(value)
        pins[name] = appliance.digest(root / "driver" / name)
    cohorts = []
    for name in COHORTS:
        child_args = campaign.parser().parse_args([])
        for key in ("backend", "namespace", "fabric_dims", "fabric_offsets", "host_timeout", "client_timeout", "full_repeats"):
            setattr(child_args, key, getattr(args, key))
        child_args.package, child_args.cases = packages / name, ",".join(CASES)
        child_args.out, child_args.plan_only = root / "cohorts" / name, True
        child = campaign.prepare(child_args)  # Prepares only; no client or tmux.
        cohorts.append(dict(id=name, path="cohorts/" + name, plan_sha256=appliance.digest(child / "plan.json"),
                            hardware_compile="pending", local_admission="not asserted by controller"))
    plan = dict(schema=1, kind="mesh-followup", campaign_id=str(uuid.uuid4()), created_at=campaign.now(),
                cohorts=cohorts, source_sha256=pins, full_repeats=args.full_repeats,
                comparison_scope="Only inside each child cohort; H4 and H8 arithmetic and PE grids differ.",
                policy="Serial frozen child controllers; child failures continue; attempted jobs are never automatically resubmitted.")
    for key in ("backend", "namespace", "fabric_dims", "fabric_offsets", "host_timeout", "client_timeout"):
        plan[key] = getattr(args, key)
    appliance.write_json(root / "plan.json", plan)
    appliance.write_json(root / "state.json", dict(schema=1, status="pending", plan_sha256=appliance.digest(root / "plan.json"),
        cohorts={name: dict(status="pending", invocations=0) for name in COHORTS}))
    load(root)
    return root


def child_command(root, entry):
    child = root / entry["path"]
    return [sys.executable, "-B", str(child / "driver/mesh_campaign.py"), "--foreground", "--resume", str(child)]


def run_followup(root, runner=campaign.execute):
    plan, state = load(root)
    # Refuse the whole parent launch if a previous child's controller still
    # owns a lock. A released local lock does not establish remote termination.
    with ExitStack() as stack:
        for entry in plan["cohorts"]:
            stack.enter_context(campaign.output_lock(root / entry["path"]))
    state.update(status="running", started_at=campaign.now())
    appliance.write_json(root / "state.json", state)
    start, code = time.monotonic(), 2
    try:
        for entry in plan["cohorts"]:
            name, child = entry["id"], root / entry["path"]
            load(root)
            slot = state["cohorts"][name]
            appliance.require(slot["invocations"] < MAX_INVOCATIONS, "follow-up resume invocation bound")
            slot.update(status="started", started_at=campaign.now(), invocations=slot["invocations"] + 1)
            argv = child_command(root, entry)
            slot["command"] = argv
            appliance.write_json(root / "state.json", state)
            print("COHORT", name, flush=True)
            child_plan, _ = campaign.load(child)
            timeout = (len(child_plan["builds"]) + len(child_plan["runs"])) * (plan["client_timeout"] + 30) + 600
            began = time.monotonic()
            try:
                # Protect the other cohort against a separate manual child
                # launch while this one runs. The selected child owns its lock.
                with ExitStack() as stack:
                    for other in plan["cohorts"]:
                        if other["id"] != name:
                            stack.enter_context(campaign.output_lock(root / other["path"]))
                    ret = runner(argv, root / "client-logs" / (name + "-%03d.log" % slot["invocations"]), timeout)
                slot["client_exit"] = ret
                _, child_state = campaign.load(child)
                summary = appliance.read_json(child / "summary.json")
                appliance.require(child_state.get("status") in ("complete", "completed_with_failures") and
                                  summary.get("status") == child_state["status"] and
                                  child_state.get("exit_code") == ret and ret in (0, 2), "child terminal evidence incomplete")
                slot["status"] = "complete" if ret == 0 and child_state["status"] == "complete" else "completed_with_failures"
                slot["summary_sha256"] = appliance.digest(child / "summary.json")
            except KeyboardInterrupt:
                slot.update(status="unknown", remote_stop_unconfirmed=True)
                raise
            except Exception as error:
                slot.update(status="unknown", error=str(error), remote_stop_unconfirmed=True)
                print("CONTINUE", name, str(error), flush=True)
            finally:
                slot.update(finished_at=campaign.now(), client_wall_seconds=time.monotonic() - began)
                appliance.write_json(root / "state.json", state)
            print("RESULT", name, slot["status"], flush=True)
        code = 0 if all(s["status"] == "complete" for s in state["cohorts"].values()) else 2
        state["status"] = "complete" if code == 0 else "completed_with_failures"
    except KeyboardInterrupt:
        state["status"], code = "interrupted", 130
    except Exception as error:
        state.update(status="incomplete", error=str(error))
    finally:
        state.update(exit_code=code, finished_at=campaign.now(),
                     controller_wall_seconds=state.get("controller_wall_seconds", 0) + time.monotonic() - start)
        appliance.write_json(root / "state.json", state)
        results = {}
        for entry in plan["cohorts"]:
            child = root / entry["path"]
            result = dict(state=state["cohorts"][entry["id"]], child_path=entry["path"])
            if (child / "summary.json").is_file():
                try:
                    result["summary"] = appliance.read_json(child / "summary.json")
                except Exception as error:
                    result["summary_error"] = str(error)
            results[entry["id"]] = result
        appliance.write_json(root / "summary.json", dict(status=state["status"], backend=plan["backend"], cohorts=results,
            comparison_scope=plan["comparison_scope"],
            remote_scope="A local timeout or released lock does not prove a remote job stopped; resume submits only child pending entries."))
    return code


def child_archive_is_current(root, name):
    """A prior ZIP cannot replace new evidence from an interrupted resume."""
    child = root / "cohorts" / name
    try:
        parent_state = appliance.read_json(root / "state.json")
        state = appliance.read_json(child / "state.json")
        if parent_state["cohorts"][name]["status"] not in ("complete", "completed_with_failures") or \
                state.get("status") not in ("complete", "completed_with_failures") or state.get("bundle_error"):
            return False
        record = appliance.read_json(child / "bundle.json")
        files = appliance.inventory(child)
        files.pop("bundle.json", None)
        if set(files) != set(record["files"]):
            return False
        for path, size in files.items():
            if record["files"][path] != dict(bytes=size, sha256=appliance.digest(child / path)):
                return False
        with zipfile.ZipFile(child.with_suffix(".zip")) as archive:
            expected = {child.name + "/" + path for path in files} | {child.name + "/bundle.json"}
            entries = archive.infolist()
            if len(entries) != len(expected) or {entry.filename for entry in entries} != expected:
                return False
            manifest_entry = archive.getinfo(child.name + "/bundle.json")
            if manifest_entry.file_size > appliance.JSON_LIMIT:
                return False
            return archive.read(manifest_entry) == appliance.bounded(child / "bundle.json")
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile):
        return False


def bundle(root):
    """One ZIP containing parent evidence and each completed child ZIP.

    If a child ZIP is unavailable or stale, preserve the current expanded
    evidence too. Byte accounting includes expanded and archived child files.
    """
    files = appliance.inventory(root, byte_limit=4 << 30, entry_limit=16384)
    child_archives = {}
    for name in COHORTS:
        archive_name = "cohorts/" + name + ".zip"
        if archive_name in files and child_archive_is_current(root, name):
            prefix = "cohorts/" + name + "/"
            files = {path: size for path, size in files.items() if not path.startswith(prefix)}
            child_archives[name] = "current ZIP"
        else:
            child_archives[name] = "current expanded evidence; any prior ZIP also retained"
    files.pop("bundle.json", None)
    appliance.write_json(root / "bundle.json", dict(schema=1,
        files={name: dict(bytes=size, sha256=appliance.digest(root / name)) for name, size in files.items()},
        child_archives=child_archives,
        scope="Parent evidence plus current child ZIPs; current expanded evidence is included when its ZIP is unavailable or stale."))
    files["bundle.json"] = (root / "bundle.json").stat().st_size
    destination = root.with_suffix(root.suffix + ".zip")
    temporary = destination.with_name(destination.name + ".tmp")
    appliance.require(not destination.is_symlink() and not temporary.exists() and not temporary.is_symlink(), "unsafe parent ZIP")
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name in sorted(files):
                archive.write(root / name, arcname=root.name + "/" + name)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def foreground(root):
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    with campaign.output_lock(root):
        previous = signal.signal(signal.SIGTERM, interrupted)
        record = appliance.read_json(root / "launcher.json") if (root / "launcher.json").exists() else {}
        record.update(status="running", pid=os.getpid(), started_at=campaign.now())
        appliance.write_json(root / "launcher.json", record)
        code = 2
        try:
            with (root / "driver.log").open("a", buffering=1) as log:
                with redirect_stdout(campaign.Tee(sys.stdout, log)), redirect_stderr(campaign.Tee(sys.stderr, log)):
                    code = run_followup(root)
        finally:
            signal.signal(signal.SIGTERM, previous)
            record.update(status="complete" if code == 0 else "interrupted" if code == 130 else "completed_with_failures",
                          finished_at=campaign.now(), exit_code=code)
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


def launch_tmux(root):
    appliance.require(shutil.which("tmux") is not None, "tmux is required on the operator node")
    with campaign.output_lock(root, ".launch.lock"):
        with campaign.output_lock(root):
            pass
        if (root / "launcher.json").exists():
            old = appliance.read_json(root / "launcher.json")
            if old.get("socket"):
                ret = subprocess.run(campaign.tmux_command(old["socket"], "has-session", "-t", "=mesh-followup"),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                appliance.require(ret.returncode != 0, "existing follow-up tmux is still present")
        socket = "mesh-followup-" + uuid.uuid4().hex[:16]
        argv = [sys.executable, "-B", str(root / "driver/mesh_followup.py"), "--foreground", "--resume", str(root)]
        record = dict(status="starting", socket=socket, session="mesh-followup", command=argv, created_at=campaign.now())
        appliance.write_json(root / "launcher.json", record)
        ret = subprocess.run(campaign.tmux_command(socket, "new-session", "-d", "-s", "mesh-followup", "-c", str(root / "driver"),
                                                   "/bin/sh", "-c", "exec " + shlex.join(argv)), capture_output=True, text=True)
        if ret.returncode:
            appliance.write_json(root / "launcher.json", dict(record, status="failed", error=ret.stderr[:4096]))
            raise ValueError("follow-up tmux launch failed")
    print("OUTPUT", root)
    print("ATTACH", shlex.join(campaign.tmux_command(socket, "attach-session", "-t", "=mesh-followup")))
    print("LOG", shlex.join(["tail", "-f", str(root / "driver.log")]))
    print("STATUS", shlex.join(["cat", str(root / "state.json")]))
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--package", type=Path, default=ROOT / "mesh_followup_cases")
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
        # A resumed execution enters its frozen parent before loading children;
        # edits in the checkout cannot replace the old campaign's controller.
        if args.resume is not None and not args.plan_only:
            appliance.require(args.out is None, "--out and --resume are mutually exclusive")
            root = appliance.regular_directory(args.resume)
            if ROOT != root / "driver":
                check_parent(root)
                command = [sys.executable, "-B", str(root / "driver/mesh_followup.py"), "--resume", str(root)]
                if args.foreground:
                    command.append("--foreground")
                os.execv(sys.executable, command)
        root = prepare(args)
        if args.plan_only:
            print("PLAN ONLY", root)
            return 0
        if args.foreground:
            if ROOT != root / "driver":
                os.execv(sys.executable, [sys.executable, "-B", str(root / "driver/mesh_followup.py"), "--foreground", "--resume", str(root)])
            return foreground(root)
        return launch_tmux(root)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print("error:", str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
