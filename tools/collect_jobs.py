#!/usr/bin/env python3
"""Find appliance job IDs in local logs without grep or ripgrep.

By default this only reads logs and prints JSON. With --collect --out NEWDIR,
query those jobs using csctl and export their logs. Files stay in NEWDIR.
No job cancellation, device selection, or administrative command is used.
"""
import argparse
import glob
import json
from pathlib import Path
import re
import subprocess
import sys

JOB = re.compile(r"\bJob id:\s*(wsjob-[A-Za-z0-9-]+)")
WORKFLOW = re.compile(r"\bworkflow id:\s*(\S+?)(?:,|\s|$)")
REMOTE = re.compile(r"\bremote log path:\s*(\S+)")
MAX_LOG = 32 << 20


def job_id(value):
    if not re.fullmatch(r"wsjob-[A-Za-z0-9-]+", value):
        raise argparse.ArgumentTypeError("job ID must be wsjob- followed by letters, digits, or hyphens")
    return value


def expand_logs(patterns):
    paths = []
    seen = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f"no log matches {pattern!r}")
        for name in matches:
            path = Path(name)
            if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_LOG:
                raise ValueError(f"log must be a regular file of at most {MAX_LOG} bytes: {path}")
            if path.resolve() not in seen:
                seen.add(path.resolve())
                paths.append(path)
    return paths


def scan_logs(paths):
    found = []
    for path in paths:
        label = path.name.lower()
        role = "compile" if "compile" in label else "launch" if "launch" in label else "unknown"
        if path.name == "client.log":
            request = path.parent / "request.json"
            if request.is_file() and not request.is_symlink() and request.stat().st_size <= 2 << 20:
                try:
                    action = json.loads(request.read_text()).get("action")
                    role = {"compile": "compile", "run": "launch"}.get(action, role)
                except (ValueError, AttributeError):
                    pass
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            match = JOB.search(line)
            if not match:
                continue
            workflow, remote = WORKFLOW.search(line), REMOTE.search(line)
            found.append(dict(log=str(path), line=number, role=role, job_id=match.group(1),
                              workflow_id=workflow.group(1) if workflow else None,
                              remote_log_path=remote.group(1) if remote else None))
    return found


def systems_from_job(value):
    """Return only the documented status.systems list of system names."""
    status = value.get("status") if isinstance(value, dict) else None
    systems = status.get("systems") if isinstance(status, dict) else None
    if isinstance(systems, list) and all(isinstance(item, str) for item in systems):
        return systems
    return None


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_command(argv, directory, stem, seconds, runner=subprocess.run):
    stdout_path, stderr_path = directory / (stem + ".stdout"), directory / (stem + ".stderr")
    entry = dict(argv=argv, timeout_seconds=seconds, stdout=stdout_path.name,
                 stderr=stderr_path.name, status="incomplete")
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            result = runner(argv, cwd=str(directory), stdout=stdout, stderr=stderr,
                            timeout=seconds, check=False)
            entry.update(status="complete" if result.returncode == 0 else "failed",
                         exit_code=result.returncode)
        except subprocess.TimeoutExpired:
            entry.update(status="timeout")
        except OSError as error:
            entry.update(status="failed", error=str(error))
    return entry


def collect(records, out, namespace, seconds, binary=False, runner=subprocess.run):
    destination = Path(out).absolute()
    if destination != destination.resolve():
        raise ValueError("output path must not traverse symlinks")
    destination.mkdir(parents=True, exist_ok=False)
    summary = dict(status="incomplete", jobs=records, collections=[])
    save_json(destination / "jobs.json", summary)
    prefix = ["csctl"] + (["-n", namespace] if namespace else [])
    identifiers = list(dict.fromkeys(record["job_id"] for record in records))
    for identifier in identifiers:
        directory = destination / identifier
        directory.mkdir()
        entry = dict(job_id=identifier, systems=None, commands=[])
        metadata = run_command(prefix + ["get", "job", identifier, "-o", "json"],
                               directory, "job", seconds, runner)
        entry["commands"].append(metadata)
        if metadata["status"] == "complete":
            try:
                path = directory / "job.stdout"
                if path.stat().st_size > MAX_LOG:
                    raise ValueError("job JSON exceeds size limit")
                entry["systems"] = systems_from_job(json.loads(path.read_text()))
                if entry["systems"] is None:
                    entry["systems_note"] = "status.systems was absent or had an unrecognized shape; inspect raw job.stdout"
            except (OSError, ValueError) as error:
                entry["metadata_error"] = str(error)
        export = prefix + ["log-export", identifier] + (["-b"] if binary else [])
        entry["commands"].append(run_command(export, directory, "log-export", seconds, runner))
        summary["collections"].append(entry)
        save_json(destination / "jobs.json", summary)
    summary["status"] = "complete" if all(
        command["status"] == "complete" and "metadata_error" not in item
        for item in summary["collections"] for command in item["commands"]) else "collection_failed"
    save_json(destination / "jobs.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", help="local log paths or quoted wildcard patterns")
    parser.add_argument("--job-id", type=job_id, action="append", default=[], help="additional explicit job ID")
    parser.add_argument("--collect", action="store_true", help="run csctl get job and log-export")
    parser.add_argument("--out", type=Path, help="new collection directory; required with --collect")
    parser.add_argument("--namespace", help="existing site namespace; otherwise use csctl default")
    parser.add_argument("--timeout", type=int, default=120, help="seconds per csctl command, 1..600")
    parser.add_argument("--binary", action="store_true", help="include binary debug artifacts in log-export (-b)")
    args = parser.parse_args(argv)
    try:
        if not args.logs and not args.job_id:
            raise ValueError("provide local logs or --job-id")
        if not 1 <= args.timeout <= 600:
            raise ValueError("timeout must be in 1..600 seconds")
        if args.collect != bool(args.out):
            raise ValueError("--collect and --out must be used together")
        if args.binary and not args.collect:
            raise ValueError("--binary requires --collect")
        records = scan_logs(expand_logs(args.logs))
        records += [dict(log=None, line=None, role="explicit", job_id=identifier,
                         workflow_id=None, remote_log_path=None) for identifier in args.job_id]
        if not records:
            raise ValueError("no 'Job id:' entries found in the supplied logs")
        if not args.collect:
            print(json.dumps(dict(jobs=records), indent=2, sort_keys=True))
            return 0
        summary = collect(records, args.out, args.namespace, args.timeout, args.binary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "complete" else 2
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
