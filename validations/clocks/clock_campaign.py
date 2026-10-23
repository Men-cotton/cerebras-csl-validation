#!/usr/bin/env python3
"""Run a fixed transfer campaign under tmux; continue after individual failures.

The operator runs this script on the SDK client node. Importing it submits
nothing. A fresh private tmux server inherits the invoking environment, as in
Modelzoo's benchmark_launcher; environment values are never written to disk.
"""
import argparse
from contextlib import contextmanager, nullcontext, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import fcntl
import hashlib
import json
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

import clock_appliance as appliance
import clock_common as common
from clock_plan import make_plan
from clock_report import analyze_run, write_report

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = (
    'clock_campaign.py', 'clock_plan.py', 'clock_report.py', 'clock_probe.py',
    'clock_common.py', 'clock_pmu.py', 'clock_appliance.py', 'clock_local.py',
    'clock_inspect.py', 'clock_source/layout.csl', 'clock_source/pe.csl',
)
TERMINAL_RUNS = {'pass', 'numeric_fail', 'incomplete', 'unknown', 'skipped_dependency'}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    common.require(len(data) <= 8 << 20, 'bounded campaign JSON')
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def output_lock(root, name='.campaign.lock'):
    with (Path(root) / name).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('another controller owns this output: ' + str(root)) from error
        yield


def check_sources(root, plan):
    expected = plan['source_sha256']
    common.require(set(expected) == set(SOURCE_FILES), 'complete frozen driver source inventory')
    for name, digest in expected.items():
        common.require(common.digest(root / 'driver' / name) == digest, 'frozen source changed: ' + name)


def prepare(args):
    if args.resume:
        root = Path(args.resume).absolute()
        common.require(root == root.resolve() and root.is_dir(), 'regular existing campaign path')
        plan = common.read_json(root / 'plan.json')
        state = common.read_json(root / 'state.json')
        common.require(state['plan_sha256'] == common.digest(root / 'plan.json'), 'frozen plan changed')
        check_sources(root, plan)
        return root
    plan = make_plan(args.profile, args.backend, args.namespace, args.fabric_dims,
                     args.fabric_offsets, args.host_timeout)
    root = Path(args.out or ('clock-campaign-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                            + '-' + uuid.uuid4().hex[:6])).absolute()
    common.require(root == root.resolve(), 'output may not traverse symlinks')
    root.mkdir(parents=True, exist_ok=False)
    for name in SOURCE_FILES:
        source = ROOT / name
        common.require(source.is_file() and not source.is_symlink(), 'regular campaign source')
        target = root / 'driver' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    plan.update(created_at=now(), campaign_id=str(uuid.uuid4()),
                source_sha256={name: common.digest(root / 'driver' / name) for name in SOURCE_FILES},
                client_timeout=args.client_timeout, local_compiler=args.compiler,
                local_python=args.worker_python,
                failure_policy='Continue independent entries after failure; never automatically retry any attempted entry.')
    atomic_json(root / 'plan.json', plan)
    state = dict(schema=1, status='pending', plan_sha256=common.digest(root / 'plan.json'),
                 builds={b['id']: dict(status='pending') for b in plan['builds']},
                 runs={r['id']: dict(status='pending') for r in plan['runs']})
    atomic_json(root / 'state.json', state)
    for name in ('builds', 'runs', 'conditions'):
        (root / name).mkdir()
    for entry in plan['runs']:
        atomic_json(root / 'conditions' / (entry['id'] + '.json'), dict(config=entry['config']))
    return root


def compile_command(root, plan, build):
    out = root / 'builds' / build['id']
    p = build['params']
    if plan['backend'] == 'local-sim':
        command = [sys.executable, str(root / 'driver/clock_local.py'), 'compile',
                   '--compiler', plan['local_compiler']]
    else:
        command = [sys.executable, str(root / 'driver/clock_appliance.py'), 'compile', plan['backend']]
        if plan.get('namespace'):
            command += ['--namespace', plan['namespace']]
        if plan.get('fabric_dims'):
            command += ['--fabric-dims', plan['fabric_dims']]
        command += ['--fabric-offsets', plan['fabric_offsets']]
    return command + ['--cols', str(p['C']), '--rows', str(p['R']), '--words', str(p['W']), '--out', str(out)]


def run_command_argv(root, plan, entry):
    build = root / 'builds' / entry['build'] / 'clock-build.json'
    if plan['backend'] == 'local-sim':
        command = [sys.executable, str(root / 'driver/clock_local.py'), 'run',
                   '--python', plan['local_python'], '--manifest', str(build)]
    else:
        command = [sys.executable, str(root / 'driver/clock_appliance.py'), 'run', plan['backend'],
                   '--artifact-record', str(build), '--host-timeout', str(plan['host_timeout'])]
    return command + ['--config-file', str(root / 'conditions' / (entry['id'] + '.json')),
                      '--out', str(root / 'runs' / entry['id'])]


def execute(command, log_path, timeout):
    """Bound the local client, not remote job lifetime. Never daemonize children."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('xb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            while child.poll() is None:
                if time.monotonic() - started >= timeout:
                    raise TimeoutError('client wall timeout; remote job termination unconfirmed')
                if log_path.stat().st_size > 16 << 20:
                    raise RuntimeError('client log exceeds 16 MiB; remote job termination unconfirmed')
                time.sleep(0.2)
            return child.returncode
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def validate_build(root, plan, build):
    folder = root / 'builds' / build['id']
    manifest = common.read_json(folder / 'clock-build.json')
    common.require(manifest['params'] == build['params'], 'planned build shape')
    if plan['backend'] == 'local-sim':
        from clock_local import inventory
        common.require(manifest['backend'] == 'local-sim', 'local build backend')
        common.require(common.read_json(folder / 'compile-status.json') ==
                       dict(status='complete', manifest_sha256=common.digest(folder / 'clock-build.json')),
                       'local compile completion')
        common.require(inventory(folder / 'build') == manifest['artifacts_sha256'], 'local compiled bytes')
        common.require(common.source_identity(root / 'driver') ==
                       {k: manifest[k] for k in ('source_sha256', 'host_sha256')}, 'local source identity')
    else:
        appliance.load_manifest(folder / 'clock-build.json', plan['backend'], root / 'driver')
    return manifest


def note_jobs(folder, role):
    try:
        return appliance.local_jobs(folder, role)
    except (OSError, ValueError):
        return []


def verify_bundle(root):
    """Verify retained bytes and the explicit omission declaration before replay."""
    root = Path(root).resolve()
    record = common.read_json(root / 'bundle.json')
    common.require(record['schema'] == 1 and record['mode'] == 'compact', 'compact bundle schema')
    files = record['files']
    common.require(isinstance(files, list) and len(files) <= 10000, 'bounded bundle inventory')
    names = set()
    total = 0
    for item in files:
        name = item['path']
        relative = Path(name)
        common.require(not relative.is_absolute() and str(relative) == name and
                       all(part not in ('.', '..') for part in relative.parts), 'bundle relative path')
        common.require(name not in names, 'unique bundle member')
        names.add(name)
        path = root / relative
        common.require(path.resolve() == path and path.is_file() and not path.is_symlink(),
                       'bundle regular member without symlinks')
        size = path.stat().st_size
        common.require(size == item['bytes'] and appliance.digest(path, 128 << 20) == item['sha256'],
                       'bundle retained bytes differ: ' + name)
        total += size
        common.require(total <= 2 << 30, 'bundle expanded bound')
    omitted = record['omitted_originals']
    common.require(isinstance(omitted, list) and len(omitted) == len(set(omitted)) and
                   len(omitted) <= 10000, 'bounded unique omitted inventory')
    for name in omitted:
        parts = Path(name).parts
        common.require(len(parts) == 3 and parts[0] == 'runs' and
                       parts[1].startswith('r') and parts[2] in ('capture', 'capture.tar.gz') and
                       str(Path(name)) == name and name not in names, 'omitted capture path')
    return record


def bundle(root, plan, *, compact_omissions=()):
    """Compact replay bundle; originals stay on disk, omissions are explicit."""
    paths, omitted = [], []
    admitted = set()
    for entry in plan['runs']:
        folder = root / 'runs' / entry['id']
        if not (folder / 'analysis.json').is_file():
            continue
        try:
            analyze_run(folder, root / 'builds' / entry['build'], entry, plan)
            admitted.add(entry['id'])
        except (ValueError, OSError, KeyError):
            if compact_omissions:
                from clock_report import read_compact_run
                read_compact_run(folder, root / 'builds' / entry['build'], entry, plan,
                                 omitted_paths=compact_omissions)
                admitted.add(entry['id'])
            # Otherwise keep original archives whose observations do not revalidate.
    for name in compact_omissions:
        if Path(name).parts[1] in admitted and not (root / name).exists():
            omitted.append(name)
    for name in ('plan.json', 'state.json', 'launcher.json', 'driver.log', 'summary.json',
                 'summary.md', 'pe.csv', 'markers.csv', 'host.csv', 'contrasts.json'):
        common.require(not (root / name).is_symlink(), 'bundle refuses symlinks')
        if (root / name).exists():
            paths.append(root / name)
    for directory in ('driver', 'conditions', 'builds', 'runs', 'client-logs'):
        base = root / directory
        common.require(not base.is_symlink(), 'bundle refuses symlinks')
        if not base.exists():
            continue
        stack = [base]
        while stack:
            current = stack.pop()
            for path in sorted(current.iterdir()):
                common.require(not path.is_symlink(), 'bundle refuses symlinks')
                if path.name == '__pycache__':
                    continue
                if path.is_dir():
                    stack.append(path)
                else:
                    common.require(path.is_file(), 'bundle only regular files')
                    if directory == 'runs' and path.parent.name.startswith('r') and path.name in ('capture', 'capture.tar.gz'):
                        if path.parent.name in admitted:
                            omitted.append(str(path.relative_to(root)))
                            continue
                    paths.append(path)
                common.require(len(paths) + len(stack) <= 10000, 'bounded bundle inventory')
    total = sum(path.stat().st_size for path in paths)
    common.require(total <= 2 << 30, 'bundle expanded bound 2 GiB')
    inventory = dict(schema=1, mode='compact', omitted_originals=omitted,
                     scope='Original capture archives remain on operator disk. Selected raw records replay independently; omitted archive hashes cannot be recomputed from this bundle.',
                     files=[dict(path=str(p.relative_to(root)), bytes=p.stat().st_size,
                                 sha256=appliance.digest(p, 128 << 20)) for p in paths])
    atomic_json(root / 'bundle.json', inventory)
    paths.append(root / 'bundle.json')
    target = root.parent / (root.name + '.zip')
    temporary = root.parent / (root.name + '.zip.tmp-' + uuid.uuid4().hex)
    common.require(not temporary.exists() and not target.is_symlink(), 'safe bundle target')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in paths:
                archive.write(path, root.name + '/' + str(path.relative_to(root)))
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def run_campaign(root, command_runner=execute, *, locked=False):
    root = Path(root).resolve()
    with nullcontext() if locked else output_lock(root):
        plan = common.read_json(root / 'plan.json')
        state = common.read_json(root / 'state.json')
        common.require(state['plan_sha256'] == common.digest(root / 'plan.json'), 'frozen plan changed')
        check_sources(root, plan)
        state.update(status='running', last_started_at=now())
        atomic_json(root / 'state.json', state)
        exit_code = 2
        interrupted = False
        try:
            for build in plan['builds']:
                slot = state['builds'][build['id']]
                folder = root / 'builds' / build['id']
                if slot['status'] in ('pass', 'started'):
                    try:
                        validate_build(root, plan, build)
                        slot.update(status='pass', recovered_or_revalidated_at=now())
                    except Exception as error:
                        slot.update(status='unknown' if slot['status'] == 'started' else 'incomplete',
                                    error=str(error), remote_stop_unconfirmed=True)
                if slot['status'] != 'pending':
                    atomic_json(root / 'state.json', state)
                    continue
                check_sources(root, plan)
                command = compile_command(root, plan, build)
                slot.update(status='started', started_at=now(), command=command)
                atomic_json(root / 'state.json', state)
                print('COMPILE', build['id'], flush=True)
                try:
                    code = command_runner(command, root / 'client-logs' / (build['id'] + '.log'), plan['client_timeout'])
                    slot['client_exit'] = code
                    common.require(code == 0, 'compile client returned ' + str(code))
                    validate_build(root, plan, build)
                    slot['status'] = 'pass'
                except KeyboardInterrupt:
                    slot.update(status='unknown', remote_stop_unconfirmed=True)
                    raise
                except Exception as error:
                    slot.update(status='incomplete', error=str(error), remote_stop_unconfirmed=True)
                    print('CONTINUE after compile failure:', build['id'], str(error), flush=True)
                finally:
                    slot.update(finished_at=now(), jobs=note_jobs(folder, 'compile'))
                    atomic_json(root / 'state.json', state)

            for number, entry in enumerate(plan['runs'], 1):
                slot = state['runs'][entry['id']]
                folder = root / 'runs' / entry['id']
                build_dir = root / 'builds' / entry['build']
                # An interrupted action is never resubmitted, even if no local files survived.
                if slot['status'] in ('pass', 'numeric_fail', 'started'):
                    previous_status = slot['status']
                    try:
                        analysis = analyze_run(folder, build_dir, entry, plan)
                        slot.update(status=analysis['status'], run_id=analysis['run_id'], revalidated_at=now())
                    except Exception as error:
                        slot.update(status='unknown' if previous_status == 'started' else 'incomplete',
                                    error=str(error), remote_stop_unconfirmed=True)
                if slot['status'] != 'pending':
                    atomic_json(root / 'state.json', state)
                    continue
                if state['builds'][entry['build']]['status'] != 'pass':
                    slot.update(status='skipped_dependency', reason='build did not validate', finished_at=now())
                    atomic_json(root / 'state.json', state)
                    continue
                check_sources(root, plan)
                common.require(common.read_json(root / 'conditions' / (entry['id'] + '.json')) ==
                               dict(config=entry['config']), 'condition changed')
                command = run_command_argv(root, plan, entry)
                slot.update(status='started', started_at=now(), command=command)
                atomic_json(root / 'state.json', state)
                print(f"RUN {number}/{len(plan['runs'])} {entry['id']} {entry['cohort']} {entry['arm']}", flush=True)
                try:
                    code = command_runner(command, root / 'client-logs' / (entry['id'] + '.log'), plan['client_timeout'])
                    slot['client_exit'] = code
                    common.require(code in (0, 1), 'run client returned ' + str(code))
                    analysis = analyze_run(folder, build_dir, entry, plan)
                    common.require(analysis['status'] == ('pass' if code == 0 else 'numeric_fail'),
                                   'client/validated result mismatch')
                    slot.update(status=analysis['status'], run_id=analysis['run_id'])
                    print('RESULT', entry['id'], slot['status'], flush=True)
                except KeyboardInterrupt:
                    slot.update(status='unknown', remote_stop_unconfirmed=True)
                    raise
                except Exception as error:
                    slot.update(status='incomplete', error=str(error), remote_stop_unconfirmed=True,
                                continued_after_failure_at=now())
                    print('CONTINUE after run failure:', entry['id'], str(error), flush=True)
                finally:
                    slot.update(finished_at=now(), jobs=note_jobs(folder, 'launch'))
                    atomic_json(root / 'state.json', state)
            failures = any(s['status'] != 'pass' for group in ('builds', 'runs') for s in state[group].values())
            state['status'] = 'completed_with_failures' if failures else 'complete'
            exit_code = 2 if failures else 0
        except KeyboardInterrupt:
            interrupted = True
            state['status'], exit_code = 'interrupted', 130
        except Exception as error:
            state.update(status='incomplete', error=str(error))
        finally:
            state.update(finished_at=now(), exit_code=exit_code)
            atomic_json(root / 'state.json', state)
            try:
                summary = write_report(root, plan, state)
                if summary.get('analysis_errors') and not interrupted:
                    state['status'], state['exit_code'], exit_code = 'completed_with_failures', 2, 2
                    atomic_json(root / 'state.json', state)
            except Exception as error:
                state['report_error'] = str(error)
                if not interrupted:
                    state['status'], state['exit_code'], exit_code = 'completed_with_failures', 2, 2
                atomic_json(root / 'state.json', state)
            try:
                archive = bundle(root, plan)
                print('BUNDLE', archive, flush=True)
            except Exception as error:
                state['bundle_error'] = str(error)
                if not interrupted:
                    state['status'], state['exit_code'], exit_code = 'completed_with_failures', 2, 2
                atomic_json(root / 'state.json', state)
                print('BUNDLE ERROR', str(error), file=sys.stderr, flush=True)
        return exit_code


def tmux_command(socket, *args):
    return ['tmux', '-L', socket, '-f', '/dev/null', *args]


def launch_tmux(root):
    common.require(shutil.which('tmux') is not None, 'tmux is required on the operator node')
    with output_lock(root, '.launch.lock'):
        with output_lock(root):
            pass
        existing = root / 'launcher.json'
        if existing.exists():
            old = common.read_json(existing)
            if old.get('socket') and subprocess.run(tmux_command(old['socket'], 'has-session', '-t', '=clock'),
                                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                raise ValueError('the existing campaign tmux session is still present')
        socket = 'clock-' + uuid.uuid4().hex[:16]
        command = [sys.executable, str(root / 'driver/clock_campaign.py'), '--foreground', '--resume', str(root)]
        request = dict(status='starting', created_at=now(), socket=socket, session='clock',
                       command=command, cwd=str(root / 'driver'), output=str(root))
        atomic_json(existing, request)
        result = subprocess.run(tmux_command(socket, 'new-session', '-d', '-s', 'clock', '-c', str(root / 'driver'),
                                             '/bin/sh', '-c', 'exec ' + shlex.join(command)), capture_output=True, text=True)
        if result.returncode:
            atomic_json(existing, dict(request, status='failed', error=result.stderr.strip(), exit_code=result.returncode))
            raise ValueError('tmux launch failed: ' + result.stderr.strip())
    print('OUTPUT', root)
    print('ATTACH', shlex.join(tmux_command(socket, 'attach-session', '-t', '=clock')))
    print('LOG', shlex.join(['tail', '-f', str(root / 'driver.log')]))
    print('STATUS', shlex.join(['cat', str(root / 'state.json')]))
    print('The tmux session exits at completion; status, logs and ZIP remain.')
    return 0


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def foreground(root):
    # Hold the same lock through launcher, run state, report and final archive.
    # A duplicate foreground controller must not modify even launcher metadata.
    with output_lock(root):
        return _foreground_locked(root)


def _foreground_locked(root):
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupted)
    launch = common.read_json(root / 'launcher.json') if (root / 'launcher.json').exists() else {}
    launch.update(status='running', started_at=now(), pid=os.getpid())
    atomic_json(root / 'launcher.json', launch)
    code = 2
    try:
        with (root / 'driver.log').open('a', buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                code = run_campaign(root, locked=True)
    finally:
        signal.signal(signal.SIGTERM, previous)
        launch.update(status='complete' if code == 0 else 'interrupted' if code == 130 else 'completed_with_failures',
                      exit_code=code, finished_at=now())
        atomic_json(root / 'launcher.json', launch)
        # Final launcher outcome must be included in the portable package too.
        try:
            bundle(root, common.read_json(root / 'plan.json'))
        except Exception as error:
            if code != 130:
                code = 2
            launch.update(status='interrupted' if code == 130 else 'completed_with_failures',
                          exit_code=code, bundle_error=str(error))
            atomic_json(root / 'launcher.json', launch)
            state = common.read_json(root / 'state.json')
            state.update(status=launch['status'], exit_code=code, bundle_error=str(error))
            atomic_json(root / 'state.json', state)
            print('final bundle:', str(error), file=sys.stderr)
    return code


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--profile', choices=('standard', 'smoke'), default='standard')
    result.add_argument('--backend', choices=('hardware', 'simulator', 'local-sim'), default='hardware')
    result.add_argument('--out', type=Path, help='fresh output directory; default timestamp plus unique suffix')
    result.add_argument('--resume', type=Path, help='continue only entries never attempted; do not retry failures')
    result.add_argument('--namespace')
    result.add_argument('--fabric-dims')
    result.add_argument('--fabric-offsets', default='4,1')
    result.add_argument('--host-timeout', type=int, default=300)
    result.add_argument('--client-timeout', type=int, default=1200,
                        help='per compile/run client wall timeout, including queue wait; remote stop remains unconfirmed')
    result.add_argument('--compiler', default='cslc', help='local-sim only')
    result.add_argument('--worker-python', default='cs_python', help='local-sim only')
    result.add_argument('--foreground', action='store_true', help='run here (for tests or an existing supervisor)')
    result.add_argument('--plan-only', action='store_true', help='save plan/source without tmux or SDK submission')
    result.add_argument('--summarize', type=Path, help='revalidate completed records and rebuild report/ZIP offline')
    return result


def enter_frozen(root, mode):
    """An old campaign always executes its own checked driver after repo updates."""
    if ROOT == root / 'driver':
        return
    plan = common.read_json(root / 'plan.json')
    state = common.read_json(root / 'state.json')
    common.require(state['plan_sha256'] == common.digest(root / 'plan.json'), 'frozen plan changed')
    check_sources(root, plan)
    command = [sys.executable, str(root / 'driver/clock_campaign.py'), mode, str(root)]
    if mode == '--resume':
        command.append('--foreground')
    os.execv(sys.executable, command)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        common.require(1 <= args.client_timeout <= 86400, 'client timeout bound')
        common.require(not (args.resume and args.out), '--resume and --out are mutually exclusive')
        if args.summarize:
            root = args.summarize.resolve()
            enter_frozen(root, '--summarize')
            with output_lock(root):
                plan, state = common.read_json(root / 'plan.json'), common.read_json(root / 'state.json')
                common.require(state['plan_sha256'] == common.digest(root / 'plan.json'), 'frozen plan changed')
                check_sources(root, plan)
                inventory = verify_bundle(root) if (root / 'bundle.json').exists() else None
                omissions = inventory['omitted_originals'] if inventory else ()
                errors = []
                for entry in plan['runs']:
                    if state['runs'][entry['id']]['status'] in ('pass', 'numeric_fail'):
                        try:
                            if omissions:
                                from clock_report import read_compact_run
                                read_compact_run(root / 'runs' / entry['id'], root / 'builds' / entry['build'],
                                                 entry, plan, omitted_paths=omissions)
                            else:
                                analyze_run(root / 'runs' / entry['id'], root / 'builds' / entry['build'], entry, plan)
                        except Exception as error:
                            errors.append(dict(id=entry['id'], error=str(error)))
                common.require(not errors, 'offline revalidation failed: ' + str(errors))
                summary = write_report(root, plan, state)
                print('BUNDLE', bundle(root, plan, compact_omissions=omissions))
                return 2 if summary.get('analysis_errors') or state['status'] != 'complete' else 0

        root = prepare(args)
        plan = common.read_json(root / 'plan.json')
        estimate = plan['estimates']
        print(f"PLAN {len(plan['builds'])} builds, {len(plan['runs'])} fresh runtimes; hardware estimate "
              f"{estimate['hardware_wall_seconds_low']/60:.0f}–{estimate['hardware_wall_seconds_high']/60:.0f} min plus queue variability.", flush=True)
        if args.plan_only:
            print('PLAN ONLY', root)
            return 0
        if args.foreground:
            enter_frozen(root, '--resume')
            return foreground(root)
        return launch_tmux(root)
    except (ValueError, OSError, KeyError) as error:
        print('error:', str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
