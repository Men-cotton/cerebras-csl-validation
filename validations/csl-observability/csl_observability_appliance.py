#!/usr/bin/env python3
"""One registered compile/run/audit action for two bounded observation probes."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import sys
import time
import types
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
FAMILY = 'csl-observability'
ENTRY = 'csl_observability_appliance.py'
PACKAGE_LIMIT, ARTIFACT_LIMIT, CAPTURE_LIMIT = 8 << 20, 64 << 20, 32 << 20
SDK_WORK_LIMIT = 128 << 20
SUPPORT = ('hardware_entry.py', 'appliance.py', 'saved_audit.py', 'portable.py')
CASES = {'queue-n32': {'N': 32}, 'arena-cap64': {'CAP': 64}}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path, limit=PACKAGE_LIMIT):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, 'bounded regular file')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    sha(path, 4 << 20)
    return json.loads(Path(path).read_text())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def support(root):
    """Reuse frozen E233 helpers under a path-specific module namespace."""
    name = '_observability_'+hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:16]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(root)/'support')]
        sys.modules[name] = package
    portable, adapter, hardware = (importlib.import_module(name+'.'+part)
                                   for part in ('portable', 'appliance', 'hardware_entry'))
    common = types.SimpleNamespace(p=portable, compiled_inventory=portable.compiled_inventory,
                                   admit_compiled_inventory=portable.admit_compiled_inventory)
    return portable, adapter, hardware, common


def bounded_tree(folder, limit):
    pending, count, total, began = [Path(folder)], 0, 0, time.monotonic()
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                count += 1
                need(count <= 4000 and time.monotonic()-began <= 10, 'tree entry/time bound')
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                else:
                    need(stat.S_ISREG(info.st_mode), 'regular tree members without links')
                    total += info.st_size
                    need(total <= limit, 'tree byte bound')
    return dict(entries=count, bytes=total)


def fresh_sdk_workdir(capture, run_id):
    capture = Path(capture).absolute()
    need(capture == capture.resolve() and str(uuid.UUID(run_id)) == run_id, 'SDK work directory owner')
    path = capture.parent/('sdk-work-'+run_id)
    need(path == path.resolve() and not path.is_relative_to(capture)
         and not capture.is_relative_to(path), 'SDK work directory outside whole capture')
    path.mkdir(exist_ok=False)
    return path


def remove_sdk_workdir(path):
    """Remove only a successfully accounted directory created by this worker."""
    result = dict(directory=path.name, limit_bytes=SDK_WORK_LIMIT, removed=False)
    try:
        need(path == path.resolve() and path.is_dir() and not path.is_symlink(), 'owned SDK work directory')
        result.update(bounded_tree(path, SDK_WORK_LIMIT), accounting='complete')
        shutil.rmtree(path)
        result['removed'] = True
    except Exception as error:
        result.update(error=repr(error))
    return result


def driver(root, case):
    folder = Path(root)/'cases'/case
    path = folder/('source/driver.py' if case == 'arena-cap64' else 'run.py')
    return folder, load(path, '_observability_driver_'+case.replace('-', '_'))


def identity(root, case):
    root = Path(root).resolve()
    need(case in CASES, 'registered case')
    package = read(root/'package.json')
    need(package.get('schema') == 1 and package.get('family') == FAMILY
         and set(package['cases']) == set(CASES), 'two-case package contract')
    need(0 < len(package['files']) <= 64, 'package entry bound')
    total = 0
    for name, pin in package['files'].items():
        path = Path(name)
        need(not path.is_absolute() and '..' not in path.parts and path.as_posix() == name
             and (root/path).resolve() == root/path, 'safe package path')
        need(sha(root/path) == pin, 'frozen package source '+name)
        total += (root/path).stat().st_size
        need(total <= PACKAGE_LIMIT, 'package byte bound')
    required = {ENTRY} | {'support/'+name for name in SUPPORT}
    need(required <= set(package['files']), 'complete adapter dependencies')
    folder, implementation = driver(root, case)
    manifest, _ = implementation.read_case(folder)
    need(manifest['params'] == CASES[case] and package['cases'][case] == sha(folder/'case.json'), 'frozen case parameters')
    return dict(package_manifest_sha256=sha(root/'package.json'), case=case,
                case_sha256=sha(folder/'case.json'), params=CASES[case],
                csl_sha256={name: sha(folder/'source'/name) for name in ('layout.csl', 'pe.csl')},
                adapter_sha256=sha(root/ENTRY), support_sha256={name: sha(root/'support'/name) for name in SUPPORT})


def controls(case, profile, delay):
    need(type(profile) is int and type(delay) is int, 'integer runtime controls')
    need((case == 'queue-n32' and (profile, delay) in ((1, 0), (1, 256), (0, 256)))
         or (case == 'arena-cap64' and (profile, delay) == (0, 0)), 'registered bounded runtime arm')
    return dict(profile=profile, delay=delay)


def archive_program(root, artifact, pinned):
    _, _, hardware, common = support(root)
    sha(artifact, ARTIFACT_LIMIT)
    return hardware.program_archive(artifact, pinned['csl_sha256'], common)


def actual_program(root, build, expected):
    _, _, hardware, common = support(root)
    bounded_tree(build, ARTIFACT_LIMIT)
    return hardware.bind_actual_program(build, {'program_inventory': expected}, common)


def compile_job(args, root=ROOT, compiler_type=None):
    pinned = identity(root, args.case)
    portable, adapter, _, _ = support(root)
    output = adapter.fresh(args.out)
    record = dict(schema=1, family=FAMILY, action='compile', backend='hardware', status='RUNNING',
                  compile_id=str(uuid.uuid4()), namespace=args.namespace, **pinned,
                  package_archive_sha256=sha(Path(root)/'package.tar.gz'),
                  compiler_args=portable.compiler_arguments(pinned['params'], 'hardware', 1))
    portable.write_json(output/'request.json', record)
    began = time.monotonic()
    try:
        with adapter.client_log(output) as log:
            cls = compiler_type or adapter.sdk_classes()[0]
            with cls(**adapter.sdk_options(args.namespace, 1)) as compiler:
                returned = Path(compiler.compile(str(Path(root)/'cases'/args.case/'source'), 'layout.csl',
                                                record['compiler_args'], str(output)))
                need(not returned.is_symlink(), 'compiler artifact symlink')
                artifact = returned.resolve()
        need(artifact.parent == output and artifact.name not in {'request.json', 'build.json', 'client.log'}, 'compiler artifact location')
        need(not log.truncated and identity(root, args.case) == pinned, 'complete log and unchanged source')
        record.update(status='PASS', artifact_path=artifact.name, artifact_sha256=sha(artifact, ARTIFACT_LIMIT),
                      program_inventory=archive_program(root, artifact, pinned))
    except BaseException as error:
        record.update(status='FAILED', error=repr(error))
        raise
    finally:
        record['compile_wall_seconds'] = time.monotonic()-began
        portable.write_json(output/'build.json', record)
    return record


def check_build(root, record, case):
    portable, _, _, common = support(root)
    need(record.get('status') == 'PASS' and record.get('action') == 'compile'
         and record.get('family') == FAMILY and record.get('backend') == 'hardware', 'successful hardware compile')
    pinned = identity(root, case)
    need(all(record.get(key) == value for key, value in pinned.items()), 'source/case/compile binding')
    need(record['compiler_args'] == portable.compiler_arguments(pinned['params'], 'hardware', 1), 'exact compiler flags')
    common.admit_compiled_inventory(record['program_inventory']['elf'])
    return pinned


def load_build(root, path, case):
    record = read(path)
    check_build(root, record, case)
    need(sha(Path(root)/'package.tar.gz') == record['package_archive_sha256'], 'package archive binding')
    name = record['artifact_path']
    need(Path(name).name == name and name not in ('', '.', '..'), 'artifact filename')
    artifact = Path(path).resolve().parent/name
    need(sha(artifact, ARTIFACT_LIMIT) == record['artifact_sha256'], 'compiler artifact identity')
    need(archive_program(root, artifact, record) == record['program_inventory'], 'archive program binding')
    return record, artifact


def runtime_binding(root, case, runtime, inventory):
    folder, implementation = driver(root, case)
    replay = implementation.audit(runtime)
    need(replay['status'] == 'PASS' and replay['backend'] == 'hardware', 'independent hardware replay')
    expected = {name: item['sha256'] for name, item in inventory['elf']['files'].items()}
    if case == 'arena-cap64':
        expected.update(inventory['metadata_sha256'])
        observed = {name: item['sha256'] for name, item in read(runtime/'program.json').items()}
        need(sha(runtime/'case/case.json') == sha(folder/'case.json'), 'runtime arena case binding')
    else:
        expected.update({name: inventory['metadata_sha256'][name] for name in ('out.json', 'bin/out_rpc.json')})
        observed = read(runtime/'build.json')['files']
        need(sha(runtime/'sources/case.json') == sha(folder/'case.json'), 'runtime queue case binding')
    need(observed == expected, 'runtime program matches compiler archive')
    return replay


def audit_capture(root, case, profile, delay, capture):
    control = controls(case, profile, delay)
    capture = Path(capture)
    bounded_tree(capture, CAPTURE_LIMIT)
    record, build, request = (read(capture/name) for name in ('worker.json', 'build.json', 'run-request.json'))
    check_build(root, build, case)
    # The independently retained launcher copies anchor returned metadata to
    # this particular run, including its compile artifact and fresh run UUID.
    need(sha(capture/'build.json') == sha(capture.parent/'build.json')
         and sha(capture/'run-request.json') == sha(capture.parent/'run-request.json'), 'returned metadata matches staged run')
    need(record['status'] == 'PASS' and record['backend'] == 'hardware', 'completed worker')
    scratch = record['sdk_workdir']
    need(scratch['directory'] == 'sdk-work-'+request['run_id']
         and scratch['accounting'] == 'complete' and scratch['removed'] is True
         and scratch['limit_bytes'] == SDK_WORK_LIMIT and type(scratch['bytes']) is int
         and 0 <= scratch['bytes'] <= SDK_WORK_LIMIT and type(scratch['entries']) is int
         and 0 <= scratch['entries'] <= 4000, 'bounded SDK scratch removed after capture exit')
    need(request['case'] == case and all(request[key] == value for key, value in control.items()), 'requested runtime arm')
    need(record['build_record_sha256'] == request['build_record_sha256'] == sha(capture/'build.json')
         and record['request_sha256'] == sha(capture/'run-request.json')
         and record['run_id'] == request['run_id'], 'returned run/compile bindings')
    need(record['actual_inventory_sha256'] == sha(capture/'actual-build-inventory.json'), 'actual program identity')
    actual = read(capture/'actual-build-inventory.json')
    need(actual == build['program_inventory'], 'worker actual program matches compiler archive')
    replay = runtime_binding(root, case, capture/'runtime', actual)
    if case == 'queue-n32':
        need(replay['profile'] == profile and replay['delay'] == delay, 'captured queue runtime arm')
    return dict(status='PASS', backend='hardware', case=case, **control, run_id=request['run_id'],
                worker_sha256=sha(capture/'worker.json'), replay=replay)


def worker(args):
    """Launcher supplies the endpoint and the already unpacked compiler artifact."""
    need(args.cmaddr, 'registered launcher endpoint required')
    build, request = read(args.build_record), read(args.request)
    need(str(uuid.UUID(request['run_id'])) == request['run_id'], 'canonical fresh run ID')
    need(sha(args.package) == build['package_archive_sha256']
         and sha(args.build_record) == request['build_record_sha256'], 'worker staged identity')
    need(sha(__file__) == build['adapter_sha256'], 'staged worker source')
    bootstrap = Path(__file__).resolve().parent/'hardware_entry.py'
    need(sha(bootstrap) == build['support_sha256']['hardware_entry.py'], 'staged extraction helper')
    helper = load(bootstrap, '_observability_bootstrap')
    output = Path(args.out).absolute()
    need(output == output.resolve(), 'worker output path')
    output.mkdir(parents=True, exist_ok=False)
    root = output.parent/('frozen-observation-'+request['run_id'])
    record = dict(status='RUNNING', backend='hardware', run_id=request['run_id'],
                  build_record_sha256=sha(args.build_record), request_sha256=sha(args.request))
    helper.write(output/'worker.json', record)
    shutil.copyfile(args.build_record, output/'build.json')
    shutil.copyfile(args.request, output/'run-request.json')
    try:
        helper.unpack(args.package, root, PACKAGE_LIMIT)
        pinned = check_build(root, build, request['case'])
        controls(request['case'], request['profile'], request['delay'])
        need(pinned['adapter_sha256'] == sha(__file__), 'unpacked adapter identity')
        actual = actual_program(root, Path(args.build).resolve(), build['program_inventory'])
        helper.write(output/'actual-build-inventory.json', actual)
        folder, implementation = driver(root, request['case'])
        sdk_workdir = fresh_sdk_workdir(output, request['run_id'])
        try:
            if request['case'] == 'arena-cap64':
                implementation.capture(folder, Path(args.build).resolve(), output/'runtime', 'hardware', args.cmaddr,
                                       sdk_workdir=sdk_workdir)
            else:
                implementation.capture(folder, output/'runtime', request['profile'], request['delay'],
                                       build=Path(args.build).resolve(), backend='hardware', cmaddr=args.cmaddr,
                                       registered_worker=True, trace=False, sdk_workdir=sdk_workdir)
        finally:
            # Both drivers restore cwd and attempt runtime.stop() before exit.
            # Failed numerical/runtime work keeps its original exception and raw evidence.
            record['sdk_workdir'] = remove_sdk_workdir(sdk_workdir)
        need(record['sdk_workdir']['removed'], 'SDK scratch accounting or cleanup failed')
        record.update(status='PASS', actual_inventory_sha256=sha(output/'actual-build-inventory.json'))
        helper.write(output/'worker.json', record)
        audit_capture(root, request['case'], request['profile'], request['delay'], output)
    except BaseException as error:
        record.update(status='FAILED', error=repr(error))
        raise
    finally:
        helper.write(output/'worker.json', record)
    return record


def run_job(args, root=ROOT, launcher_type=None):
    control = controls(args.case, args.profile, args.delay)
    need(type(args.host_timeout) is int and 1 <= args.host_timeout <= 600, 'worker timeout at most 600 seconds')
    build, artifact = load_build(root, args.build_record, args.case)
    portable, adapter, helper, _ = support(root)
    output = adapter.fresh(args.out)
    shutil.copyfile(args.build_record, output/'build.json')
    request = dict(schema=1, run_id=str(uuid.uuid4()), case=args.case, **control,
                   build_record_sha256=sha(output/'build.json'))
    portable.write_json(output/'run-request.json', request)
    command = ['cs_python', ENTRY, 'worker', '--package', 'package.tar.gz', '--build-record', 'build.json',
               '--request', 'run-request.json', '--build', 'out', '--out', 'capture', '--cmaddr', '%CMADDR%']
    shell = (f'if timeout --signal=INT --kill-after=15s {args.host_timeout}s {shlex.join(command)} > host.log 2>&1; '
             'then code=0; else code=$?; fi; printf \'%s\\n\' "$code" > host-exit.txt')
    summary = dict(status='RUNNING', command=shell, run_id=request['run_id'], downloads={}, errors={})
    portable.write_json(output/'collection.json', summary)
    pending = None
    try:
        with adapter.client_log(output) as log:
            cls = launcher_type or adapter.sdk_classes()[1]
            with cls(str(artifact), simulator=False, **adapter.sdk_options(build['namespace'])) as launcher:
                try:
                    for path in (Path(root)/ENTRY, Path(root)/'support/hardware_entry.py', Path(root)/'package.tar.gz',
                                 output/'build.json', output/'run-request.json'):
                        launcher.stage(str(path))
                    load_build(root, args.build_record, args.case)
                    need(sha(output/'build.json') == request['build_record_sha256']
                         and read(output/'run-request.json') == request, 'staged request unchanged')
                    response = str(launcher.run(shell) or '')
                    need(len(response.encode()) <= 16 << 20, 'bounded launcher response')
                    (output/'launcher-response.txt').write_text(response)
                except BaseException as error:
                    pending = error
                    summary['launch_error'] = repr(error)
                finally:
                    for name, bound in (('host.log', 16 << 20), ('host-exit.txt', 128), ('capture', CAPTURE_LIMIT)):
                        try:
                            path = adapter.downloaded_path(launcher.download_artifact(name, str(output/name)), output/name, bound)
                            need(path.name not in {row['path'] for row in summary['downloads'].values()}, 'download alias')
                            summary['downloads'][name] = dict(path=path.name, sha256=sha(path, bound), retained=True)
                        except BaseException as error:
                            summary['errors'][name] = repr(error)
        need(not log.truncated, 'complete appliance job-ID log')
        if pending is not None:
            raise RuntimeError('registered run failed; partial downloads retained') from pending
        need(not summary['errors'], 'complete collection')
        need((output/summary['downloads']['host-exit.txt']['path']).read_text().strip() == '0', 'worker success')
        extracted = helper.unpack(output/summary['downloads']['capture']['path'], output/'capture-extracted', CAPTURE_LIMIT)
        need((extracted/'capture').is_dir() and {p.name for p in extracted.iterdir()} == {'capture'}, 'single capture prefix')
        (extracted/'capture').rename(output/'capture-data')
        extracted.rmdir()
        summary['audit'] = audit_capture(root, args.case, args.profile, args.delay, output/'capture-data')
        summary['status'] = 'PASS'
        (output/summary['downloads']['capture']['path']).unlink()
        summary['downloads']['capture']['retained'] = False
    except BaseException as error:
        summary.update(status='FAILED', error=repr(error))
        raise
    finally:
        portable.write_json(output/'collection.json', summary)
    return summary


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('compile', 'run', 'audit', 'worker'))
    parser.add_argument('--case', choices=tuple(CASES))
    parser.add_argument('--profile', type=int, default=0)
    parser.add_argument('--delay', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--namespace', default='job-operator')
    parser.add_argument('--host-timeout', type=int, default=600)
    for flag in ('build-record', 'capture', 'package', 'build', 'request'):
        parser.add_argument('--'+flag, type=Path)
    parser.add_argument('--cmaddr')
    args = parser.parse_args()
    if args.action == 'compile': result = compile_job(args)
    elif args.action == 'run': result = run_job(args)
    elif args.action == 'worker': result = worker(args)
    else:
        need(not args.out.exists(), 'fresh audit output')
        result = audit_capture(ROOT, args.case, args.profile, args.delay, args.capture)
        support(ROOT)[0].write_json(args.out, result)
    print(json.dumps({key: result[key] for key in ('status', 'backend', 'run_id') if key in result}))


if __name__ == '__main__':
    main()
