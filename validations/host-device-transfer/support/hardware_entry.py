"""Registered compile/run/audit entrypoint; no controller or automatic submission."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import sys
import tarfile
import time
import uuid

ROOT = Path(__file__).resolve().parent
FAMILY = 'native-cora-observation'
PREFIX = 'models/hierarchical_gnn/grouped_csr/native_cora_appliance'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path, limit=256 << 20):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, 'bounded regular file')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    require(Path(path).stat().st_size <= 16 << 20, 'bounded JSON')
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def unpack(archive, output, limit=256 << 20):
    """Bound time, members, logical bytes and path types before extracting."""
    require(Path(archive).stat().st_size <= limit, 'bounded archive')
    output = Path(output).absolute()
    require(output == output.resolve(), 'extraction path')
    output.mkdir(parents=True, exist_ok=False)
    seen, total, began = set(), 0, time.monotonic()
    with tarfile.open(archive, 'r:*') as stream:
        members = []
        for member in stream:
            name = PurePosixPath(member.name)
            require(member.isfile() or member.isdir(), 'regular archive members only')
            require(not name.is_absolute() and '..' not in name.parts and str(name) not in ('', '.')
                    and member.name not in seen, 'unique safe archive path')
            seen.add(member.name)
            total += member.size
            require(len(seen) <= 2000 and total <= limit and time.monotonic()-began <= 60, 'archive accounting bound')
            members.append(member)
        for member in members:
            target = output/member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, target.open('xb') as destination:
                    shutil.copyfileobj(source, destination, 1 << 20)
            require(time.monotonic()-began <= 60, 'bounded extraction time')
    return output


def modules(root):
    sys.path.insert(0, str(root))
    from models.hierarchical_gnn.grouped_csr.native_cora_appliance import saved_audit, appliance
    from models.hierarchical_gnn.grouped_csr.native_cora_appliance.observation import run
    return saved_audit, appliance, run


def program_tree(build, common):
    """Generated ELF inventory plus all generated application/helper RPC metadata."""
    build = Path(build).absolute()
    inventory = common.compiled_inventory(build)
    common.admit_compiled_inventory(inventory)
    began, count, total, metadata = time.monotonic(), 0, 0, {}
    def failed(error):
        raise error
    for folder, dirs, files in os.walk(build, followlinks=False, onerror=failed):
        for name in dirs+files:
            count += 1
            require(count <= 20000 and not (Path(folder)/name).is_symlink()
                    and time.monotonic()-began <= 45, 'bounded program metadata traversal')
        for name in files:
            if name not in ('out.json', 'out_rpc.json'):
                continue
            path = Path(folder)/name
            total += path.stat().st_size
            require(total <= 16 << 20, 'generated metadata byte bound')
            metadata[path.relative_to(build).as_posix()] = sha(path, 2 << 20)
    require({'out.json', 'west/out.json', 'east/out.json'} <= set(metadata), 'generated I/O metadata required')
    return dict(elf=inventory, metadata_sha256=metadata)


def program_archive(path, sources, common):
    """Bind returned compiler archive source, generated ELF bytes and metadata.

    This follows shah-resident-cora/archive.py's generated-program extraction;
    only generated kernel ELF sections are inspected, never SDK binaries.
    """
    sha(path, 256 << 20)
    require(set(sources) == {'layout.csl', 'pe.csl'}, 'complete CSL source pins')
    began, total, seen, elfs, metadata, csl = time.monotonic(), 0, set(), {}, {}, {}
    with tarfile.open(path, 'r:*') as archive:
        for count, member in enumerate(archive):
            name = PurePosixPath(member.name)
            require(name.parts and not name.is_absolute() and '..' not in name.parts
                    and str(name) == member.name.rstrip('/'), 'safe compiler member')
            require(str(name) not in seen and (member.isdir() or member.isfile()), 'unique regular compiler member')
            seen.add(str(name))
            total += member.size
            require(count < 20000 and 0 <= member.size and total <= 256 << 20
                    and time.monotonic()-began <= 45, 'compiler archive accounting bound')
            if not member.isfile():
                continue
            selected = name.suffix == '.elf' or name.name in ('out.json', 'out_rpc.json')
            source = len(name.parts) >= 2 and name.parts[-2] == 'csl' and name.name in sources
            if not (selected or source):
                continue
            require(member.size <= (32 << 20 if name.suffix == '.elf' else 2 << 20), 'compiler selected-member bound')
            with archive.extractfile(member) as stream:
                payload = stream.read(member.size+1)
            require(len(payload) == member.size and time.monotonic()-began <= 45, 'complete compiler member')
            digest = hashlib.sha256(payload).hexdigest()
            if name.suffix == '.elf':
                elfs[str(name)] = dict(sha256=digest, bytes=len(payload), **common.p._elf_storage(payload))
            elif selected:
                metadata[str(name)] = digest
            if source:
                require(name.name not in csl and digest == sources[name.name], 'compiled CSL differs from source pins')
                csl[name.name] = digest
    roots = {str(PurePosixPath(name).parent.parent) for name in metadata
             if PurePosixPath(name).parts[-2:] == ('west', 'out.json')}
    roots = {root for root in roots if root+'/east/out.json' in metadata and root+'/out.json' in metadata}
    require(len(roots) == 1 and set(csl) == set(sources) and elfs, 'unique complete generated program and CSL')
    prefix = next(iter(roots))+'/'
    require(all(name.startswith(prefix) for name in (*elfs, *metadata)), 'single generated program tree')
    files = {name[len(prefix):]: item for name, item in elfs.items()}
    inventory = dict(files=files, artifact_bytes=sum(item['bytes'] for item in files.values()),
                     max_static_sram_end=max(item['static_sram_end'] for item in files.values()),
                     scope='worker-observed generated ELF hashes and allocated low-address sections before launch; stack/runtime high-water unmeasured')
    common.admit_compiled_inventory(inventory)
    return dict(elf=inventory, metadata_sha256={name[len(prefix):]: digest for name, digest in metadata.items()})


def bind_actual_program(build, record, common):
    actual = program_tree(build, common)
    require(actual == record['program_inventory'], 'unpacked program differs from compiler archive')
    return actual


def identity(root, label, profile):
    require(re.fullmatch('[a-z][a-z0-9-]{0,63}', label) and profile in (0, 1), 'case/profile')
    package = read(root/'package.json')
    require(package.get('schema') == 1 and package.get('family') == FAMILY, 'observer package')
    require(len(package['files']) < 1000, 'package inventory bound')
    total = 0
    for name, digest in package['files'].items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts, 'safe source path')
        require(sha(root/name) == digest, 'frozen package file '+name)
        total += (root/name).stat().st_size
        require(total < 128 << 20, 'package size bound')
    common, _, observer = modules(root)
    case = root/'cases'/label
    native = common.load(case)
    require(package['cases'][label]['case_sha256'] == sha(case/'case.json'), 'package case identity')
    require(native['admission']['forecast_bytes']+132 < 44*1024, 'observer forecast')
    manifest = observer.manifest_for(case, profile)
    return dict(package_manifest_sha256=sha(root/'package.json'), case=label, profile=profile,
                case_sha256=sha(case/'case.json'), params=manifest['params'],
                warmup=package['warmup'], repeats=package['repeats'],
                csl_sha256={name: sha(root/PREFIX/'observation'/name) for name in ('layout.csl', 'pe.csl')})


def compile_job(args, root=ROOT, compiler_type=None):
    pinned = identity(root, args.case, args.profile)
    common, adapter, _ = modules(root)
    output = adapter.fresh(args.out)
    record = dict(schema=1, family=FAMILY, action='compile', backend='hardware', status='RUNNING',
                  compile_id=str(uuid.uuid4()), namespace=args.namespace, **pinned,
                  package_archive_sha256=sha(root/'package.tar.gz'),
                  compiler_args=common.compiler_arguments(pinned['params'], 'hardware', 1))
    write(output/'request.json', record)
    began = time.monotonic()
    try:
        with adapter.client_log(output) as log:
            cls = compiler_type or adapter.sdk_classes()[0]
            with cls(**adapter.sdk_options(args.namespace, 1)) as compiler:
                returned = Path(compiler.compile(str(root/PREFIX/'observation'), 'layout.csl',
                                                record['compiler_args'], str(output)))
                require(not returned.is_symlink(), 'artifact symlink')
                artifact = returned.resolve()
        require(artifact.parent == output and artifact.name not in ('request.json', 'build.json', 'client.log'), 'artifact location')
        require(not log.truncated and pinned == identity(root, args.case, args.profile), 'source unchanged and complete compiler log')
        record.update(status='PASS', artifact_path=artifact.name, artifact_sha256=sha(artifact, 1 << 30),
                      program_inventory=program_archive(artifact, pinned['csl_sha256'], common))
    except BaseException as error:
        record.update(status='FAILED', error=repr(error))
        raise
    finally:
        record['compile_wall_seconds'] = time.monotonic()-began
        write(output/'build.json', record)
    return record


def load_build(root, path, label, profile):
    record = read(path)
    require(record.get('status') == 'PASS' and record.get('action') == 'compile'
            and record.get('family') == FAMILY and record.get('backend') == 'hardware', 'successful hardware compile')
    require(all(record.get(k) == v for k, v in identity(root, label, profile).items()), 'source/case/compile join')
    common, _, _ = modules(root)
    require(record['compiler_args'] == common.compiler_arguments(record['params'], 'hardware', 1), 'exact compiler flags')
    require(sha(root/'package.tar.gz') == record['package_archive_sha256'], 'staged package binding')
    name = record['artifact_path']
    require(Path(name).name == name and name not in ('', '.', '..'), 'artifact filename')
    artifact = Path(path).resolve().parent/name
    require(sha(artifact, 1 << 30) == record['artifact_sha256'], 'compiled artifact binding')
    require(program_archive(artifact, record['csl_sha256'], common) == record['program_inventory'], 'compiler archive program binding')
    return record, artifact


def worker(args):
    # This entry is launched only by SdkLauncher in the registered job. There is
    # no SSH client, endpoint discovery or submission loop.
    require(args.cmaddr, 'registered launcher must supply %CMADDR%')
    build_record = read(args.build_record)
    request = read(args.request)
    require(sha(args.package) == build_record['package_archive_sha256']
            and sha(args.build_record) == request['build_record_sha256'], 'worker staged identity')
    output = Path(args.out).absolute()
    require(output == output.resolve(), 'worker output')
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status='RUNNING', backend='hardware', run_id=request['run_id'],
                  build_record_sha256=sha(args.build_record), request_sha256=sha(args.request))
    write(output/'worker.json', record)
    root = output.parent/'frozen-observation'
    try:
        unpack(args.package, root)
        require(sha(root/'hardware_entry.py') == sha(__file__), 'staged worker source')
        pinned = identity(root, build_record['case'], build_record['profile'])
        require(all(build_record.get(k) == v for k, v in pinned.items()), 'worker frozen source/case join')
        common, _, observer = modules(root)
        require(build_record['compiler_args'] == common.compiler_arguments(pinned['params'], 'hardware', 1), 'worker exact compile flags')
        case = root/'cases'/pinned['case']
        build = Path(args.build).resolve()
        common.check_abi(build, observer.manifest_for(case, pinned['profile']))
        inventory = bind_actual_program(build, build_record, common)
        write(output/'actual-build-inventory.json', inventory)
        shutil.copyfile(args.build_record, output/'build.json')
        shutil.copyfile(args.request, output/'run-request.json')
        observer.capture(case, build, output/'runtime', pinned['profile'], pinned['warmup'], pinned['repeats'],
                         backend='hardware', cmaddr=args.cmaddr)
        record.update(status='PASS', actual_inventory_sha256=sha(output/'actual-build-inventory.json'))
    except BaseException as error:
        record.update(status='FAILED', error=repr(error))
        raise
    finally:
        write(output/'worker.json', record)
    return record


def compare_numerical_summary(fresh, recorded):
    """Accept cross-host FP64 diagnostic roundoff after independently gating raw outputs."""
    require(fresh.keys() == recorded.keys(), 'numerical summary keys')
    require({k: v for k, v in fresh.items() if k != 'errors'} ==
            {k: v for k, v in recorded.items() if k != 'errors'}, 'numerical outcomes differ')
    require(fresh['errors'].keys() == recorded['errors'].keys(), 'numerical tensor keys')
    for name, metrics in fresh['errors'].items():
        require(metrics.keys() == recorded['errors'][name].keys()
                == {'max_abs', 'max_relative_floor_1e_12', 'relative_l2'}, 'numerical diagnostic keys')
        for metric, value in metrics.items():
            other = recorded['errors'][name][metric]
            require(type(value) in (int, float) and type(other) in (int, float)
                    and math.isfinite(value) and math.isfinite(other)
                    and value >= 0 and other >= 0, 'finite nonnegative diagnostics')
            # A tiny cross-BLAS FP64 reference change can be amplified by the
            # 1e-12 denominator floor. Fresh replay, not the remote relative
            # diagnostic, is canonical. Raw ATOL/RTOL and predictions were
            # independently gated by numerical() before this comparison.
            if metric != 'max_relative_floor_1e_12':
                require(abs(value-other) <= 1e-12 + 1e-8*max(abs(value), abs(other)),
                        'numerical diagnostic differs: '+name+'.'+metric)
        remote = recorded['errors'][name]
        require(remote['max_relative_floor_1e_12'] <= remote['max_abs']/1e-12*(1+1e-12),
                'relative diagnostic exceeds absolute/floor bound')


def audit_capture(root, label, profile, capture):
    import numpy as np
    common, _, observer = modules(root)
    pinned = identity(root, label, profile)
    capture = Path(capture)
    worker_record = read(capture/'worker.json')
    build = read(capture/'build.json')
    request = read(capture/'run-request.json')
    require(worker_record['status'] == 'PASS' and worker_record['backend'] == 'hardware', 'completed registered capture')
    require(build.get('status') == 'PASS' and build.get('action') == 'compile'
            and build.get('backend') == 'hardware' and build.get('family') == FAMILY, 'returned successful compile')
    require(all(build.get(k) == v for k, v in pinned.items()), 'returned source/case/compile join')
    require(build['compiler_args'] == common.compiler_arguments(pinned['params'], 'hardware', 1), 'returned exact compiler flags')
    require(worker_record['build_record_sha256'] == sha(capture/'build.json') == request['build_record_sha256']
            and worker_record['request_sha256'] == sha(capture/'run-request.json')
            and worker_record['run_id'] == request['run_id'], 'returned request bindings')
    require(worker_record['actual_inventory_sha256'] == sha(capture/'actual-build-inventory.json'), 'actual ELF inventory binding')
    actual_program = read(capture/'actual-build-inventory.json')
    common.admit_compiled_inventory(actual_program['elf'])
    require(actual_program == build['program_inventory'], 'returned program matches compile-derived inventory')
    case = root/'cases'/label
    expected, context = common.independent_expected(case, common.load(case))
    execution = read(capture/'runtime/execution.json')
    require(execution['status'] == 'PASS' and execution['backend'] == 'hardware'
            and execution['profile'] == profile and execution['case_sha256'] == pinned['case_sha256']
            and execution['sources'] == observer.sources() and execution['params'] == pinned['params'], 'returned runtime source identity')
    require(execution['memory']['static_inventory'] == actual_program['elf'], 'runtime actual ELF inventory matches compiled program')
    require(sha(capture/'runtime/raw.npz') == execution['raw_sha256'], 'raw capture binding')
    require(execution['input_sha256'] == {name: common.array_sha(context['original'][name])
            for name in ('x', 'src', 'dst', 'coeff', 'w0', 'w1')}, 'original tensor identity')
    raw = common.load_npz(capture/'runtime/raw.npz', limit=common.CAPTURE_LIMIT)
    manifest = observer.manifest_for(case, profile)
    pes = manifest['params']['C']*manifest['params']['R']
    arrays = common.load_npz(case/'inputs.npz')
    for prefix in ('before_', 'after_'):
        for name, value in arrays.items():
            expected_carrier = value.astype(np.uint32) if value.dtype == np.uint16 else value
            require(np.array_equal(raw[prefix+name].view(np.uint32), expected_carrier.view(np.uint32)), 'offline immutable '+name)
    observer.completion_check(raw['initial_completion'], 0, pes)
    require(execution['warmup'] == pinned['warmup'] and execution['repeats'] == pinned['repeats']
            and len(execution['iterations']) == pinned['warmup']+pinned['repeats'], 'complete repetition cohort')
    fields = dict(manifest['inputs'], **manifest['outputs'])
    expected_apis = (['construct']+['get_id:'+name for name in fields]+['load', 'run']
        +['h2d:'+name for name in arrays]+['d2h:before_'+name for name in arrays]+['d2h:initial_completion'])
    expected_raw = {'initial_completion'} | {prefix+name for prefix in ('before_', 'after_') for name in arrays}
    for epoch in range(1, pinned['warmup']+pinned['repeats']+1):
        expected_apis += ['launch:f_run', 'd2h:completion']
        expected_apis += [f'd2h:e{epoch}_'+name for name in manifest['outputs'] if name != 'completion']
        expected_raw.update(f'e{epoch}_'+name for name in manifest['outputs'])
    expected_apis += ['d2h:after_'+name for name in arrays]+['stop']
    require(set(raw) == expected_raw, 'exact raw repeated capture inventory')
    timings = execution['api_timings']
    require([row['api'] for row in timings] == expected_apis, 'complete SDK API sequence')
    prior_end = 0
    for row in timings:
        require(type(row['start_ns']) is int and type(row['end_ns']) is int
                and row['end_ns'] >= row['start_ns'] >= prior_end
                and row['seconds'] == (row['end_ns']-row['start_ns'])/1e9, 'ordered host API clock')
        prior_end = row['end_ns']
    launches = [index for index, row in enumerate(timings) if row['api'] == 'launch:f_run']
    previous, numerical_replay = None, []
    for epoch, iteration in enumerate(execution['iterations'], 1):
        require(iteration['epoch'] == epoch and iteration['warmup'] == (epoch <= pinned['warmup']), 'ordered epoch classification')
        launch = timings[launches[epoch-1]]
        complete = timings[launches[epoch-1]+1]
        require(complete['start_ns'] == launch['end_ns']
                and iteration['launch_call_seconds'] == launch['seconds']
                and iteration['completion_read_seconds'] == complete['seconds']
                and iteration['launch_to_completion_seconds'] == (complete['end_ns']-launch['start_ns'])/1e9,
                'fresh completion timing window')
        observed = {name: raw[f'e{epoch}_'+name] for name in manifest['outputs']}
        observer.completion_check(observed['completion'], epoch, pes)
        fresh = observer.numerical(observed, expected, context)
        compare_numerical_summary(fresh, iteration['numerics'])
        numerical_replay.append(dict(epoch=epoch, numerics=fresh))
        decoded = observer.decode_profile(observed, profile)
        require(all(iteration.get(k) == v for k, v in decoded.items()), 'offline phase decoder replay')
        if previous is not None:
            for name in ('pre', 'logits', 'panel_sum'):
                require(np.array_equal(observed[name].view(np.uint32), previous[name].view(np.uint32)), 'offline same-input repeat bits')
        previous = observed
    return dict(status='PASS', backend='hardware', case_sha256=pinned['case_sha256'], profile=profile,
                epochs=len(execution['iterations']), run_id=request['run_id'], raw_sha256=execution['raw_sha256'],
                execution_sha256=sha(capture/'runtime/execution.json'), numerical_replay=numerical_replay)


def run_job(args, root=ROOT, launcher_type=None):
    record, artifact = load_build(root, args.build_record, args.case, args.profile)
    common, adapter, _ = modules(root)
    output = adapter.fresh(args.out)
    require(1 <= args.host_timeout <= 1800, 'bounded worker timeout')
    shutil.copyfile(args.build_record, output/'build.json')
    request = dict(schema=1, run_id=str(uuid.uuid4()), build_record_sha256=sha(output/'build.json'))
    write(output/'run-request.json', request)
    command = ['cs_python', 'hardware_entry.py', 'worker', '--package', 'package.tar.gz', '--build-record', 'build.json',
               '--request', 'run-request.json', '--build', 'out', '--out', 'capture', '--cmaddr', '%CMADDR%']
    shell = (f'if timeout --signal=INT --kill-after=15s {args.host_timeout}s {shlex.join(command)} > host.log 2>&1; '
             'then code=0; else code=$?; fi; printf \'%s\\n\' "$code" > host-exit.txt')
    summary = dict(status='RUNNING', command=shell, run_id=request['run_id'], downloads={}, errors={})
    write(output/'collection.json', summary)
    pending = None
    try:
        with adapter.client_log(output) as log:
            cls = launcher_type or adapter.sdk_classes()[1]
            with cls(str(artifact), simulator=False, **adapter.sdk_options(record['namespace'])) as launcher:
                try:
                    for path in (root/'hardware_entry.py', root/'package.tar.gz', output/'build.json', output/'run-request.json'):
                        launcher.stage(str(path))
                    # Recheck identity after staging and before starting work.
                    load_build(root, args.build_record, args.case, args.profile)
                    response = str(launcher.run(shell) or '')
                    require(len(response.encode()) <= 16 << 20, 'bounded launcher response')
                    (output/'launcher-response.txt').write_text(response)
                except BaseException as error:
                    pending = error
                    summary['launch_error'] = repr(error)
                finally:
                    for name, bound in (('host.log', 16 << 20), ('host-exit.txt', 128), ('capture', 256 << 20)):
                        try:
                            path = adapter.downloaded_path(launcher.download_artifact(name, str(output/name)), output/name, bound)
                            summary['downloads'][name] = dict(path=path.name, sha256=sha(path, bound))
                        except BaseException as error:
                            summary['errors'][name] = repr(error)
        require(not log.truncated, 'complete client log')
        if pending is not None:
            raise RuntimeError('registered run failed; partial downloads retained') from pending
        require(not summary['errors'], 'complete collection')
        require((output/summary['downloads']['host-exit.txt']['path']).read_text().strip() == '0', 'worker success')
        # SDK directory downloads have a leading capture/ directory.
        extracted = unpack(output/summary['downloads']['capture']['path'], output/'capture-extracted')
        require((extracted/'capture').is_dir(), 'SDK capture archive prefix')
        (extracted/'capture').rename(output/'capture-data')
        require(not list(extracted.iterdir()), 'only expected capture prefix')
        extracted.rmdir()
        summary['audit'] = audit_capture(root, args.case, args.profile, output/'capture-data')
        summary['status'] = 'PASS'
        # Preserve one extracted authoritative raw copy after successful replay.
        (output/summary['downloads']['capture']['path']).unlink()
        summary['downloads']['capture']['retained'] = False
    except BaseException as error:
        summary.update(status='FAILED', error=repr(error))
        raise
    finally:
        write(output/'collection.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('compile', 'run', 'audit', 'worker'))
    parser.add_argument('--case')
    parser.add_argument('--profile', type=int, choices=(0, 1), default=1)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--namespace', default='job-operator')
    parser.add_argument('--build-record', type=Path)
    parser.add_argument('--host-timeout', type=int, default=600)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--cmaddr')
    args = parser.parse_args()
    if args.action == 'compile': result = compile_job(args)
    elif args.action == 'run': result = run_job(args)
    elif args.action == 'worker': result = worker(args)
    else:
        require(not args.out.exists(), 'fresh audit output')
        result = audit_capture(ROOT, args.case, args.profile, args.capture)
        write(args.out, result)
    print(json.dumps({key: result[key] for key in ('status', 'backend', 'run_id') if key in result}))


if __name__ == '__main__':
    main()
