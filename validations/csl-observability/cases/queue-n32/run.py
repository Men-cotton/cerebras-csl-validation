#!/usr/bin/env python3
"""Bounded two-PE capture/replay. Hardware entry is callable only by a registered worker."""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import time
import uuid
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE_NAMES = ('source/layout.csl', 'source/pe.csl', 'oracle.py', 'transport_oracle.py', 'run.py', 'elf_reader.py')
RUNTIME_ELFS = {'generated/MEMCPY_XY_ROUTES.elf', 'generated/coord.elf', 'generated/default.elf'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path, limit=64 << 20):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, 'bounded regular file '+str(path))
    before = path.stat()
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            result.update(chunk)
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'file changed during hash')
    return result.hexdigest()


def read(path):
    digest(path, 4 << 20)
    return json.loads(Path(path).read_text())


def write(path, value, replace=False):
    with Path(path).open('w' if replace else 'x') as target:
        json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
        target.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(folder, n):
    root = HERE.parents[5]
    folder = Path(folder).resolve()
    oracle = load('queue_prepare_oracle', HERE/'oracle.py')
    oracle.symbols(n)
    need(not folder.exists(), 'fresh case')
    (folder/'source').mkdir(parents=True)
    origins = [HERE/'layout.csl', HERE/'pe.csl', HERE/'oracle.py', HERE.parent/'oracle.py',
               HERE/'run.py', root/'models/hierarchical_gnn/grouped_csr/driver.py']
    for name, source in zip(SOURCE_NAMES, origins):
        shutil.copyfile(source, folder/name)
    write(folder/'case.json', dict(schema=1, n=n, params={'N':n}, symbols=oracle.symbols(n),
          source_sha256={name:digest(folder/name) for name in SOURCE_NAMES},
          scope='2x1 WSE3; runtime PROFILE0/1 and delay0/256 use the same compiled ELF; direct queue masks, not link congestion'))
    return folder/'case.json'


def read_case(folder):
    case = read(folder/'case.json')
    need(set(case['source_sha256']) == set(SOURCE_NAMES), 'exact frozen source inventory')
    need(all(digest(folder/name) == value for name, value in case['source_sha256'].items()), 'frozen source identity')
    oracle = load('queue_case_oracle', folder/'oracle.py')
    need(case['params'] == {'N':case['n']} and case['symbols'] == oracle.symbols(case['n']), 'case parameters and ABI')
    return case, oracle


def inventory(build, elf):
    """Read generated ELF sections only; bound the complete build-tree accounting."""
    need(build.is_dir() and not build.is_symlink(), 'regular build directory')
    pending, files, entries, size, began = [build], [], 0, 0, time.monotonic()
    while pending:
        need(time.monotonic()-began <= 10, 'build scan deadline')
        with os.scandir(pending.pop()) as children:
            for child in children:
                entries += 1
                need(entries <= 2000 and time.monotonic()-began <= 10, 'build scan bounds')
                info = child.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(child.path))
                else:
                    need(stat.S_ISREG(info.st_mode), 'regular build member; no symlinks')
                    size += info.st_size
                    need(size <= 64 << 20, 'build byte bound')
                    files.append(Path(child.path))
    metadata = {str(p.relative_to(build)):p for p in files if str(p.relative_to(build)) in RUNTIME_ELFS}
    need(not metadata or set(metadata) == RUNTIME_ELFS, 'complete known SDK metadata ELF set')
    for path in metadata.values():
        elf.runtime_elf_identity(path)
    images = {str(p.relative_to(build)):p for p in files if p.suffix == '.elf' and str(p.relative_to(build)) not in metadata}
    need(images, 'application/helper ELF inventory')
    storage = {name:elf.elf_storage(path) for name, path in images.items()}
    need(all(row['static_sram_end'] < 44*1024 for row in storage.values()), 'strict 44KiB static endpoint gate')
    retained = dict(images, **{'out.json':build/'out.json', 'bin/out_rpc.json':build/'bin/out_rpc.json'})
    return dict(files={name:digest(path) for name, path in retained.items()}, elf_storage=storage,
                accounting=dict(entries=entries, visible_bytes=size, complete=True))


def expected_calls(symbols):
    names = ['construct'] + ['id:'+name for name in symbols] + ['load', 'run', 'initial_completion']
    for i in range(3):
        names += [f'in{i}:'+name for name in symbols if name != 'completion']
        names += [f'prepare{i}', f'launch{i}', f'out{i}:completion']
        names += [f'out{i}:'+name for name in symbols if name != 'completion']
    return names+['stop']


def validate_raw(raw, n, profile, delay, oracle):
    symbols = oracle.symbols(n)
    names = {'initial_completion'} | {f'{direction}{i}_{name}' for direction in ('input', 'output')
                                              for i in range(3) for name in symbols}
    need(set(raw) == names, 'complete three-invocation raw arrays')
    need(raw['initial_completion'].dtype == np.uint32 and raw['initial_completion'].shape == (2, 1)
         and np.array_equal(raw['initial_completion'], np.zeros((2, 1), np.uint32)), 'fresh initial completion')
    rows = []
    for i in range(3):
        expected = oracle.fixture(n, i, profile, delay)
        outputs = {}
        for name, width in symbols.items():
            for direction in ('input', 'output'):
                a = raw[f'{direction}{i}_{name}']
                need(a.dtype == np.uint32 and a.shape == (2, width), 'raw shape/type '+name)
            need(np.array_equal(raw[f'input{i}_{name}'], np.asarray(expected[name], np.uint32)), 'original input bytes '+name)
            outputs[name] = raw[f'output{i}_{name}'].tolist()
        rows.append(oracle.audit(outputs, n, i, profile, delay))
    return rows


def load_raw(path, symbols):
    """Reject forged array extents before NumPy allocates from an NPY header."""
    expected = {'initial_completion':(2, 1)}
    expected.update({f'{direction}{i}_{name}':(2, width) for direction in ('input', 'output')
                     for i in range(3) for name, width in symbols.items()})
    digest(path, 4 << 20)
    with zipfile.ZipFile(path) as archive:
        items = archive.infolist()
        need(len(items) == len(expected) and {item.filename for item in items} == {name+'.npy' for name in expected}
             and sum(item.file_size for item in items) <= 4 << 20, 'bounded raw archive')
        for item in items:
            payload = archive.read(item)
            stream = io.BytesIO(payload)
            version = np.lib.format.read_magic(stream)
            need(version in ((1, 0), (2, 0)), 'raw NPY format')
            header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = header(stream, max_header_size=4096)
            wanted = expected[item.filename[:-4]]
            need(dtype == np.dtype('uint32') and not fortran and shape == wanted
                 and stream.tell()+4*wanted[0]*wanted[1] == len(payload), 'bounded raw NPY extent')
    with np.load(path, allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def audit(folder):
    folder = Path(folder).resolve()
    record, build = read(folder/'capture.json'), read(folder/'build.json')
    need(record['status'] == 'COMPLETED', 'completed capture required')
    need(record['backend'] in ('simulator', 'hardware'), 'explicit captured backend')
    need(record['backend'] != 'hardware' or record['registered_worker'], 'registered hardware worker')
    need(digest(folder/'raw.npz', 4 << 20) == record['raw_sha256'], 'raw identity')
    need(digest(folder/'build.json') == record['build_sha256'], 'build identity')
    case, oracle = read_case(folder/'sources')
    need(build['case_sha256'] == digest(folder/'sources/case.json') and
         record['case_sha256'] == build['case_sha256'], 'capture/build/case binding')
    need(record['n'] == case['n'] and record['source_sha256'] == case['source_sha256'], 'capture source binding')
    need(build['source_sha256'] == case['source_sha256'] and build['elf_storage'] and
         all(0 < row['static_sram_end'] < 44*1024 for row in build['elf_storage'].values()), 'saved static ELF admission')
    raw = load_raw(folder/'raw.npz', case['symbols'])
    rows = validate_raw(raw, record['n'], record['profile'], record['delay'], oracle)
    calls = expected_calls(case['symbols'])
    need(record['expected_api'] == calls and [r['name'] for r in record['api']] == calls
         and all(r['status'] == 'complete' and r['end_ns'] >= r['start_ns'] for r in record['api']),
         'closed exact native API sequence')
    by_name = {row['name']:row for row in record['api']}
    expected_timings = [dict(epoch=i+1, launch_to_completion_seconds=
                       (by_name[f'out{i}:completion']['end_ns']-by_name[f'launch{i}']['start_ns'])/1e9)
                       for i in range(3)]
    need(record['timings'] == expected_timings and
         all(row['launch_to_completion_seconds'] > 0 for row in expected_timings), 'completed host windows replay')
    return dict(status='PASS', backend=record['backend'], run_id=record['run_id'], n=record['n'],
                profile=record['profile'], delay=record['delay'], invocations=rows,
                raw_sha256=record['raw_sha256'], capture_sha256=digest(folder/'capture.json'),
                case_sha256=record['case_sha256'], build_sha256=record['build_sha256'],
                scope='three same-runtime epochs; queue empty/full snapshots and local drain latency, not physical-link stall fraction')


def finalize(folder, build, elf, pinned):
    """A failed identity check, replay, or audit write invalidates completion."""
    try:
        need(inventory(build, elf)['files'] == pinned['files'], 'compiled artifacts changed during capture')
        result = audit(folder)
        write(folder/'audit.json', result)
        return result
    except BaseException as error:
        record = read(folder/'capture.json')
        record.update(status='FAILED', post_capture_error=type(error).__name__+': '+str(error))
        write(folder/'capture.json', record, replace=True)
        raise


def capture(case_dir, folder, profile, delay, build=None, backend='simulator', cmaddr=None,
            registered_worker=False, trace=False, sdk=None):
    """Hardware callers are registered workers; this module never submits jobs."""
    need(backend in ('simulator', 'hardware') and (backend == 'hardware') == bool(cmaddr), 'explicit backend/address')
    need(backend != 'hardware' or registered_worker, 'hardware capture requires registered worker')
    need(not trace or backend == 'simulator', 'fabric trace is simulator-only')
    case_dir, folder = Path(case_dir).resolve(), Path(folder).resolve()
    build = Path(build).resolve() if build is not None else case_dir/'build'
    case, oracle = read_case(case_dir)
    oracle.fixture(case['n'], 0, profile, delay)
    need(digest(HERE/'run.py') == case['source_sha256']['run.py'], 'capture driver matches frozen case')
    elf = load('queue_elf_reader', case_dir/'elf_reader.py')
    pinned = inventory(build, elf)
    params, rpc = read(build/'out.json'), read(build/'bin/out_rpc.json')
    need({k:int(v) for k, v in params['params'].items()} == case['params'], 'compiled parameters')
    symbols = {r['name']:r for r in rpc['rpc_symbols'] if r['kind'] == 'Var'}
    functions = {r['name']:r for r in rpc['rpc_symbols'] if r['kind'] == 'Func'}
    need(set(symbols) == set(case['symbols']) and all(r['type'] == '[*]u32' for r in symbols.values()), 'compiled symbol ABI')
    need(set(functions) == {'prepare', 'run_probe'} and all(not r.get('inputs') for r in functions.values()), 'compiled RPC ABI')
    need(not folder.exists(), 'fresh capture output')
    (folder/'sources/source').mkdir(parents=True)
    for name in SOURCE_NAMES+('case.json',):
        shutil.copyfile(case_dir/name, folder/'sources'/name)
    pinned.update(case_sha256=digest(case_dir/'case.json'), source_sha256=case['source_sha256'])
    write(folder/'build.json', pinned)
    if sdk is None:
        from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    else:
        SdkRuntime, MemcpyDataType, MemcpyOrder = sdk
    journal, raw, timings, owned, error = [], {}, [], None, None
    started, original_cwd = time.monotonic(), Path.cwd()

    def call(name, function, *args, **kwargs):
        row = dict(name=name, status='started', start_ns=time.monotonic_ns())
        journal.append(row)
        try:
            value = function(*args, **kwargs)
            row['status'] = 'complete'
            return value
        except BaseException as exc:
            row.update(status='failed', error=type(exc).__name__+': '+str(exc))
            raise
        finally:
            row['end_ns'] = time.monotonic_ns()

    def construct():
        nonlocal owned
        options = dict(cmaddr=cmaddr) if backend == 'hardware' else dict(simfab_numthreads=1, suppress_simfab_trace=not trace)
        owned = SdkRuntime(str(build), **options)
        return owned

    try:
        os.chdir(folder)
        call('construct', construct)
        ids = {name:call('id:'+name, owned.get_id, name) for name in case['symbols']}
        need(ids == {name:row['id'] for name, row in symbols.items()}, 'runtime IDs match compiled ABI')
        call('load', owned.load);call('run', owned.run)
        opts = dict(streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
        previous = np.full((2, 1), 0xDEAD, np.uint32)
        call('initial_completion', owned.memcpy_d2h, previous.ravel(), ids['completion'], 0, 0, 2, 1, 1, **opts)
        raw['initial_completion'] = previous.copy()
        need(np.array_equal(previous, np.zeros((2, 1), np.uint32)), 'fresh runtime completion zero')
        for i in range(3):
            inputs = oracle.fixture(case['n'], i, profile, delay)
            raw[f'input{i}_completion'] = previous.copy()
            for name, width in case['symbols'].items():
                if name == 'completion':
                    continue
                a = np.ascontiguousarray(inputs[name], dtype=np.uint32)
                raw[f'input{i}_{name}'] = a.copy()
                call(f'in{i}:'+name, owned.memcpy_h2d, ids[name], a.ravel(), 0, 0, 2, 1, width, **opts)
            call(f'prepare{i}', owned.launch, 'prepare', nonblock=False)
            previous = np.full((2, 1), 0xDEAD, np.uint32)
            call(f'launch{i}', owned.launch, 'run_probe', nonblock=False)
            before = journal[-1]['start_ns']
            call(f'out{i}:completion', owned.memcpy_d2h, previous.ravel(), ids['completion'], 0, 0, 2, 1, 1, **opts)
            after = journal[-1]['end_ns']
            raw[f'output{i}_completion'] = previous.copy()
            need(np.array_equal(previous, np.full((2, 1), i+1, np.uint32)), 'fresh all-PE completion')
            timings.append(dict(epoch=i+1, launch_to_completion_seconds=(after-before)/1e9))
            for name, width in case['symbols'].items():
                if name == 'completion':
                    continue
                a = np.zeros((2, width), np.uint32)
                call(f'out{i}:'+name, owned.memcpy_d2h, a.ravel(), ids[name], 0, 0, 2, 1, width, **opts)
                raw[f'output{i}_{name}'] = a
            oracle.audit({name:raw[f'output{i}_{name}'].tolist() for name in case['symbols']}, case['n'], i, profile, delay)
    except BaseException as exc:
        error = type(exc).__name__+': '+str(exc)
    finally:
        if owned is not None:
            try:
                call('stop', owned.stop)
            except BaseException as exc:
                error = (error+'; ' if error else '')+'stop: '+str(exc)
        os.chdir(original_cwd)
        np.savez(folder/'raw.npz', **raw)
        write(folder/'capture.json', dict(status='FAILED' if error else 'COMPLETED', error=error,
              backend=backend, registered_worker=registered_worker, trace=trace, run_id=str(uuid.uuid4()),
              n=case['n'], profile=profile, delay=delay, case_sha256=pinned['case_sha256'],
              source_sha256=case['source_sha256'], build_sha256=digest(folder/'build.json'),
              raw_sha256=digest(folder/'raw.npz', 4 << 20), api=journal, expected_api=expected_calls(case['symbols']),
              timings=timings, wall_seconds=time.monotonic()-started))
    need(error is None, error or '')
    return finalize(folder, build, elf, pinned)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare');p.add_argument('--case', type=Path, required=True);p.add_argument('--n', type=int, required=True)
    p = sub.add_parser('capture')
    p.add_argument('--case', type=Path, required=True);p.add_argument('--build', type=Path)
    p.add_argument('--out', type=Path, required=True);p.add_argument('--profile', type=int, required=True)
    p.add_argument('--delay', type=int, required=True)
    p.add_argument('--trace', action='store_true', help='simulator only; requires explicit bounded trace budget in outer runner')
    p = sub.add_parser('audit');p.add_argument('--capture', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(prepare(args.case, args.n))
    elif args.action == 'capture':
        print(json.dumps(capture(args.case, args.out, args.profile, args.delay, build=args.build, trace=args.trace), indent=2))
    else:
        print(json.dumps(audit(args.capture), indent=2))


if __name__ == '__main__':
    main()
