#!/usr/bin/env python3
"""Prepare, capture, and independently replay a bounded owned-arena probe.

No submission, compiler, polling, or scheduler implementation is provided.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import time
import zipfile

HERE = Path(__file__).resolve().parent
SOURCES = ('driver.py', 'oracle.py', 'layout.csl', 'pe.csl')
LIMIT = 4 << 20
RUNTIME_ELFS = frozenset(('generated/MEMCPY_XY_ROUTES.elf',
                          'generated/coord.elf', 'generated/default.elf'))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path, limit=LIMIT):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit, 'bounded regular file')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    digest(path)
    return json.loads(Path(path).read_text())


def write(path, value):
    data = json.dumps(value, sort_keys=True, indent=2)+'\n'
    need(len(data.encode()) < LIMIT, 'bounded JSON')
    Path(path).write_text(data)


def fresh(path):
    path = Path(path).absolute()
    need(path == path.resolve(), 'output path without symlinks')
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_oracle(path):
    spec = importlib.util.spec_from_file_location('arena_frozen_oracle', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(folder, capacity):
    oracle = load_oracle(HERE/'oracle.py')
    oracle.plan(capacity)
    folder = fresh(folder)
    (folder/'source').mkdir()
    for name in SOURCES:
        shutil.copyfile(HERE/name, folder/'source'/name)
    case = dict(schema=1, family='arena-observe', capacity=capacity, params=dict(CAP=capacity),
                sources={name: digest(folder/'source'/name) for name in SOURCES},
                run_plan=[list(row) for row in oracle.RUNS],
                scope='one PE, one owned u16 arena; no total SRAM or stack peak claim')
    write(folder/'case.json', case)
    return case


def read_case(folder):
    folder = Path(folder).resolve()
    case = read(folder/'case.json')
    need(case.get('schema') == 1 and case.get('family') == 'arena-observe', 'case family')
    need(set(case['sources']) == set(SOURCES), 'complete source inventory')
    for name, pin in case['sources'].items():
        need(digest(folder/'source'/name) == pin, 'frozen source '+name)
    oracle = load_oracle(folder/'source/oracle.py')
    oracle.plan(case['capacity'])
    need(case['params'] == dict(CAP=case['capacity']) and case['run_plan'] == [list(row) for row in oracle.RUNS], 'fixed case contract')
    need(case['sources']['driver.py'] == digest(Path(__file__)), 'running driver matches frozen driver')
    return case, oracle


def fields(capacity):
    return dict(control=(2, 'u16'), rows=(55, 'u16'), values=(capacity+52, 'u16'),
                status=(6, 'u16'), clocks=(6, 'u16'), completion=(1, 'u32'))


def elf_sections(payload):
    need(len(payload) >= 52 and payload[:4] == b'\x7fELF' and payload[4] in (1, 2)
         and payload[5] in (1, 2) and payload[6] == 1, 'generated ELF')
    wide, endian = payload[4] == 2, '<' if payload[5] == 1 else '>'
    header_format = endian+('HHIQQQIHHHHHH' if wide else 'HHIIIIIHHHHHH')
    need(len(payload) >= 16+struct.calcsize(header_format), 'ELF header bound')
    header = struct.unpack_from(header_format, payload, 16)
    offset, size, count = header[5], header[10], header[11]
    fmt = endian+('IIQQQQIIQQ' if wide else 'IIIIIIIIII')
    need(0 < count <= 4096 and size >= struct.calcsize(fmt)
         and offset >= 16+struct.calcsize(header_format) and offset+size*count <= len(payload), 'ELF section extent')
    sections = [struct.unpack_from(fmt, payload, offset+index*size) for index in range(count)]
    need(all(s[1] == 8 or s[5] == 0 or s[4]+s[5] <= len(payload) for s in sections), 'ELF section payload bound')
    return sections


def elf_end(payload):
    ends = []
    for section in elf_sections(payload):
        if section[2] & 2 and section[5] and section[3] < 48*1024:
            ends.append(section[3]+section[5])
    need(ends and max(ends) < 44*1024, 'compiled static SRAM admission')
    return max(ends)


def program(build, case, runtime_added=None):
    build = Path(build).resolve()
    need({k: int(v) for k, v in read(build/'out.json')['params'].items()} == case['params'], 'compiled parameter ABI')
    symbols = read(build/'bin/out_rpc.json')['rpc_symbols']
    wanted = {name: ('Var', '[*]'+dtype, ()) for name, (_, dtype) in fields(case['capacity']).items()}
    wanted['f_run'] = ('Func', 'void', ())
    need(len(symbols) == len(wanted) and len({s['id'] for s in symbols}) == len(wanted), 'unique RPC IDs')
    need({s['name']: (s['kind'], s['type'], tuple(x['type'] for x in s.get('inputs', []))) for s in symbols} == wanted, 'compiled pointer ABI')
    result, metadata, count, total, began = {}, {}, 0, 0, time.monotonic()
    def failed(error):
        raise error
    for directory, dirs, files in os.walk(build, followlinks=False, onerror=failed):
        for name in dirs+files:
            count += 1
            need(count <= 4000 and time.monotonic()-began <= 20 and not (Path(directory)/name).is_symlink(), 'bounded generated tree')
        for name in files:
            path = Path(directory)/name
            info = path.lstat()
            need(stat.S_ISREG(info.st_mode), 'regular generated file')
            total += info.st_size
            need(total < 128 << 20 and time.monotonic()-began <= 20, 'generated bytes/deadline')
            if not (name.endswith('.elf') or name in ('out.json', 'out_rpc.json')):
                continue
            pin = digest(path, 32 << 20)
            relative = path.relative_to(build).as_posix()
            if relative in RUNTIME_ELFS:
                need(runtime_added is not None, 'runtime metadata present before execution')
                payload = path.read_bytes()
                need(all(s[2] & 2 == 0 for s in elf_sections(payload)), 'runtime metadata contains SHF_ALLOC')
                metadata[relative] = dict(sha256=pin, bytes=len(payload), no_shf_alloc=True)
                continue
            result[relative] = dict(sha256=pin, bytes=path.stat().st_size)
            if name.endswith('.elf'):
                result[relative]['static_sram_end'] = elf_end(path.read_bytes())
    need(not metadata or set(metadata) == RUNTIME_ELFS, 'complete known SDK metadata additions')
    if runtime_added is not None:
        runtime_added.update(metadata)
    need(time.monotonic()-began <= 20, 'generated traversal deadline')
    need(any(name.endswith('.elf') for name in result) and {'out.json', 'west/out.json', 'east/out.json'} <= set(result), 'generated program with SDK helpers')
    return result


def load_raw(path):
    import numpy as np
    digest(path)
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        need(len(entries) <= 32 and len({e.filename for e in entries}) == len(entries)
             and all('/' not in e.filename and e.filename.endswith('.npy') for e in entries)
             and sum(e.file_size for e in entries) < LIMIT, 'bounded raw NPZ')
        for entry in entries:
            payload = archive.read(entry)
            stream = io.BytesIO(payload)
            version = np.lib.format.read_magic(stream)
            need(version in ((1, 0), (2, 0)), 'raw NPY format')
            header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                      else np.lib.format.read_array_header_2_0)
            shape, fortran, dtype = header(stream, max_header_size=4096)
            need(dtype == np.dtype('uint32') and not fortran and len(shape) == 1
                 and type(shape[0]) is int and 0 <= shape[0] <= 1024
                 and stream.tell()+4*shape[0] == len(payload), 'bounded raw NPY extent')
    with np.load(path, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    need(all(v.dtype == np.uint32 and v.ndim == 1 for v in raw.values()), 'raw carrier ABI')
    return raw


def audit(folder):
    folder = Path(folder).resolve()
    record = read(folder/'execution.json')
    case, oracle = read_case(folder/'case')
    need(record['status'] == 'PASS' and record['backend'] in ('simulator', 'hardware'), 'completed capture')
    need(record['case_sha256'] == digest(folder/'case/case.json') and record['sources'] == case['sources'], 'source/case bindings')
    need(record['program'] == read(folder/'program.json') and record['program_sha256'] == digest(folder/'program.json'), 'actual generated program binding')
    need(record['raw_sha256'] == digest(folder/'raw.npz'), 'raw binding')
    metadata = record['runtime_generated_elf']
    need(not metadata or set(metadata) == RUNTIME_ELFS, 'recorded known SDK metadata')
    need(all(v['no_shf_alloc'] is True and type(v['bytes']) is int and 0 < v['bytes'] <= 32 << 20
             and len(v['sha256']) == 64 for v in metadata.values()), 'runtime metadata identity')
    raw = load_raw(folder/'raw.npz')
    need(raw['initial_completion'].tolist() == [0], 'new runtime initial epoch')
    expected_keys = {'initial_completion'} | {f'e{epoch}_{name}' for epoch in range(1, 5) for name in fields(case['capacity'])}
    need(set(raw) == expected_keys and len(record['iterations']) == 4, 'complete fixed repetition inventory')
    checks = []
    previous_end = 0
    for epoch, ((mode, seed), iteration) in enumerate(zip(oracle.RUNS, record['iterations']), 1):
        data = {name: raw[f'e{epoch}_{name}'].tolist() for name in fields(case['capacity'])}
        check = oracle.validate(data, case['capacity'], mode, seed, epoch)
        need(iteration['audit'] == check and iteration['epoch'] == epoch, 'independent exact replay')
        start, end = iteration['host_start_ns'], iteration['host_completion_ns']
        need(type(start) is int and type(end) is int and previous_end <= start < end, 'ordered host completion windows')
        need(iteration['host_launch_to_completion_seconds'] == (end-start)/1e9, 'host duration')
        previous_end = end
        checks.append(check)
    need((raw['e1_values'] == raw['e2_values']).all() and (raw['e1_values'] == raw['e4_values']).all(), 'same payload across observer modes and reset')
    return dict(status='PASS', backend=record['backend'], capacity=case['capacity'], epochs=4,
                raw_sha256=record['raw_sha256'], execution_sha256=digest(folder/'execution.json'),
                checks=checks, scope=case['scope'])


def capture(case_folder, build, folder, backend, cmaddr=None):
    import numpy as np
    need(backend in ('simulator', 'hardware') and (backend == 'hardware') == bool(cmaddr), 'explicit backend/worker endpoint')
    case_folder, build = Path(case_folder).resolve(), Path(build).resolve()
    case, oracle = read_case(case_folder)
    inventory = program(build, case)
    folder = fresh(folder)
    shutil.copytree(case_folder/'source', folder/'case/source')
    shutil.copyfile(case_folder/'case.json', folder/'case/case.json')
    write(folder/'program.json', inventory)
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    raw, runtime = {}, None
    record = dict(status='RUNNING', backend=backend, sources=case['sources'], case_sha256=digest(folder/'case/case.json'),
                  program=inventory, program_sha256=digest(folder/'program.json'), iterations=[],
                  timing_scope='same-PE allocation/query/write/copy interval; host launch through fresh epoch read includes command and D2H',
                  memory_scope=case['scope'])
    runtime_directory = folder/'runtime-work'
    runtime_directory.mkdir()
    original_cwd = Path.cwd()
    schema = fields(case['capacity'])
    def options(dtype):
        return dict(streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT if dtype == 'u16' else MemcpyDataType.MEMCPY_32BIT,
                    order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    def fetch(name):
        count, dtype = schema[name]
        value = np.full(count, 0xDEAD, np.uint32)
        runtime.memcpy_d2h(value, ids[name], 0, 0, 1, 1, count, **options(dtype))
        return value
    def persist():
        np.savez_compressed(folder/'raw.npz', **raw)
        record['raw_sha256'] = digest(folder/'raw.npz')
        write(folder/'execution.json', record)
    try:
        os.chdir(runtime_directory)
        runtime = SdkRuntime(str(build), **(dict(cmaddr=cmaddr) if backend == 'hardware'
                            else dict(simfab_numthreads=1, suppress_simfab_trace=True)))
        ids = {name: runtime.get_id(name) for name in schema}
        runtime.load()
        runtime.run()
        raw['initial_completion'] = fetch('completion')
        need(raw['initial_completion'].tolist() == [0], 'initial completed epoch')
        for epoch, (mode, seed) in enumerate(oracle.RUNS, 1):
            control = np.asarray([mode, seed], np.uint32)
            runtime.memcpy_h2d(ids['control'], control, 0, 0, 1, 1, 2, **options('u16'))
            for name, (count, dtype) in schema.items():
                if name not in ('control', 'completion'):
                    runtime.memcpy_h2d(ids[name], np.full(count, 0xDEAD, np.uint32), 0, 0, 1, 1, count, **options(dtype))
            completed = np.full(1, 0xDEAD, np.uint32)
            begin = time.monotonic_ns()
            runtime.launch('f_run', nonblock=False)
            runtime.memcpy_d2h(completed, ids['completion'], 0, 0, 1, 1, 1, **options('u32'))
            end = time.monotonic_ns()
            raw[f'e{epoch}_completion'] = completed
            for name in schema:
                if name != 'completion':
                    raw[f'e{epoch}_{name}'] = fetch(name)
            values = {name: raw[f'e{epoch}_{name}'].tolist() for name in schema}
            checked = oracle.validate(values, case['capacity'], mode, seed, epoch)
            record['iterations'].append(dict(epoch=epoch, audit=checked, host_start_ns=begin,
                host_completion_ns=end, host_launch_to_completion_seconds=(end-begin)/1e9))
            persist()
        runtime_added = {}
        need(program(build, case, runtime_added) == inventory, 'generated program unchanged')
        record['runtime_generated_elf'] = runtime_added
        record['status'] = 'PASS'
    except BaseException as error:
        record.update(status='FAILED', error=repr(error))
        raise
    finally:
        if runtime is not None:
            try:
                runtime.stop()
            except BaseException as error:
                record.update(status='FAILED', stop_error=repr(error))
        os.chdir(original_cwd)
        persist()
    need(record['status'] == 'PASS', 'complete capture and stop')
    try:
        checked = audit(folder)
    except BaseException as error:
        record.update(status='FAILED', audit_error=repr(error))
        persist()
        raise
    write(folder/'audit.json', checked)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    p = actions.add_parser('prepare')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--capacity', type=int, choices=(64, 128), default=64)
    p = actions.add_parser('capture')
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--backend', choices=('simulator', 'hardware'), required=True)
    p.add_argument('--cmaddr')
    p = actions.add_parser('audit')
    p.add_argument('--capture', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare': result = prepare(args.out, args.capacity)
    elif args.action == 'capture': result = capture(args.case, args.build, args.out, args.backend, args.cmaddr)
    else: result = audit(args.capture)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
