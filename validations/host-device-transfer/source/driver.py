#!/usr/bin/env python3
"""Completed transfer campaigns with independent frame and capture audit."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np

SOURCE_FILES = ('layout.csl', 'pe.csl', 'driver.py', 'oracle.py')
TIMING_SCOPES = {
    'echo': 'worker perf_counter_ns around sequential H2D and blocking full echo for every chunk',
    'ack': 'worker perf_counter_ns before H2D through blocking all-PE delivery ACK; full readback excluded',
}
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_echo_oracle_' + hashlib.sha256(str(HERE).encode()).hexdigest()[:12], HERE/'oracle.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 256 << 20, 'bounded regular file')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')
    temporary.replace(path)


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def program_files(build):
    result, count, total = {}, 0, 0
    began = time.monotonic()
    for folder, directories, files in os.walk(build, followlinks=False):
        for name in directories + files:
            path = Path(folder)/name
            count += 1
            need(count <= 20000 and time.monotonic()-began < 45 and not path.is_symlink(), 'program scan bound')
        for name in files:
            path = Path(folder)/name
            if path.suffix == '.elf' or name in ('out.json', 'out_rpc.json'):
                total += path.stat().st_size
                need(total <= 256 << 20, 'program byte bound')
                result[path.relative_to(build).as_posix()] = dict(sha256=sha(path), bytes=path.stat().st_size)
    need(result and 'out.json' in result, 'compiled program metadata')
    return result


def channel_mapping(metadata_root, plan):
    result = {}
    for side, direction, count_name, height_prefix, color_prefix in (
            ('west', 'h2d', 'num_demux', 'height_demux_', 'FABRIC_H2D_DATA_ID_'),
            ('east', 'd2h', 'num_mux', 'height_mux_', 'FABRIC_D2H_DATA_ID_')):
        params = read(Path(metadata_root)/side/'out.json')['params']
        need(int(params['height']) == plan['height'] and int(params[count_name]) == plan['channels'],
             'generated channel count and height')
        need(int(params['width_'+side+'_buf']) == plan[side], 'generated buffer width')
        if side == 'west':
            need(int(params['num_inputs']) >= 1, 'generated streaming input registration')
        offset, groups = 0, []
        for channel in range(plan['channels']):
            height = int(params[height_prefix+str(channel)])
            need(height > 0, 'positive channel row partition')
            groups.append(dict(row_begin=offset, row_end=offset+height,
                               routing_color=int(params[color_prefix+str(channel)])))
            offset += height
        need(offset == plan['height'], 'channel partitions cover fixed ROI')
        result[direction] = groups
    return result

def check_program(metadata, plan):
    params = read(Path(metadata)/'out.json')['params']
    for name, expected in oracle.compiler_params(plan).items():
        need(int(params[name]) == expected, 'compiled parameter '+name)
    return channel_mapping(metadata, plan)


def expected_reply(expected, plan, geometry):
    if plan['reply'] == 'echo':
        return expected
    return expected.reshape(plan['pes'], geometry['frame_words'])[:, 0].copy()


def capture(source_root, build, output, case, order, backend='simulator', cmaddr=None,
            sdk_workdir=None, smoke=False, runtime_api=None, run_id=None):
    plan = oracle.plan(case, order, smoke)
    if run_id is None:
        run_id = str(uuid.uuid4())
    need(type(run_id) is str and str(uuid.UUID(run_id)) == run_id, 'canonical capture run ID')
    need(backend in ('simulator', 'hardware') and (backend != 'hardware' or (cmaddr and not smoke)),
         'execution backend')
    source_root, build, output = (Path(p).resolve() for p in (source_root, build, output))
    output.mkdir(parents=True, exist_ok=False)
    (output/'source').mkdir()
    (output/'witnesses').mkdir()
    for name in SOURCE_FILES:
        sha(source_root/name)
        shutil.copyfile(source_root/name, output/'source'/name)
    source_sha = {name: sha(output/'source'/name) for name in SOURCE_FILES}
    need(source_sha['oracle.py'] == sha(HERE/'oracle.py') and source_sha['driver.py'] == sha(__file__),
         'running driver source identity')
    inventory = program_files(build)
    write(output/'program.json', inventory)
    (output/'metadata').mkdir()
    for name in inventory:
        if name.endswith('.json'):
            destination = output/'metadata'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(build/name, destination)
    mapping = check_program(output/'metadata', plan)
    record = dict(schema=1, status='RUNNING', backend=backend, case=case, order=order,
                  capture_id=run_id, plan=plan, source_sha256=source_sha,
                  program_sha256=sha(output/'program.json'), rows=[], witnesses={}, setup_seconds={},
                  channel_mapping=mapping, timing_scope=TIMING_SCOPES[plan['reply']],
                  physical_wire_bytes=None, outstanding_exchanges=1,
                  full_readback_outside_timing=plan['reply'] == 'ack')
    write(output/'capture.json', record)
    if sdk_workdir is None:
        sdk_workdir = output.parent/(output.name+'-sdk-work')
        sdk_workdir.mkdir(exist_ok=False)
    sdk_workdir = Path(sdk_workdir).resolve()
    need(sdk_workdir.is_dir() and not sdk_workdir.is_relative_to(output)
         and not output.is_relative_to(sdk_workdir), 'SDK working files outside capture')
    if runtime_api is None:
        from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    else:
        SdkRuntime, MemcpyDataType, MemcpyOrder = runtime_api
    previous, runner = Path.cwd(), None
    generation = order * 1000000 + 1
    try:
        os.chdir(sdk_workdir)
        options = {'cmaddr': cmaddr} if cmaddr else dict(simfab_numthreads=1, suppress_simfab_trace=True)
        runner = SdkRuntime(str(build), **options)
        for action in ('load', 'run'):
            start = time.perf_counter()
            getattr(runner, action)()
            record['setup_seconds'][action] = time.perf_counter()-start
        config_id, status_id = runner.get_id('config_words'), runner.get_id('status')
        frame_id = runner.get_id('frame_data') if plan['reply'] == 'ack' else None
        config, status = np.empty(plan['pes'], dtype=np.uint32), np.empty(plan['pes'], dtype=np.uint32)
        kwargs = dict(data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR)
        h2d = getattr(runner, {'unicast': 'memcpy_h2d', 'row-bcast': 'memcpy_h2d_rowbcast',
                              'col-bcast': 'memcpy_h2d_colbcast'}[plan['mode']])
        rows, columns = plan['height'], plan['width']
        for size in plan['sizes']:
            geometry = oracle.geometry(size, plan)
            n, count = geometry['payload_words_per_pe'], geometry['frame_words']
            runner.launch('configure', np.uint32(n), nonblock=True)
            runner.memcpy_d2h(config, config_id, 0, 0, columns, rows, 1,
                              streaming=False, nonblock=False, **kwargs)
            runner.memcpy_d2h(status, status_id, 0, 0, columns, rows, 1,
                              streaming=False, nonblock=False, **kwargs)
            need(np.all(config == n) and np.all(status == 0), 'completed configuration on every PE')
            replies = [np.empty(plan['pes']*geometry['reply_words'], dtype=np.uint32)
                       for _ in range(plan['chunks'])]
            readbacks = ([np.empty(plan['pes']*count, dtype=np.uint32) for _ in replies]
                         if plan['reply'] == 'ack' else [])
            for iteration in range(plan['warmup']+plan['repeats']):
                pairs = oracle.frames(size, generation, plan)
                need(len(pairs) == len(replies), 'all chunks prepared before timing')
                for reply in replies:
                    reply.fill(0xDEADBEEF)
                started = time.perf_counter_ns()
                for (sent, _), reply in zip(pairs, replies):
                    h2d(0, sent, 0, 0, columns, rows, count,
                        streaming=True, nonblock=True, **kwargs)
                    runner.memcpy_d2h(reply, 1, 0, 0, columns, rows, geometry['reply_words'],
                                      streaming=True, nonblock=False, **kwargs)
                ended = time.perf_counter_ns()
                # ACK cases always have one chunk: copying the stable frame is
                # deliberately outside the delivery interval and before reuse.
                need(not readbacks or len(pairs) == 1, 'ACK readback requires one retained frame')
                for observed in readbacks:
                    runner.memcpy_d2h(observed, frame_id, 0, 0, columns, rows, count,
                                      streaming=False, nonblock=False, **kwargs)
                chunks = []
                equal = True
                for index, ((sent, expected), reply) in enumerate(zip(pairs, replies)):
                    readback = readbacks[index] if readbacks else None
                    same = bool(np.array_equal(reply, expected_reply(expected, plan, geometry))
                                and (readback is None or np.array_equal(readback, expected)))
                    equal = equal and same
                    chunks.append(dict(index=index, generation=generation+index,
                                       sent_sha256=oracle.digest(sent), received_sha256=oracle.digest(reply),
                                       readback_sha256=oracle.digest(readback) if readback is not None else None,
                                       equal=same))
                row = dict(payload_bytes=size, generation=generation, iteration=iteration,
                           phase='warmup' if iteration < plan['warmup'] else 'measured',
                           elapsed_ns=ended-started, chunks=chunks, equal=equal)
                record['rows'].append(row)
                if not equal or iteration == plan['warmup']+plan['repeats']-1:
                    witnesses = []
                    for index, reply in enumerate(replies):
                        witness = dict(index=index, generation=generation+index)
                        for kind, array in [('reply', reply)] + ([('readback', readbacks[index])] if readbacks else []):
                            name = f'witnesses/size-{size}-generation-{generation+index}-{kind}.npy'
                            np.save(output/name, array, allow_pickle=False)
                            witness[kind] = dict(path=name, sha256=sha(output/name))
                        witnesses.append(witness)
                    record['witnesses'][str(size)] = witnesses
                need(equal, 'received reply or full readback differs from expected frame')
                generation += plan['generation_stride']
            write(output/'capture.json', record)
        record['status'] = 'EXECUTED'
    except BaseException as error:
        record.update(status='FAILED', error=repr(error))
        raise
    finally:
        try:
            if runner is not None:
                start = time.perf_counter()
                runner.stop()
                record['setup_seconds']['stop'] = time.perf_counter()-start
        except BaseException as error:
            record.update(status='FAILED', stop_error=repr(error))
            raise
        finally:
            os.chdir(previous)
            write(output/'capture.json', record)
    return record


def audit(capture):
    capture = Path(capture).resolve()
    record = read(capture/'capture.json')
    need(record.get('schema') == 1 and record.get('status') == 'EXECUTED', 'complete successful execution')
    plan = oracle.plan(record['case'], record['order'], record['plan']['smoke'])
    need(record['plan'] == plan and record['backend'] in ('simulator', 'hardware'), 'exact capture plan')
    need(record['backend'] != 'hardware' or not plan['smoke'], 'hardware must use full schedule')
    need(record['outstanding_exchanges'] == 1 and record['physical_wire_bytes'] is None, 'measurement scope')
    need(record['timing_scope'] == TIMING_SCOPES[plan['reply']] and
         record['full_readback_outside_timing'] is (plan['reply'] == 'ack'), 'exact timing scope')
    need(str(uuid.UUID(record['capture_id'])) == record['capture_id'], 'canonical capture ID')
    need(set(record['setup_seconds']) == {'load', 'run', 'stop'}
         and all(type(value) in (int, float) and np.isfinite(value) and value >= 0
                 for value in record['setup_seconds'].values()), 'complete runtime lifecycle')
    need(set(record['source_sha256']) == set(SOURCE_FILES), 'complete source identity')
    for name, pin in record['source_sha256'].items():
        need(sha(capture/'source'/name) == pin and sha(HERE/name) == pin, 'frozen source identity '+name)
    need(sha(capture/'program.json') == record['program_sha256'], 'program record identity')
    inventory = read(capture/'program.json')
    for name, entry in inventory.items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts, 'program path')
        if name.endswith('.json'):
            need(sha(capture/'metadata'/name) == entry['sha256'], 'retained generated metadata identity')
    mapping = check_program(capture/'metadata', plan)
    need(record['channel_mapping'] == mapping, 'captured channel mapping')
    total = plan['warmup']+plan['repeats']
    need(len(record['rows']) == len(plan['sizes'])*total, 'complete logical round count')
    need(set(record['witnesses']) == {str(size) for size in plan['sizes']}, 'raw witnesses per size')
    generation, cursor, summaries = record['order']*1000000+1, 0, []
    for size in plan['sizes']:
        elapsed, geometry = [], oracle.geometry(size, plan)
        for iteration in range(total):
            row = record['rows'][cursor]
            phase = 'warmup' if iteration < plan['warmup'] else 'measured'
            need(row['payload_bytes'] == size and row['generation'] == generation
                 and row['iteration'] == iteration and row['phase'] == phase, 'exact ordered schedule')
            need(type(row['elapsed_ns']) is int and 0 < row['elapsed_ns'] < 600*10**9,
                 'bounded positive completed interval')
            pairs = oracle.frames(size, generation, plan)
            need(row['equal'] is True and len(row['chunks']) == len(pairs), 'complete verified chunks')
            for index, ((sent, expected), chunk) in enumerate(zip(pairs, row['chunks'])):
                reply = expected_reply(expected, plan, geometry)
                need(chunk['index'] == index and chunk['generation'] == generation+index
                     and chunk['equal'] is True, 'ordered chunk identity')
                need(chunk['sent_sha256'] == oracle.digest(sent)
                     and chunk['received_sha256'] == oracle.digest(reply)
                     and chunk['readback_sha256'] == (oracle.digest(expected) if plan['reply'] == 'ack' else None),
                     'independent input/reply/readback digests')
                if iteration == total-1:
                    witnesses = record['witnesses'][str(size)]
                    need(len(witnesses) == len(pairs), 'complete raw chunk witnesses')
                    witness = witnesses[index]
                    need(witness['index'] == index and witness['generation'] == generation+index,
                         'raw witness chunk identity')
                    kinds = [('reply', reply)] + ([('readback', expected)] if plan['reply'] == 'ack' else [])
                    need(set(witness) == {'index', 'generation'} | {kind for kind, _ in kinds},
                         'exact raw witness fields')
                    for kind, expected_raw in kinds:
                        name = f'witnesses/size-{size}-generation-{generation+index}-{kind}.npy'
                        item = witness[kind]
                        need(item['path'] == name and sha(capture/name) == item['sha256'], 'raw witness identity')
                        observed = np.load(capture/name, allow_pickle=False)
                        need(observed.dtype == np.uint32 and observed.shape == expected_raw.shape
                             and np.array_equal(observed, expected_raw), 'independent raw witness equality')
            if phase == 'measured':
                elapsed.append(row['elapsed_ns'])
            generation, cursor = generation+plan['generation_stride'], cursor+1
        summaries.append(dict(**geometry, samples=len(elapsed),
                              median_round_ns=float(np.median(elapsed)), p95_round_ns=float(np.percentile(elapsed, 95)),
                              min_round_ns=min(elapsed), max_round_ns=max(elapsed)))
    return dict(status='PASS', backend=record['backend'], case=record['case'], order=record['order'],
                smoke=plan['smoke'], capture_id=record['capture_id'], logical_rounds=cursor,
                exchanges=cursor*plan['chunks'], summary=summaries, channel_mapping=mapping,
                timing_scope=record['timing_scope'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    run = sub.add_parser('capture')
    run.add_argument('--source', type=Path, default=HERE)
    run.add_argument('--build', type=Path, required=True)
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--case', choices=tuple(oracle.CASES)+tuple(oracle.SMOKE_CASES), required=True)
    run.add_argument('--order', type=int, default=0)
    run.add_argument('--smoke', action='store_true')
    check = sub.add_parser('audit')
    check.add_argument('--capture', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'capture':
        capture(args.source, args.build, args.out, args.case, args.order, smoke=args.smoke)
        result = audit(args.out)
        write(args.out/'audit.json', result)
    else:
        result = audit(args.capture)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
