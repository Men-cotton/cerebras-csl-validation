"""Worker for local SIM, appliance SIM and hardware; no frequency conversion."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import numpy as np
import clock_common as common

ROOT = Path(__file__).resolve().parent


def host_identity():
    result = dict(python=sys.version, platform=platform.platform(),
                  clock='perf_counter_ns and process_time_ns; neither is a CPU cycle counter',
                  container=Path(os.environ.get('SINGULARITY_CONTAINER', os.environ.get('APPTAINER_CONTAINER', ''))).name,
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                  load_average=list(os.getloadavg()) if hasattr(os, 'getloadavg') else None)
    for package in ('cerebras_sdk', 'cerebras_appliance', 'numpy'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = 'unknown'
    return result


def sdk():
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    return SdkRuntime, dict(streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                           order=MemcpyOrder.ROW_MAJOR)


def capture(args, sdk_factory=sdk):
    manifest = common.read_json(args.manifest)
    request = common.read_json(args.config_file)
    common.require(set(request) == {'config', 'run_id', 'manifest_sha256'}, 'request keys')
    common.require(str(uuid.UUID(request['run_id'])) == request['run_id'], 'canonical run UUID')
    common.require(common.digest(args.manifest) == request['manifest_sha256'], 'manifest identity')
    p, cfg = common.validate_params(manifest['params']), common.validate_config(request['config'])
    common.require(cfg == request['config'] and args.backend == manifest['backend'], 'canonical config/backend')
    common.require(args.backend in common.BACKENDS, 'backend')
    common.require((args.backend == 'hardware') == bool(args.cmaddr), 'hardware requires cmaddr; SIM has none')
    for n in ('clock_probe.py', 'clock_common.py', 'clock_pmu.py'):
        common.require(common.digest(ROOT/n) == manifest['host_sha256'][n], 'host source identity: '+n)
    # Appliance workers receive compiled artifacts, not the compiler's CSL tree.
    common.require(set(manifest['source_sha256']) == {'clock_source/layout.csl', 'clock_source/pe.csl'}, 'source pins')
    compiled = common.read_json(Path(args.name)/'out.json')['params']
    common.require({k:str(v) for k,v in compiled.items()} == {k:str(v) for k,v in p.items()}, 'compiled parameters')
    symbols = common.read_json(Path(args.name)/'bin/out_rpc.json')['rpc_symbols']
    common.require(len(symbols) == 7 and len({v['name'] for v in symbols}) == 7
                   and len({v['id'] for v in symbols}) == 7, 'unique RPC symbols')
    abi = {v['name']: (v['kind'],v['type'],[a['type'] for a in v.get('inputs',[])]) for v in symbols}
    common.require(abi == {**{n:('Var','[*]u32',[]) for n in ('data','seen','report')},
                           'init':('Func','void',[]), 'tic':('Func','void',['u16']),
                           'toc':('Func','void',['u16']), 'internal':('Func','void',['u16','u16'])}, 'compiled RPC ABI')
    count = 1 if cfg['arm'] == 'rectangle' else p['R'] if cfg['arm'] == 'strips' else p['C']*p['R']
    common.require(cfg['pause_at'] != 'between_h2d' or count > 1, 'between_h2d needs two pieces')
    destination = Path(args.out).absolute()
    destination.mkdir(parents=True, exist_ok=False)
    meta = dict(schema=1, status='incomplete', backend=args.backend, params=p, config=cfg,
                run_id=request['run_id'], manifest_sha256=request['manifest_sha256'],
                source_sha256=manifest['source_sha256'], host_sha256=manifest['host_sha256'],
                host=host_identity(), cmaddr_provided=bool(args.cmaddr), clock_bits=48,
                clock_scope='per-PE interval; clocks are not synchronized across PEs')
    common.write_json(destination/'capture.json', meta)
    events, raw, arrays = [], {}, {}
    runtime, failure, pmu = None, None, None
    previous = Path.cwd()

    def record(value):
        if isinstance(value, np.ndarray):
            return arrays[id(value)]
        if isinstance(value, np.generic):
            return int(value)
        if value is None or type(value) in (str, int, bool, float):
            return value
        if type(value).__name__ in ('MemcpyOrder','MemcpyDataType'):
            return str(value)
        return {'type': type(value).__name__}

    def invoke(name, fn, *a, **kw):
        event = dict(name=name, args=[record(v) for v in a], kwargs={k:record(v) for k,v in kw.items()},
                     wall_start_ns=time.perf_counter_ns(), cpu_start_ns=time.process_time_ns(), status='incomplete')
        events.append(event)
        if pmu is not None:
            pmu.begin()
        try:
            value = fn(*a, **kw)
            event.update(status='pass', result_type=type(value).__name__, result=record(value))
            return value
        except BaseException as error:
            event.update(error=repr(error))
            raise
        finally:
            if pmu is not None:
                event['pmu'] = pmu.end()
            event.update(wall_end_ns=time.perf_counter_ns(), cpu_end_ns=time.process_time_ns())

    def remember(array, role):
        arrays[id(array)] = dict(role=role, dtype=str(array.dtype), shape=list(array.shape),
                                initial_sha256=common.digest_bytes(array.tobytes()))
        return array

    def pause(step, where):
        if cfg['pause_at'] == where:
            invoke(f'pause:{step}:{where}', lambda delay: time.sleep(delay) if delay else None, cfg['pause_seconds'])

    def launch(mode, name, *arguments):
        task = invoke(name, runtime.launch, *arguments, nonblock=(mode in ('queued', 'wait')))
        if mode == 'wait':
            invoke(name+':wait', runtime.task_wait, task)

    flags = {}
    try:
        if cfg.get('host_pmu',False):
            from clock_pmu import CounterGroup
            pmu = CounterGroup()
            invoke('pmu-empty', lambda: None)
        runtime_type, flags = sdk_factory()
        build = str(Path(args.name).absolute())
        os.chdir(destination)
        options = dict(cmaddr=args.cmaddr)
        if args.backend != 'hardware':
            options.update(simfab_numthreads=1, suppress_simfab_trace=True)
        # Do not include an appliance address in the durable API records.
        def construct():
            nonlocal runtime
            runtime = runtime_type(build, **options)
        invoke('construct', construct)
        ids = {n: invoke('get_id:'+n, runtime.get_id, n) for n in ('data', 'seen', 'report')}
        common.require(ids == {s['name']:s['id'] for s in symbols if s['kind'] == 'Var'}, 'runtime and compiled IDs')
        invoke('load', runtime.load)
        invoke('run', runtime.run)
        invoke('init', runtime.launch, 'init', nonblock=False)
        for step in range(3):
            wanted = common.fixture(p, step)
            payloads = common.pieces(wanted, cfg['arm']) if cfg['kind'] == 'transfer' else []
            preload = remember((np.bitwise_not(wanted) if cfg['kind'] == 'transfer' else wanted.copy()).reshape(-1), f'preload:{step}')
            for i, piece in enumerate(payloads):
                remember(piece[4], f'h2d:{step}:{i}')
            names = ['data', 'seen', 'report']
            if cfg['kind'] != 'internal' and cfg['start'] == 'witness':
                names += ['start']
            if cfg['pause_at'] == 'after_end_witness':
                names += ['end_before_pause']
            buffers = {}
            for n in names:
                size = p['W'] if n in ('data', 'seen') else common.REPORT_WORDS
                buffers[n] = remember(np.full(p['R']*p['C']*size, 0xffffffff, dtype=np.uint32), f'{n}:{step}')
                raw[f'{n}_{step}'] = buffers[n].reshape(p['R'], p['C'], size)
            invoke(f'preload:{step}', runtime.memcpy_h2d, ids['data'], preload,
                   0, 0, p['C'], p['R'], p['W'], nonblock=False, **flags)
            def read(name, symbol='report'):
                invoke(f'd2h:{step}:{name}', runtime.memcpy_d2h, buffers[name], ids[symbol],
                       0, 0, p['C'], p['R'], p['W'] if symbol != 'report' else common.REPORT_WORDS,
                       nonblock=False, **flags)
            pause(step, 'before_start')
            if cfg['kind'] != 'internal':
                launch(cfg['start'], f'tic:{step}', 'tic', np.uint16(step+1))
                if cfg['start'] == 'witness':
                    read('start')
                pause(step, 'after_start')
                for i, (x,y,c,r,payload) in enumerate(payloads):
                    task = invoke(f'h2d:{step}:{i}', runtime.memcpy_h2d, ids['data'], payload,
                                  x,y,c,r,p['W'], nonblock=cfg['transfer'] != 'blocking', **flags)
                    if cfg['transfer'] == 'wait':
                        invoke(f'h2d:{step}:{i}:wait', runtime.task_wait, task)
                    if i == 0:
                        pause(step, 'between_h2d')
                pause(step, 'before_end')
                launch('blocking', f'toc:{step}', 'toc', np.uint16(cfg['work']))
            else:
                launch(cfg['start'], f'internal:{step}', 'internal', np.uint16(step+1), np.uint16(cfg['work']))
            pause(step, 'after_end_return')
            if cfg['pause_at'] == 'after_end_witness':
                read('end_before_pause')
                pause(step, 'after_end_witness')
            read('report')
            read('data', 'data')
            read('seen', 'seen')
    except BaseException as error:
        failure = error
        meta['error'] = repr(error)
    finally:
        try:
            if runtime is not None:
                invoke('stop', runtime.stop)
        except BaseException as error:
            failure = error
            meta['stop_error'] = repr(error)
        finally:
            if pmu is not None:
                pmu.close()
            os.chdir(previous)
            np.savez(destination/'raw.npz', **raw)
            common.write_json(destination/'calls.json', events)
            meta.update(raw_sha256=common.digest(destination/'raw.npz'), calls_sha256=common.digest(destination/'calls.json'))
    if failure is None:
        try:
            common.validate_calls(events, p, cfg)
        except ValueError as error:
            meta.update(status='incomplete', error=str(error))
        else:
            try:
                meta['steps'] = common.validate_arrays(raw, p, cfg)
                meta['status'] = 'pass'
            except ValueError as error:
                meta.update(status='observed_numeric_fail', error=str(error))
    common.write_json(destination/'capture.json', meta)
    print(dict(status=meta['status'], run_id=meta['run_id'], out=str(destination)), flush=True)
    return {'pass':0, 'observed_numeric_fail':1, 'incomplete':2}[meta['status']]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--backend', choices=common.BACKENDS, required=True)
    parser.add_argument('--cmaddr')
    args = parser.parse_args(argv)
    try:
        return capture(args)
    except Exception as error:
        print('error: '+str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
