"""Worker for local SIM, appliance SIM and hardware; no frequency conversion."""
import argparse
import ctypes
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import threading
import time
import uuid

try:
    import resource
except ImportError:
    resource = None

import numpy as np
import clock_common as common

ROOT = Path(__file__).resolve().parent


def host_identity():
    result = dict(python=sys.version, platform=platform.platform(),
                  clock='perf_counter_ns and process_time_ns; neither is a CPU cycle counter',
                  container=Path(os.environ.get('SINGULARITY_CONTAINER', os.environ.get('APPTAINER_CONTAINER', ''))).name,
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                  load_average=list(os.getloadavg()) if hasattr(os, 'getloadavg') else None,
                  machine=platform.machine(), logical_cpu_count=os.cpu_count(),
                  process_id=os.getpid(), native_thread_id=threading.get_native_id())
    try:
        # CPU identity only; do not collect hostnames, network settings or serials.
        with open('/proc/cpuinfo') as stream:
            cpu_text = stream.read((1 << 20) + 1)
        fields = {'vendor_id', 'model name', 'cpu family', 'model', 'stepping',
                  'CPU implementer', 'CPU architecture', 'CPU variant', 'CPU part', 'CPU revision'}
        cpu = {}
        for line in cpu_text[:1 << 20].splitlines():
            key, separator, value = line.partition(':')
            if separator and key.strip() in fields:
                cpu.setdefault(key.strip(), value.strip())
        result['cpu_identity'] = dict(fields=cpu, truncated=len(cpu_text) > 1 << 20)
    except OSError as error:
        result['cpu_identity'] = dict(status='unavailable', error_type=type(error).__name__)
    for package in ('cerebras_sdk', 'cerebras_appliance', 'numpy'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = 'unknown'
    return result


def runtime_identity(runtime_type):
    """Report loaded module facts; a file hash is not an SDK release number."""
    result = []
    for name in sorted({runtime_type.__module__, 'cerebras.sdk', 'cerebras.sdk.runtime'}):
        module = sys.modules.get(name)
        if module is None:
            continue
        version = vars(module).get('__version__')
        record = dict(module=name, version=version if isinstance(version, str) else 'unknown')
        source = vars(module).get('__file__')
        if isinstance(source, str):
            path = Path(source)
            record['file_name'] = path.name
            try:
                record['file_sha256'] = common.digest(path.resolve())
            except (OSError, ValueError) as error:
                record['fingerprint_error'] = type(error).__name__
        result.append(record)
    return result


class ThreadObserver:
    """Optional endpoint observations outside the PMU interval.

    Equal endpoint CPUs do not exclude an intervening migration. The thread
    clock includes user and kernel CPU time and is not a CPU cycle counter.
    """
    def __init__(self):
        self.cpu = None
        try:
            self.cpu = ctypes.CDLL(None, use_errno=True).sched_getcpu
            self.cpu.argtypes = []
            self.cpu.restype = ctypes.c_int
        except (AttributeError, OSError):
            self.cpu = None

    def sample(self):
        result = dict(thread_time_ns=None, cpu=None, voluntary_context_switches=None,
                      involuntary_context_switches=None, minor_faults=None, major_faults=None)
        if hasattr(time, 'thread_time_ns'):
            result['thread_time_ns'] = time.thread_time_ns()
        if self.cpu is not None:
            cpu = self.cpu()
            result['cpu'] = cpu if cpu >= 0 else None
        if resource is not None and hasattr(resource, 'RUSAGE_THREAD'):
            try:
                usage = resource.getrusage(resource.RUSAGE_THREAD)
                for field, source in (('voluntary_context_switches', 'ru_nvcsw'),
                                      ('involuntary_context_switches', 'ru_nivcsw'),
                                      ('minor_faults', 'ru_minflt'), ('major_faults', 'ru_majflt')):
                    result[field] = int(getattr(usage, source))
            except OSError:
                pass
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
    common.require(len(symbols) in (7, 9) and len({v['name'] for v in symbols}) == len(symbols)
                   and len({v['id'] for v in symbols}) == len(symbols), 'unique RPC symbols')
    abi = {v['name']: (v['kind'],v['type'],[a['type'] for a in v.get('inputs',[])]) for v in symbols}
    base_abi = {**{n:('Var','[*]u32',[]) for n in ('data','seen','report')},
                'init':('Func','void',[]), 'tic':('Func','void',['u16']),
                'toc':('Func','void',['u16']), 'internal':('Func','void',['u16','u16'])}
    marker_abi = dict(base_abi, marks=('Var','[*]u32',[]), mark=('Func','void',['u16','u16']))
    common.require(abi in (base_abi, marker_abi), 'compiled RPC ABI')
    common.require(not cfg.get('markers', False) or abi == marker_abi, 'compiled marker ABI required')
    count = common.piece_count(p, cfg)
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
    observer = ThreadObserver() if cfg.get('pmu_controls', 0) else None
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
        if observer is not None:
            event['thread_before'] = observer.sample()
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
            if observer is not None:
                event['thread_after'] = observer.sample()
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
        meta['host']['runtime_modules'] = runtime_identity(runtime_type)
        if observer is not None:
            meta['host']['thread_observation'] = ('before/after each invocation, outside PMU interval; '
                'RUSAGE_THREAD counters and thread_time_ns include kernel activity; '
                'endpoint CPUs do not count all migrations')
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
        selected = ['data', 'seen', 'report'] + (['marks'] if cfg.get('markers', False) else [])
        ids = {n: invoke('get_id:'+n, runtime.get_id, n) for n in selected}
        common.require(ids == {s['name']:s['id'] for s in symbols if s['name'] in selected}, 'runtime and compiled IDs')
        invoke('load', runtime.load)
        invoke('run', runtime.run)
        invoke('init', runtime.launch, 'init', nonblock=False)
        for step in range(3):
            for sample in range(cfg.get('pmu_controls', 0)):
                invoke(f'pmu-empty:{step}:before:{sample}', lambda: None)
            wanted = common.fixture(p, step)
            payloads = common.pieces(wanted, cfg['arm'], cfg.get('piece_order', 'forward')) if cfg['kind'] == 'transfer' else []
            preload = remember((np.bitwise_not(wanted) if cfg['kind'] == 'transfer' else wanted.copy()).reshape(-1), f'preload:{step}')
            for i, piece in enumerate(payloads):
                remember(piece[4], f'h2d:{step}:{i}')
            names = ['data', 'seen', 'report']
            if cfg['kind'] != 'internal' and cfg['start'] == 'witness':
                names += ['start']
            if cfg['pause_at'] == 'after_end_witness':
                names += ['end_before_pause']
            if cfg.get('markers', False):
                names += ['marks']
            buffers = {}
            for n in names:
                size = (p['W'] if n in ('data', 'seen') else p['C']*p['R']*common.MARK_WORDS
                        if n == 'marks' else common.REPORT_WORDS)
                buffers[n] = remember(np.full(p['R']*p['C']*size, 0xffffffff, dtype=np.uint32), f'{n}:{step}')
                raw[f'{n}_{step}'] = buffers[n].reshape(p['R'], p['C'], size)
            invoke(f'preload:{step}', runtime.memcpy_h2d, ids['data'], preload,
                   0, 0, p['C'], p['R'], p['W'], nonblock=False, **flags)
            def read(name, symbol='report'):
                size = (common.REPORT_WORDS if symbol == 'report' else p['C']*p['R']*common.MARK_WORDS
                        if symbol == 'marks' else p['W'])
                invoke(f'd2h:{step}:{name}', runtime.memcpy_d2h, buffers[name], ids[symbol],
                       0, 0, p['C'], p['R'], size,
                       nonblock=False, **flags)
            pause(step, 'before_start')
            if cfg['kind'] != 'internal':
                launch(cfg['start'], f'tic:{step}', 'tic', np.uint16(step+1))
                if cfg['start'] == 'witness':
                    read('start')
                pause(step, 'after_start')
                for i in range(count if cfg.get('markers', False) else len(payloads)):
                    if cfg['kind'] == 'transfer':
                        x,y,c,r,payload = payloads[i]
                        task = invoke(f'h2d:{step}:{i}', runtime.memcpy_h2d, ids['data'], payload,
                                      x,y,c,r,p['W'], nonblock=cfg['transfer'] != 'blocking', **flags)
                        if cfg['transfer'] == 'wait':
                            invoke(f'h2d:{step}:{i}:wait', runtime.task_wait, task)
                    if cfg.get('markers', False):
                        invoke(f'mark:{step}:{i}', runtime.launch, 'mark', np.uint16(i),
                               np.uint16(step+1), nonblock=True)
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
            if cfg.get('markers', False):
                read('marks', 'marks')
            for sample in range(cfg.get('pmu_controls', 0)):
                invoke(f'pmu-empty:{step}:after:{sample}', lambda: None)
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
