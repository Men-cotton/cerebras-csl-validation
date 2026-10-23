"""Worker integration with an in-memory SDK double and synthetic PMU records.

These tests exercise capture production and offline validation, not measured
cycles, Linux PMU permission, SDK command ordering, or hardware correctness.
"""
import argparse
from contextlib import ExitStack
from enum import Enum
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_common as common
import clock_probe as probe
import test_clock_common as base


class MemcpyDataType(Enum):
    MEMCPY_32BIT = 1


class MemcpyOrder(Enum):
    ROW_MAJOR = 1


class FakeRuntime:
    instances = []
    fail_marker = False

    def __init__(self, build, **kwargs):
        self.data = np.zeros((2, 2, 4), dtype=np.uint32)
        self.seen = self.data.copy()
        self.report = np.zeros((2, 2, 16), dtype=np.uint32)
        self.marks = np.full((2, 2, 32), 0xffffffff, dtype=np.uint32)
        self.stopped = False
        self.instances.append(self)

    def get_id(self, name):
        return dict(base.IDS, marks=57)[name]

    def load(self):
        pass

    def run(self):
        pass

    def stop(self):
        self.stopped = True

    def task_wait(self, task):
        pass

    def memcpy_h2d(self, symbol, payload, x, y, c, r, words, **kwargs):
        assert symbol == 17 and words == 4
        self.data[y:y+r, x:x+c] = payload.reshape(r, c, words)

    def memcpy_d2h(self, out, symbol, x, y, c, r, words, **kwargs):
        src = {17: self.data, 23: self.seen, 41: self.report, 57: self.marks}[symbol]
        out[:] = src[y:y+r, x:x+c].reshape(-1)

    def launch(self, name, *args, **kwargs):
        if name == 'init':
            self.report[:, :, 11] = 0x5a17c10c
            self.report[:, :, 15] = 0xc10c5a17
            return
        if name in ('tic', 'internal'):
            self.step = int(args[0])-1
            self.marks[:] = 0xffffffff
            for y in range(2):
                for x in range(2):
                    start = (base.U48-100+1000*self.step+100000*(2*y+x)) & base.U48
                    self.report[y, x, :3] = base.limbs(start)
                    self.report[y, x, 6] = self.step+1
                    self.report[y, x, 9] = int(name == 'internal')
            if name == 'tic':
                return
        if name == 'mark':
            if self.fail_marker:
                raise RuntimeError('synthetic marker failure')
            index, generation = map(int, args)
            assert generation == self.step+1
            for y in range(2):
                for x in range(2):
                    start = sum(int(self.report[y, x, i]) << (16*i) for i in range(3))
                    timestamp = (start+20*(index+1)) & base.U48
                    self.marks[y, x, 8*index:8*index+8] = base.limbs(timestamp) + [
                        generation, index, int(self.data[y, x, 0]), int(self.data[y, x, -1]), 0x4d41524b]
            return
        assert name in ('toc', 'internal')
        work = int(args[-1])
        self.seen[:] = self.data
        for y in range(2):
            for x in range(2):
                pe = 2*y+x
                start = sum(int(self.report[y, x, i]) << (16*i) for i in range(3))
                work_start = (start+200+pe) & base.U48
                end = (work_start+20*work+pe+3) & base.U48
                self.report[y, x, 3:6] = base.limbs(end)
                self.report[y, x, 7:11] = [self.step+1, work, int(name == 'internal'), 4*work]
                self.report[y, x, 12:15] = base.limbs(work_start)


class FakeCounters:
    def begin(self):
        pass

    def end(self):
        return dict(status='ok', scope='thread-self-user', cycles=900, instructions=400,
                    time_enabled_ns=100, time_running_ns=100, scaled=False)

    def close(self):
        pass


def sdk_double():
    return FakeRuntime, dict(streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                             order=MemcpyOrder.ROW_MAJOR)


class WorkerTests(unittest.TestCase):
    def run_worker(self, config, marker_abi=True, fail_marker=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root/'build'
            (build/'bin').mkdir(parents=True)
            common.write_json(build/'out.json', {'params': base.PARAMS})
            symbols = [dict(name=name, id=value, kind='Var', type='[*]u32')
                       for name, value in base.IDS.items()]
            functions = [('init', []), ('tic', ['u16']), ('toc', ['u16']), ('internal', ['u16', 'u16'])]
            if marker_abi:
                symbols.append(dict(name='marks', id=57, kind='Var', type='[*]u32'))
                functions.append(('mark', ['u16', 'u16']))
            for index, (name, inputs) in enumerate(functions):
                symbols.append(dict(name=name, id=100+index, kind='Func', type='void',
                                    inputs=[{'type': value} for value in inputs]))
            common.write_json(build/'bin/out_rpc.json', {'rpc_symbols': symbols})
            manifest = dict(backend='local-sim', params=base.PARAMS, **common.source_identity(probe.ROOT))
            common.write_json(root/'manifest.json', manifest)
            request = dict(config=common.validate_config(config), run_id=base.RUN_ID,
                           manifest_sha256=common.digest(root/'manifest.json'))
            common.write_json(root/'request.json', request)
            args = argparse.Namespace(manifest=root/'manifest.json', config_file=root/'request.json',
                                      name=str(build), out=str(root/'capture'), backend='local-sim', cmaddr=None)
            sample = dict(thread_time_ns=100, cpu=1, voluntary_context_switches=0,
                          involuntary_context_switches=0, minor_faults=0, major_faults=0)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(probe, 'host_identity', return_value={'mock': True}))
                stack.enter_context(mock.patch.object(probe, 'runtime_identity', return_value=[]))
                observer = stack.enter_context(mock.patch.object(probe, 'ThreadObserver'))
                observer.return_value.sample.return_value = sample
                stack.enter_context(mock.patch.object(FakeRuntime, 'fail_marker', fail_marker))
                stack.enter_context(mock.patch.dict(sys.modules, {'clock_pmu': types.SimpleNamespace(CounterGroup=FakeCounters)}))
                stack.enter_context(mock.patch('builtins.print'))
                code = probe.capture(args, sdk_factory=sdk_double)
                if not request['config'].get('pmu_controls'):
                    observer.assert_not_called()
            meta = common.read_json(root/'capture/capture.json')
            calls = common.read_json(root/'capture/calls.json')
            raw_bytes = (root/'capture/raw.npz').read_bytes()
            verified = common.validate_capture_content(meta, calls, raw_bytes, manifest, request)
            return code, meta, calls, verified

    def test_legacy_and_new_abi_marker_off_retain_old_event_sequence(self):
        for marker_abi in (False, True):
            code, meta, calls, verified = self.run_worker({}, marker_abi=marker_abi)
            self.assertEqual((code, verified['status']), (0, 'pass'))
            self.assertEqual([c['name'] for c in calls], [c['name'] for c in base.call_fixture(base.config())])
            self.assertNotIn('marks', meta['steps'][0])

    def test_reversed_markers_capture_partial_prefix_and_final_arrays(self):
        for kind in ('transfer', 'empty'):
            for arm in ('rectangle', 'strips', 'tiles'):
                code, meta, calls, verified = self.run_worker(dict(kind=kind, arm=arm, markers=True,
                                                                  piece_order='reverse', transfer='wait'))
                self.assertEqual((code, verified['status']), (0, 'pass'))
                count = {'rectangle': 1, 'strips': 2, 'tiles': 4}[arm]
                self.assertEqual(len(meta['steps'][0]['marks']), count)
                names = [c['name'] for c in calls]
                for step in range(3):
                    self.assertGreater(names.index(f'd2h:{step}:marks'), names.index(f'toc:{step}'))
                if kind == 'empty':
                    self.assertFalse(any(n.startswith('h2d:') for n in names))

    def test_markers_need_new_abi(self):
        with self.assertRaisesRegex(ValueError, 'compiled marker ABI'):
            self.run_worker({'markers': True}, marker_abi=False)

    def test_repeated_empty_controls_and_thread_metadata(self):
        code, meta, calls, verified = self.run_worker(dict(host_pmu=True, pmu_controls=3))
        self.assertEqual((code, verified['status']), (0, 'pass'))
        self.assertEqual(sum(c['name'].startswith('pmu-empty') for c in calls), 19)
        self.assertTrue(all('thread_before' in c and 'thread_after' in c for c in calls))
        self.assertIn('outside PMU interval', meta['host']['thread_observation'])

    def test_marker_exception_preserves_incomplete_capture_and_stops_runtime(self):
        code, meta, calls, verified = self.run_worker({'markers': True}, fail_marker=True)
        self.assertEqual((code, verified['status']), (2, 'incomplete'))
        self.assertIn('synthetic marker failure', meta['error'])
        self.assertTrue(FakeRuntime.instances[-1].stopped)
        self.assertEqual(calls[-1]['name'], 'stop')

    def test_runtime_file_fingerprint_does_not_guess_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'runtime.so'
            path.write_bytes(b'synthetic module bytes')
            module = types.ModuleType('test_fake_runtime')
            module.__file__ = str(path)
            runtime_type = type('Runtime', (), {'__module__': module.__name__})
            with mock.patch.dict(sys.modules, {module.__name__: module}):
                result = next(r for r in probe.runtime_identity(runtime_type) if r['module'] == module.__name__)
            self.assertEqual(result, dict(module=module.__name__, version='unknown', file_name='runtime.so',
                                          file_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def test_thread_observer_is_graceful_without_kernel_interfaces(self):
        with mock.patch.object(probe.ctypes, 'CDLL', side_effect=OSError('unavailable')), \
                mock.patch.object(probe, 'resource', None), \
                mock.patch.object(probe.time, 'thread_time_ns', return_value=42):
            observer = probe.ThreadObserver()
            result = observer.sample()
        self.assertEqual(result['thread_time_ns'], 42)
        self.assertTrue(all(v is None for k, v in result.items() if k != 'thread_time_ns'))


if __name__ == '__main__':
    unittest.main()
