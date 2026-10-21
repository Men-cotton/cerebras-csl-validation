"""Independent synthetic record tests; never import or call a Cerebras SDK.

The integer fixture, clock events, and API sequence below are constructed here,
not by the producer's fixture/packing/expected-call helpers. These tests validate
record admission; synthetic clocks are not measured simulator/device evidence.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_common as common


PARAMS = {'C': 2, 'R': 2, 'W': 4}
U32, U48 = 2**32 - 1, 2**48 - 1
IDS = {'data': 17, 'seen': 23, 'report': 41}
RUN_ID = '1ef13d62-d1e4-4c24-a994-354b0af30d42'


def config(**changes):
    value = dict(kind='transfer', start='witness', transfer='queued',
                 arm='rectangle', pause_at='none', pause_seconds=0.0,
                 work=1, steps=3)
    return dict(value, **changes)


def input_words(step):
    values = [((i * 2654435761 + 0x12345678) & U32) ^ (0xA5A55A5A if step == 1 else 0)
              for i in range(16)]
    return np.array(values, dtype=np.uint32).reshape(2, 2, 4)


def limbs(value):
    return [(value >> shift) & 65535 for shift in (0, 16, 32)]


def raw_fixture(cfg):
    raw, expected = {}, []
    for step in range(3):
        values = input_words(step)
        raw[f'data_{step}'], raw[f'seen_{step}'] = values.copy(), values.copy()
        report = np.zeros((2, 2, 16), dtype=np.uint32)
        pe_rows = []
        for y in range(2):
            row = []
            for x in range(2):
                pe = 2*y + x
                start = (U48 - 100 + 1000*step + 100000*pe) & U48
                active = 20*cfg['work'] + pe + 3
                before = 200 + pe
                work_start = (start + before) & U48
                end = (work_start + active) & U48
                report[y, x] = (limbs(start) + limbs(end) +
                    [step+1, step+1, cfg['work'], int(cfg['kind'] == 'internal'), 4*cfg['work'], 0x5A17C10C] +
                    limbs(work_start) + [0xC10C5A17])
                row.append(dict(start=start, end=end, work_start=work_start,
                                span_cycles=before+active, work_cycles=active, pre_work_cycles=before))
            pe_rows.append(row)
        raw[f'report_{step}'] = report
        if cfg['kind'] != 'internal' and cfg['start'] == 'witness':
            witness = report.copy()
            witness[:, :, 7] = step
            # Fields not yet published belong to the previous completed call.
            if step:
                for i in (3, 4, 5, 8, 10, 12, 13, 14):
                    witness[:, :, i] = raw[f'report_{step-1}'][:, :, i]
            else:
                witness[:, :, [3, 4, 5, 8, 10, 12, 13, 14]] = 0
            raw[f'start_{step}'] = witness
        if cfg['pause_at'] == 'after_end_witness':
            raw[f'end_before_pause_{step}'] = report.copy()
        expected.append(dict(step=step, pe=pe_rows,
                             max_span_cycles=max(p['span_cycles'] for row in pe_rows for p in row),
                             max_work_cycles=max(p['work_cycles'] for row in pe_rows for p in row)))
    return raw, expected


def descriptor(value, role):
    return dict(role=role, dtype='uint32', shape=list(value.shape),
                initial_sha256=hashlib.sha256(value.tobytes()).hexdigest())


def call_fixture(cfg):
    events = []
    wall, cpu = 1000, 0
    flags = dict(streaming=False, nonblock=False,
                 data_type='MemcpyDataType.MEMCPY_32BIT', order='MemcpyOrder.ROW_MAJOR')

    def add(name, args=(), kwargs=None, result=None):
        nonlocal wall, cpu
        duration = 1000 + (int(cfg['pause_seconds']*10**9) if name.startswith('pause:') else 0)
        events.append(dict(name=name, args=list(args), kwargs=dict(kwargs or {}), status='pass',
                           wall_start_ns=wall, wall_end_ns=wall+duration,
                           cpu_start_ns=cpu, cpu_end_ns=cpu+100,
                           result_type='int' if type(result) is int else 'NoneType', result=result))
        wall += duration + 100
        cpu += 200

    def pause(step, location):
        if cfg['pause_at'] == location:
            add(f'pause:{step}:{location}', [cfg['pause_seconds']])

    def read(step, name):
        symbol = name if name in ('data', 'seen') else 'report'
        count = 4 if symbol != 'report' else 16
        guard = np.full(4*count, U32, dtype=np.uint32)
        add(f'd2h:{step}:{name}', [descriptor(guard, f'{name}:{step}'), IDS[symbol], 0, 0, 2, 2, count], flags)

    add('construct')
    for name, value in IDS.items():
        add('get_id:'+name, [name], result=value)
    add('load')
    add('run')
    add('init', ['init'], {'nonblock': False})
    for step in range(3):
        wanted = input_words(step)
        preload = (np.bitwise_not(wanted) if cfg['kind'] == 'transfer' else wanted).reshape(-1)
        add(f'preload:{step}', [IDS['data'], descriptor(preload, f'preload:{step}'), 0, 0, 2, 2, 4], flags)
        pause(step, 'before_start')
        command = 'internal' if cfg['kind'] == 'internal' else 'tic'
        arguments = [command, step+1] + ([cfg['work']] if command == 'internal' else [])
        add(f'{command}:{step}', arguments, {'nonblock': cfg['start'] in ('queued', 'wait')})
        if cfg['start'] == 'wait':
            add(f'{command}:{step}:wait', [{'type': 'Task'}])
        if cfg['kind'] != 'internal':
            if cfg['start'] == 'witness':
                read(step, 'start')
            pause(step, 'after_start')
            if cfg['kind'] == 'transfer':
                if cfg['arm'] == 'rectangle':
                    packets = [(0, 0, 2, 2, wanted.reshape(-1))]
                elif cfg['arm'] == 'strips':
                    packets = [(0, y, 2, 1, wanted[y].reshape(-1)) for y in range(2)]
                else:
                    packets = [(x, y, 1, 1, wanted[y, x]) for y in range(2) for x in range(2)]
                for i, (x, y, c, r, payload) in enumerate(packets):
                    name = f'h2d:{step}:{i}'
                    add(name, [IDS['data'], descriptor(payload, name), x, y, c, r, 4],
                        dict(flags, nonblock=cfg['transfer'] != 'blocking'))
                    if cfg['transfer'] == 'wait':
                        add(name+':wait', [{'type': 'Task'}])
                    if i == 0:
                        pause(step, 'between_h2d')
            pause(step, 'before_end')
            add(f'toc:{step}', ['toc', cfg['work']], {'nonblock': False})
        pause(step, 'after_end_return')
        if cfg['pause_at'] == 'after_end_witness':
            read(step, 'end_before_pause')
            pause(step, 'after_end_witness')
        for name in ('report', 'data', 'seen'):
            read(step, name)
    add('stop')
    return events


def npz_bytes(raw):
    output = io.BytesIO()
    np.savez(output, **raw)
    return output.getvalue()


def envelope(cfg=None):
    cfg = cfg or config()
    raw, expected = raw_fixture(cfg)
    calls = call_fixture(cfg)
    manifest = dict(backend='local-sim', params=dict(PARAMS),
                    source_sha256={'clock_source/layout.csl': '1'*64, 'clock_source/pe.csl': '2'*64},
                    host_sha256={'clock_probe.py': '3'*64, 'clock_common.py': '4'*64})
    request = dict(config=cfg, run_id=RUN_ID, manifest_sha256='5'*64)
    meta = dict(schema=1, backend=manifest['backend'], params=dict(PARAMS), config=cfg,
                run_id=RUN_ID, manifest_sha256=request['manifest_sha256'],
                source_sha256=copy.deepcopy(manifest['source_sha256']),
                host_sha256=copy.deepcopy(manifest['host_sha256']), status='pass', steps=expected)
    return meta, calls, raw, manifest, request


def admitted(record):
    meta, calls, raw, manifest, request = record
    payload = npz_bytes(raw)
    # Rehash deliberate mutations so semantic checks, not stale hash checks, fail.
    meta['raw_sha256'] = hashlib.sha256(payload).hexdigest()
    meta['calls_sha256'] = hashlib.sha256(json.dumps(calls, sort_keys=True).encode()).hexdigest()
    return common.validate_capture_content(meta, calls, payload, manifest, request)


def event(calls, name):
    return next(item for item in calls if item['name'] == name)


class ClockRawTests(unittest.TestCase):
    def test_independent_aba_and_u48_wrap(self):
        for cfg in (config(), config(kind='empty'), config(kind='internal', start='blocking', work=3),
                    config(start='wait', transfer='wait', arm='tiles')):
            with self.subTest(cfg=cfg):
                record = envelope(cfg)
                self.assertEqual(admitted(record), {'status': 'pass', 'steps': record[0]['steps']})
                first = record[0]['steps'][0]['pe'][0][0]
                self.assertLess(first['end'], first['start'])
                self.assertEqual(first['work_cycles'], 20*cfg['work']+3)
                self.assertEqual(first['pre_work_cycles'], 200)

    def test_full_high16_bits_in_input_and_snapshot(self):
        for name in ('data_0', 'seen_1', 'seen_2'):
            with self.subTest(name=name):
                record = envelope()
                record[2][name][-1, -1, -1] ^= np.uint32(1 << 31)
                with self.assertRaises(ValueError):
                    admitted(record)

    def test_stale_generation_and_reordered_pe_data(self):
        for stale in (True, False):
            record = envelope()
            record[2]['seen_1'] = (record[2]['seen_0'].copy() if stale
                                   else record[2]['seen_1'][::-1].copy())
            with self.subTest(stale=stale), self.assertRaises(ValueError):
                admitted(record)

    def test_report_generations_work_kind_and_guards(self):
        for word in (6, 7, 8, 9, 10, 11, 15):
            record = envelope()
            record[2]['report_2'][1, 1, word] ^= np.uint32(1)
            with self.subTest(word=word), self.assertRaises(ValueError):
                admitted(record)

    def test_start_witness_generation_clock_and_guards(self):
        for word in (0, 6, 7, 11, 15):
            record = envelope()
            record[2]['start_1'][1, 0, word] ^= np.uint32(1)
            with self.subTest(word=word), self.assertRaises(ValueError):
                admitted(record)

    def test_end_witness_is_immutable(self):
        record = envelope(config(pause_at='after_end_witness', pause_seconds=0.25))
        self.assertEqual(admitted(record)['status'], 'pass')
        record[2]['end_before_pause_2'][1, 1, 3] ^= np.uint32(1)
        with self.assertRaisesRegex(ValueError, 'frozen end record identity'):
            admitted(record)

    def test_clock_high_limb_and_active_outside_span(self):
        for mutation in ('high16', 'outside'):
            record = envelope(config(start='blocking'))
            p = record[2]['report_0'][1, 1]
            if mutation == 'high16':
                p[3] = 65536
            else:
                start = int(p[0]) | int(p[1]) << 16 | int(p[2]) << 32
                p[12:15] = limbs((start-10) & U48)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                admitted(record)

    def test_half_range_zero_and_reversed_intervals(self):
        for end, start in ((1, 1), (2**47, 0), (2**47+1, 0), (0, 1)):
            with self.subTest(end=end, start=start), self.assertRaises(ValueError):
                common.delta(end, start)
        self.assertEqual(common.delta(7, U48-2), 10)

    def test_cross_generation_clock_continuity(self):
        record = envelope(config(start='blocking'))
        prior = record[2]['report_0'][0, 0]
        end = int(prior[3]) | int(prior[4]) << 16 | int(prior[5]) << 32
        current = record[2]['report_1'][0, 0]
        current[:3], current[12:15], current[3:6] = limbs(end), limbs(end+200), limbs(end+223)
        with self.assertRaisesRegex(ValueError, 'positive unambiguous clock interval'):
            common.validate_arrays(record[2], PARAMS, record[4]['config'])

    def test_missing_extra_and_wrong_dtype_arrays(self):
        for mutation in ('missing', 'extra', 'dtype'):
            record = envelope()
            if mutation == 'missing':
                del record[2]['seen_2']
            elif mutation == 'extra':
                record[2]['unclaimed'] = record[2]['report_0'].copy()
            else:
                record[2]['data_1'] = record[2]['data_1'].astype(np.uint16)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                admitted(record)


class ClockReceiptTests(unittest.TestCase):
    def test_same_data_requires_matching_call_granularity(self):
        rectangle = envelope(config(arm='rectangle'))
        strips = envelope(config(arm='strips'))
        tiles = envelope(config(arm='tiles'))
        for record in (rectangle, strips, tiles):
            self.assertEqual(admitted(record)['status'], 'pass')
            for name, value in rectangle[2].items():
                np.testing.assert_array_equal(value, record[2][name])
        strips[1][:] = copy.deepcopy(rectangle[1])
        with self.assertRaises(ValueError):
            admitted(strips)

    def test_forged_native_flags(self):
        mutations = (('data_type', 'MemcpyDataType.MEMCPY_16BIT'),
                     ('order', 'MemcpyOrder.COL_MAJOR'), ('streaming', True), ('nonblock', False))
        for key, value in mutations:
            record = envelope()
            event(record[1], 'h2d:1:0')['kwargs'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                admitted(record)

    def test_forged_payload_guard_symbol_and_rectangle(self):
        for mutation in ('payload', 'guard', 'symbol', 'rectangle', 'lookup', 'init'):
            record = envelope()
            if mutation == 'payload':
                event(record[1], 'h2d:0:0')['args'][1]['initial_sha256'] = '0'*64
            elif mutation == 'guard':
                event(record[1], 'd2h:2:seen')['args'][0]['initial_sha256'] = '0'*64
            elif mutation == 'symbol':
                event(record[1], 'd2h:0:seen')['args'][1] = IDS['data']
            elif mutation == 'rectangle':
                event(record[1], 'h2d:0:0')['args'][3] = 1
            elif mutation == 'lookup':
                event(record[1], 'get_id:seen')['result'] = IDS['data']
            else:
                event(record[1], 'init')['kwargs']['nonblock'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                admitted(record)

    def test_missing_or_incomplete_api_event(self):
        for missing in (True, False):
            record = envelope()
            if missing:
                record[1].pop()
            else:
                event(record[1], 'd2h:1:report')['status'] = 'incomplete'
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                admitted(record)

    def test_pause_requires_elapsed_host_time(self):
        record = envelope(config(pause_at='after_start', pause_seconds=0.25))
        self.assertEqual(admitted(record)['status'], 'pass')
        pause = event(record[1], 'pause:1:after_start')
        pause['wall_end_ns'] = pause['wall_start_ns'] + 1000
        with self.assertRaises(ValueError):
            admitted(record)

    def test_time_fields_are_finite_nonnegative_integers(self):
        for field, value in (('wall_end_ns', float('inf')), ('wall_end_ns', 1.5),
                             ('cpu_start_ns', -1), ('cpu_end_ns', float('inf'))):
            record = envelope()
            record[1][-1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                admitted(record)

    def test_rehashed_capture_identity_and_cycle_claim_forgery(self):
        for mutation in ('run_id', 'source', 'backend', 'cycles'):
            record = envelope()
            if mutation == 'run_id':
                record[0]['run_id'] = '4ac6aef7-3867-43ec-a1a3-bda414511f2b'
            elif mutation == 'source':
                record[0]['source_sha256']['clock_source/pe.csl'] = '0'*64
            elif mutation == 'backend':
                record[0]['backend'] = 'hardware'
            else:
                record[0]['steps'][1]['max_work_cycles'] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                admitted(record)

    def test_numerical_failure_and_incomplete_remain_distinct(self):
        record = envelope()
        record[2]['seen_0'][0, 0, 0] ^= np.uint32(1 << 16)
        record[0]['status'] = 'observed_numeric_fail'
        self.assertEqual(admitted(record)['status'], 'observed_numeric_fail')
        good = envelope()
        good[0]['status'] = 'observed_numeric_fail'
        with self.assertRaises(ValueError):
            admitted(good)
        partial = envelope()
        partial[0]['status'] = 'incomplete'
        partial[1].clear()
        partial[2].clear()
        self.assertEqual(admitted(partial), {'status': 'incomplete', 'numerical_validation': 'not complete'})


class ClockArchiveTests(unittest.TestCase):
    @staticmethod
    def raw_npy(header, payload=b''):
        body = repr(header).encode('latin1')
        body += b' ' * ((-(10+len(body)+1)) % 64) + b'\n'
        return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(body)) + body + payload

    def test_npy_shape_bomb_rejected_before_numpy_load(self):
        for header, payload in (
            ({'descr': '<u4', 'fortran_order': False, 'shape': (2**40, 1, 1)}, b''),
            ({'descr': '<u4', 'fortran_order': False, 'shape': (2, 2, 4)}, b'\0'*4),
            ({'descr': '|O', 'fortran_order': False, 'shape': (1, 1, 1)}, b'\0'*8)):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w') as archive:
                archive.writestr('data_0.npy', self.raw_npy(header, payload))
            with self.subTest(header=header), mock.patch.object(np, 'load', side_effect=AssertionError('must preflight first')):
                with self.assertRaises(ValueError):
                    common.read_raw(stream.getvalue())

    def test_bounded_valid_npz_round_trip(self):
        raw, _ = raw_fixture(config())
        decoded = common.read_raw(npz_bytes(raw))
        self.assertEqual(set(decoded), set(raw))
        for name, value in raw.items():
            np.testing.assert_array_equal(value, decoded[name])


if __name__ == '__main__':
    unittest.main()
