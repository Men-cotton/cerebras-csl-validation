"""Synthetic marker/order/diagnostic admission; no SDK, PMU or device execution."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_common as common
import test_clock_common as base


def rectangles(cfg):
    if cfg['arm'] == 'rectangle':
        result = [(0, 0, 2, 2)]
    elif cfg['arm'] == 'strips':
        result = [(0, 0, 2, 1), (0, 1, 2, 1)]
    else:
        result = [(0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)]
    return result[::-1] if cfg.get('piece_order') == 'reverse' else result


def marked_raw(cfg):
    raw, expected = base.raw_fixture(cfg)
    for step in range(3):
        wanted = base.input_words(step)
        state = np.bitwise_not(wanted) if cfg['kind'] == 'transfer' else wanted.copy()
        records = np.full((2, 2, 4, 8), 0xffffffff, dtype=np.uint32)
        derived = []
        for index, (x0, y0, width, height) in enumerate(rectangles(cfg)):
            if cfg['kind'] == 'transfer':
                state[y0:y0+height, x0:x0+width] = wanted[y0:y0+height, x0:x0+width]
            rows = []
            for y in range(2):
                row = []
                for x in range(2):
                    start = expected[step]['pe'][y][x]['start']
                    elapsed = (index + 1) * 20
                    timestamp = (start + elapsed) & base.U48
                    first, last = int(state[y, x, 0]), int(state[y, x, -1])
                    records[y, x, index] = base.limbs(timestamp) + [step+1, index, first, last, 0x4d41524b]
                    row.append(dict(timestamp=timestamp, from_start_cycles=elapsed,
                                    from_previous_cycles=20, first_word=first, last_word=last))
                rows.append(row)
            derived.append(dict(index=index, piece_rectangle=[x0, y0, width, height], pe=rows))
        raw[f'marks_{step}'] = records.reshape(2, 2, 32)
        expected[step]['marks'] = derived
    return raw, expected


def extended_calls(cfg):
    """Extend the independent legacy fixture, without producer sequence helpers."""
    events = []
    flags = dict(streaming=False, nonblock=False,
                 data_type='MemcpyDataType.MEMCPY_32BIT', order='MemcpyOrder.ROW_MAJOR')

    def add(name, args=(), kwargs=None, result=None):
        events.append(dict(name=name, args=list(args), kwargs=dict(kwargs or {}), status='pass',
                           result=result, result_type='int' if type(result) is int else 'NoneType'))

    if cfg.get('host_pmu'):
        add('pmu-empty')
    for event in base.call_fixture(cfg):
        name = event['name']
        if name.startswith('preload:'):
            step = int(name.split(':')[1])
            for i in range(cfg.get('pmu_controls', 0)):
                add(f'pmu-empty:{step}:before:{i}')
        if name.startswith('h2d:') and not name.endswith(':wait'):
            _, step, index = name.split(':')
            step, index = int(step), int(index)
            x, y, c, r = rectangles(cfg)[index]
            payload = base.input_words(step)[y:y+r, x:x+c].copy().reshape(-1)
            event['args'] = [base.IDS['data'], base.descriptor(payload, name), x, y, c, r, 4]
        if name.startswith('toc:') and cfg.get('markers') and cfg['kind'] == 'empty':
            step = int(name.split(':')[1])
            for i in range(len(rectangles(cfg))):
                add(f'mark:{step}:{i}', ['mark', i, step+1], {'nonblock': True})
        events.append(event)
        if name == 'get_id:report' and cfg.get('markers'):
            add('get_id:marks', ['marks'], result=57)
        if name.startswith('h2d:') and cfg.get('markers'):
            parts = name.split(':')
            if name.endswith(':wait') or cfg['transfer'] != 'wait':
                step, i = int(parts[1]), int(parts[2])
                add(f'mark:{step}:{i}', ['mark', i, step+1], {'nonblock': True})
        if name.startswith('d2h:') and name.endswith(':seen'):
            step = int(name.split(':')[1])
            if cfg.get('markers'):
                guard = np.full(128, 0xffffffff, dtype=np.uint32)
                add(f'd2h:{step}:marks', [base.descriptor(guard, f'marks:{step}'), 57, 0, 0, 2, 2, 32], flags)
            for i in range(cfg.get('pmu_controls', 0)):
                add(f'pmu-empty:{step}:after:{i}')
    wall = 1000
    for index, event in enumerate(events):
        duration = 1000 + (int(cfg['pause_seconds']*1e9) if event['name'].startswith('pause:') else 0)
        event.update(wall_start_ns=wall, wall_end_ns=wall+duration,
                     cpu_start_ns=index*200, cpu_end_ns=index*200+100)
        wall += duration + 100
        if cfg.get('host_pmu'):
            event['pmu'] = dict(status='ok', scope='thread-self-user', cycles=900, instructions=400,
                                time_enabled_ns=100, time_running_ns=100, scaled=False)
        if cfg.get('pmu_controls'):
            event['thread_before'] = dict(thread_time_ns=index*200, cpu=2,
                voluntary_context_switches=index, involuntary_context_switches=0,
                minor_faults=0, major_faults=0)
            event['thread_after'] = dict(event['thread_before'], thread_time_ns=index*200+100, cpu=3)
    return events


class ExtensionTests(unittest.TestCase):
    def test_absent_options_retain_legacy_config_and_raw(self):
        self.assertEqual(common.validate_config({}), base.config())
        record = base.envelope()
        self.assertEqual(base.admitted(record)['status'], 'pass')

    def test_optional_config_bounds(self):
        for invalid in ({'piece_order': 'random'}, {'markers': 1},
                        {'kind': 'internal', 'start': 'blocking', 'markers': True},
                        {'pmu_controls': True}, {'pmu_controls': -1}, {'pmu_controls': 9},
                        {'pmu_controls': 1}, {'host_pmu': True, 'pmu_controls': 1.0}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                common.validate_config(invalid)

    def test_partial_prefix_snapshots_forward_reverse_and_empty(self):
        for kind in ('empty', 'transfer'):
            for arm in ('rectangle', 'strips', 'tiles'):
                for order in ('forward', 'reverse'):
                    cfg = base.config(kind=kind, arm=arm, markers=True, piece_order=order)
                    raw, expected = marked_raw(cfg)
                    with self.subTest(kind=kind, arm=arm, order=order):
                        self.assertEqual(common.validate_arrays(raw, base.PARAMS, cfg), expected)
                        common.validate_calls(extended_calls(cfg), base.PARAMS, cfg)

    def test_marker_payload_generation_guard_limb_and_unused_slot(self):
        for word in range(8):
            cfg = base.config(arm='strips', markers=True)
            raw, _ = marked_raw(cfg)
            # Timestamp limb errors are made unambiguously invalid.
            raw['marks_1'][1, 1, word] = 65536 if word < 3 else raw['marks_1'][1, 1, word] ^ np.uint32(1)
            with self.subTest(word=word), self.assertRaises(ValueError):
                common.validate_arrays(raw, base.PARAMS, cfg)
        raw, _ = marked_raw(cfg)
        raw['marks_0'][0, 0, 16] = 0
        with self.assertRaisesRegex(ValueError, 'unused marker'):
            common.validate_arrays(raw, base.PARAMS, cfg)

    def test_marker_order_and_window_are_same_pe_intervals(self):
        cfg = base.config(arm='tiles', markers=True)
        for mutation in ('equal', 'reverse', 'after_work'):
            raw, expected = marked_raw(cfg)
            first = expected[0]['marks'][0]['pe'][0][0]['timestamp']
            timestamp = first if mutation == 'equal' else (first-1) & base.U48
            if mutation == 'after_work':
                timestamp = expected[0]['pe'][0][0]['work_start']
            raw['marks_0'][0, 0, 8:11] = base.limbs(timestamp)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                common.validate_arrays(raw, base.PARAMS, cfg)

    def test_reverse_submission_requires_reversed_payload_and_rectangle(self):
        cfg = base.config(arm='tiles', piece_order='reverse')
        common.validate_calls(extended_calls(cfg), base.PARAMS, cfg)
        with self.assertRaises(ValueError):
            common.validate_calls(base.call_fixture(cfg), base.PARAMS, cfg)

    def test_exact_marker_call_order_and_transfer_flags(self):
        cfg = base.config(arm='strips', transfer='wait', markers=True,
                          pause_at='between_h2d', pause_seconds=0.01)
        calls = extended_calls(cfg)
        common.validate_calls(calls, base.PARAMS, cfg)
        names = [call['name'] for call in calls]
        i = names.index('h2d:0:0')
        self.assertEqual(names[i:i+4], ['h2d:0:0', 'h2d:0:0:wait', 'mark:0:0', 'pause:0:between_h2d'])
        for mutation in ('flag', 'generation', 'readback'):
            bad = copy.deepcopy(calls)
            if mutation == 'readback':
                base.event(bad, 'd2h:0:marks')['args'][-1] -= 1
            elif mutation == 'flag':
                base.event(bad, 'mark:0:0')['kwargs']['nonblock'] = False
            else:
                base.event(bad, 'mark:0:0')['args'][-1] = 3
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                common.validate_calls(bad, base.PARAMS, cfg)

    def test_interleaved_controls_are_outside_all_three_windows(self):
        cfg = base.config(markers=True, host_pmu=True, pmu_controls=2)
        calls = extended_calls(cfg)
        common.validate_calls(calls, base.PARAMS, cfg)
        names = [call['name'] for call in calls]
        self.assertEqual(sum(n.startswith('pmu-empty') for n in names), 13)
        for step in range(3):
            self.assertLess(names.index(f'pmu-empty:{step}:before:1'), names.index(f'preload:{step}'))
            self.assertGreater(names.index(f'pmu-empty:{step}:after:0'), names.index(f'd2h:{step}:marks'))
        bad = copy.deepcopy(calls)
        bad[-1]['thread_after']['thread_time_ns'] = 0
        with self.assertRaisesRegex(ValueError, 'monotonic thread'):
            common.validate_calls(bad, base.PARAMS, cfg)
        bad = copy.deepcopy(calls)
        del bad[-1]['thread_after']
        with self.assertRaisesRegex(ValueError, 'diagnostics match'):
            common.validate_calls(bad, base.PARAMS, cfg)

    def test_unavailable_thread_fields_and_changing_cpu_are_admitted(self):
        fields = dict(thread_time_ns=None, cpu=None, voluntary_context_switches=None,
                      involuntary_context_switches=None, minor_faults=None, major_faults=None)
        common.validate_thread_samples(fields, fields)
        common.validate_thread_samples(dict(fields, cpu=10), dict(fields, cpu=2))
        with self.assertRaises(ValueError):
            common.validate_thread_samples(dict(fields, cpu=-1), fields)

    def test_capture_claimed_marker_derivation_is_checked_against_raw(self):
        cfg = base.config(arm='strips', markers=True)
        record = list(base.envelope(cfg))
        record[2], record[0]['steps'] = marked_raw(cfg)
        record[1] = extended_calls(cfg)
        self.assertEqual(base.admitted(record)['status'], 'pass')
        record[0]['steps'][0]['marks'][0]['pe'][0][0]['from_previous_cycles'] += 1
        with self.assertRaisesRegex(ValueError, 'raw-derived'):
            base.admitted(record)


if __name__ == '__main__':
    unittest.main()
