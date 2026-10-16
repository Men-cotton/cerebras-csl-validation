"""Host-only checks for capture admission; no Cerebras runtime is imported."""
import copy
from pathlib import Path
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnostic as diag


MARKER = 0xfefedcba
UNUSED = 0xcafebabe
PAYLOAD_GUARDS = [0x12345678, 0x23456789, 0x6789abcd, 0x789abcde]
MASK48 = (1 << 48) - 1


def f32_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def input_word(panel, pe, index):
    return 257 + 17*index + 100*pe + (1000 if panel == 1 else 0)


def input_value(panel, pe, index):
    return (64 + 3*index/16 + 4*pe) * (-0.5 if panel == 1 else 1)


def oracle_stream(panel, off, n, packing):
    """Build wire words independently of diagnostic.expected_stream."""
    rows = [input_word(panel, 0, off+i) for i in range(n)]
    words = [n]
    if packing == 'scalar16':
        words.extend(rows)
    else:
        for i in range(0, n, 2):
            high = rows[i+1] if i+1 < n else 0
            words.append(rows[i] + high*65536)
    words.extend(f32_bits(input_value(panel, 0, off+i)) for i in range(n))
    return words + [MARKER]


def observed_fixture(panel=0, off=3, n=8, mode=0, variant='payload', packing='packed16', stream=None):
    """Represent a completed capture, including untouched memory on both PEs."""
    observed = {
        'source_rows': np.array([[[input_word(panel, pe, i) for i in range(32)]
                                 for pe in range(2)]], dtype=np.uint16),
        'source_values': np.array([[[input_value(panel, pe, i) for i in range(32)]
                                   for pe in range(2)]], dtype=np.float32),
    }
    for name in ('local_copy', 'compact_rows', 'received_rows'):
        observed[name] = np.full((1, 2, 20), 0xd00d, dtype=np.uint16)
        observed[name][:, :, [0, 1, 18, 19]] = [0x1234, 0x2345, 0x6789, 0x789a]
    observed['received_values'] = np.full((1, 2, 20), -12345.5, dtype=np.float32)
    observed['received_values'][:, :, [0, 1, 18, 19]] = [-10001, -10002, -10003, -10004]
    for pe in range(2):
        observed['local_copy'][0, pe, 2:2+n] = observed['source_rows'][0, pe, off:off+n]
    if mode:
        observed['compact_rows'][0, 0, 2:2+n] = observed['source_rows'][0, 0, off:off+n]
    if variant != 'payload':
        observed['received_rows'][0, 1, 2:2+n] = observed['source_rows'][0, 0, off:off+n]
        observed['received_values'][0, 1, 2:2+n] = observed['source_values'][0, 0, off:off+n]
    callbacks = 3 if n else 1
    observed['status'] = np.array([[[0, 4, off, n, mode, 0, callbacks, 0],
                                    [0, 4, off, n, mode, n, 0, callbacks]]], dtype=np.uint32)
    observed['diag_state'] = np.zeros((1, 2, 8), dtype=np.uint32)
    observed['diag_state'][:, :, 0] = {'control': 0, 'events': 1, 'payload': 2}[variant]
    observed['diag_state'][:, :, 6] = 1
    observed['rx_payload'] = np.full((1, 2, 68), UNUSED, dtype=np.uint32)
    observed['rx_payload'][:, :, [0, 1, 66, 67]] = PAYLOAD_GUARDS
    observed['diag_events'] = np.zeros((1, 2, 128), dtype=np.uint32)
    if variant == 'payload':
        words = oracle_stream(panel, off, n, packing) if stream is None else stream
        observed['rx_payload'][0, 1, 2:2+len(words)] = words
        observed['diag_state'][0, 1, 1] = len(words)
        observed['diag_state'][0, 1, 3] = 1
        observed['status'][0, 0, 6] = callbacks+1
        observed['status'][0, 1, 7] = 0
    if variant == 'events':
        ids = [0, 1, 2, 3, 4] if n else [0, 1, 4]
        phases = [1, 1, 2, 3, 4] if n else [1, 1, 4]
        observed['diag_state'][:, :, 4] = len(ids)
        events = observed['diag_events'].reshape(2, 16, 8)
        for pe in range(2):
            for i, (event, phase) in enumerate(zip(ids, phases)):
                stamp = (MASK48-9 + pe*123456 + i*20) & MASK48
                events[pe, i] = [event, phase, stamp & 65535,
                                 (stamp >> 16) & 65535, stamp >> 32, 0x0004, 0x0400, 0]
    return observed


class StreamOracleTests(unittest.TestCase):
    def test_literal_length_eight_stream(self):
        expected = [8, 0x01450134, 0x01670156, 0x01890178, 0x01ab019a,
                    0x42812000, 0x42818000, 0x4281e000, 0x42824000,
                    0x4282a000, 0x42830000, 0x42836000, 0x4283c000, MARKER]
        np.testing.assert_array_equal(diag.expected_stream(0, 3, 8, 0, 'packed16'),
                                      np.array(expected, dtype=np.uint32))

    def test_controls_zero_tail_maximum_and_aba(self):
        cases = [(31, 0, 0), (31, 1, 0), (3, 7, 0), (2, 9, 0), (3, 8, 1), (16, 16, 0)]
        for panel in (0, 1, 2):
            for off, n, mode in cases:
                for packing in ('scalar16', 'packed16'):
                    with self.subTest(panel=panel, case=(off, n, mode), packing=packing):
                        got = diag.expected_stream(panel, off, n, mode, packing)
                        self.assertEqual(got.dtype, np.uint32)
                        self.assertEqual(got.tolist(), oracle_stream(panel, off, n, packing))
                        observed = observed_fixture(panel, off, n, mode, packing=packing)
                        errors, details = diag.analyze_diagnostics(observed, panel, off, n, mode, 'payload')
                        self.assertFalse(errors, errors)
                        self.assertIn(packing, details['matching_representations'])
        np.testing.assert_array_equal(diag.expected_stream(0, 3, 8, 0, 'packed16'),
                                      diag.expected_stream(2, 3, 8, 1, 'packed16'))
        self.assertNotEqual(diag.expected_stream(0, 3, 8, 0, 'packed16').tolist(),
                            diag.expected_stream(1, 3, 8, 0, 'packed16').tolist())

    def test_short_stream_has_ambiguous_representation(self):
        for n in (0, 1):
            observed = observed_fixture(n=n)
            errors, details = diag.analyze_diagnostics(observed, 0, 3, n, 0, 'payload')
            self.assertFalse(errors)
            self.assertEqual(set(details['matching_representations']), {'scalar16', 'packed16'})

    def test_reject_invalid_case_and_packing(self):
        for off, n, mode, packing in [(-1, 8, 0, 'packed16'), (31, 2, 0, 'packed16'),
                                     (3, 17, 0, 'packed16'), (3, -1, 0, 'packed16'),
                                     (3, 8, 2, 'packed16'), (3, 8, 0, 'auto')]:
            with self.subTest(case=(off, n, mode, packing)), self.assertRaises(ValueError):
                diag.expected_stream(0, off, n, mode, packing)


class PayloadAdmissionTests(unittest.TestCase):
    def analyze(self, observed, panel=0, variant='payload'):
        return diag.analyze_diagnostics(observed, panel, 3, 8, 0, variant)

    def test_known_extra_zero_boundary_is_not_accepted_as_packing(self):
        words = [8, 0x01450134, 0x01670156, 0x01890178, 0x0000019a, 0x000001ab,
                 0x42812000, 0x42818000, 0x4281e000, 0x42824000,
                 0x4282a000, 0x42830000, 0x42836000, 0x4283c000, MARKER]
        observed = observed_fixture(stream=words)
        errors, details = self.analyze(observed)
        self.assertEqual(set(errors), {'payload_stream'})
        self.assertEqual(details['matching_representations'], [])
        self.assertEqual(details['payload_words'][4:6], ['0x0000019a', '0x000001ab'])

    def test_missing_duplicate_reordered_and_mixed_words(self):
        normal = oracle_stream(0, 3, 8, 'packed16')
        mutations = [normal[:2]+normal[3:], normal[:2]+[normal[2]]+normal[2:],
                     normal[:2]+[normal[3], normal[2]]+normal[4:],
                     normal[:1]+[308, 325]+normal[2:]]
        for words in mutations:
            with self.subTest(words=words):
                errors, _ = self.analyze(observed_fixture(stream=words))
                self.assertIn('payload_stream', errors)

    def test_marker_is_exactly_once_at_end(self):
        normal = oracle_stream(0, 3, 8, 'packed16')
        mutations = [normal[:-1], normal+[MARKER], [MARKER]+normal[1:], normal+[123]]
        for words in mutations:
            with self.subTest(words=words):
                errors, _ = self.analyze(observed_fixture(stream=words))
                self.assertIn('payload_marker', errors)

    def test_reject_bad_count_overflow_marker_count_done_and_reserved(self):
        for index, value in [(1, 0), (1, 1), (1, 65), (1, 65535), (2, 1), (3, 0), (3, 2),
                             (4, 1), (5, 1), (6, 0), (7, 1)]:
            with self.subTest(state_index=index, value=value):
                observed = observed_fixture()
                observed['diag_state'][0, 1, index] = value
                errors, _ = self.analyze(observed)
                self.assertTrue(errors)

    def test_sender_storage_and_receiver_guards_unused_tail(self):
        for pe, index in [(0, 2), (0, 0), (0, 67), (1, 0), (1, 1), (1, 66), (1, 67), (1, 20)]:
            with self.subTest(pe=pe, index=index):
                observed = observed_fixture()
                observed['rx_payload'][0, pe, index] ^= np.uint32(1)
                errors, _ = self.analyze(observed)
                self.assertIn('payload_storage', errors)

    def test_stale_a_payload_rejected_in_b_panel(self):
        observed = observed_fixture(panel=1, stream=oracle_stream(0, 3, 8, 'packed16'))
        errors, _ = self.analyze(observed, panel=1)
        self.assertEqual(set(errors), {'payload_stream'})

    def test_source_local_copy_output_guard_and_completion_checked(self):
        for key, pe, index in [('source_rows', 0, 0), ('source_values', 1, 31),
                               ('local_copy', 0, 2), ('compact_rows', 1, 3),
                               ('received_rows', 1, 18), ('received_values', 0, 19),
                               ('status', 0, 6), ('status', 1, 7)]:
            with self.subTest(key=key, pe=pe, index=index):
                observed = observed_fixture()
                observed[key][0, pe, index] += 1
                errors, _ = self.analyze(observed)
                self.assertIn(key, errors)

    def test_reject_wrong_shape_and_dtype(self):
        for key in ('rx_payload', 'diag_state', 'diag_events', 'source_rows', 'received_values'):
            with self.subTest(key=key, defect='shape'):
                observed = observed_fixture()
                observed[key] = observed[key].reshape(-1)
                with self.assertRaises(ValueError):
                    self.analyze(observed)
            with self.subTest(key=key, defect='dtype'):
                observed = observed_fixture()
                observed[key] = observed[key].astype(np.float64)
                with self.assertRaises(ValueError):
                    self.analyze(observed)

    def test_events_and_control_preserve_original_numeric_failure(self):
        for variant in ('control', 'events'):
            with self.subTest(variant=variant):
                observed = observed_fixture(variant=variant)
                errors, _ = self.analyze(observed, variant=variant)
                self.assertFalse(errors)
                observed['received_rows'][0, 1, 9] = 0
                words = observed['received_values'].view(np.uint32)
                words[0, 1, 3:10] = words[0, 1, 2:9].copy()
                words[0, 1, 2] = 427
                errors, _ = self.analyze(observed, variant=variant)
                self.assertEqual(set(errors), {'received_rows', 'received_values'})
                self.assertEqual(errors['received_rows']['count'], 1)
                self.assertEqual(errors['received_values']['count'], 8)


class EventAdmissionTests(unittest.TestCase):
    def analyze(self, observed, n=8):
        return diag.analyze_diagnostics(observed, 0, 3, n, 0, 'events')

    def test_local_timing_wrap_and_independent_pe_origins(self):
        observed = observed_fixture(variant='events')
        errors, details = self.analyze(observed)
        self.assertFalse(errors)
        for timeline in details['events']:
            self.assertEqual([e['cycles_from_start'] for e in timeline], [0, 20, 40, 60, 80])
            self.assertEqual([e['cycles_from_previous'] for e in timeline], [0, 20, 20, 20, 20])
        self.assertGreater(details['events'][0][0]['timestamp'], details['events'][0][1]['timestamp'])
        self.assertNotEqual(details['events'][0][0]['timestamp'], details['events'][1][0]['timestamp'])

    def test_zero_length_has_header_and_done_only(self):
        observed = observed_fixture(variant='events', n=0)
        errors, details = self.analyze(observed, n=0)
        self.assertFalse(errors)
        self.assertEqual([e['event'] for e in details['events'][0]], [0, 1, 4])

    def test_wrong_sequence_and_phase_rejected(self):
        for column, replacement, expected_error in [(0, 1, 'event_sequence'), (1, 1, 'event_phases')]:
            observed = observed_fixture(variant='events')
            observed['diag_events'].reshape(2, 16, 8)[1, 2, column] = replacement
            errors, _ = self.analyze(observed)
            self.assertIn(expected_error, errors)

    def test_bad_event_count_overflow_tail_reserved_and_word_bounds(self):
        original = observed_fixture(variant='events')
        defects = [('state', 4, 17), ('state', 5, 1), ('reserved', 7, 1),
                   ('stamp', 2, 65536), ('iq', 5, 65536), ('oq', 6, 65536), ('tail', 0, 9)]
        for defect, index, value in defects:
            with self.subTest(defect=defect):
                observed = copy.deepcopy(original)
                if defect == 'state':
                    observed['diag_state'][0, 0, index] = value
                else:
                    row = 5 if defect == 'tail' else 0
                    observed['diag_events'].reshape(2, 16, 8)[0, row, index] = value
                errors, _ = self.analyze(observed)
                self.assertIn('event_record', errors)

    def test_backward_timestamp_rejected(self):
        observed = observed_fixture(variant='events')
        events = observed['diag_events'].reshape(2, 16, 8)
        events[1, 1, 2:5] = events[1, 0, 2:5]
        events[1, 1, 2] -= 1
        errors, _ = self.analyze(observed)
        self.assertIn('event_record', errors)

    def test_short_event_count_cannot_hide_missing_completion(self):
        observed = observed_fixture(variant='events')
        observed['diag_state'][0, 1, 4] = 4
        observed['diag_events'].reshape(2, 16, 8)[1, 4] = 0
        errors, _ = self.analyze(observed)
        self.assertIn('diag_state', errors)
        self.assertIn('event_sequence', errors)

    def test_non_event_variants_reject_stale_events(self):
        for variant in ('control', 'payload'):
            observed = observed_fixture(variant=variant)
            observed['diag_events'][0, 0, 0] = 1
            errors, _ = diag.analyze_diagnostics(observed, 0, 3, 8, 0, variant)
            self.assertIn('unexpected_events', errors)


if __name__ == '__main__':
    unittest.main()
