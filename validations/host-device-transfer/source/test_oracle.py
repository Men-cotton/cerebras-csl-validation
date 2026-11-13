import hashlib
import json
import unittest

import numpy as np

import oracle


def payload_bytes_by_pe(frame, pes, words):
    payload = frame.reshape(pes, words + 1)[:, 1:]
    return payload.astype('<u4').view(np.uint8)


class TransferContractTests(unittest.TestCase):
    def test_campaigns_have_independent_controls_and_no_smoke_arms(self):
        self.assertEqual({name: len(cases) for name, cases in oracle.CAMPAIGNS.items()},
                         dict(geometry=18, buffers=10, split=5, broadcast=4))
        flattened = sum(oracle.CAMPAIGNS.values(), [])
        self.assertEqual(len(set(flattened)), 37)
        self.assertEqual(set(flattened), set(oracle.CASES))
        self.assertTrue(set(oracle.CASES).isdisjoint(oracle.SMOKE_CASES))
        self.assertEqual(len(oracle.SMOKE_CASES), 7)
        for case in oracle.CASES:
            plan = oracle.plan(case)
            self.assertEqual(plan['pes'], plan['width'] * plan['height'])
            self.assertGreaterEqual(plan['pes'], 4096)
            self.assertEqual(plan['capacity'], 4097)
            self.assertEqual(plan['generation_stride'], 16)
            self.assertEqual((plan['warmup'], plan['repeats']), (10, 30))
            self.assertEqual(json.loads(json.dumps(plan)), plan)

    def test_exact_ratio_rounding_is_minimal_and_both_orientations_are_present(self):
        for ratio, shape in ((1, (64, 64)), (2, (46, 92)), (4, (32, 128)),
                             (8, (23, 184)), (16, (16, 256))):
            self.assertEqual(oracle.rounded_shape(ratio), shape)
            width, height = shape
            self.assertEqual(height, ratio * width)
            self.assertGreaterEqual(width * height, 4096)
            self.assertLess((width - 1) * (width - 1) * ratio, 4096)
            for w, h in (shape, shape[::-1]):
                for channels in (1, 16):
                    plan = oracle.plan(f'geometry-w{w}-h{h}-c{channels}')
                    self.assertEqual((plan['width'], plan['height']), (w, h))
                    self.assertEqual(plan['extra_pes'], 136 if ratio in (2, 8) else 0)
        rounded = oracle.geometry(64 << 20, oracle.plan('geometry-w46-h92-c16'))
        self.assertEqual(rounded['payload_words_per_pe'], 3965)
        self.assertEqual(rounded['padding_bytes'], 10656)
        square = oracle.geometry(64 << 20, oracle.plan('geometry-w64-h64-c16'))
        self.assertEqual((square['payload_words_per_pe'], square['padding_bytes']), (4096, 0))

    def test_buffer_axis_and_compiler_contract(self):
        pairs = {(oracle.CASES[name]['west'], oracle.CASES[name]['east'])
                 for name in oracle.CAMPAIGNS['buffers']}
        self.assertEqual(pairs, {(0, 0), (1, 0), (0, 1), (1, 1), (4, 0), (0, 4),
                                (4, 4), (16, 0), (0, 16), (16, 16)})
        expected = dict(WIDTH=64, HEIGHT=64, CAPACITY=4097, REPLY_ACK=0,
                        MEMCPYH2D_DATA_1_ID=0, MEMCPYD2H_DATA_1_ID=1)
        for name in oracle.CAMPAIGNS['buffers']:
            self.assertEqual(oracle.compiler_params(oracle.plan(name)), expected)
        ack = oracle.compiler_params(oracle.plan('broadcast-col-bcast'))
        self.assertEqual(ack, {**expected, 'REPLY_ACK': 1})

    def test_size_orders_and_smoke_schedules_are_explicit(self):
        for case in oracle.CASES:
            ascending = oracle.plan(case)
            sizes = ascending['sizes']
            first_power = 6 if ascending['campaign'] == 'broadcast' else 0
            self.assertEqual(sizes, [0] + [1 << bit for bit in range(first_power, 27)])
            self.assertEqual(oracle.plan(case, 1)['sizes'], list(reversed(sizes)))
            shuffled = oracle.plan(case, 2)['sizes']
            self.assertEqual(sorted(shuffled), sizes)
            self.assertNotEqual(shuffled, sizes)
            self.assertNotEqual(shuffled, list(reversed(sizes)))
            self.assertEqual(shuffled, oracle.plan(case, 2)['sizes'])
        for case in oracle.SMOKE_CASES:
            with self.assertRaises(ValueError):
                oracle.plan(case)
            plan = oracle.plan(case, smoke=True)
            minimum = 1 if plan['pattern'] == 'unique' else 2
            self.assertEqual(plan['sizes'], [0, minimum, 64, 1024])
            self.assertEqual((plan['width'], plan['height'], plan['channels']), (2, 2, 1))
            self.assertEqual((plan['warmup'], plan['repeats']), (1, 3))

    def test_zero_and_tiny_echo_account_actual_carriers(self):
        plan = oracle.plan('smoke-echo', smoke=True)
        zero = oracle.geometry(0, plan)
        self.assertEqual((zero['payload_bytes'], zero['frame_words'], zero['padding_bytes']), (0, 1, 0))
        self.assertEqual((zero['h2d_transfer_bytes'], zero['d2h_transfer_bytes'], zero['readback_bytes']),
                         (16, 16, 0))
        tiny = oracle.geometry(1, plan)
        self.assertEqual(tiny['payload_words_per_pe'], 1)
        self.assertEqual((tiny['h2d_payload_bytes'], tiny['h2d_padding_bytes'], tiny['padding_bytes']),
                         (1, 15, 15))
        self.assertEqual((tiny['control_h2d_bytes'], tiny['control_d2h_bytes']), (16, 16))
        self.assertEqual((tiny['h2d_transfer_bytes'], tiny['d2h_transfer_bytes']), (32, 32))
        sent, expected = oracle.frames(1, 17, plan)[0]
        np.testing.assert_array_equal(sent, [17, 0xc3, 17, 0, 17, 0, 17, 0])
        np.testing.assert_array_equal(expected, [17, 0xc3, 0x85ebca7a, 0,
                                               0x0bd794c7, 0, 0x91c35f50, 0])

    def test_all_pe_headers_are_fresh_and_distinguish_swaps(self):
        for case in oracle.SMOKE_CASES:
            plan = oracle.plan(case, smoke=True)
            for chunk, (sent, expected) in enumerate(oracle.frames(0, 17, plan)):
                self.assertTrue(np.all(sent == 17 + chunk))
                headers = [((pe * 0x85ebca6b) & 0xffffffff) ^ (17 + chunk) for pe in range(4)]
                np.testing.assert_array_equal(expected, headers)
                self.assertNotEqual(oracle.digest(expected), oracle.digest(expected[::-1]))
                damaged = expected.copy()
                damaged[1] = damaged[0]
                self.assertNotEqual(oracle.digest(expected), oracle.digest(damaged))
                newer = oracle.frames(0, 17 + oracle.GENERATION_STRIDE, plan)[chunk][1]
                self.assertTrue(np.all(newer != expected))

    def test_split_zero_and_subchunk_payloads_have_equal_extents_and_exact_totals(self):
        plan = oracle.plan('smoke-split16', smoke=True)
        zero = oracle.geometry(0, plan)
        self.assertEqual(zero['per_chunk_payload_bytes'], [0] * 16)
        self.assertEqual((zero['h2d_transfer_bytes'], zero['d2h_transfer_bytes']), (256, 256))
        tiny = oracle.geometry(1, plan)
        self.assertEqual(tiny['per_chunk_payload_bytes'], [1] + [0] * 15)
        self.assertEqual((tiny['frame_words'], tiny['control_h2d_bytes'], tiny['h2d_padding_bytes']),
                         (2, 256, 255))
        self.assertEqual(tiny['h2d_transfer_bytes'], 512)
        parts = oracle.frames(1, 17, plan)
        self.assertEqual({sent.size for sent, _ in parts}, {8})
        self.assertEqual(parts[0][0].reshape(4, 2)[0, 1], 0xc3)
        for sent, _ in parts[1:]:
            self.assertTrue(np.all(sent.reshape(4, 2)[:, 1:] == 0))
        eight = oracle.geometry(8, plan)
        self.assertEqual(eight['per_chunk_payload_bytes'], [2, 2, 2, 2] + [0] * 12)
        base = oracle.frames(8, 17, oracle.plan('smoke-echo', smoke=True))[0][0]
        base_bytes = payload_bytes_by_pe(base, 4, 1)
        parts = oracle.frames(8, 17, plan)
        for index, (sent, _) in enumerate(parts):
            actual = payload_bytes_by_pe(sent, 4, 1)
            if index < 4:
                np.testing.assert_array_equal(actual[:, 0], base_bytes[:, index])
                self.assertTrue(np.all(actual[:, 1:] == 0))
            else:
                self.assertTrue(np.all(actual == 0))

    def test_split_preserves_identical_per_pe_payload_across_counts_and_rounds(self):
        for size in (0, 1, 2, 4, 8, 16, 64, 1024):
            for generation in (17, 17 + oracle.GENERATION_STRIDE):
                baseline_plan = oracle.plan('split-k1')
                baseline_shape = oracle.geometry(size, baseline_plan)
                baseline = oracle.frames(size, generation, baseline_plan)[0][0]
                wanted = payload_bytes_by_pe(baseline, 4096, baseline_shape['payload_words_per_pe'])
                span = wanted.shape[1]
                valid = (np.arange(4096 * span).reshape(4096, span) < size)
                for count in (1, 2, 4, 8, 16):
                    plan = oracle.plan(f'split-k{count}')
                    shape = oracle.geometry(size, plan)
                    parts = oracle.frames(size, generation, plan)
                    chunk_span = (span + count - 1) // count
                    slices, payload_counts = [], []
                    for index, (sent, _) in enumerate(parts):
                        begin, end = min(index * chunk_span, span), min((index + 1) * chunk_span, span)
                        actual = payload_bytes_by_pe(sent, 4096, shape['payload_words_per_pe'])
                        slices.append(actual[:, :end-begin])
                        self.assertTrue(np.all(actual[:, end-begin:] == 0))
                        payload_counts.append(int(np.count_nonzero(valid[:, begin:end])))
                    got = np.concatenate(slices, axis=1)
                    np.testing.assert_array_equal(got, wanted, err_msg=str((size, generation, count)))
                    self.assertEqual(shape['per_chunk_payload_bytes'], payload_counts)
                    self.assertEqual(sum(shape['per_chunk_payload_bytes']), size)
                    self.assertEqual(sum(sent.nbytes for sent, _ in parts), shape['h2d_transfer_bytes'])
                    self.assertEqual(sum(expected.nbytes for _, expected in parts), shape['d2h_transfer_bytes'])

    def test_split_reconstructs_full_words_on_each_small_shape_pe(self):
        whole = oracle.plan('smoke-echo', smoke=True)
        split = oracle.plan('smoke-split16', smoke=True)
        for size in (64, 1024, 65536):
            base_shape, split_shape = oracle.geometry(size, whole), oracle.geometry(size, split)
            base = oracle.frames(size, 91, whole)[0][0]
            wanted = payload_bytes_by_pe(base, 4, base_shape['payload_words_per_pe'])
            width = (wanted.shape[1] + 15) // 16
            slices = [payload_bytes_by_pe(sent, 4, split_shape['payload_words_per_pe'])[:, :width]
                      for sent, _ in oracle.frames(size, 91, split)]
            np.testing.assert_array_equal(np.concatenate(slices, axis=1), wanted)
            # A global-stream split would move bytes between PEs at these sizes.
            self.assertFalse(np.array_equal(wanted[0], wanted[1]))

    def test_broadcast_and_replicated_upload_have_identical_receiver_images(self):
        for pattern in ('row', 'col'):
            compact = oracle.plan(f'smoke-{pattern}-bcast', smoke=True)
            expanded = oracle.plan(f'smoke-{pattern}-unicast', smoke=True)
            for size in (0, 2, 4, 64, 1024, 65536):
                sent, expected = oracle.frames(size, 17, compact)[0]
                normal, wanted = oracle.frames(size, 17, expanded)[0]
                np.testing.assert_array_equal(expected, wanted)
                self.assertEqual(sent.nbytes * 2, normal.nbytes)
                shape = oracle.geometry(size, compact)
                self.assertEqual(shape['h2d_payload_bytes'], size // 2)
                self.assertEqual(shape['h2d_transfer_bytes'], sent.nbytes)
                self.assertEqual(shape['d2h_transfer_bytes'], 16)
                self.assertEqual(shape['readback_bytes'], expected.nbytes)
                self.assertEqual(shape['delivered_payload_bytes'], size)
                self.assertEqual(shape['control_h2d_bytes'], 8)
                self.assertEqual(shape['delivered_control_bytes'], 16)
                receiver = expected.reshape(2, 2, shape['frame_words'])
                if pattern == 'row':
                    np.testing.assert_array_equal(receiver[:, 0, 1:], receiver[:, 1, 1:])
                else:
                    np.testing.assert_array_equal(receiver[0, :, 1:], receiver[1, :, 1:])
        row = oracle.frames(2, 17, oracle.plan('smoke-row-bcast', smoke=True))[0][1].reshape(4, 2)
        col = oracle.frames(2, 17, oracle.plan('smoke-col-bcast', smoke=True))[0][1].reshape(4, 2)
        np.testing.assert_array_equal(row[:, 1], [0xc3, 0xc3, 0, 0])
        np.testing.assert_array_equal(col[:, 1], [0xc3, 0, 0xc3, 0])

    def test_full_frame_and_padding_preserve_payload_without_aliasing(self):
        plan = oracle.plan('smoke-echo', smoke=True)
        for size in (1, 2, 4, 8, 65536):
            shape = oracle.geometry(size, plan)
            sent, expected = oracle.frames(size, 0xffffffff, plan)[0]
            self.assertFalse(np.shares_memory(sent, expected))
            self.assertTrue(sent.flags.c_contiguous)
            self.assertTrue(expected.flags.c_contiguous)
            payload = sent.reshape(4, shape['frame_words'])[:, 1:].astype('<u4').tobytes()
            self.assertEqual(payload[size:], bytes(shape['padding_bytes']))
            np.testing.assert_array_equal(sent.reshape(4, -1)[:, 1:], expected.reshape(4, -1)[:, 1:])
        self.assertEqual(oracle.geometry(65536, plan)['frame_words'], 4097)

    def test_digest_and_plan_copies_do_not_depend_on_host_byte_order(self):
        raw = b'\x04\x03\x02\x01\xff\xff\xff\xff'
        for dtype in ('<u4', '>u4'):
            self.assertEqual(oracle.digest(np.array([0x01020304, 0xffffffff], dtype=dtype)),
                             hashlib.sha256(raw).hexdigest())
        changed = oracle.plan('geometry-w64-h64-c16')
        changed['sizes'].append(3)
        changed['width'] = 2
        original = oracle.plan('geometry-w64-h64-c16')
        self.assertNotIn(3, original['sizes'])
        self.assertEqual(original['width'], 64)

    def test_invalid_inputs_and_changed_configuration_fail_closed(self):
        plan = oracle.plan('smoke-echo', smoke=True)
        for size in (True, False, -1, 3, 65537, 1 << 27, 1.0, '1', None, np.int64(1)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                oracle.geometry(size, plan)
        for generation in (True, -1, 1 << 32, 1.0, '1', None, np.uint32(1)):
            with self.subTest(generation=generation), self.assertRaises(ValueError):
                oracle.frames(0, generation, plan)
        with self.assertRaises(ValueError):
            oracle.frames(0, 0xffffffff - 14, oracle.plan('smoke-split16', smoke=True))
        oracle.frames(0, 0xffffffff - 15, oracle.plan('smoke-split16', smoke=True))
        for case in (True, None, [], '', 'split-k3', 'geometry-w45-h90-c16'):
            with self.subTest(case=case), self.assertRaises(ValueError):
                oracle.plan(case)
        for order in (True, False, -1, 3, 0.0, '0', None):
            with self.subTest(order=order), self.assertRaises(ValueError):
                oracle.plan('split-k1', order)
        for smoke in (0, 1, None, 'yes'):
            with self.subTest(smoke=smoke), self.assertRaises(ValueError):
                oracle.plan('split-k1', smoke=smoke)
        for key, value in (('width', 3), ('channels', True), ('reply', 'echo'), ('mode', 'col-bcast')):
            changed = oracle.plan('broadcast-row-bcast')
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                oracle.geometry(0, changed)
        with self.assertRaises(ValueError):
            oracle.geometry(1, oracle.plan('broadcast-row-bcast'))
        for array in ([1], np.array([1], np.int32), np.array([1], np.uint64), np.array([1.0])):
            with self.assertRaises(ValueError):
                oracle.digest(array)


if __name__ == '__main__':
    unittest.main()
