import hashlib
import json
import math
import unittest

import numpy as np

import oracle


class PingPongContractTests(unittest.TestCase):
    def test_zero_has_only_a_fresh_control_word(self):
        shape = oracle.geometry(0)
        self.assertEqual(shape, {
            "payload_bytes": 0, "active_pes": 1, "active_rows": 1, "active_cols": 1,
            "payload_words_per_pe": 0, "frame_words": 1,
            "control_h2d_bytes": 4, "control_d2h_bytes": 4,
            "padding_bytes": 0, "transfer_bytes_per_direction": 4,
        })
        np.testing.assert_array_equal(oracle.expected_frame(0, 17), [17])
        np.testing.assert_array_equal(oracle.expected_frame(0, 0xffffffff), [0xffffffff])

    def test_geometry_accounts_for_padding_and_topology_boundaries(self):
        for size, pes, words, padding in (
                (1, 1, 1, 3), (2, 1, 1, 2), (4, 1, 1, 0),
                (8192, 1, 2048, 0), (16384, 1, 4096, 0),
                (0, 64, 0, 0), (1, 64, 1, 255), (4, 4, 1, 12),
                (16384, 4, 1024, 0), (32768, 4, 2048, 0),
                (65536, 4, 4096, 0), (1048576, 64, 4096, 0),
                (1, 4096, 1, 16383), (67108864, 4096, 4096, 0)):
            with self.subTest(size=size, pes=pes):
                shape = oracle.geometry(size, pes)
                self.assertEqual((shape["active_pes"], shape["payload_words_per_pe"],
                                  shape["padding_bytes"]), (pes, words, padding))
                self.assertEqual((shape["active_rows"], shape["active_cols"]),
                                 (math.isqrt(pes), math.isqrt(pes)))
                self.assertEqual(shape["frame_words"], words + 1)
                self.assertEqual(shape["control_h2d_bytes"], 4 * pes)
                self.assertEqual(shape["control_d2h_bytes"], 4 * pes)
                self.assertEqual(shape["transfer_bytes_per_direction"], size + padding + 4 * pes)

    def test_known_payload_bytes_generation_and_row_order(self):
        np.testing.assert_array_equal(oracle.expected_frame(1, 17), [17, 0xc3])
        np.testing.assert_array_equal(oracle.expected_frame(2, 17), [17, 0xb3c3])
        np.testing.assert_array_equal(oracle.expected_frame(4, 17), [17, 0x1bdcb3c3])
        np.testing.assert_array_equal(oracle.expected_frame(8, 17),
                                      [17, 0x1bdcb3c3, 0xb9853908])
        frame = oracle.expected_frame(65536, 17, 4)
        self.assertEqual(frame.shape, (16388,))
        self.assertEqual(frame.dtype, np.dtype(np.uint32))
        self.assertTrue(frame.flags.c_contiguous)
        self.assertEqual(frame[4097], 0x85ebca7a)
        self.assertEqual(frame[4098], 0x95e0ed7e)
        changed = oracle.expected_frame(65536, 18, 4)
        self.assertTrue(np.all(frame != changed))

    def test_tiny_payload_uses_fixed_rows_with_zero_tail_padding(self):
        for pes in oracle.PE_COUNTS:
            with self.subTest(pes=pes):
                control = oracle.expected_frame(0, 29, pes)
                headers29 = [29 ^ ((pe * 0x85ebca6b) & 0xffffffff) for pe in range(pes)]
                np.testing.assert_array_equal(control, headers29)
                self.assertEqual(len(set(int(value) for value in control)), pes)
                frame = oracle.expected_frame(1, 17, pes).reshape(pes, 2)
                headers17 = [17 ^ ((pe * 0x85ebca6b) & 0xffffffff) for pe in range(pes)]
                np.testing.assert_array_equal(frame[:, 0], headers17)
                self.assertEqual(frame[0, 1], 0xc3)
                self.assertTrue(np.all(frame[1:, 1] == 0))
                payload_bytes = frame[:, 1:].astype("<u4").tobytes()
                self.assertEqual(payload_bytes, b"\xc3" + b"\x00" * (4 * pes - 1))
                self.assertEqual(oracle.geometry(1, pes)["active_pes"], pes)

    def test_full_frontier_has_every_header_and_no_padding(self):
        frame = oracle.expected_frame(1048576, 0xffffffff, 64).reshape(64, 4097)
        headers = [0xffffffff ^ ((pe * 0x85ebca6b) & 0xffffffff) for pe in range(64)]
        np.testing.assert_array_equal(frame[:, 0], headers)
        self.assertEqual(frame.nbytes, 1048576 + 64 * 4)
        self.assertEqual(len(set(int(v) for v in frame[:, 1])), 64)

    def test_zero_header_known_values_detect_pe_permutation_and_duplication(self):
        expected = oracle.expected_frame(0, 17, 4)
        np.testing.assert_array_equal(expected, [17, 0x85ebca7a, 0x0bd794c7, 0x91c35f50])
        self.assertNotEqual(oracle.digest(expected[::-1]), oracle.digest(expected))
        duplicate = expected.copy()
        duplicate[1] = duplicate[0]
        self.assertNotEqual(oracle.digest(duplicate), oracle.digest(expected))

    def test_plans_preserve_sizes_counts_and_deterministic_order(self):
        sizes = [0] + [1 << exponent for exponent in range(21)]
        ascending = oracle.plan("powers-p64", 0)
        self.assertEqual(ascending["sizes"], sizes)
        self.assertEqual((ascending["warmup"], ascending["repeats"]), (10, 30))
        self.assertEqual((ascending["width"], ascending["height"], ascending["pes"],
                          ascending["capacity"], ascending["channels"]), (8, 8, 64, 4097, 1))
        self.assertEqual(oracle.plan("powers-p64", 1)["sizes"], list(reversed(sizes)))
        permuted = oracle.plan("powers-p64", 2)
        self.assertEqual(sorted(permuted["sizes"]), sizes)
        self.assertEqual(permuted, oracle.plan("powers-p64", 2))
        self.assertNotEqual(permuted["sizes"], sizes)
        self.assertNotEqual(permuted["sizes"], list(reversed(sizes)))
        zero = oracle.plan("zero-p1")
        self.assertEqual((zero["sizes"], zero["warmup"], zero["repeats"]), ([0], 20, 200))
        self.assertEqual((zero["height"], zero["capacity"], zero["channels"]), (1, 4097, 1))
        self.assertEqual(json.loads(json.dumps(ascending)), ascending)
        ascending["sizes"].append(3)
        ascending["height"] = 0
        self.assertEqual(oracle.plan("powers-p64")["sizes"], sizes)
        self.assertEqual(oracle.CASES["powers-p64"]["height"], 8)

    def test_each_sweep_holds_pe_count_channels_and_common_sizes_fixed(self):
        self.assertEqual(set(oracle.CASES), {
            "zero-p1", "zero-p4", "zero-p16", "zero-p64",
            "zero-p256", "zero-p1024", "zero-p4096",
            "powers-p1", "powers-p4", "powers-p16", "powers-p64",
            "powers-p256", "powers-p1024", "powers-p4096",
            "powers-p4096-c2", "powers-p4096-c4", "powers-p4096-c8", "powers-p4096-c16",
        })
        common = [0] + [1 << exponent for exponent in range(15)]
        for pes in oracle.PE_COUNTS:
            with self.subTest(pes=pes):
                schedule = oracle.plan(f"powers-p{pes}")
                self.assertEqual(schedule["sizes"][:len(common)], common)
                self.assertEqual(schedule["sizes"][-1], 16384 * pes)
                self.assertEqual(schedule["channels"], 1)
                self.assertEqual((schedule["width"], schedule["height"]),
                                 (math.isqrt(pes), math.isqrt(pes)))
                self.assertEqual({oracle.geometry(size, pes)["active_pes"]
                                  for size in schedule["sizes"]}, {pes})
                self.assertEqual(oracle.geometry(schedule["sizes"][-1], pes)["frame_words"], 4097)
                zero = oracle.plan(f"zero-p{pes}")
                self.assertEqual((zero["pes"], zero["channels"]), (pes, 1))
                with self.assertRaises(ValueError):
                    oracle.geometry(32768 * pes, pes)

    def test_channel_comparison_changes_only_channels_and_case_name(self):
        comparison = {f"powers-p4096-c{channels}": channels for channels in (2, 4, 8, 16)}
        for name, parameters in oracle.CASES.items():
            self.assertEqual(parameters["channels"], comparison.get(name, 1))
        for order in (0, 1, 2):
            baseline = oracle.plan("powers-p4096", order)
            self.assertEqual(max(baseline["sizes"]), 64 << 20)
            self.assertEqual((baseline["width"], baseline["height"], baseline["pes"]), (64, 64, 4096))
            self.assertEqual((baseline["warmup"], baseline["repeats"]), (10, 30))
            for name, channels in comparison.items():
                with self.subTest(case=name, order=order):
                    changed = oracle.plan(name, order)
                    self.assertEqual(changed["case"], name)
                    self.assertEqual(changed["channels"], channels)
                    changed["case"] = baseline["case"]
                    changed["channels"] = baseline["channels"]
                    self.assertEqual(changed, baseline)

    def test_smoke_remains_explicit_and_has_the_large_boundary(self):
        for case in oracle.CASES:
            for order in (0, 1, 2):
                sample = oracle.plan(case, order, smoke=True)
                self.assertTrue(sample["smoke"])
                self.assertEqual((sample["warmup"], sample["repeats"]), (1, 3))
                maximum = 16384 * sample["pes"]
                wanted = [0] if case.startswith("zero-") else sorted({
                    0, 1, 4, 16384, min(32768, maximum), maximum})
                self.assertEqual(sorted(sample["sizes"]), wanted)

    def test_digest_has_canonical_little_endian_bytes(self):
        raw = b"\x04\x03\x02\x01\x00\x00\x00\x00\xff\xff\xff\xff"
        expected = hashlib.sha256(raw).hexdigest()
        for dtype in ("<u4", ">u4"):
            values = np.array([0x01020304, 0, 0xffffffff], dtype=dtype)
            self.assertEqual(oracle.digest(values), expected)
        storage = np.array([0x01020304, 7, 0, 7, 0xffffffff, 7], dtype=np.uint32)
        self.assertEqual(oracle.digest(storage[::2]), expected)
        self.assertNotEqual(oracle.digest(np.array([0, 0x01020304, 0xffffffff], np.uint32)), expected)

    def test_invalid_sizes_generations_plans_and_digest_carriers(self):
        for size in (True, False, -1, 3, 16385, 1 << 21, 4.0, "4", None, np.int64(4)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                oracle.geometry(size)
        for pes in (True, False, 0, -1, 2, 3, 65, 8192, 1.0, "1", None, np.int64(4)):
            with self.subTest(pes=pes), self.assertRaises(ValueError):
                oracle.geometry(0, pes)
        for generation in (True, False, -1, 1 << 32, 1.0, "1", None, np.uint32(1)):
            with self.subTest(generation=generation), self.assertRaises(ValueError):
                oracle.expected_frame(4, generation)
        for case in ("", "other", "zero", "powers", "powers-p2", "powers-p4096-c1",
                     "powers-p4096-c3", "powers-p64-c16", True, None, []):
            with self.subTest(case=case), self.assertRaises(ValueError):
                oracle.plan(case)
        for order in (True, False, -1, 3, 0.0, "0", None):
            with self.subTest(order=order), self.assertRaises(ValueError):
                oracle.plan("powers-p64", order)
        for smoke in (0, 1, "yes", None):
            with self.subTest(smoke=smoke), self.assertRaises(ValueError):
                oracle.plan("powers-p64", smoke=smoke)
        for values in ([1], np.array([1], dtype=np.int32), np.array([1.0]),
                       np.array([True]), np.array([1], dtype=np.uint64)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                oracle.digest(values)


if __name__ == "__main__":
    unittest.main()
