import unittest
from unittest.mock import patch

import numpy as np

from oracle import (DONE, EMPTY, END_ROW, controlled_fixture, decode, fixture,
                    h_tiles, pack, random_fixture, router_events)


class OracleTests(unittest.TestCase):
    def test_duplicate_empty_shard_boundaries_and_skew(self):
        case = fixture(R=4, F=3, V=3, Q=4, B=3)
        np.testing.assert_allclose(decode(case["params"], case["streams"], case["h"]),
                                   case["reference"], rtol=0, atol=0)
        self.assertEqual(case["edges"][0][:2], case["edges"][1][:2])
        self.assertEqual(case["metrics"]["nnz_per_worker_row"][1:3], [0, 0])
        self.assertGreater(case["metrics"]["max_to_mean_worker_work"], 1)
        self.assertGreater(case["metrics"]["h2d_padding_pairs"], 0)
        self.assertTrue(np.any(np.all(case["reference"] == 0, axis=1)))
        self.assertEqual(case["edges"][-4][1], case["params"]["K"] - 1)
        np.testing.assert_array_equal(h_tiles(case["params"], case["h"])[-1, :, -1], 0)

    def test_empty_runs_expand_to_q_done(self):
        case = fixture(R=2, F=2, V=3, Q=8, B=2, empty=True)
        for stream in case["streams"]:
            np.testing.assert_array_equal(stream, [[END_ROW, 8], [END_ROW, 8]])
            self.assertEqual(router_events(case["params"], 0, stream[0]), [(DONE, None)] * 8)
        np.testing.assert_array_equal(decode(case["params"], case["streams"], case["h"]), 0)

    def test_several_shapes_against_dense_reference(self):
        for R, F, V, Q, B in [(1, 1, 3, 1, 1), (2, 3, 2, 2, 3), (4, 2, 3, 4, 3)]:
            with self.subTest(R=R, F=F, V=V, Q=Q, B=B):
                case = fixture(R, F, V, Q, B)
                a = np.zeros((case["params"]["N"], case["params"]["K"]), np.float64)
                for row, col, value in case["edges"]:
                    a[row, col] += value
                expected = a @ case["h"].astype(np.float64)
                np.testing.assert_array_equal(case["reference"], expected)
                np.testing.assert_allclose(decode(case["params"], case["streams"], case["h"]), expected,
                                           rtol=0, atol=0)

    def test_reject_corrupt_lengths_indices_and_padding(self):
        case = fixture()
        params = case["params"]
        invalid = [np.array([END_ROW], np.uint32),
                   np.array([END_ROW, 0], np.uint32),
                   np.array([END_ROW, params["Q"] + 1], np.uint32),
                   np.array([END_ROW, params["Q"] - 1], np.uint32),
                   np.array([params["V"], 0, END_ROW, params["Q"]], np.uint32),
                   np.array([EMPTY, 0, END_ROW, params["Q"]], np.uint32),
                   np.array([END_ROW, params["Q"], EMPTY, 1], np.uint32),
                   np.array([END_ROW, params["Q"], EMPTY, 0, 0, 0], np.uint32),
                   np.array([0, 0x7f800000, END_ROW, params["Q"]], np.uint32)]
        for words in invalid:
            with self.subTest(words=words.tolist()), self.assertRaises(ValueError):
                router_events(params, 0, words)
        with self.assertRaises(ValueError):
            decode(params, case["streams"][:-1], case["h"])
        with self.assertRaises(ValueError):
            router_events(params, params["R"] - 1,
                          np.array([params["K"], 0, END_ROW, params["Q"]], np.uint32))

    def test_reject_invalid_host_inputs(self):
        case = fixture()
        for edges in [[(-1, 0, 1)], [(0, case["params"]["K"], 1)], [(0, 0, float("nan"))]]:
            with self.assertRaises(ValueError):
                pack(case["params"], edges)
        with self.assertRaises(ValueError):
            fixture(Q=3)
        with self.assertRaises(ValueError):
            fixture(V=65536)

    def test_random_shapes_against_independent_dense_reference(self):
        for shape in [(4, 4, 8, 4, 4), (8, 8, 16, 8, 8)]:
            with self.subTest(shape=shape):
                case = random_fixture(*shape)
                params = case["params"]
                dense = np.zeros((params["N"], params["K"]), np.float64)
                for row, col, value in case["edges"]:
                    dense[row, col] += value
                reference = dense @ case["h"].astype(np.float64)
                np.testing.assert_allclose(case["reference"], reference, rtol=1e-13, atol=1e-13)
                np.testing.assert_allclose(decode(params, case["streams"], case["h"]), reference,
                                           rtol=2e-5, atol=3e-6)
                self.assertEqual(case["reference"].shape, (params["B"] * params["Q"], params["F"]))
                self.assertEqual(case["edges"][0][:2], case["edges"][1][:2])
                self.assertEqual(case["metrics"]["nnz_per_worker_row"][1], 0)
                self.assertGreater(case["metrics"]["nnz_per_worker_row"][0],
                                   max(case["metrics"]["nnz_per_worker_row"][1:]))
                np.testing.assert_array_equal(case["reference"][3], 0)
                self.assertEqual(case["edges"][-1][1], params["K"] - 1)

    def test_random_seed_reproducibility_and_unclamped_degree(self):
        first = random_fixture(2, 2, 3, 2, 3, seed=19, degree=17)
        second = random_fixture(2, 2, 3, 2, 3, seed=19, degree=17)
        np.testing.assert_array_equal(first["h"], second["h"])
        self.assertEqual(first["edges"], second["edges"])
        for left, right in zip(first["streams"], second["streams"]):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(sum(row == 0 for row, _, _ in first["edges"]), 4 * 17)
        changed = random_fixture(2, 2, 3, 2, 3, seed=20, degree=17)
        self.assertFalse(np.array_equal(first["h"], changed["h"]))
        for kwargs in ({"degree": 0}, {"degree": 2.5}, {"seed": -1}, {"degree": 1_000_000}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                random_fixture(2, 2, 3, 2, 3, **kwargs)

    def test_controlled_dense_reference_duplicates_and_empty_shards(self):
        for shape, skew in [((1, 1, 2, 1, 1), False),
                            ((2, 3, 3, 2, 3), False),
                            ((2, 3, 3, 2, 3), True),
                            ((4, 2, 3, 4, 2), True)]:
            with self.subTest(shape=shape, skew=skew):
                case = controlled_fixture(*shape, degree=5, skew=skew)
                params = case["params"]
                dense = np.zeros((params["N"], params["K"]), np.float64)
                for row, col, value in case["edges"]:
                    dense[row, col] += value
                expected = dense @ case["h"].astype(np.float64)
                np.testing.assert_allclose(case["reference"], expected, rtol=1e-13, atol=1e-13)
                np.testing.assert_allclose(decode(params, case["streams"], case["h"]),
                                           expected, rtol=2e-5, atol=3e-6)
                self.assertEqual(params["K"], params["R"] * params["V"])
                self.assertLess(len({(row, col) for row, col, _ in case["edges"]}),
                                len(case["edges"]))

    def test_controlled_balance_pair_preserves_occurrences_and_payload_contract(self):
        R, F, V, Q, B, degree = 4, 3, 5, 4, 3, 2
        N = B * Q
        balanced = controlled_fixture(R, F, V, Q, B, degree=degree)
        skewed = controlled_fixture(R, F, V, Q, B, degree=degree, skew=True)
        np.testing.assert_array_equal(balanced["h"], skewed["h"])
        self.assertEqual(len(balanced["edges"]), N * R * degree)
        self.assertEqual(len(balanced["edges"]), len(skewed["edges"]))
        for left, right in zip(balanced["edges"], skewed["edges"]):
            row, col, value = left
            self.assertEqual(right, (row, col - (col // V % 2) * V, value))
        bm, sm = balanced["metrics"], skewed["metrics"]
        self.assertEqual(bm["nnz_per_worker_row"], [N * degree] * R)
        self.assertEqual(sm["nnz_per_worker_row"], [2 * N * degree, 0] * (R // 2))
        self.assertEqual(bm["max_to_mean_worker_work"], 1)
        self.assertEqual(sm["max_to_mean_worker_work"], 2)
        self.assertEqual(bm["h2d_padding_pairs"], 0)
        self.assertEqual(bm["h2d_stream_payload_bytes"], 8 * N * R * (degree + 1))
        self.assertEqual(sm["h2d_stream_payload_bytes"], 8 * N * R * (2 * degree + 1))
        self.assertEqual(sm["h2d_padding_pairs"], B * (R // 2) * (Q * (2 * degree + 1) - 1))
        for name in ("nnz_occurrences", "h2d_edge_payload_bytes", "h_upload_padded_payload_bytes",
                     "d2h_output_payload_bytes", "row_broadcast_edge_deliveries"):
            self.assertEqual(bm[name], sm[name])

    def test_controlled_fixed_n_chunking_and_reproducibility(self):
        for skew in (False, True):
            baseline = controlled_fixture(4, 3, 5, 2, 8, seed=19, skew=skew)
            for Q, B in ((1, 16), (2, 8), (4, 4), (8, 2), (16, 1)):
                case = controlled_fixture(4, 3, 5, Q, B, seed=19, skew=skew)
                self.assertEqual(case["edges"], baseline["edges"])
                np.testing.assert_array_equal(case["h"], baseline["h"])
                np.testing.assert_array_equal(case["reference"], baseline["reference"])
                self.assertEqual(case["metrics"]["h2d_stream_payload_bytes"],
                                 baseline["metrics"]["h2d_stream_payload_bytes"])
            repeat = controlled_fixture(4, 3, 5, 2, 8, seed=19, skew=skew)
            for left, right in zip(baseline["streams"], repeat["streams"]):
                np.testing.assert_array_equal(left, right)
            changed = controlled_fixture(4, 3, 5, 2, 8, seed=20, skew=skew)
            self.assertFalse(np.array_equal(baseline["h"], changed["h"]))

    def test_controlled_rejects_invalid_inputs_before_allocation(self):
        defaults = dict(R=2, F=2, V=3, Q=2, B=3)
        invalid = [{name: value} for name in defaults for value in (True, 1.5, 0, -1)]
        invalid += [dict(Q=3), dict(V=65536), dict(degree=True), dict(degree=1.5),
                    dict(degree=0), dict(seed=True), dict(seed=1.5), dict(seed=-1),
                    dict(skew=1), dict(R=3, skew=True), dict(degree=1_000_000),
                    dict(F=2_000_000), dict(B=2_000_000),
                    dict(R=np.int64(2**62), V=np.int64(4))]
        with patch("oracle.np.random.default_rng", side_effect=AssertionError("allocation before admission")):
            for override in invalid:
                with self.subTest(override=override), self.assertRaises(ValueError):
                    controlled_fixture(**{**defaults, **override})

    def test_controlled_feature_width_preserves_matrix_and_feature_prefix(self):
        for skew in (False, True):
            baseline = controlled_fixture(4, 8, 5, 4, 3, seed=19, skew=skew)
            for F in (1, 2, 4, 8):
                case = controlled_fixture(4, F, 5, 4, 3, seed=19, skew=skew)
                self.assertEqual(case["edges"], baseline["edges"])
                np.testing.assert_array_equal(case["h"], baseline["h"][:, :F])
                np.testing.assert_array_equal(case["reference"], baseline["reference"][:, :F])
                for left, right in zip(case["streams"], baseline["streams"]):
                    np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
