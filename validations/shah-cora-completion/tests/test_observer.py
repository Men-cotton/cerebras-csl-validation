import unittest

import numpy as np

from observer import clocks, epoch_check


class ObserverTests(unittest.TestCase):
    def test_all_roles_must_report_current_epoch(self):
        p = dict(F=16, R=2)
        values = np.full((4, 17, 1), 4, np.uint32)
        epoch_check(values, p, 4)
        for y, x in [(0, 0), (0, 1), (2, 1), (2, 0)]:
            with self.subTest(y=y, x=x):
                stale = values.copy()
                stale[y, x] = 3
                with self.assertRaises(ValueError):
                    epoch_check(stale, p, 4)
        with self.assertRaises(ValueError):
            epoch_check(values[:1], p, 4)

    def test_all_48_bits_of_pe_clocks(self):
        packed = np.array([[[65534, 65535, 1, 2, 0, 2]]], np.uint16)
        decoded = clocks(packed)
        self.assertEqual(int(decoded[0, 0, 1] - decoded[0, 0, 0]), 4)


if __name__ == "__main__":
    unittest.main()
