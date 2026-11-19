"""Offline validation of each registered grouped packet schedule and corrupted header rejection."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import saved_audit as native


class GroupedSchedule(unittest.TestCase):
    def test_frozen_plan_and_corruption(self):
        for case in sorted((ROOT/"cases").iterdir()):
            with self.subTest(case=case.name):
                manifest=native.load(case)
                arrays=native.load_npz(case/"inputs.npz")
                expected,_=native.independent_expected(case,manifest,arrays)
                self.assertEqual(expected["traffic"][:,0].sum(),expected["traffic"][:,1].sum())
                sender=next(pe for pe in range(len(arrays["counts"])) if arrays["counts"][pe,0])
                arrays["packet_header"][sender,0]^=1
                with self.assertRaises(ValueError):
                    native.independent_expected(case,manifest,arrays)


if __name__=="__main__":
    unittest.main()
