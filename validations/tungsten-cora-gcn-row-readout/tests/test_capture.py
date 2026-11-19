"""Verify complete captured wires and reject incorrect row and nonroot results."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from appliance import sha, verify_case
from worker import unpack


class RowReadout(unittest.TestCase):
    def test_all_frozen_cases_and_corruption(self):
        for case in sorted((HERE/'cases').iterdir()):
            with self.subTest(case=case.name):
                verify_case(case,sha(case/'build.json'))
                p = json.loads((case/'build.json').read_text())['params']
                fixed = 17+p['L']*(p['H']+p['O'])
                with np.load(case/'run/frame-1.npz') as f, np.load(case/'reference.npz') as ref:
                    frame = f['reply']
                    for offset,error in [(fixed,ValueError),(fixed+1,AssertionError),
                                         (fixed+1+p['K']*p['T']+17,AssertionError)]:
                        faulty = frame.copy()
                        faulty[offset] ^= np.uint32(1)
                        with self.assertRaises(error):
                            unpack(faulty,1,p,ref)
