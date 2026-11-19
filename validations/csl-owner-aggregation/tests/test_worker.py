import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
if not (ROOT/'appliance.py').is_file():
    ROOT = ROOT.parent
sys.path.insert(0,str(ROOT))
from appliance import verify
from worker import features, fixture, oracle


class Aggregation(unittest.TestCase):
    def test_all_owners_and_frames(self):
        for owners in (1,2,4,8,16):
            ptr,src,weights=fixture(owners)
            self.assertEqual(int(ptr[:,-1].sum()),224)
            for trial in (1,2,3):
                x=features(trial)
                y=np.zeros((64,4),np.float32)
                for owner in range(owners):
                    for row in range(64//owners):
                        for e in range(ptr[owner,row],ptr[owner,row+1]):
                            y[owner*(64//owners)+row]+=weights[owner,e]*x[src[owner,e]]
                np.testing.assert_array_equal(y.view(np.uint32),oracle(x).view(np.uint32))

    def test_saved_captures(self):
        paths=sorted((ROOT/'evidence').glob('p*-m?.json'))
        self.assertGreater(len(paths),0)
        for path in paths:
            result=json.loads(path.read_text())
            verify(result,result['owners'],result['mode'],True)
        result['trials'][0]['cycles'][0] += 1
        with self.assertRaises(AssertionError):
            verify(result,result['owners'],result['mode'],True)
