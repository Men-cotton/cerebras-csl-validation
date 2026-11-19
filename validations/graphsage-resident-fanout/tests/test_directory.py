"""Exercise compressed source ranges against the original graph and physical packet headers."""
from pathlib import Path
import json
import sys
import unittest

import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(FAMILY))
import saved_audit


class FanoutDirectory(unittest.TestCase):
    def test_wrong_source_with_valid_boundaries(self):
        for name in json.loads((FAMILY/'package.json').read_text())['cases']:
            with self.subTest(case=name):
                case = FAMILY/'cases'/name
                manifest = saved_audit.load(case)
                arrays = saved_audit.load_npz(case/'inputs.npz')
                ptr = arrays['tx_ptr'].copy()
                counts = np.diff(ptr.astype(np.int64), axis=1)
                candidates = np.argwhere(counts[:,1:] != 0)
                self.assertGreater(len(candidates),0)
                pe, previous_source = candidates[0]
                ptr[pe, previous_source+1] += 1
                np.testing.assert_array_equal(ptr[:,[0,-1]],arrays['tx_ptr'][:,[0,-1]])
                self.assertTrue(np.all(np.diff(ptr.astype(np.int64),axis=1)>=0))
                with self.assertRaisesRegex(ValueError,'exact source, destination|valid source owner slot'):
                    saved_audit.independent_expected(case,manifest,dict(arrays,tx_ptr=ptr))
