import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import saved_audit as native


class FixedH16(unittest.TestCase):
    def test_frozen_split_and_projection_work(self):
        package=json.loads((ROOT/'package.json').read_text())
        self.assertTrue(package['cases'])
        for name,digest in package['cases'].items():
            with self.subTest(case=name):
                case=ROOT/'cases'/name;manifest=native.load(case);self.assertEqual(native.sha(case/'case.json'),digest)
                arrays=native.load_npz(case/'inputs.npz');expected,_=native.independent_expected(case,manifest,arrays)
                self.assertEqual(int(expected['stats'][:,4].sum()),16*int(arrays['x_ptr'][:,-1].astype(np.int64).sum()))
                hops=23*int(np.count_nonzero(np.isin(arrays['split_control'][:,0],[2,3])))
                self.assertGreater(hops,0)
                np.testing.assert_array_equal(expected['split_words'].sum(axis=0),[hops,hops])
