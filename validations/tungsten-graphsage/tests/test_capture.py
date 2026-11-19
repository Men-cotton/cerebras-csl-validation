import json
from pathlib import Path
import sys
import unittest

import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from appliance import verify_case
from worker import sha,unpack


class GraphSageTest(unittest.TestCase):
    def test_frozen_cases(self):
        root=HERE/'cases' if (HERE/'cases').is_dir() else HERE/'bundles'
        cases=[p.parent.parent for p in root.glob('*/run/capture.json') if json.loads(p.read_text())['stage']=='complete']
        self.assertTrue(cases)
        for case in cases:
            with self.subTest(case=case.name):
                verify_case(case,sha(case/'build.json'))
                p=json.loads((case/'build.json').read_text())['params']
                with np.load(case/'run/frame-2.npz',allow_pickle=False) as f,np.load(case/'reference.npz',allow_pickle=False) as ref:
                    corrupt=f['reply'].copy()
                    corrupt[10]=np.uint32(0x7fc00000)
                    with self.assertRaises(AssertionError):
                        unpack(corrupt,2,p,ref)
