import json
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from appliance import sha, verify_case
from worker import unpack


class GcnTest(unittest.TestCase):
    def test_frozen_cases_and_frame_discriminator(self):
        # Frozen watcher packages include only cases which passed the SDK replay.
        cases = HERE/'cases'
        if not cases.exists():
            cases = HERE/'bundles'
        complete = [b.parent.parent for b in cases.glob('*/run/capture.json')
                    if json.loads(b.read_text()).get('stage') == 'complete']
        self.assertTrue(complete)
        for case in complete:
            with self.subTest(case=case.name):
                verify_case(case,sha(case/'build.json'))
                params = json.loads((case/'build.json').read_text())['params']
                with np.load(case/'run/frame-1.npz') as raw, np.load(case/'reference.npz') as ref:
                    wrong = raw['reply']
                    wrong[-1] ^= np.uint32(1)
                    with self.assertRaises(AssertionError):
                        unpack(wrong,1,params,ref)
