"""Check the original CSR, the measured input variants, and a modified device return."""
from pathlib import Path
import sys
import unittest

import numpy as np

HOME=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HOME))
from appliance import check
from contract import frame, loaded, references
from paired_worker import sha


class PhysicalPlacement(unittest.TestCase):
    def test_actual_sdk_frames(self):
        cases=list((HOME/'cases').iterdir())
        self.assertTrue(cases)
        for case in cases:
            with self.subTest(case=case.name):
                doc,_,_=check(case,sha(case/'release.json'))
                self.assertGreaterEqual(len(doc['mode_order']),2)

    def test_changed_layer(self):
        case=next((HOME/'cases').iterdir())
        _,context,bindings=references(case,HOME.parent/'h16-hub-cora')
        _,expected,nodes=bindings['swap']
        raw=loaded(case/'swap-sdk.npz')
        raw['pre'][0,0]+=np.float32(1.0)
        with self.assertRaises(AssertionError):
            frame(raw,expected,nodes,context,1.)
