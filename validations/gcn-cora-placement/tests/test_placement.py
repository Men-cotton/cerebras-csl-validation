"""Check the original CSR, the measured input variants, and a modified device return."""
from pathlib import Path
import sys
import unittest

import numpy as np

HOME=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HOME))
from appliance import check, check_measurement
from contract import frame, loaded, references
from paired_worker import sha


class PhysicalPlacement(unittest.TestCase):
    def test_reference_roundoff(self):
        saved = {'per_pe_cycles': [17], 'max_abs_fp64': {'panel_sum': 1.3639037259594033e-6}}
        measured = dict(saved, max_abs_fp64={'panel_sum': 1.3639037241830465e-6})
        check_measurement(measured, saved, 'original')
        for changed in (dict(measured, per_pe_cycles=[18]),
                        dict(measured, max_abs_fp64={'panel_sum': 1.4e-6}),
                        dict(measured, max_abs_fp64={})):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                check_measurement(changed, saved, 'original')

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
