from pathlib import Path
import sys
import unittest
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import appliance as app


class TwoHopComparison(unittest.TestCase):
    def test_all_registered_sdk_returns(self):
        releases=sorted((HERE/'cases').glob('*/release.json'))
        self.assertTrue(releases)
        for path in releases:
            case,rel,desc,expected=app.checked_case(path.parent.name,app.device.sha(path))
            self.assertEqual(set(expected['logits']),{.5,1.})
            self.assertEqual(set(expected['hidden']),{.5,1.})
            self.assertEqual(desc['roles']['hardware']['fabric'],[762,1172])
            self.assertEqual(expected['hidden'][1.].size,rel['hidden_nodes']*desc['final']['H'])
            self.assertEqual(expected['logits'][1.].shape[0],rel['destinations'])
            with self.assertRaisesRegex(ValueError,'release SHA'):
                app.checked_case(case.name,'0'*64)

    def test_instrumented_worker_and_graph_boundaries(self):
        case=HERE/'cases/small-arxiv-p2'
        _,rel,desc,expected=app.checked_case(case.name,app.device.sha(case/'release.json'))
        r,_=app.observed(case,case/'bound-sdk',expected,'simulator',True)
        self.assertEqual([v['echo'] for v in r['frames']],[False,True,True,False,False])
        self.assertEqual([v['hidden_round_trip_words'] for v in r['frames']],[0,128,128,0,0])
        self.assertEqual([r['raw_vertices'],r['hidden_vertices'],r['final_vertices']],[14,4,2])
        self.assertFalse(np.array_equal(expected['hidden'][1.],expected['hidden'][.5]))
        with self.assertRaisesRegex(ValueError,'target role'):
            app.observed(case,case/'bound-sdk',expected,'hardware',True)

    def test_changed_echo_reference_is_rejected(self):
        case=HERE/'cases/small-arxiv-p2'
        _,_,_,original=app.checked_case(case.name,app.device.sha(case/'release.json'))
        expect={'logits':original['logits'],'hidden':{k:v.copy() for k,v in original['hidden'].items()}}
        expect['hidden'][1.][0]^=np.uint32(1)
        with self.assertRaises(AssertionError):
            app.observed(case,case/'bound-sdk',expect,'simulator',True)
