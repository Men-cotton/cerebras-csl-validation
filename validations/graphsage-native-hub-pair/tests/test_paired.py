import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from appliance import references


class PairedGraphSAGE(unittest.TestCase):
    def test_original_graph_two_assignments(self):
        cases=sorted((ROOT/'cases').iterdir())
        self.assertTrue(cases)
        for case in cases:
            with self.subTest(case=case.name):
                pin=hashlib.sha256((case/'release.json').read_bytes()).hexdigest()
                doc,_,original=references(case,pin)
                nodes=np.load(original/'node_ids.npy',allow_pickle=False)
                with np.load(case/'split.npz',allow_pickle=False) as arrays:
                    self.assertEqual(int(nodes[0,arrays['split_info'][0,1]]),163)
                self.assertGreater(doc['modes']['split']['plan']['moved'],0)
                actual=json.loads((case/'sdk-audit.json').read_text())
                self.assertEqual(actual['nodes'],2708)
                self.assertEqual(actual['modes']['split'][0]['moved_partial_word_hops'],46)
                self.assertEqual(actual['modes']['control'][0]['moved_partial_word_hops'],0)

    def test_entry_clock_wrap(self):
        from paired_audit import entry_intervals
        words=np.array([[5,0,0,19,0,0,0xfff0,0xffff,0xffff]],dtype=np.uint16)
        self.assertEqual(entry_intervals(words)['per_pe_including_barrier'],[35])
        self.assertEqual(entry_intervals(words)['per_pe_barrier_cycles'],[21])
        words[0,0]=20
        with self.assertRaisesRegex(ValueError,'clock order'):
            entry_intervals(words)

    def test_changed_release_pin(self):
        case=next((ROOT/'cases').iterdir())
        with self.assertRaisesRegex(ValueError,'release pin'):
            references(case,'0'*64)

    def test_changed_restored_A_layer(self):
        from sample_audit import compare_returned_A
        for case in (ROOT/'cases').iterdir():
            cert=json.loads((case/'release.json').read_text())
            if cert.get('sdk_measurement') != 'two stopped-core A frames':
                continue
            with tempfile.TemporaryDirectory() as temp:
                folder=Path(temp)
                for mode in ('control','split'):
                    for trial in (1,3):
                        file=folder/f'{mode}-{trial}.npz'
                        if mode=='split' and trial==3:
                            with np.load(case/'sim'/(mode+'-1.npz')) as source:
                                raw=dict(source)
                            raw['logits'].view('<u4')[0,0] ^= np.uint32(1)
                            np.savez_compressed(file,**raw)
                        else:
                            file.symlink_to(case/'sim'/(mode+'-1.npz'))
                with self.assertRaises(AssertionError):
                    compare_returned_A(case,folder)


if __name__=='__main__':
    unittest.main()
