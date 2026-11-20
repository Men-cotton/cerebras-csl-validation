from pathlib import Path
import sys
import unittest
from hashlib import sha256
HERE=Path(__file__).resolve().parents[1]/'native'
sys.path.insert(0,str(HERE))
import physical

class TestNativeRows(unittest.TestCase):
    def test_original_window(self):
        releases=sorted((HERE/'cases').glob('*/release.json'))
        self.assertTrue(releases)
        for p in releases:
            with self.subTest(case=p.parent.name):
                _,_,sdk,_,scores=physical.check(p.parent.name,sha256(p.read_bytes()).hexdigest())
                self.assertEqual(sdk['params']['L'],4)
                self.assertEqual(sdk['params']['P'],4)
                self.assertEqual(set(scores),{.5,1.})

    def test_release(self):
        p=next((HERE/'cases').glob('*/release.json'))
        with self.assertRaisesRegex(ValueError,'pin'):
            physical.check(p.parent.name,'0'*64)
