from pathlib import Path
import sys
import unittest
from hashlib import sha256

FAMILY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(FAMILY))
import appliance


class TestPorts(unittest.TestCase):
    def test_replay(self):
        releases=sorted((FAMILY/"cases").glob("*/release.json"))
        self.assertTrue(releases)
        for p in releases:
            with self.subTest(case=p.parent.name):
                desc, scores=appliance.check(p.parent,sha256(p.read_bytes()).hexdigest())
                self.assertEqual(set(scores),{0.5,1.0})
                self.assertIn(desc["lanes"],appliance.PORTS)

    def test_wrong_release(self):
        p=next((FAMILY/"cases").glob("*/release.json"))
        with self.assertRaisesRegex(ValueError,"pin"):
            appliance.check(p.parent,"0"*64)
