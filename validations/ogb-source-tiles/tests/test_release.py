from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import appliance


class TestCase(unittest.TestCase):
    def test_all_original_outputs(self):
        cases=sorted((HERE/"cases").glob("*/release.json"))
        self.assertTrue(cases)
        for release in cases:
            build,v=appliance.check(release.parent,appliance.device.sha(release))
            self.assertEqual(build["params"]["FRAMES"],6)
            self.assertEqual(set(v),{.5,1.})

    def test_pin(self):
        release=next((HERE/"cases").glob("*/release.json"))
        with self.assertRaisesRegex(ValueError,"pin"):
            appliance.check(release.parent,"0"*64)
