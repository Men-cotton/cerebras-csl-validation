from pathlib import Path
import sys
import unittest

FAMILY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAMILY))
from physical import driver


class TestEightPorts(unittest.TestCase):
    def test_original_windows(self):
        releases = sorted((FAMILY/"cases").glob("*/release.json"))
        self.assertEqual([p.parent.name for p in releases], ["arxiv-rcm128-p16-l8","products-rcm128-p16-l8"])
        for p in releases:
            with self.subTest(case=p.parent.name):
                desc, scores = driver.check(p.parent,driver.device.sha(p))
                self.assertEqual((desc["lanes"],desc["owners_per_lane"]), (8,2))
                self.assertEqual(set(scores), {0.5,1.0})
                self.assertEqual(scores[1.0].shape[0],128)
                with self.assertRaisesRegex(ValueError, "pin"):
                    driver.check(p.parent, "0"*64)
