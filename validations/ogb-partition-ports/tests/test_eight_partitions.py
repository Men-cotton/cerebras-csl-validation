from pathlib import Path
import sys
import unittest

FAMILY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAMILY))
import eight


class TestEightPortAssignments(unittest.TestCase):
    def test_windows(self):
        releases = sorted((FAMILY / "eight-cases").glob("*/release.json"))
        self.assertTrue(releases)
        for release in releases:
            with self.subTest(case=release.parent.name):
                build, scores = eight.check(release.parent, eight.device.sha(release))
                self.assertEqual((build["lanes"], build["owners_per_lane"]), (8, 2))
                self.assertEqual(set(scores), {0.5, 1.0})
                with self.assertRaisesRegex(ValueError, "pin"):
                    eight.check(release.parent, "0" * 64)
