import json
from pathlib import Path
import sys
import unittest
NATIVE = Path(__file__).resolve().parents[1]/"native"
sys.path.insert(0,str(NATIVE))
import physical
class CompleteBudgetControl(unittest.TestCase):
    def test_all_accepted_cases(self):
        releases=list((NATIVE/"cases").glob("*/release.json"))
        self.assertEqual(len(releases),1)
        case=releases[0].parent
        _,_,sdk,fixture,expected=physical.check(case.name,physical.sha(releases[0]))
        self.assertEqual(sdk["params"]["P"]*sdk["params"]["L"],16)
        self.assertEqual(expected[1.].shape,(128,47))
    def test_invalid_release(self):
        name=next((NATIVE/"cases").glob("*/release.json")).parent.name
        with self.assertRaisesRegex(ValueError,"release pin"):
            physical.check(name,"0"*64)
