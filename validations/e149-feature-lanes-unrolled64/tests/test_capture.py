import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from appliance import verified

class ReturnedFrames(unittest.TestCase):
    def test_frozen_simulators(self):
        cycles=verified()
        self.assertEqual(set(cycles),{"csl","tungsten"})
        for values in cycles.values():
            self.assertEqual(len(values),3)
            self.assertTrue(all(v>0 for v in values))
