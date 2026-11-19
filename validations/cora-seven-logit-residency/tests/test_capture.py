import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from appliance import verified

class LongResidency(unittest.TestCase):
    def test_all_three_129_panel_simulator_runs(self):
        captured=verified()
        self.assertEqual(len(captured),3)
        for result in captured.values():
            self.assertEqual(len(result["max_owner_compute"]),129)
