import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from appliance import verified

class OwnerWindow(unittest.TestCase):
    def test_full_twelve_frames(self):
        rows=verified()
        self.assertEqual(len(rows),3)
        self.assertEqual({len(r["max_owner_compute"]) for r in rows.values()},{12})
