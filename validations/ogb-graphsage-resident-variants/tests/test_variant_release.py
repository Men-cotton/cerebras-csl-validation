from pathlib import Path
import subprocess
import sys
import unittest


class Variant(unittest.TestCase):
    def test_frozen_variants(self):
        family=Path(__file__).resolve().parents[1]
        cases=sorted((family/"cases").glob("*/parent.json"))
        self.assertTrue(cases)
        for item in cases:
            subprocess.run([sys.executable,"-B",family/"variant.py","check","--case",item.parent.name],check=True)
