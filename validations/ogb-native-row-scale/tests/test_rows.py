import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

class TestNativeRows(unittest.TestCase):
    def test_released_windows(self):
        cases = sorted((ROOT / "cases").glob("*/release.json"))
        self.assertGreaterEqual(len(cases), 2)
        for p in cases:
            with self.subTest(case=p.parent.name):
                subprocess.run([sys.executable, "-B", ROOT / "physical.py", "check", "--case", p.parent.name, "--release-sha256", hashlib.sha256(p.read_bytes()).hexdigest()], check=True)
