"""Replay all five official frames from each remaining arxiv batch."""
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

HERE=Path(__file__).resolve().parents[1]/"batched"

class RemainingArxiv(unittest.TestCase):
    def test_three_complete_batches(self):
        for idx in (1,2,3):
            name=f"arxiv-rcm32-b{idx}-p32"
            pin=hashlib.sha256((HERE/"cases"/name/"release.json").read_bytes()).hexdigest()
            subprocess.run([sys.executable,"-B",str(HERE/"single.py"),"check","--case",name,"--release-sha256",pin],check=True)
