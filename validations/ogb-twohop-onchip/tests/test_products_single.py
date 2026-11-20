"""Verify both SDK references while registering only the products one-port system image."""
from pathlib import Path
import hashlib
import subprocess
import sys
import unittest
BASE=Path(__file__).resolve().parents[1]/"batched"
class SinglePortProducts(unittest.TestCase):
    def test_both_sdk_controls(self):
        name="products-rcm32-b8-p32"
        worker=BASE/"products_single.py"
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        subprocess.run([sys.executable,"-B",str(worker),"check","--case",name,"--release-sha256",sha(BASE/"cases"/name/"release.json"),"--driver-sha256",sha(worker)],check=True)
