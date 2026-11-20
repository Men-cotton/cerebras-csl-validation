"""Verify the additive release without polluting the already released modules' import path."""
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

EXT = Path(__file__).resolve().parents[1] / "parallel/placement"


class PlacementRelease(unittest.TestCase):
    def test_machine_image_and_nine_frame_placement(self):
        case = "arxiv-p16-e24"
        pin = hashlib.sha256((EXT/"cases"/case/"release.json").read_bytes()).hexdigest()
        subprocess.run([sys.executable,"-B",str(EXT/"appliance.py"),"check","--case",case,
                        "--release-sha256",pin],check=True)
