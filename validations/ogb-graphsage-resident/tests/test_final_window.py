"""Check the RCM final-layer release and reject failures at its system boundary."""
from pathlib import Path
import subprocess
import sys
import unittest

BASE = Path(__file__).resolve().parents[1] / "window-cache/rcm-final"


class FinalWindow(unittest.TestCase):
    def test_physical_image_and_two_sdk_frames(self):
        import hashlib
        pin = hashlib.sha256((BASE / "release.json").read_bytes()).hexdigest()
        subprocess.run([sys.executable, "-B", str(BASE / "appliance.py"), "check", "--release", pin], check=True)

    def test_prefix_is_ineligible_as_completed_physical_measurement(self):
        source = """
import sys, tempfile, subprocess
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from appliance import HERE, SHARED, audit, prepare, sha, verify
with tempfile.TemporaryDirectory() as tmp:
    stage = Path(tmp)
    pin = sha(HERE / "release.json")
    prepare(stage, pin)
    try:
        verify(stage, SHARED / "base/physical/out", pin)
    except ValueError:
        pass
    else:
        raise AssertionError("local hardware build accepted as returned system firmware")
    try:
        audit(stage, SHARED / "base/sim", HERE, HERE / "sdk", stage / "audit.json")
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("two SDK frames accepted as complete")
"""
        subprocess.run([sys.executable, "-B", "-c", source, str(BASE)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
