import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whole"))
from physical import checked_case, unpack


class WholeGraph(unittest.TestCase):
    def test_paired_representative_windows(self):
        _, release, native, direct = checked_case()
        self.assertEqual(release["windows"], 331)
        self.assertEqual(release["frames"], 339)
        self.assertEqual(native["params"], direct["params"])
        self.assertEqual(release["local_sdk"]["csl"]["nodes"], 1407)
        self.assertEqual(len(release["representative_windows"]), 3)

    def test_reject_unexpected_capture_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with zipfile.ZipFile(tmp / "raw.zip", "w") as output:
                output.writestr("w0001.npz.extra", b"x")
            with self.assertRaises(ValueError):
                unpack(tmp / "raw.zip", tmp / "out")
