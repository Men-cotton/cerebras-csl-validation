from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"resident"))
from physical import checked_case,unpack


class ResidentTwoHop(unittest.TestCase):
    def test_original_nodes_and_dataflow(self):
        files=sorted((ROOT/"cases").glob("*/release.json"))
        self.assertTrue(files)
        for path in files:
            _,rel,csl,tungsten=checked_case(path.parent.name)
            self.assertEqual(csl["params"],tungsten["params"])
            self.assertEqual(len(rel["sdk_peer"]["phases"]),9)
            self.assertEqual(rel["sdk_peer"]["phases"]["t1-s2"]["rows"],4)

    def test_wrong_capture_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with zipfile.ZipFile(root/"raw.zip","w") as z:
                z.writestr("result.json",b"{}")
                z.writestr("t1-s0-w0100.npz.extra",b"1")
            with self.assertRaises(ValueError):
                unpack(root/"raw.zip",root/"out")
