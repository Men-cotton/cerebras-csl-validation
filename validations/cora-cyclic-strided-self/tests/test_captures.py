import json
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
class OriginalCora(unittest.TestCase):
    def test_each_geometry(self):
        for folder in sorted(ROOT.glob("r[0-9]*")):
            subprocess.run([sys.executable,"-B","appliance.py","check"],cwd=folder,check=True)
            data=json.loads((folder/"simulator-comparison.json").read_text())
            self.assertEqual(data["cases"]["csl"]["summary"]["words"],56868)
            self.assertEqual(data["cases"]["tungsten"]["summary"]["validation"],[402,500])
    def test_same_source_across_available_geometries(self):
        releases=[json.loads((folder/"release.json").read_text()) for folder in sorted(ROOT.glob("r[0-9]*"))]
        self.assertTrue(releases)
        self.assertEqual(len({v["input_sha256"] for v in releases}),1)
        self.assertEqual(len({tuple(sorted(v["csl"].items())) for v in releases}),1)
        self.assertEqual(len({v["direct"][str(v["shards"])]["source"]["aggregation.w"] for v in releases}),1)
    def test_reject_simulator_as_physical(self):
        code="from appliance import *\nfrom contract import load_graph\nl=read(ROOT/'release.json');r=l['shards']\nfor lang in ('csl','tungsten'):\n try:\n  check_capture(ROOT/'sim'/f'{lang}-r{r}',lang,r,'hardware',l,load_graph(ROOT/'fixture.npz'))\n except ValueError:\n  continue\n raise AssertionError('simulator accepted as physical')"
        for folder in sorted(ROOT.glob("r[0-9]*")):
            subprocess.run([sys.executable,"-B","-c",code],cwd=folder,check=True)
