import json
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
class SelectedCora(unittest.TestCase):
    def test_both_complete_selected_programs(self):
        selection=json.loads((ROOT/"selection.json").read_text())
        for lang in ("csl","tungsten"):
            info=json.loads((ROOT/lang/"target.json").read_text())
            self.assertEqual(info["name"],selection["best"][lang])
            subprocess.run([sys.executable,"-B","appliance.py","check"],cwd=ROOT/lang,check=True)
            local=json.loads((ROOT/lang/"sim/result.json").read_text())
            self.assertEqual(local["summary"]["words"],56868)
    def test_simulator_is_rejected_as_physical(self):
        code="from pathlib import Path\nfrom appliance import *\nfrom contract import load_graph\nt=read(ROOT/'target.json')\ntry:\n check_capture(ROOT/'sim',t['language'],16,'hardware',read(ROOT/'release.json'),load_graph(ROOT/'fixture.npz'))\nexcept ValueError:\n pass\nelse:\n raise AssertionError('simulator accepted as hardware')"
        for lang in ("csl","tungsten"):
            subprocess.run([sys.executable,"-B","-c",code],cwd=ROOT/lang,check=True)
    def test_same_original_logits(self):
        a,b=[json.loads((ROOT/lang/"sim/result.json").read_text())["summary"] for lang in ("csl","tungsten")]
        self.assertEqual(a["owner_sha256"],b["owner_sha256"])
        self.assertEqual(a["validation"],[402,500])
