import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
class MappedTungsten(unittest.TestCase):
    def test_simulator_and_original_graph(self):
        subprocess.run([sys.executable,"-B","appliance.py","check"],cwd=ROOT/"case",check=True)
    def test_image_and_assignment(self):
        prior=json.loads((ROOT/"comparators.json").read_text());f=ROOT/"case";read=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        self.assertEqual(read(f/"source-map.npz"),prior["same_source_map_sha256"])
        self.assertEqual(read(f/"direct-r16/hardware/out/program.elf"),prior["same_physical_elf_sha256"])
    def test_simulator_cannot_supply_physical_result(self):
        code="from appliance import *\ntry:\n check_capture(ROOT/'sim','tungsten',16,'hardware',read(ROOT/'release.json'),load_graph(ROOT/'fixture.npz'))\nexcept ValueError:\n pass\nelse:\n raise AssertionError('simulator accepted as hardware')"
        subprocess.run([sys.executable,"-B","-c",code],cwd=ROOT/"case",check=True)
