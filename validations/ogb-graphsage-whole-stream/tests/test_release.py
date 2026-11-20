import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]


class FullLayerStream(unittest.TestCase):
    def test_frozen_native_and_peer_controls(self):
        subprocess.run([sys.executable,"-B",ROOT/"worker.py","check"],check=True)

    def test_existing_graph_is_reused(self):
        release=json.loads((ROOT/"case/release.json").read_text())
        self.assertEqual(release["payload"]["whole.npz"]["source"],"parent")
        self.assertEqual((release["windows"],release["frames"]),(331,339))
        self.assertEqual(release["native_sdk"]["nodes"],1407)
