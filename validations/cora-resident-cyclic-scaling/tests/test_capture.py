import sys
from pathlib import Path
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "tests" else HERE
sys.path.insert(0, str(ROOT))
from appliance import check_capture, read, verified
from contract import check_frame, banks, checksum, load_graph, window


class FullCora(unittest.TestCase):
    def test_all_four_resident_programs(self):
        for summary in verified().values():
            self.assertEqual(summary["words"], 56868)
            self.assertEqual(summary["edges"], 13264)
            self.assertEqual(summary["validation"], [402, 500])
            self.assertEqual(len(summary["windows"]), 11)

    def test_simulator_trace_rejected_as_physical(self):
        with self.assertRaises(ValueError):
            check_capture(
                ROOT / "sim/tungsten-r4",
                "tungsten",
                4,
                "hardware",
                read(ROOT / "release.json"),
                load_graph(ROOT / "fixture.npz"),
            )

    def test_graph_word_witness_rejects_corruption(self):
        graph = load_graph(ROOT / "fixture.npz")
        sl = window(graph, 4, 0)
        hashes = [
            np.array([[checksum(bank[y, x].view(np.uint32)) for x in range(4)] for y in range(4)])
            for bank in banks(graph, 4)
        ]
        with np.load(ROOT / "sim/tungsten-r4/raw.npz") as raw:
            reply = raw["w0_t1"].copy()
        reply[4] ^= np.uint32(1)
        with self.assertRaises(AssertionError):
            check_frame(graph, sl, 0, 1, 4, reply, hashes)
