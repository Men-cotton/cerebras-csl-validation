from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest

import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAMILY))
import appliance as candidate


class SpacedWest(unittest.TestCase):
    def test_complete_control_and_both_actual_images(self):
        case = candidate.CASE
        build, arrays = candidate.check(case, candidate.device.sha(case / "release.json"))
        self.assertEqual(build["roles"]["hardware"]["fabric"], [762, 1172])
        self.assertEqual({k:v.shape for k,v in arrays.items()}, {0.5:(128,40), 1.0:(128,40)})
        self.assertFalse(np.array_equal(arrays[1.0], arrays[0.5]))

    def test_adjacent_second_lane_is_distinct_from_physical_pitch(self):
        raw = candidate.read(candidate.CASE / "hardware/out/out_port_map.json")
        moved = copy.deepcopy(raw)
        moved["lvds_ports"][2]["loc"] = 2
        self.assertNotEqual(candidate.mapping(raw), candidate.mapping(moved))

    def test_reject_wrong_release(self):
        with self.assertRaisesRegex(ValueError, "registered pitch-18 release"):
            candidate.check(candidate.CASE, "0"*64)

    def test_missing_capture_cannot_claim_hardware_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "result.json").write_text(json.dumps({"role":"hardware","stopped":False,"frames":[]}))
            with self.assertRaisesRegex(ValueError, "incorrect firmware or unfinished capture"):
                candidate.report.replay(candidate.CASE/"fixture.npz", candidate.CASE, out)
