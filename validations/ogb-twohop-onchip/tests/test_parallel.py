import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]/"parallel"
spec = importlib.util.spec_from_file_location("two_row_physical",ROOT/"appliance.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class MeasuredTwoRowGraph(unittest.TestCase):
    def test_registered_images_and_all_nine_frames(self):
        for r in sorted((ROOT/"cases").glob("*/release.json")):
            with self.subTest(case=r.parent.name):
                case,peer,desc,report = app.checked_case(r.parent.name,app.device.sha(r))
                self.assertEqual(len(report["frames"]),9)
                self.assertEqual(report["first_owners"],desc["first"]["P"])
                self.assertEqual(report["two_port_geometry"],[desc["first"]["P"]//2,4])
                self.assertEqual(report["one_port_geometry"],[desc["first"]["P"],2])
                self.assertEqual(app.device.sha(peer/"build.json"),report["original_build_sha256"])
                for f in report["frames"]:
                    self.assertEqual(len(f["input_words_by_port"]),2)
                    self.assertGreater(f["collector_from_data_tag_cycles"],0)
                    self.assertLess(f["collector_from_data_tag_cycles"],f["collector_first_to_last_cycles"])
                with self.assertRaisesRegex(ValueError,"two-row release SHA"):
                    app.checked_case(case.name,"0"*64)

    def test_real_instrumented_calibration_and_physical_guard(self):
        case=ROOT/"cases/small-arxiv-p2"
        case,peer,_,_=app.checked_case(case.name,app.device.sha(case/"release.json"))
        result=app.observed(case,peer,case/"bound-sdk","simulator",True)
        self.assertEqual(sum(f["echo"] for f in result["frames"]),4)
        self.assertEqual({f["concurrent"] for f in result["frames"]},{False,True})
        for i in (2,3,4,5):
            with np.load(case/"bound-sdk"/f"frame-{i}.npz") as raw:
                self.assertEqual(raw["echo"].size,64)
        with self.assertRaisesRegex(ValueError,"target and one-port SDK reference"):
            app.observed(case,peer,case/"bound-sdk","hardware",True)
