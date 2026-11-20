import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from appliance import check_capture,sha,verify_case


class PhysicalScaleTest(unittest.TestCase):
    def test_exact_targets_and_required_hardware_sequence(self):
        for rows in (32,64):
            case=HERE/'cases'/f'full-c16r{rows}-t8-cluster-mean-fused-44k'
            b=sha(case/'build.json');proof=sha(case/'proof.json')
            verified=verify_case(case,b,proof)
            self.assertEqual(verified['required_hardware_trials'],[1,2,3])
            self.assertEqual(verified['complete_control_trials'],[1,2,3])
            with self.assertRaises(ValueError):
                verify_case(case,b,'0'*64)
            saved=json.loads((case/'run/capture.json').read_text())
            saved.update(backend='hardware',stage='complete',stop_success=True,trials=[])
            with tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);(folder/'capture.json').write_text(json.dumps(saved))
                with self.assertRaisesRegex(ValueError,'incomplete frame sequence'):
                    check_capture(case/'build.json',case/'reference.npz',folder,'hardware')
