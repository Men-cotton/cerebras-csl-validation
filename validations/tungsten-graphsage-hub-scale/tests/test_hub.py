import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from appliance import check_capture,sha,verify_case


class DeviceHubTest(unittest.TestCase):
    def test_partial_local_is_not_complete_hardware(self):
        for rows in (32,64):
            case=HERE/'cases'/f'full-c16r{rows}-t8-cluster-mean-factor-hub-split'
            result=verify_case(case,sha(case/'build.json'),sha(case/'proof.json'))
            self.assertEqual(result['required_hardware_trials'],[1,2,3])
            self.assertEqual(result['split_control_trial'],1)
            record=json.loads((case/'run/capture.json').read_text())
            record.update(backend='hardware',stage='complete',stop_success=True,trials=[])
            with tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);(folder/'capture.json').write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError,'incomplete frame sequence'):
                    check_capture(case/'build.json',case/'reference.npz',folder,'hardware')
