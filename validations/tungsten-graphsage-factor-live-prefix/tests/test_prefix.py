"""Check local prefix admission and the strict physical three-trial boundary."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

FAMILY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(FAMILY))
from appliance import check_capture, sha, verify_case


class InterruptedPrefix(unittest.TestCase):
    def test_prefix_and_hardware_boundary(self):
        case=FAMILY/'cases/full-c16r64-t8-cluster-mean-factor'
        result=verify_case(case,sha(case/'build.json'),sha(case/'prefix.json'))
        self.assertEqual(result['local_completed_trials'],[1,2])
        self.assertEqual(result['hardware_required_trials'],[1,2,3])
        with self.assertRaises(ValueError):
            verify_case(case,sha(case/'build.json'),'0'*64)
        with self.assertRaises(ValueError):
            check_capture(case/'build.json',case/'reference.npz',case/'run','hardware')
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            shortened=json.loads((case/'run/capture.json').read_text())
            shortened.update(backend='hardware',stage='complete',stop_success=True)
            (folder/'capture.json').write_text(json.dumps(shortened))
            with self.assertRaisesRegex(ValueError,'incomplete frame sequence'):
                check_capture(case/'build.json',case/'reference.npz',folder,'hardware')
