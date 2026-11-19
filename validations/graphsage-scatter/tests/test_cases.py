"""Applied in the frozen watcher family; recheck the independent FP64 oracle."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import appliance
import saved_audit


class PairedCase(unittest.TestCase):
    def test_frozen_projection_and_logits(self):
        for name, pin in json.loads((HERE / 'simulator.json').read_text()).items():
            with self.subTest(case=name):
                binding = appliance.identity(HERE, name)
                self.assertEqual(binding['case_sha256'], pin['case_sha256'])
                case = HERE / 'cases' / name
                manifest = saved_audit.load(case)
                expected, _ = saved_audit.independent_expected(case, manifest, saved_audit.load_npz(case / 'inputs.npz'))
                with np.load(case / 'expected.npz') as frozen:
                    for key, values in expected.items():
                        np.testing.assert_allclose(frozen[key], values, rtol=1e-12, atol=1e-12)
