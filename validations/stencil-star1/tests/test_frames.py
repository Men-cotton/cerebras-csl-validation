import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
spec = importlib.util.spec_from_file_location('stencil_star1_appliance',HERE/'appliance.py')
appliance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(appliance)

class TestFrames(unittest.TestCase):
    def test_frozen_frames(self):
        cases = sorted((HERE/'cases').glob('*/case.json'))
        self.assertTrue(cases)
        for path in cases:
            with self.subTest(case=path.parent.name):
                doc = appliance.verify_sources(path.parent,appliance.sha(path))
                rows = appliance.verify_frames(path.parent,path.parent/'sim','simulator')
                self.assertEqual(len(rows),3)
                self.assertFalse(any(v['max_cycles']<=0 for v in rows))
                self.assertEqual(doc['params']['E'],8)

    def test_wrong_case_pin(self):
        path = next((HERE/'cases').glob('*/case.json'))
        with self.assertRaises(ValueError):
            appliance.verify_sources(path.parent,'0'*64)
