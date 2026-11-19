"""Recheck paired images, all compute outputs, and the twelve repeated setup records."""
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from audit import load_verified


class FrozenCase(unittest.TestCase):
    def test_physical_images_and_setup(self):
        for owners in (1,2,4,8,16):
            with self.subTest(owners=owners):
                case=ROOT/'cases'/f'p{owners}'
                load_verified(case,owners)
                capture=json.loads((case/'verification/result.json').read_text())
                self.assertEqual(len(capture['trials']),12)
                self.assertEqual(len(capture['trials'][0]['setup_cycles']),owners)
