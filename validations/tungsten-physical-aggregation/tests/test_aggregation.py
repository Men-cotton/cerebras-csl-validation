import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit import check_capture, load_verified
from contract import WIDTHS, expected, verify


class AggregationTest(unittest.TestCase):
    def test_paired_images_and_saved_frames(self):
        for width in WIDTHS:
            with self.subTest(width=width):
                candidates = [ROOT/'bundles', ROOT/'cases']
                root = next(base/f'p{width}' for base in candidates if (base/f'p{width}').is_dir())
                build = load_verified(root, width)
                capture = json.loads((root/'verification/result.json').read_text())
                self.assertEqual(len(check_capture(build,capture,'simulator')), 3)
                # A single changed owner output must be rejected even when PE
                # order, frame length, tags and clock words are all preserved.
                words = np.array(capture['trials'][0]['reply'], dtype='<u4')
                words[-1] ^= np.uint32(1)
                with self.assertRaisesRegex(ValueError,'owner output mismatch'):
                    verify(width,1,words)

    def test_restore_changes_inputs_then_replays(self):
        self.assertFalse(np.array_equal(expected(1),expected(2)))
        np.testing.assert_array_equal(expected(1),expected(3))
        self.assertTrue(np.all(expected(1)[::8] == 0))
