"""Replay an actual SDK capture. Reject stale or duplicated PE witnesses."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from appliance import replay, verify
from worker import diagnostics, sha, witnesses


class Completion(unittest.TestCase):
    def test_release_and_pilot(self):
        full = verify(sha(HERE/'release.json'))
        pilot = HERE/'pilot'
        build = json.loads((pilot/'build.json').read_text())
        self.assertEqual(full['source'], build['source'])
        self.assertEqual([full['params'][k] for k in ('C','R')], [16,32])
        r = replay(pilot/'run', build, 'simulator', pilot/'reference.npz',
                   sha(pilot/'build.json'), sha(HERE/'worker.py'))
        self.assertEqual(len(r['dispatch_to_all_pe_ns']), 5)
        self.assertEqual(full['roles']['hardware']['elf_sha256'], sha(HERE/'bundle/out/program.elf'))
        self.assertEqual(full['comparison_reference_sha256'], sha(HERE/'reference.npz'))

    def test_wrong_epoch_peer_and_logit(self):
        pilot = HERE/'pilot'
        p = json.loads((pilot/'build.json').read_text())['params']
        with np.load(pilot/'run/frame-2.npz') as raw, np.load(pilot/'reference.npz') as ref:
            with self.assertRaises(ValueError):
                witnesses(raw['witness'], p, 1)
            words = raw['witness']
            words[-1] = words[-2]
            with self.assertRaises(ValueError):
                witnesses(words, p, 2)
            reply = raw['reply']
            reply[17+p['L']*p['H']] ^= 1
            with self.assertRaises(AssertionError):
                diagnostics(reply, p, 2, ref)
