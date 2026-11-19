"""Validate frozen native source, twelve resident replies and the setup timestamps."""
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from appliance import FILES, sha, verify


class FrozenCase(unittest.TestCase):
    def test_resident_panels(self):
        for owners in (1,2,4,8,16):
            with self.subTest(owners=owners):
                case=ROOT/'evidence'/f'p{owners}'
                build=json.loads((case/'local-build.json').read_text())
                capture=json.loads((case/'local-capture.json').read_text())
                self.assertEqual(build['source_sha256'],{f:sha(ROOT/f) for f in FILES})
                self.assertEqual([r['trial'] for r in capture['trials']],list(range(1,13)))
                verify(capture,owners,7,True)
