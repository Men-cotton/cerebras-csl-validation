import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

HOME_DIR=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HOME_DIR))
from appliance import verify_case
from contract import sha, verify


class PairedImageTest(unittest.TestCase):
    def test_frozen_sdk_frames(self):
        cases=sorted((HOME_DIR/'cases').glob('*/release.json'))
        self.assertTrue(cases)
        for release in cases:
            with self.subTest(case=release.parent.name):verify_case(release.parent,sha(release))

    def test_changed_value_with_updated_file_digest_fails(self):
        case=next((HOME_DIR/'cases').glob('*/release.json')).parent
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'capture';shutil.copytree(case/'sdk',folder)
            raw=folder/'raw.npz'
            with np.load(raw,allow_pickle=False) as source:values={name:source[name].copy() for name in source.files}
            values['replies'][0,0,10]^=np.uint32(1)
            np.savez_compressed(raw,**values)
            path=folder/'capture.json';record=json.loads(path.read_bytes());record['raw_sha256']=sha(raw);path.write_text(json.dumps(record))
            with self.assertRaises(AssertionError):verify(case/'build.json',case/'fixture.npz',folder,'simulator')

    def test_sdk_capture_requires_simulator_role(self):
        case=next((HOME_DIR/'cases').glob('*/release.json')).parent
        with self.assertRaisesRegex(ValueError,'target ELF identity'):
            verify(case/'build.json',case/'fixture.npz',case/'sdk','hardware')
