import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('endpoint_host', HERE/'appliance.py')
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


class Controls(unittest.TestCase):
    def setUp(self):
        self.case = HERE/'cases/west-two'
        self.build = json.loads((self.case/'build.json').read_bytes())

    def test_both_stopped(self):
        host.check(self.case,host.sha(self.case/'release.json'))

    def test_wrong_pair_data_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture=Path(tmp)/'data'
            shutil.copytree(self.case/'sdk',capture)
            path=capture/'fixed/trace.json'
            doc=json.loads(path.read_bytes())
            doc['transactions'][1]['output'][0]^=1
            host.save(path,doc)
            with self.assertRaisesRegex(ValueError,'eight returned data words'):
                host.replay(capture,'simulator',self.build)
