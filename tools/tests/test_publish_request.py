import base64
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import publish_request as pub


class PublishRequest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name)/'repo'
        self.repo.mkdir()
        def git(*args):
            return subprocess.check_output(['git','-C',str(self.repo),'-c','user.name=Tester',
                                            '-c','user.email=tests@example.invalid',*args])
        self.git=git
        git('init','-q')
        git('commit','--allow-empty','-qm','Initial')
        self.path='requests/demo/demo.json'
        request=self.repo/self.path
        request.parent.mkdir(parents=True)
        request.write_text('{"id":"demo"}\n')
        (self.repo/'image.bin').write_bytes(bytes([255,0,2]))
        git('add',self.path,'image.bin')

    def test_exact_message_and_binary(self):
        self.git('commit','-qm','Register demo','-m','ALCF-Request: demo\nALCF-Manifest: '+self.path)
        out=Path(self.tmp.name)/'objects'
        reg=pub.plan(self.repo,'HEAD',out)
        self.assertEqual(reg['manifest'],self.path)
        commit=json.loads((out/'commit.json').read_bytes())
        self.assertEqual(commit['message'].encode(),self.git('cat-file','commit','HEAD').split(b'\n\n',1)[1])
        self.assertEqual(commit['parents'],[self.git('rev-parse','HEAD^').decode().strip()])
        tree=json.loads((out/'tree.json').read_bytes())
        files={row['path']:row for row in tree['tree']}
        self.assertEqual(files[self.path]['content'],'{"id":"demo"}\n')
        binary=json.loads((out/(files['image.bin']['sha']+'.json')).read_bytes())
        self.assertEqual(base64.b64decode(binary['content']),bytes([255,0,2]))

    def test_wrong_registration_refused(self):
        self.git('commit','-qm','Register demo','-m','ALCF-Request: other\nALCF-Manifest: '+self.path)
        with self.assertRaisesRegex(ValueError,'final commit paragraph'):
            pub.verify(self.repo,'HEAD')
