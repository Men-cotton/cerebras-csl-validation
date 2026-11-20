import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT=Path(__file__).resolve().parents[1]


class HeaderCrossover(unittest.TestCase):
    def test_five_complete_captures(self):
        for case in ("on","off","cyclic","mapped"):
            subprocess.run([sys.executable,"-B","appliance.py","check"],cwd=ROOT/case,check=True)

    def test_input_only_native_header_control(self):
        on,off=[json.loads((ROOT/n/"release.json").read_text()) for n in ("on","off")]
        self.assertEqual(on["mode"],"self-simd")
        self.assertEqual(off["mode"],"all-fmac")
        for field in ("input_sha256","csl","csl_simulator","direct","port_sha256"):
            self.assertEqual(on[field],off[field])
        proof=json.loads((ROOT/"header-only.json").read_text())
        self.assertTrue(proof["same_elf_per_language"] and proof["only_host_runtime_count_bits_differ"])
        self.assertEqual(len(on["csl_simulator"]["16"]["elf"]),64)

    def test_simulated_capture_rejected_as_hardware(self):
        for subject,lang,capture in (("off","csl","sim/csl-r16"),("on","csl","sim/csl-r16"),
                ("off","tungsten","sim/tungsten-r16"),("cyclic","tungsten","sim"),("mapped","tungsten","sim")):
            code=("from appliance import *\nfrom contract import load_graph\ntry:\n"
                  f" check_capture(ROOT/{capture!r},{lang!r},16,'hardware',read(ROOT/'release.json'),load_graph(ROOT/'fixture.npz'))\n"
                  "except ValueError:\n pass\nelse:\n raise AssertionError('simulator accepted as physical')")
            subprocess.run([sys.executable,"-B","-c",code],cwd=ROOT/subject,check=True)

    def test_both_native_hosts_accept_the_same_compile_record(self):
        code='''
import io,sys,tarfile,tempfile,types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import appliance as app
class ReachedLauncher(Exception):pass
def launcher(program,**kw):
    assert Path(program)==archive and not kw['simulator']
    raise ReachedLauncher
with tempfile.TemporaryDirectory() as temp:
    folder=Path(temp);archive=folder/'archive.tar'
    with tarfile.open(archive,'w') as t:
        for name in ('out/out.json','out/east/out.json','out/west/out.json'):
            entry=tarfile.TarInfo(name);entry.size=2;t.addfile(entry,io.BytesIO(b'{}'))
        for elf in (app.ROOT/'csl-r16/out/bin').glob('*.elf'):t.add(elf,'out/bin/'+elf.name)
    build=folder/'build.json'
    record=dict(artifact=archive.name,sha256=app.sha(archive),application_elfs=app.application_elfs(archive),
                shards=16,flags=app.flags(16,True),release_sha256=app.sha(app.ROOT.parent/'off/release.json'))
    sdk=types.ModuleType('cerebras.sdk.client');sdk.SdkLauncher=launcher;sdk.SdkCompiler=object
    with patch.dict(sys.modules,{'cerebras.sdk.client':sdk}),patch.object(app,'verified',return_value={}):
        for wrong in (False,True):
            if wrong:record['release_sha256']='0'*64
            app.save(build,record)
            args=SimpleNamespace(action='run-csl',out=folder/str(wrong),shards=16,namespace='test',build=build)
            try:app.main(args)
            except ReachedLauncher:assert not wrong
            except ValueError:assert wrong
            else:raise AssertionError('expected verifier or launcher')
'''
        for mode in ("on","off"):
            subprocess.run([sys.executable,"-B","-c",code],cwd=ROOT/mode,check=True)
