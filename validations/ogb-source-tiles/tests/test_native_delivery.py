import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

import numpy as np

FAMILY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(FAMILY/'native'))
from inventory import actual, program_bytes, read, sha
from compare_delivery import checked
from source_table import assign, audit, device
from paired_worker import original_reply, sequence
from physical import NAMES, checked_case, compiler_args


def fixture_tar(dest,folder,flags,src,altered=False):
    with tarfile.open(dest,'w:gz') as archive:
        def add(name,payload):
            member=tarfile.TarInfo(name);member.size=len(payload)
            archive.addfile(member,io.BytesIO(payload))
        add('program/compile_command.sh',('cslc src/layout.csl '+' '.join(flags)+'\n').encode())
        for n in src:
            payload=(folder/'src'/n).read_bytes()
            add('program/src/'+n,payload+(b'x' if altered and n=='layout.csl' else b''))
        config=read(folder/'build.json')['csl_params']
        import json
        params=json.dumps({'params':{k:str(v) for k,v in config.items()}}).encode()
        for name in ('out.json','east/out.json','west/out.json','bin/out_rpc.json'):
            add('program/out/'+name,params)
        for name in ('bin/out_0_0.elf','east/bin/out_0_0.elf','west/bin/out_0_0.elf'):
            add('program/out/'+name,b'fixture-'+name.encode())


class NativeDelivery(unittest.TestCase):
    def test_frozen_two_languages_and_delivery_controls(self):
        for name in NAMES:
            case=FAMILY/'native/cases'/name
            _,rel,sdk,fixture,expected=checked_case(name,sha(case/'release.json'))
            self.assertEqual(len(rel['original_ids']),128)
            self.assertEqual(len(set(rel['original_ids'])),128)
            self.assertEqual(sdk['params']['P'],16)
            self.assertEqual({0.5,1.},set(expected))
            for v in expected.values():self.assertEqual(v.shape[0],128)
            with self.assertRaisesRegex(ValueError,'release pin'):
                checked_case(name,'0'*64)

    def test_complete_real_paired_sdk_capture(self):
        pilot=FAMILY/'native/pilot';fixture=pilot/'fixture.npz';meta=read(pilot/'build.json')
        pins=read(pilot/'provenance.json')
        for path,h in pins['files'].items():self.assertEqual(sha(pilot/path),h)
        self.assertEqual(actual(pilot/'out'),meta['inventory'])
        for n,h in meta['worker_sha256'].items():
            path=FAMILY/'native'/n
            self.assertEqual(sha(path if path.is_file() else FAMILY/n),h)
        z=device.load(fixture);cfg,g,_,_=assign(z,2,16)
        reference={}
        for tag,scale in ((1,1.),(3,.5)):
            for name in ('native','direct'):
                with np.load(pilot/'reference'/f'{name}-{tag}.npz',allow_pickle=False) as raw:
                    values,_=audit(raw['reply'],z,cfg,g,tag-1)
                if scale in reference:np.testing.assert_array_equal(reference[scale],values.view(np.uint32))
                else:reference[scale]=values.view(np.uint32)
        record=checked(pilot,fixture,pilot/'capture',reference,'simulator')
        self.assertEqual(len(record['transactions']),12);self.assertEqual(record['classes'],47)

    def test_unique_wire_tags_and_balanced_order(self):
        frames=list(sequence());self.assertEqual(len(frames),12)
        self.assertEqual([r[-1] for r in frames[::2]].count('row'),3)
        for index in range(0,12,2):
            self.assertEqual(frames[index][:-1],frames[index+1][:-1])
            self.assertNotEqual(frames[index][-1],frames[index+1][-1])
        a=np.array([[3,8],[3,2]],np.uint32)
        b=original_reply(a,3,1)
        np.testing.assert_array_equal(a[:,0],[3,3]);np.testing.assert_array_equal(b[:,0],[2,2])
        with self.assertRaisesRegex(ValueError,'transaction tag'):
            original_reply(a,4,1)

    def test_compiler_source_flags_layout_and_all_roles(self):
        case=FAMILY/'native/cases'/NAMES[0]
        sdk=read(case/'build.json')
        flags=compiler_args(sdk)
        with tempfile.TemporaryDirectory() as t:
            archive=Path(t)/'hardware.tar.gz'
            fixture_tar(archive,case,flags,sdk['source_sha256'])
            files=program_bytes(archive,sdk['source_sha256'],flags,sdk['csl_params'])
            self.assertIn('bin/out_0_0.elf',files);self.assertIn('west/bin/out_0_0.elf',files)
            self.assertIn('east/bin/out_0_0.elf',files)
            with self.assertRaisesRegex(ValueError,'compiler flag'):
                program_bytes(archive,sdk['source_sha256'],['--channels=9'],sdk['csl_params'])
            fixture_tar(archive,case,flags,sdk['source_sha256'],altered=True)
            with self.assertRaisesRegex(ValueError,'physical application sources'):
                program_bytes(archive,sdk['source_sha256'],flags,sdk['csl_params'])
