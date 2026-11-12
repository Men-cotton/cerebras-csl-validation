"""Hardware adapter tests use local stubs, never SDK clients or job submission."""
from contextlib import contextmanager
import io
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import hardware_entry as hw
from models.hierarchical_gnn.grouped_csr.native_cora_appliance import appliance, saved_audit


def program_payloads():
    # Minimal generated ELF fixture with one allocated NOBITS section. This is
    # test data for the bounded ELF section reader, not an executable program.
    ident = b'\x7fELF'+bytes([1, 1, 1])+bytes(9)
    header = struct.pack('<HHIIIIIHHHHHH', 2, 0, 1, 0, 0, 52, 0, 52, 0, 0, 40, 1, 0)
    section = struct.pack('<IIIIIIIIII', 0, 8, 2, 0, 0, 16, 0, 0, 4, 0)
    return {'artifact/out/out.json': b'{"params":{}}',
            'artifact/out/west/out.json': b'{}', 'artifact/out/east/out.json': b'{}',
            'artifact/out/bin/out_rpc.json': b'{"rpc_symbols":[]}',
            'artifact/out/bin/pe.elf': ident+header+section,
            'artifact/csl/layout.csl': b'layout fixture', 'artifact/csl/pe.csl': b'pe fixture'}


def archive_program(path, payloads):
    with tarfile.open(path, 'w:gz') as archive:
        for name, payload in payloads.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def source_pins(payloads):
    return {name: hashlib.sha256(payloads['artifact/csl/'+name]).hexdigest() for name in ('layout.csl', 'pe.csl')}


class ProgramBindingTests(unittest.TestCase):
    def test_archive_matches_actual_generated_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payloads = program_payloads()
            archive_program(root/'program.tgz', payloads)
            expected = hw.program_archive(root/'program.tgz', source_pins(payloads), saved_audit)
            hw.unpack(root/'program.tgz', root/'extracted')
            actual = hw.bind_actual_program(root/'extracted/artifact/out', dict(program_inventory=expected), saved_audit)
            self.assertEqual(actual, expected)
            self.assertEqual(actual['elf']['max_static_sram_end'], 16)

    def test_reject_changed_runtime_elf_or_metadata(self):
        for changed in ('bin/pe.elf', 'bin/out_rpc.json'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                payloads = program_payloads()
                archive_program(root/'program.tgz', payloads)
                expected = hw.program_archive(root/'program.tgz', source_pins(payloads), saved_audit)
                hw.unpack(root/'program.tgz', root/'extracted')
                build = root/'extracted/artifact/out'
                path = build/changed
                path.write_bytes(path.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError, 'unpacked program differs'):
                    hw.bind_actual_program(build, dict(program_inventory=expected), saved_audit)

    def test_reject_archive_compiled_from_different_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payloads = program_payloads()
            pins = source_pins(payloads)
            payloads['artifact/csl/pe.csl'] += b' changed'
            archive_program(root/'program.tgz', payloads)
            with self.assertRaisesRegex(ValueError, 'compiled CSL differs'):
                hw.program_archive(root/'program.tgz', pins, saved_audit)


class NumericalSummaryTests(unittest.TestCase):
    def test_only_small_diagnostic_roundoff_is_allowed(self):
        import copy
        fresh = dict(status='PASS', errors={'logits': {'max_abs': 1.54e-6,
                     'max_relative_floor_1e_12': 0.001, 'relative_l2': 6.64e-8}},
                     argmax_mismatches=0, accuracy={'correct':402, 'count':500, 'fraction':.804})
        shifted = copy.deepcopy(fresh)
        shifted['errors']['logits']['max_abs'] += 2e-15
        shifted['errors']['logits']['max_relative_floor_1e_12'] += .0001
        hw.compare_numerical_summary(fresh, shifted)
        for change in ('large', 'nan', 'negative', 'outcome', 'missing', 'impossible_relative'):
            bad = copy.deepcopy(fresh)
            if change == 'large': bad['errors']['logits']['max_abs'] = .01
            elif change == 'nan': bad['errors']['logits']['max_abs'] = float('nan')
            elif change == 'negative': bad['errors']['logits']['max_abs'] = -1.
            elif change == 'outcome': bad['argmax_mismatches'] = 1
            elif change == 'impossible_relative': bad['errors']['logits']['max_relative_floor_1e_12'] = 1e10
            else: del bad['errors']['logits']['relative_l2']
            with self.subTest(change=change), self.assertRaises(ValueError):
                hw.compare_numerical_summary(fresh, bad)


class ArchiveTests(unittest.TestCase):
    def test_extract_regular_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with tarfile.open(root/'good.tgz', 'w:gz') as archive:
                member = tarfile.TarInfo('source/run.py')
                member.size = 3
                archive.addfile(member, io.BytesIO(b'abc'))
            hw.unpack(root/'good.tgz', root/'result')
            self.assertEqual((root/'result/source/run.py').read_bytes(), b'abc')

    def test_reject_traversal_links_duplicates(self):
        for mode in ('traversal', 'link', 'duplicate'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with tarfile.open(root/'bad.tgz', 'w:gz') as archive:
                    member = tarfile.TarInfo('../escape' if mode == 'traversal' else 'safe')
                    if mode == 'link':
                        member.type = tarfile.SYMTYPE
                        member.linkname = '/etc/passwd'
                    archive.addfile(member)
                    if mode == 'duplicate': archive.addfile(member)
                with self.assertRaises(ValueError): hw.unpack(root/'bad.tgz', root/'out')
                self.assertFalse((root.parent/'escape').exists())


class AdapterTests(unittest.TestCase):
    def run_cohort(self, launch_failure=False, escape=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'package.tar.gz').write_bytes(b'package fixture')
            (root/'hardware_entry.py').write_bytes(b'worker fixture')
            pinned = dict(package_manifest_sha256='p', case='tiny', profile=0, case_sha256='c',
                          params=dict(C=2, R=2, PROFILE=0), warmup=1, repeats=2)
            payloads = program_payloads()
            pinned['csl_sha256'] = source_pins(payloads)
            commands, staged = [], []

            class Compiler:
                def __init__(self, **kwargs): self.options = kwargs
                def __enter__(self): return self
                def __exit__(self, *_args): pass
                def compile(self, directory, source, flags, out):
                    commands.append((directory, source, flags))
                    path = root/'escaped.bin' if escape else Path(out)/'artifact.bin'
                    archive_program(path, payloads)
                    return str(path)

            class Launcher:
                def __init__(self, *_args, **_kwargs): pass
                def __enter__(self): return self
                def __exit__(self, *_args): pass
                def stage(self, path): staged.append(Path(path).name)
                def run(self, command):
                    commands.append(command)
                    if launch_failure: raise RuntimeError('fixture failure')
                    return 'fixture completed'
                def download_artifact(self, name, requested):
                    path = Path(requested)
                    if name == 'capture':
                        path = path.with_suffix('.tar.gz')
                        with tarfile.open(path, 'w:gz') as archive:
                            directory = tarfile.TarInfo('capture')
                            directory.type = tarfile.DIRTYPE
                            archive.addfile(directory)
                            member = tarfile.TarInfo('capture/worker.json')
                            member.size = 2
                            archive.addfile(member, io.BytesIO(b'{}'))
                    else:
                        path.write_text('0\n' if name == 'host-exit.txt' else 'fixture log')
                    return str(path)

            with patch.object(hw, 'identity', return_value=pinned), patch.object(hw, 'modules', return_value=(saved_audit, appliance, None)):
                args = SimpleNamespace(case='tiny', profile=0, namespace='job-operator', out=root/'compile')
                if escape:
                    with self.assertRaises(ValueError): hw.compile_job(args, root, Compiler)
                    self.assertEqual(hw.read(root/'compile/build.json')['status'], 'FAILED')
                    return
                built = hw.compile_job(args, root, Compiler)
                self.assertEqual(built['status'], 'PASS')
                self.assertIn('PROFILE:0', commands[0][2])
                runargs = SimpleNamespace(case='tiny', profile=0, build_record=root/'compile/build.json', out=root/'run', host_timeout=90)
                with patch.object(hw, 'audit_capture', return_value=dict(status='PASS')):
                    if launch_failure:
                        with self.assertRaises(RuntimeError): hw.run_job(runargs, root, Launcher)
                        self.assertTrue((root/'run/capture.tar.gz').exists())
                        self.assertEqual(hw.read(root/'run/collection.json')['status'], 'FAILED')
                    else:
                        result = hw.run_job(runargs, root, Launcher)
                        self.assertEqual(result['status'], 'PASS')
                        self.assertTrue((root/'run/capture-data/worker.json').is_file())
                        self.assertFalse((root/'run/capture.tar.gz').exists())
                self.assertEqual(staged, ['hardware_entry.py', 'package.tar.gz', 'build.json', 'run-request.json'])
                self.assertIn('%CMADDR%', commands[1])
                self.assertNotIn('ssh ', commands[1])

    def test_compile_stage_run_collect_with_local_stubs(self): self.run_cohort()
    def test_failed_launch_preserves_partial_download(self): self.run_cohort(launch_failure=True)
    def test_reject_compiler_output_escape(self): self.run_cohort(escape=True)


if __name__ == '__main__':
    unittest.main()
