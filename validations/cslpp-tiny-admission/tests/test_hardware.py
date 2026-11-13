"""Offline generated-source, archive and single-compile boundary regressions."""
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import hardware_entry as hw


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def elf_fixture(endpoint=16):
    ident = b'\x7fELF'+bytes([1, 1, 1])+bytes(9)
    header = struct.pack('<HHIIIIIHHHHHH', 2, 0, 1, 0, 0, 52, 0, 52, 0, 0, 40, 1, 0)
    section = struct.pack('<IIIIIIIIII', 0, 8, 2, 0, 0, endpoint, 0, 0, 4, 0)
    return ident+header+section


def package_fixture(root):
    files, arms = {}, {}

    def put(name, value):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = value if isinstance(value, bytes) else json.dumps(value).encode()
        path.write_bytes(payload)
        files[name] = digest(payload)
        return files[name]

    case_sha = put('case/case.json', {})
    direct_layout = b'''layout {
  @set_rectangle(C, R);
  for (@range(u16, C)) |x| {
    for (@range(u16, R)) |y| {
      @set_tile_code(x, y, "pe.csl", .{});
    }
  }
}'''
    for arm in hw.ARMS:
        if arm == 'direct':
            source = {'layout.csl': direct_layout, 'pe.csl': b'direct PE'}
        else:
            layout = 'layout { @set_rectangle(2, 2);\n'
            source = {}
            for x, y in sorted(hw.COORDS):
                name = f'pe_{x}_{y}.csl'
                source[name] = f'{arm} PE{x}{y}'.encode()
                layout += f'@set_tile_code({x}, {y}, "{name}", .{{}});\n'
            source['layout.csl'] = (layout+'}').encode()
        sources = {name: put(f'generated/{arm}/{name}', value) for name, value in source.items()}
        original = source if arm == 'direct' else {'layout.csl': direct_layout, 'pe.csl': arm.encode()}
        original_hashes = {name: put(f'originals/{arm}/{name}', value) for name, value in original.items()}
        references = {f'out_{i}_0.elf': put(f'references/{arm}/build/bin/out_{i}_0.elf', elf_fixture())
                      for i in range(4)}
        provenance = dict(arm=arm, status='PASS', backend='simulator', params=hw.PARAMS,
                          case_sha256=case_sha, source=original_hashes,
                          generated_source={} if arm == 'direct' else sources,
                          inventory=dict(files={f'bin/{name}': dict(sha256=value) for name, value in references.items()}))
        put(f'references/{arm}/compile.json', provenance)
        arms[arm] = dict(sources=sources, layout=hw.layout_mapping(source['layout.csl'], arm),
                         reference_files=references)
    package = dict(schema=1, family=hw.FAMILY, files=files, params=hw.PARAMS,
                   arms=arms, case_sha256=case_sha)
    hw.write(root/'package.json', package)
    return package


def program_payloads(root, arm):
    result = {f'artifact/csl/{name}': path.read_bytes() for path in (root/'generated'/arm).iterdir()
              for name in (path.name,)}
    result.update({'artifact/out/out.json': b'{}', 'artifact/out/west/out.json': b'{}',
                   'artifact/out/east/out.json': b'{}', 'artifact/out/bin/out_rpc.json': b'{}'})
    result.update({f'artifact/out/bin/out_{i}_0.elf': elf_fixture() for i in range(4)})
    return result


def make_archive(path, payloads, extra_member=None):
    with tarfile.open(path, 'w:gz') as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if extra_member:
            archive.addfile(extra_member)


class SourceIdentityTests(unittest.TestCase):
    def test_all_arms_and_source_mutations(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(hw.common, 'load',
                return_value=dict(kind='tiny', params={k: v for k, v in hw.PARAMS.items() if k != 'PROFILE'})):
            root = Path(temp)
            package_fixture(root)
            for arm in hw.ARMS:
                self.assertEqual(hw.identity(root, arm)['arm'], arm)
            path = root/'generated/async/pe_0_0.csl'
            payload = path.read_bytes()
            for change in ('changed', 'missing', 'swapped', 'extra'):
                with self.subTest(change=change):
                    if change == 'changed': path.write_bytes(payload+b'x')
                    elif change == 'missing': path.unlink()
                    elif change == 'swapped': path.write_bytes((root/'generated/async/pe_0_1.csl').read_bytes())
                    else: (root/'generated/async/extra.csl').write_bytes(b'extra')
                    with self.assertRaises(ValueError): hw.identity(root, 'async')
                    if change == 'extra': (root/'generated/async/extra.csl').unlink()
                    else: path.write_bytes(payload)

    def test_rehashed_source_still_requires_original_provenance(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(hw.common, 'load',
                return_value=dict(kind='tiny', params={k: v for k, v in hw.PARAMS.items() if k != 'PROFILE'})):
            root = Path(temp)
            package = package_fixture(root)
            name = 'generated/async/pe_0_0.csl'
            (root/name).write_bytes(b'changed')
            package['files'][name] = digest(b'changed')
            package['arms']['async']['sources']['pe_0_0.csl'] = digest(b'changed')
            hw.write(root/'package.json', package)
            with self.assertRaisesRegex(ValueError, 'established local reference'):
                hw.identity(root, 'async')

    def test_full_cora_parameters_and_layout_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = package_fixture(root)
            package['params'] = dict(hw.PARAMS, C=16, R=32)
            hw.write(root/'package.json', package)
            with self.assertRaisesRegex(ValueError, 'full Cora'):
                hw.identity(root, 'async')
        text = b'layout { @set_rectangle(16, 32); }'
        with self.assertRaisesRegex(ValueError, 'tiny rectangle'):
            hw.layout_mapping(text, 'async')


class ArchiveTests(unittest.TestCase):
    def test_complete_archive_and_source_failures(self):
        for change in ('none', 'missing', 'extra', 'changed', 'nested', 'second-root', 'outside'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                package = package_fixture(root)
                payloads = program_payloads(root, 'async')
                if change == 'missing': del payloads['artifact/csl/pe_0_0.csl']
                elif change == 'extra': payloads['artifact/csl/extra.csl'] = b'extra'
                elif change == 'changed': payloads['artifact/csl/pe_0_0.csl'] += b'changed'
                elif change == 'nested': payloads['artifact/csl/nested/pe_0_0.csl'] = b'nested'
                elif change == 'second-root': payloads['another/csl/pe_0_0.csl'] = payloads['artifact/csl/pe_0_0.csl']
                elif change == 'outside': payloads['artifact/extra.csl'] = b'extra'
                make_archive(root/'program.tgz', payloads)
                if change == 'none':
                    inventory, elfs = hw.program_archive(root/'program.tgz', package['arms']['async']['sources'])
                    self.assertEqual(len(elfs), 4)
                    self.assertEqual(inventory['elf']['max_static_sram_end'], 16)
                else:
                    with self.assertRaises(ValueError):
                        hw.program_archive(root/'program.tgz', package['arms']['async']['sources'])

    def test_archive_traversal_links_duplicates_and_helper_overflow(self):
        for change in ('traversal', 'link', 'duplicate', 'helper-overflow'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                package = package_fixture(root)
                payloads = program_payloads(root, 'direct')
                info = None
                if change == 'traversal': payloads['../escape'] = b'x'
                elif change == 'link':
                    info = tarfile.TarInfo('artifact/link')
                    info.type, info.linkname = tarfile.SYMTYPE, '/etc/passwd'
                elif change == 'duplicate': info = tarfile.TarInfo('artifact/csl/layout.csl')
                else: payloads['artifact/out/west/bin/helper.elf'] = elf_fixture(44*1024)
                make_archive(root/'program.tgz', payloads, info)
                with self.assertRaises(ValueError):
                    hw.program_archive(root/'program.tgz', package['arms']['direct']['sources'])


class CompileTests(unittest.TestCase):
    def exercise(self, failure=None):
        with tempfile.TemporaryDirectory() as temp, patch.object(hw.common, 'load',
                return_value=dict(kind='tiny', params={k: v for k, v in hw.PARAMS.items() if k != 'PROFILE'})):
            root = Path(temp)
            package_fixture(root)
            calls = []

            class Compiler:
                def __init__(self, **options): self.options = options
                def __enter__(self): return self
                def __exit__(self, *_): pass
                def compile(self, directory, source, flags, output):
                    calls.append((directory, source, flags))
                    path = root/'escape.tgz' if failure == 'escape' else Path(output)/'program.tgz'
                    make_archive(path, program_payloads(root, 'async'))
                    return str(path)

            args = SimpleNamespace(arm='async', namespace='job-operator', out=root/'compile')
            gate = dict(status='PASS', application_programs={}, scope='mock equivalence',
                        coroutine_capacity_status='NOT_ESTABLISHED', admission_status='HOLD',
                        admission_reason='backend-generated call closure unverified')
            with patch.object(hw, 'verify_stack', side_effect=ValueError('target code changed') if failure == 'stack' else None,
                              return_value=gate):
                if failure:
                    with self.assertRaises(ValueError): hw.compile_job(args, root, Compiler)
                    record = hw.read(root/'compile/build.json')
                    self.assertEqual(record['status'], 'FAILED')
                    self.assertEqual(record['compiler_status'], 'PASS')
                    if failure == 'stack':
                        self.assertTrue((root/'compile/program.tgz').is_file())
                        self.assertEqual(record['stack_status'], 'FAILED')
                else:
                    record = hw.compile_job(args, root, Compiler)
                    self.assertEqual(record['status'], 'EVIDENCE_COMPLETE')
                    self.assertEqual(record['admission_status'], 'HOLD')
                    result = hw.audit_build(root, 'async', root/'compile/build.json')
                    self.assertEqual(result['status'], 'EVIDENCE_COMPLETE')
                    self.assertEqual(result['admission_status'], 'HOLD')
                    forged = dict(record, admission_status='PASS')
                    hw.write(root/'compile/build.json', forged)
                    with self.assertRaisesRegex(ValueError, 'replayed target stack admission'):
                        hw.audit_build(root, 'async', root/'compile/build.json')
                    hw.write(root/'compile/build.json', record)
                    payloads = program_payloads(root, 'async')
                    payloads['artifact/out/bin/out_0_0.elf'] += b'changed'
                    make_archive(root/'compile/program.tgz', payloads)
                    with self.assertRaisesRegex(ValueError, 'archive identity'):
                        hw.audit_build(root, 'async', root/'compile/build.json')
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], 'layout.csl')
            self.assertEqual(calls[0][2], hw.common.compiler_arguments(hw.PARAMS, 'hardware', 1))

    def test_one_compile_and_offline_replay_reject_target_mutation(self): self.exercise()
    def test_stack_failure_retains_successful_compiler_artifact(self): self.exercise('stack')
    def test_compiler_artifact_cannot_escape(self): self.exercise('escape')


class PlacementTests(unittest.TestCase):
    def test_coordinate_mapping_uses_fabric_not_elf_names(self):
        payloads = {f'bin/arbitrary{i}.elf': str((y+1)*762+x+4).encode()
                    for i, (x, y) in enumerate(sorted(hw.COORDS))}
        with patch.object(hw.stack_gate, 'inspect', side_effect=lambda data: dict(tile_indices=[int(data)])):
            self.assertEqual(set(hw.application_map(payloads, 762)), hw.COORDS)
            with self.assertRaises(ValueError): hw.application_map(payloads, 9)
            payloads['bin/arbitrary0.elf'] = payloads['bin/arbitrary1.elf']
            with self.assertRaisesRegex(ValueError, 'unique'):
                hw.application_map(payloads, 762)


if __name__ == '__main__':
    unittest.main()
