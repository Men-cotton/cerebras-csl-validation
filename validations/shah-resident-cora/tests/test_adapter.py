"""Offline contract tests; SDK classes are never imported."""
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import uuid
import zipfile

import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAMILY))
import archive
import appliance
import saved_audit


def elf(address=256):
    ident = b'\x7fELF' + bytes([1, 1, 1]) + bytes(9)
    header = struct.pack('<HHIIIIIHHHHHH', 2, 0, 1, 0, 0, 52, 0, 52, 0, 0, 40, 1, 0)
    section = struct.pack('<IIIIIIIIII', 0, 1, 2, address, 92, 4, 0, 0, 4, 0)
    return ident + header + section + bytes(4)


def make_tar(path, entries):
    with tarfile.open(path, 'w:gz') as out:
        for name, data in entries:
            item = tarfile.TarInfo(name)
            item.size = len(data)
            out.addfile(item, io.BytesIO(data))


class AdapterTests(unittest.TestCase):
    def test_audit_diagnostic_roundoff_is_accepted(self):
        remote = dict(status='PASS', schema=1, observations={
            'gcn0': dict(max_abs_errors=dict(projected1=1.2974250299113699e-8,
                                            logits=1.4224621047026487e-8)),
            'ah160': dict(max_abs_error=1e-8)})
        fresh = copy.deepcopy(remote)
        fresh['observations']['gcn0']['max_abs_errors'].update(
            projected1=1.2974250271358123e-8, logits=1.4224621005393123e-8)
        fresh['observations']['ah160']['max_abs_error'] += 4e-17
        archive.compare_audits(fresh, remote)
        for original, replay in ((0., 1e-12), (1., 1. + 5e-9)):
            archive.compare_audits(dict(status='PASS', max_abs_error=original),
                                   dict(status='PASS', max_abs_error=replay))

    def test_audit_nondiagnostic_fields_and_schema_require_exact_equality(self):
        remote = dict(status='PASS', schema=1, run_id='run', case_sha256='case',
                      raw_sha256='tensor', backend='hardware', observations={
                          'ah160': dict(max_abs_error=1e-8, max_pe_cycles=300,
                                        stats_unique=[[0x600D, 1]], projected_sha256='input', seconds=1.)})
        mutations = [lambda x: x.update(status='FAIL'), lambda x: x.update(schema=2),
                     lambda x: x.update(run_id='other'), lambda x: x.update(case_sha256='other'),
                     lambda x: x.update(raw_sha256='other'), lambda x: x.update(backend='simulator'),
                     lambda x: x.update(extra=1), lambda x: x.pop('schema'),
                     lambda x: x.update(schema=1.), lambda x: x.update(schema=True),
                     lambda x: x['observations']['ah160'].update(max_pe_cycles=301),
                     lambda x: x['observations']['ah160']['stats_unique'][0].append(0),
                     lambda x: x['observations']['ah160']['stats_unique'][0].__setitem__(1, 2),
                     lambda x: x['observations']['ah160'].update(projected_sha256='other'),
                     lambda x: x['observations']['ah160'].update(seconds=1. + 1e-13)]
        archive.compare_audits(copy.deepcopy(remote), remote)
        for index, mutate in enumerate(mutations):
            fresh = copy.deepcopy(remote)
            mutate(fresh)
            with self.subTest(index=index), self.assertRaises(ValueError):
                archive.compare_audits(fresh, remote)

    def test_audit_material_error_changes_and_invalid_diagnostics_rejected(self):
        for field in ('max_abs_error', 'max_abs_errors'):
            def report(value):
                return dict(status='PASS', **{field: {'pre': value} if field.endswith('s') else value})
            for original, replay in ((0., 1.01e-12), (1., 1. + 3e-8), (1e-8, 1e-7),
                                     (0., 0), (0., True), (-1., -1.),
                                     (float('nan'), float('nan')), (float('inf'), float('inf')),
                                     (float('-inf'), float('-inf')), ('same', 'same'), ({}, {})):
                with self.subTest(field=field, original=original, replay=replay), self.assertRaises(ValueError):
                    archive.compare_audits(report(original), report(replay))
        with self.assertRaises(ValueError):
            archive.compare_audits(dict(status='PASS', max_abs_errors={'pre': 0.}),
                                   dict(status='PASS', max_abs_errors={'logits': 0.}))
        with self.assertRaises(ValueError):
            archive.compare_audits(dict(status='PASS', max_abs_errors=[]),
                                   dict(status='PASS', max_abs_errors=[]))
        with self.assertRaises(ValueError):
            archive.compare_audits(dict(status='PASS', seconds=float('nan')),
                                   dict(status='PASS', seconds=float('nan')))

    def test_compiler_archive_and_runtime_tree_have_identical_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = {'out/bin/out.elf': elf(), 'out/out.json': b'{}',
                        'out/west/out.json': b'{"params":{}}', 'out/east/out.json': b'{"params":{}}',
                        'out/out_rpc.json': b'{}', 'csl/layout.csl': b'layout', 'csl/pe.csl': b'pe'}
            artifact = root / 'compile.tar.gz'
            make_tar(artifact, [('wrapper/' + k, v) for k, v in payloads.items()])
            pins = {n: hashlib.sha256(payloads['csl/' + n]).hexdigest() for n in ('layout.csl', 'pe.csl')}
            expected = archive.program_archive(artifact, pins)
            tree = archive.unpack(artifact, root / 'extracted') / 'wrapper/out'
            self.assertEqual(expected, archive.program_tree(tree))
            (tree / 'west/out.json').write_text('{"changed":true}')
            self.assertNotEqual(expected, archive.program_tree(tree))

    def test_compiled_source_and_sram_mutations_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mutation in ('source', 'storage'):
                artifact = root / (mutation + '.tar.gz')
                entries = [('out/bin/out.elf', elf(44 * 1024 if mutation == 'storage' else 256)),
                           ('out/out.json', b'{}'), ('out/west/out.json', b'{}'), ('out/east/out.json', b'{}'),
                           ('csl/layout.csl', b'changed' if mutation == 'source' else b'layout'), ('csl/pe.csl', b'pe')]
                make_tar(artifact, entries)
                pins = {n: hashlib.sha256(v).hexdigest() for n, v in [('layout.csl', b'layout'), ('pe.csl', b'pe')]}
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    archive.program_archive(artifact, pins)

    def test_unsafe_duplicate_and_oversize_archive_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, entries in enumerate(([('../escape', b'x')], [('same', b'a'), ('same', b'b')], [('big', bytes(65))])):
                path = root / f'{index}.tar'
                make_tar(path, entries)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    archive.unpack(path, root / f'out-{index}', max_bytes=64 if index == 2 else archive.MAX_BYTES)
            path = root / 'link.tar'
            with tarfile.open(path, 'w') as out:
                entry = tarfile.TarInfo('link'); entry.type = tarfile.SYMTYPE; entry.linkname = '/etc/passwd'; out.addfile(entry)
            with self.assertRaises(ValueError):
                archive.unpack(path, root / 'out-link')
            self.assertFalse((root / 'escape').exists())

    def test_run_binding_rejects_wrong_uuid_program_mode_and_case(self):
        run_id = str(uuid.uuid4())
        record = dict(backend='hardware', case=dict(case_sha256='a' * 64), program_inventory=dict(elf={'files': {}}, metadata_sha256={}))
        request = dict(run_id=run_id, mode='all', compile_record=record)
        collection = dict(run_id=run_id)
        status = dict(run_id=run_id, mode='all', compile_record=record, status='PASS', program_inventory=record['program_inventory'])
        execution = dict(run_id=run_id, mode='all', status='CAPTURED', compiled_inventory=record['program_inventory']['elf'],
                         backend='hardware', case_sha256='a' * 64)
        saved_audit.bind(request, collection, status, execution, record)
        for key, value in [('run_id', str(uuid.uuid4())), ('mode', 'gcn'), ('compiled_inventory', {}), ('case_sha256', 'b' * 64)]:
            wrong = {**execution, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                saved_audit.bind(request, collection, status, wrong, record)

    def test_worker_command_and_geometry_are_bounded(self):
        command = appliance.host_command('hardware', 86400)
        self.assertEqual(command.count('cs_python'), 1)
        self.assertIn('--cmaddr %CMADDR%', command)
        self.assertNotIn('--cmaddr', appliance.host_command('simulator', 30))
        for value in (0, 86401, True):
            with self.assertRaises(ValueError):
                appliance.host_command('hardware', value)
        class Driver:
            @staticmethod
            def compiler_flags(params, hardware=False, channels=1):
                return ['--params=' + str(params['rows']), '--channels=' + str(channels), '-o', 'out']
        with patch.object(appliance, 'frozen_driver', return_value=Driver):
            self.assertIn('--params=64', appliance.flags({'rows': 64}, 'hardware', 1))
            with self.assertRaises(ValueError):
                appliance.flags({'rows': 64}, 'hardware', 8)

    def test_npz_header_bound_before_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / 'valid.npz'
            np.savez(valid, array=np.arange(8, dtype=np.float32))
            archive.check_npz(valid)
            malformed = root / 'bad.npz'
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(header, dict(descr='<f4', fortran_order=False, shape=(2**30,)))
            with zipfile.ZipFile(malformed, 'w') as out:
                out.writestr('array.npy', header.getvalue() + bytes(4))
            with self.assertRaises(ValueError):
                archive.check_npz(malformed)
            objects = root / 'objects.npz'
            np.savez(objects, array=np.array([None], dtype=object))
            with self.assertRaises(ValueError):
                archive.check_npz(objects)

    def test_registered_variant_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for family in ('shah-resident-cora', 'shah-csr-cora'):
                (root / 'validation.json').write_text(json.dumps(dict(id=family)))
                self.assertEqual(appliance.family_id(root), family)
            (root / 'validation.json').write_text(json.dumps(dict(id='unrelated-family')))
            with self.assertRaises(ValueError):
                appliance.family_id(root)

    def test_download_path_collision_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requested = root / 'host.log'
            requested.write_text('log')
            self.assertEqual(appliance.downloaded_path(None, requested, 128), requested)
            (root / 'request.json').write_text('{}')
            with self.assertRaises(ValueError):
                appliance.downloaded_path(str(root / 'request.json'), requested, 128)
            with self.assertRaises(ValueError):
                appliance.downloaded_path('../outside', requested, 128)


if __name__ == '__main__':
    unittest.main()
