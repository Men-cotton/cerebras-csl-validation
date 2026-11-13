"""Offline compiler/launcher fixtures: no SDK client or appliance connection."""
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import struct
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ADAPTER = HERE/'host_device_ping_pong_appliance.py'
if not ADAPTER.exists():
    ADAPTER = HERE.parent/'host_device_ping_pong_appliance.py'
spec = importlib.util.spec_from_file_location('ping_pong_test_adapter', ADAPTER)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def tar_payloads(path, payloads):
    with tarfile.open(path, 'w:gz') as archive:
        for name, payload in payloads.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def make_package(root):
    root.mkdir()
    shutil.copyfile(ADAPTER, root/app.ENTRY)
    (root/'support').mkdir()
    if (ADAPTER.parent/'support').is_dir():
        origins = {name: ADAPTER.parent/'support'/name for name in app.SUPPORT}
    else:
        native = ADAPTER.parent.parent/'models/hierarchical_gnn/grouped_csr/native_cora_appliance'
        origins = {name: native/('observation/hardware_entry.py' if name == 'hardware_entry.py' else name)
                   for name in app.SUPPORT}
    for name, source in origins.items():
        shutil.copyfile(source, root/'support'/name)
    (root/'source').mkdir()
    for name in app.SOURCES:
        (root/'source'/name).write_text('# synthetic '+name+'\n')
    files = {str(path.relative_to(root)): app.sha(path) for path in root.rglob('*') if path.is_file()}
    write(root/'package.json', dict(schema=1, family=app.FAMILY, files=files, cases=app.CASES))
    tar_payloads(root/'package.tar.gz', {name: (root/name).read_bytes() for name in [*files, 'package.json']})
    return root


def program_payloads(root, case):
    ident = b'\x7fELF'+bytes([1, 1, 1])+bytes(9)
    header = struct.pack('<HHIIIIIHHHHHH', 2, 0, 1, 0, 0, 52, 0, 52, 0, 0, 40, 1, 0)
    section = struct.pack('<IIIIIIIIII', 0, 8, 2, 0, 0, 16, 0, 0, 4, 0)
    return {'artifact/out/out.json': json.dumps(dict(params=app.CASES[case])).encode(),
            'artifact/out/west/out.json': b'{}', 'artifact/out/east/out.json': b'{}',
            'artifact/out/bin/out_rpc.json': b'{"rpc_symbols":[]}',
            'artifact/out/bin/pe.elf': ident+header+section,
            **{'artifact/csl/'+name: (root/'source'/name).read_bytes()
               for name in ('layout.csl', 'pe.csl')}}


def compile_fixture(root, output, case='zero-p1', changed_source=False):
    calls = []
    class Compiler:
        def __init__(self, **options):
            calls.append(('options', options))
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def compile(self, directory, source, flags, destination):
            calls.append(('compile', directory, source, flags))
            payloads = program_payloads(root, case)
            if changed_source:
                payloads['artifact/csl/pe.csl'] += b' changed'
            path = Path(destination)/'actual-compiler-result.tar.gz'
            tar_payloads(path, payloads)
            return str(path)
    args = SimpleNamespace(case=case, namespace='job-operator', out=output)
    record = app.compile_job(args, root=root, compiler_type=Compiler)
    return record, calls


class PingPongApplianceTests(unittest.TestCase):
    def test_fixed_cases_orders_and_compiler_flags(self):
        grids = ((1, 1), (4, 2), (16, 4), (64, 8), (256, 16), (1024, 32), (4096, 64))
        expected = {f'{mode}-p{pes}': {'WIDTH': edge, 'HEIGHT': edge, 'CAPACITY': 4097}
                    for mode in ('zero', 'powers') for pes, edge in grids}
        expected.update({f'powers-p4096-c{channels}': {'WIDTH': 64, 'HEIGHT': 64, 'CAPACITY': 4097}
                         for channels in (2, 4, 8, 16)})
        expected = {case: dict(params, MEMCPYH2D_DATA_1_ID=0, MEMCPYD2H_DATA_1_ID=1)
                    for case, params in expected.items()}
        expected_channels = {case: 1 for case in expected}
        expected_channels.update({'powers-p4096-c2': 2, 'powers-p4096-c4': 4,
                                  'powers-p4096-c8': 8, 'powers-p4096-c16': 16})
        self.assertEqual(app.CASES, expected)
        self.assertEqual(app.CHANNELS, expected_channels)
        for case, params in expected.items():
            for order in range(3):
                self.assertEqual(app.controls(case, order), {'order': order})
            flags = app.compiler_arguments(case)
            self.assertIn(f'--channels={expected_channels[case]} ', flags)
            self.assertIn('--fabric-dims=762,1172 ', flags)
            self.assertIn(f'CAPACITY:4097,HEIGHT:{params["HEIGHT"]},MEMCPYD2H_DATA_1_ID:1,'
                          f'MEMCPYH2D_DATA_1_ID:0,WIDTH:{params["WIDTH"]} ', flags)
        for case, order in (('other', 0), ('zero-p1', -1), ('powers-p64', 3), ('zero-p1', True), ('zero-p1', 1.0)):
            with self.subTest(case=case, order=order), self.assertRaises(ValueError):
                app.controls(case, order)
        for old_case in ('zero', 'powers'):
            with self.assertRaises(ValueError):
                app.controls(old_case, 0)

    def test_run_request_rejects_wrong_schedule_and_identity(self):
        good = dict(schema=1, run_id=str(uuid.uuid4()), case='zero-p1', order=0, build_record_sha256='a'*64)
        app.check_request(good, 'zero-p1', 0)
        for changes in ({'case': 'powers-p64'}, {'order': 1}, {'order': True}, {'schema': True},
                        {'run_id': 'not-a-uuid'}, {'build_record_sha256': '0'}, {'extra': 0}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                app.check_request(dict(good, **changes), 'zero-p1', 0)

    def test_compiler_returned_path_and_actual_program_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = make_package(base/'family')
            record, calls = compile_fixture(root, base/'compile', 'powers-p64')
            self.assertEqual(record['artifact_path'], 'actual-compiler-result.tar.gz')
            self.assertEqual(len([row for row in calls if row[0] == 'compile']), 1)
            self.assertEqual(calls[0][1]['resource_cpu'], 1000)
            loaded, artifact = app.load_build(root, base/'compile/build.json', 'powers-p64')
            helper = app.support(root)[2]
            helper.unpack(artifact, base/'unpacked', app.ARTIFACT_LIMIT)
            build = base/'unpacked/artifact/out'
            self.assertEqual(app.actual_program(root, build, loaded['program_inventory']), record['program_inventory'])
            for name in ('bin/pe.elf', 'east/out.json'):
                path = build/name
                original = path.read_bytes()
                path.write_bytes(original+b' ')
                with self.assertRaisesRegex(ValueError, 'unpacked program differs'):
                    app.actual_program(root, build, record['program_inventory'])
                path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, 'source/case/compile binding'):
                app.load_build(root, base/'compile/build.json', 'zero-p1')

    def test_wrong_compiled_source_latches_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = make_package(base/'family')
            with self.assertRaisesRegex(ValueError, 'compiled CSL differs'):
                compile_fixture(root, base/'compile', changed_source=True)
            self.assertEqual(app.read(base/'compile/build.json')['status'], 'FAILED')

    def test_package_mutation_missing_source_and_archive_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = make_package(base/'family')
            app.identity(root, 'zero-p1')
            manifest = app.read(root/'package.json')
            changed = dict(manifest, files={key: value for key, value in manifest['files'].items()
                                            if key != 'source/oracle.py'})
            write(root/'package.json', changed)
            with self.assertRaisesRegex(ValueError, 'complete adapter dependencies'):
                app.identity(root, 'zero-p1')
            write(root/'package.json', manifest)
            (root/'source/pe.csl').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'frozen package source'):
                app.identity(root, 'zero-p1')
            tar_payloads(base/'bad.tar.gz', {'../escape': b'bad'})
            with self.assertRaises(ValueError):
                app.support(root)[2].unpack(base/'bad.tar.gz', base/'unpack', app.CAPTURE_LIMIT)
            self.assertFalse((base/'escape').exists())

    def run_launcher(self, failure=None):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = make_package(base/'family')
            compile_fixture(root, base/'compile')
            staged, commands, downloads = [], [], []
            class Launcher:
                def __init__(self, artifact, **options):
                    self.artifact, self.options = artifact, options
                def __enter__(self): return self
                def __exit__(self, *_): pass
                def stage(self, path): staged.append(Path(path).name)
                def run(self, command):
                    commands.append(command)
                    if failure == 'launch': raise RuntimeError('mock launch failure')
                    return 'mock job identifier'
                def download_artifact(self, name, requested):
                    downloads.append(name)
                    path = Path(requested)
                    if name == 'capture':
                        path = path.with_suffix('.tar.gz')
                        tar_payloads(path, {'capture/worker.json': b'{"partial":true}'})
                    else:
                        path.write_text('1\n' if name == 'host-exit.txt' and failure == 'worker' else
                                        '0\n' if name == 'host-exit.txt' else 'partial SDK log')
                    return str(path)
            args = SimpleNamespace(case='zero-p1', order=0, host_timeout=600,
                                   build_record=base/'compile/build.json', out=base/'run')
            with patch.object(app, 'audit_capture', return_value={'status': 'PASS'}) as audit:
                if failure:
                    with self.assertRaises((ValueError, RuntimeError)):
                        app.run_job(args, root=root, launcher_type=Launcher)
                    self.assertEqual(app.read(base/'run/collection.json')['status'], 'FAILED')
                    self.assertTrue((base/'run/capture.tar.gz').is_file())
                    audit.assert_not_called()
                else:
                    result = app.run_job(args, root=root, launcher_type=Launcher)
                    self.assertEqual(result['status'], 'PASS')
                    self.assertFalse((base/'run/capture.tar.gz').exists())
                    self.assertTrue((base/'run/capture-data/worker.json').is_file())
                    audit.assert_called_once()
            self.assertEqual(downloads, ['host.log', 'host-exit.txt', 'capture'])
            self.assertEqual(len(commands), 1)
            self.assertIn('600s', commands[0])
            self.assertIn('%CMADDR%', commands[0])
            self.assertEqual(set(staged), {app.ENTRY, 'hardware_entry.py', 'package.tar.gz', 'build.json', 'run-request.json'})

    def test_one_launcher_action_collects_and_replays(self):
        self.run_launcher()

    def test_launcher_failure_keeps_partial_capture(self):
        self.run_launcher('launch')

    def test_worker_failure_keeps_partial_capture(self):
        self.run_launcher('worker')

    def test_worker_scratch_capture_and_outer_identity(self):
        # Exercise the real outer audit with a fake numerical driver. The driver
        # oracle and timing protocol are independently tested by the family.
        for case in ('zero-p1', 'powers-p64'):
            for failure in (False, True):
                with self.subTest(case=case, failure=failure), tempfile.TemporaryDirectory() as temporary:
                    base = Path(temporary)
                    root = make_package(base/'family')
                    compiled, _ = compile_fixture(root, base/'compile', case)
                    staged = base/'staged'
                    staged.mkdir()
                    for source, name in ((root/app.ENTRY, app.ENTRY), (root/'support/hardware_entry.py', 'hardware_entry.py'),
                                         (root/'package.tar.gz', 'package.tar.gz'), (base/'compile/build.json', 'build.json')):
                        shutil.copyfile(source, staged/name)
                    helper = app.support(root)[2]
                    helper.unpack(base/'compile'/compiled['artifact_path'], staged/'unpacked', app.ARTIFACT_LIMIT)
                    build = staged/'unpacked/artifact/out'
                    run_id, order = str(uuid.uuid4()), 2
                    write(staged/'run-request.json', dict(schema=1, run_id=run_id, case=case, order=order,
                                                       build_record_sha256=app.sha(staged/'build.json')))
                    original_driver = app.driver
                    observed_cwds = []
                    def fake_driver(frozen_root, label):
                        folder, implementation = original_driver(frozen_root, label)
                        def capture(source, actual_build, runtime, capture_case, capture_order,
                                    *, backend, cmaddr, sdk_workdir, smoke, run_id):
                            self.assertEqual((source, actual_build, capture_case, capture_order), (folder, build, case, order))
                            self.assertEqual((backend, cmaddr, smoke), ('hardware', 'fixture', False))
                            self.assertEqual(run_id, app.read(staged/'run-request.json')['run_id'])
                            runtime = Path(runtime)
                            runtime.mkdir()
                            write(runtime/'partial.json', {'attempted': True})
                            previous = Path.cwd()
                            try:
                                os.chdir(sdk_workdir)
                                observed_cwds.append(Path.cwd())
                                with Path('sdk-config.conf').open('xb') as stream: stream.truncate(50 << 20)
                                if failure: raise RuntimeError('fake runtime failed after stop attempt')
                            finally:
                                os.chdir(previous)
                            inventory = compiled['program_inventory']
                            expected = {name: row['sha256'] for name, row in inventory['elf']['files'].items()}
                            expected.update(inventory['metadata_sha256'])
                            write(runtime/'program.json', {name: {'sha256': digest} for name, digest in expected.items()})
                            (runtime/'source').mkdir()
                            for name in app.SOURCES:
                                shutil.copyfile(source/name, runtime/'source'/name)
                            write(runtime/'capture.json', dict(source_sha256={name: app.sha(source/name) for name in app.SOURCES}))
                        implementation.capture = capture
                        implementation.audit = lambda runtime: dict(status='PASS', backend='hardware', case=case,
                                                                     order=order, capture_id=run_id)
                        return folder, implementation
                    args = SimpleNamespace(cmaddr='fixture', package=staged/'package.tar.gz', build_record=staged/'build.json',
                                           request=staged/'run-request.json', build=build, out=staged/'capture')
                    previous = Path.cwd()
                    with patch.object(app, '__file__', str(staged/app.ENTRY)), patch.object(app, 'driver', side_effect=fake_driver):
                        if failure:
                            with self.assertRaisesRegex(RuntimeError, 'fake runtime failed'):
                                app.worker(args)
                        else:
                            self.assertEqual(app.worker(args)['status'], 'PASS')
                            frozen_root = staged/('frozen-ping-pong-'+run_id)
                            self.assertEqual(app.audit_capture(frozen_root, case, order, args.out)['status'], 'PASS')
                            with patch.object(app, 'runtime_binding', return_value={'capture_id': str(uuid.uuid4())}):
                                with self.assertRaisesRegex(ValueError, 'runtime capture belongs to requested run'):
                                    app.audit_capture(frozen_root, case, order, args.out)
                            with self.assertRaisesRegex(ValueError, 'requested runtime arm'):
                                app.audit_capture(frozen_root, case, 0, args.out)
                            copied = args.out/'runtime/source/oracle.py'
                            copied.write_text('changed')
                            with self.assertRaisesRegex(ValueError, 'runtime frozen source bytes'):
                                app.audit_capture(frozen_root, case, order, args.out)
                            shutil.copyfile(root/'source/oracle.py', copied)
                            worker = app.read(args.out/'worker.json')
                            write(args.out/'worker.json', dict(worker, run_id=str(uuid.uuid4())))
                            with self.assertRaisesRegex(ValueError, 'returned run/compile bindings'):
                                app.audit_capture(frozen_root, case, order, args.out)
                            write(args.out/'worker.json', worker)
                            with (args.out/'unexpected-sdk-file').open('xb') as stream: stream.truncate(app.CAPTURE_LIMIT+1)
                            with self.assertRaisesRegex(ValueError, 'tree byte bound'):
                                app.audit_capture(frozen_root, case, order, args.out)
                    self.assertEqual(Path.cwd(), previous)
                    scratch = staged/('sdk-work-'+run_id)
                    self.assertEqual(observed_cwds, [scratch])
                    self.assertFalse(scratch.exists())
                    record = app.read(args.out/'worker.json')
                    self.assertEqual(record['status'], 'FAILED' if failure else 'PASS')
                    self.assertEqual(record['sdk_workdir']['bytes'], 50 << 20)
                    self.assertTrue(record['sdk_workdir']['removed'])
                    self.assertTrue((args.out/'runtime/partial.json').is_file())

    def test_sdk_scratch_rejects_alias_existing_path_and_accounts_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run_id = str(uuid.uuid4())
            capture = base/'capture'
            capture.mkdir()
            work = app.fresh_sdk_workdir(capture, run_id)
            with self.assertRaises(FileExistsError): app.fresh_sdk_workdir(capture, run_id)
            with self.assertRaises(ValueError): app.fresh_sdk_workdir(work, run_id)
            with (work/'overflow').open('xb') as stream: stream.truncate(app.SDK_WORK_LIMIT+1)
            result = app.remove_sdk_workdir(work)
            self.assertFalse(result['removed'])
            self.assertIn('tree byte bound', result['error'])
            self.assertTrue(work.exists())
            (work/'overflow').unlink()
            work.rmdir()
            work.symlink_to(capture, target_is_directory=True)
            with self.assertRaises(ValueError): app.fresh_sdk_workdir(capture, run_id)


if __name__ == '__main__':
    unittest.main()
