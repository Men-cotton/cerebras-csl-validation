"""Local compiler/launcher fixtures only; never import an SDK client."""
import hashlib
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
ADAPTER = HERE/'csl_observability_appliance.py'
if not ADAPTER.exists():
    ADAPTER = HERE.parent/'csl_observability_appliance.py'
spec = importlib.util.spec_from_file_location('observability_test_adapter', ADAPTER)
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
    for case, params in app.CASES.items():
        folder = root/'cases'/case
        (folder/'source').mkdir(parents=True)
        for name in ('layout.csl', 'pe.csl'):
            (folder/'source'/name).write_text('synthetic '+case+' '+name)
        # Framework fixture only: real family oracles have their own tests.
        shim = folder/('source/driver.py' if case == 'arena-cap64' else 'run.py')
        shim.write_text('import json\ndef read_case(folder):\n    return json.loads((folder/"case.json").read_text()), None\n')
        write(folder/'case.json', dict(params=params))
    files = {str(path.relative_to(root)): app.sha(path) for path in root.rglob('*') if path.is_file()}
    write(root/'package.json', dict(schema=1, family=app.FAMILY, files=files,
          cases={case: app.sha(root/'cases'/case/'case.json') for case in app.CASES}))
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
            **{'artifact/csl/'+name: (root/'cases'/case/'source'/name).read_bytes()
               for name in ('layout.csl', 'pe.csl')}}


def compile_fixture(root, output, case='queue-n32', changed_source=False):
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


class ObservabilityApplianceTests(unittest.TestCase):
    def test_fixed_runtime_arms(self):
        for profile, delay in ((1, 0), (1, 256), (0, 256)):
            app.controls('queue-n32', profile, delay)
        app.controls('arena-cap64', 0, 0)
        for case, profile, delay in (('queue-n32', 0, 0), ('queue-n32', 1, 1), ('arena-cap64', 1, 0),
                                     ('other', 0, 0), ('queue-n32', True, 0)):
            with self.subTest(case=case, profile=profile, delay=delay), self.assertRaises(ValueError):
                app.controls(case, profile, delay)

    def test_compiler_returned_path_source_and_actual_program_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = make_package(base/'family')
            record, calls = compile_fixture(root, base/'compile')
            self.assertEqual(record['artifact_path'], 'actual-compiler-result.tar.gz')
            self.assertEqual(len([row for row in calls if row[0] == 'compile']), 1)
            self.assertIn('--fabric-dims=762,1172', record['compiler_args'])
            self.assertEqual(calls[0][1]['resource_cpu'], 1000)
            loaded, artifact = app.load_build(root, base/'compile/build.json', 'queue-n32')
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

    def test_wrong_compiled_source_latches_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = make_package(base/'family')
            with self.assertRaisesRegex(ValueError, 'compiled CSL differs'):
                compile_fixture(root, base/'compile', changed_source=True)
            self.assertEqual(app.read(base/'compile/build.json')['status'], 'FAILED')

    def test_package_mutation_and_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = make_package(base/'family')
            app.identity(root, 'arena-cap64')
            (root/'cases/arena-cap64/source/pe.csl').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'frozen package source'):
                app.identity(root, 'arena-cap64')
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
            args = SimpleNamespace(case='queue-n32', profile=1, delay=256, host_timeout=600,
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

    def test_worker_separates_large_sdk_scratch_with_real_outer_audit(self):
        # Family numerical code is covered by the family SDK-fake suites. This
        # fixture exercises worker ownership, actual ELF/metadata joins, the
        # real outer audit and its 32 MiB bound, without an SDK job.
        for case in app.CASES:
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
                    run_id = str(uuid.uuid4())
                    profile, delay = (1, 0) if case == 'queue-n32' else (0, 0)
                    write(staged/'run-request.json', dict(run_id=run_id, case=case, profile=profile, delay=delay,
                                                       build_record_sha256=app.sha(staged/'build.json')))
                    original_driver = app.driver
                    observed_cwds = []
                    def fake_driver(frozen_root, label):
                        folder, implementation = original_driver(frozen_root, label)
                        def capture(*args, sdk_workdir, **kwargs):
                            runtime = Path(args[2] if label == 'arena-cap64' else args[1])
                            runtime.mkdir()
                            write(runtime/'partial.json', {'attempted':True})
                            previous = Path.cwd()
                            try:
                                os.chdir(sdk_workdir)
                                observed_cwds.append(Path.cwd())
                                with Path('sdk-config.conf').open('xb') as stream: stream.truncate(50 << 20)
                                if failure: raise RuntimeError('fake runtime failed after stop attempt')
                            finally:
                                os.chdir(previous)
                            program = compiled['program_inventory']
                            expected = {name: row['sha256'] for name, row in program['elf']['files'].items()}
                            if label == 'arena-cap64':
                                expected.update(program['metadata_sha256'])
                                write(runtime/'program.json', {name:{'sha256':digest} for name,digest in expected.items()})
                                saved_case = runtime/'case/case.json'
                            else:
                                expected.update({name:program['metadata_sha256'][name]
                                                 for name in ('out.json', 'bin/out_rpc.json')})
                                write(runtime/'build.json', {'files':expected})
                                saved_case = runtime/'sources/case.json'
                            saved_case.parent.mkdir(parents=True)
                            shutil.copyfile(folder/'case.json', saved_case)
                        implementation.capture = capture
                        implementation.audit = lambda runtime: dict(status='PASS', backend='hardware', profile=profile, delay=delay)
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
                            frozen_root = staged/('frozen-observation-'+run_id)
                            self.assertEqual(app.audit_capture(frozen_root, case, profile, delay, args.out)['status'], 'PASS')
                            with (args.out/'unexpected-sdk-file').open('xb') as stream: stream.truncate(50 << 20)
                            with self.assertRaisesRegex(ValueError, 'tree byte bound'):
                                app.audit_capture(frozen_root, case, profile, delay, args.out)
                    self.assertEqual(Path.cwd(), previous)
                    scratch = staged/('sdk-work-'+run_id)
                    self.assertEqual(observed_cwds, [scratch])
                    self.assertFalse(scratch.exists())
                    record = app.read(args.out/'worker.json')
                    self.assertEqual(record['status'], 'FAILED' if failure else 'PASS')
                    self.assertEqual(record['sdk_workdir']['bytes'], 50 << 20)
                    self.assertEqual(record['sdk_workdir']['entries'], 1)
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
