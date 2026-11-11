#!/usr/bin/env python3
"""One foreground compile, run, or independent saved audit for resident Shah GCN."""
import argparse
import importlib.util
from contextlib import contextmanager
import logging
from pathlib import Path
import shlex
import sys
import tempfile
import uuid

from archive import digest, program_archive, read, require, unpack, write

ROOT = Path(__file__).resolve().parent
SOURCES = ('layout.csl', 'pe.csl')
HOSTS = ('driver.py', 'oracle.py', 'proof.py', 'worker.py', 'archive.py')


def family_id(root=ROOT):
    value = read(root / 'validation.json')['id']
    require(value in ('shah-resident-cora', 'shah-csr-cora'), 'registered family identity')
    return value


def frozen_driver(root=ROOT):
    # Each command is a fresh process; load the exact admitted execution files,
    # including when a host test has already imported a different family.
    sys.path.insert(0, str(root))
    for name in ('oracle', 'proof', 'driver'):
        spec = importlib.util.spec_from_file_location(name, root / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules['driver']


def identity(root=ROOT):
    return dict(source_sha256={n: digest(root / n, 2 << 20) for n in SOURCES},
                host_sha256={n: digest(root / n, 2 << 20) for n in HOSTS},
                audit_sha256=digest(root / 'saved_audit.py', 2 << 20),
                adapter_sha256=digest(root / 'appliance.py', 2 << 20))


def fresh(path):
    path = Path(path).absolute()
    require(path == path.resolve(), 'nonsymlink destination')
    path.mkdir(parents=True, exist_ok=False)
    return path


def case_record(label, root=ROOT):
    catalog = read(root / 'cases.json')
    require(catalog['schema'] == 1 and label in catalog['cases'], 'registered case label')
    item = catalog['cases'][label]
    relative = Path(item['bundle'])
    require(relative.parts == ('cases', label + '.tar.gz'), 'case bundle path')
    bundle = root / relative
    require(digest(bundle) == item['sha256'], 'frozen case bundle changed')
    return item, bundle


def verify_case(path, item, root=ROOT):
    require(digest(path / 'case.json') == item['case_sha256'], 'frozen case manifest changed')
    m = read(path / 'case.json')
    require(m['params'] == item['params'], 'case parameters changed')
    for name, pin in m['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'case file path')
        require(digest(path / relative) == pin, 'case payload changed: ' + name)
    for name, pin in m['source_sha256'].items():
        require(Path(name).name == name and digest(path / 'source' / name) == pin
                and digest(root / name) == pin, 'case execution source changed: ' + name)
    return m


def sdk_options(namespace):
    options = dict(resource_cpu=1000, resource_mem=8 << 30, disable_version_check=False)
    if namespace:
        options['mgmt_namespace'] = namespace
    return options


def sdk_classes():
    from cerebras.appliance import logger  # noqa: F401
    from cerebras.sdk.client import SdkCompiler, SdkLauncher
    return SdkCompiler, SdkLauncher


@contextmanager
def client_log(destination):
    handler = logging.FileHandler(destination / 'client.log')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s'))
    logging.getLogger().addHandler(handler)
    try:
        yield
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def flags(params, backend, channels, root=ROOT):
    require(channels in (1, 2, 4), 'channel count')
    driver = frozen_driver(root)
    words = driver.compiler_flags(params, hardware=backend == 'hardware', channels=channels)
    require(isinstance(words, list) and words[-2:] == ['-o', 'out'], 'one generated output directory')
    return shlex.join(words)


def compile_program(args, root=ROOT, compiler_type=None):
    item, bundle = case_record(args.case, root)
    pins = identity(root)
    out = fresh(args.out)
    command = flags(item['params'], args.backend, args.channels, root)
    record = dict(schema=1, family=family_id(root), backend=args.backend, namespace=args.namespace,
                  case_label=args.case, case=item, channels=args.channels, compiler_args=command, **pins)
    write(out / 'request.json', record)
    try:
        if compiler_type is None:
            compiler_type = sdk_classes()[0]
        with tempfile.TemporaryDirectory(prefix='shah-resident-case-') as temporary:
            case = unpack(bundle, Path(temporary) / 'case')
            verify_case(case, item, root)
            with client_log(out), compiler_type(**sdk_options(args.namespace)) as compiler:
                artifact = Path(compiler.compile(str(case / 'source'), 'layout.csl', command, str(out))).resolve()
        require(artifact.parent == out, 'compiler artifact outside output')
        require(identity(root) == pins and digest(bundle) == item['sha256'], 'source/case changed during compilation')
        record.update(artifact_path=artifact.name, artifact_sha256=digest(artifact),
                      program_inventory=program_archive(artifact, pins['source_sha256']))
        write(out / 'build.json', record)
    except Exception as error:
        write(out / 'compile-error.json', dict(error=repr(error)))
        raise
    print(out / 'build.json', flush=True)
    return 0


def load_record(path, backend=None, root=ROOT):
    path = Path(path).resolve()
    record = read(path)
    require(record.get('schema') == 1 and record.get('family') == family_id(root)
            and record.get('backend') in ('hardware', 'simulator')
            and (backend is None or record['backend'] == backend), 'build schema/backend')
    for key, value in identity(root).items():
        require(record.get(key) == value, 'compiled execution/audit source drift: ' + key)
    item, _ = case_record(record['case_label'], root)
    require(record['case'] == item and record['compiler_args'] == flags(item['params'], record['backend'], record['channels'], root),
            'compile case/flags binding')
    name = record['artifact_path']
    require(isinstance(name, str) and Path(name).name == name, 'adjacent artifact filename')
    artifact = path.parent / name
    require(digest(artifact) == record['artifact_sha256'], 'artifact hash')
    require(program_archive(artifact, record['source_sha256']) == record['program_inventory'], 'actual compiler program inventory')
    return record, artifact


def host_command(backend, seconds):
    require(type(seconds) is int and 1 <= seconds <= 86400, 'worker timeout 1..86400')
    words = ['cs_python', 'worker.py', '--record', 'run-request.json', '--name', 'out', '--out', 'capture']
    if backend == 'hardware':
        words += ['--cmaddr', '%CMADDR%']
    return (f'if timeout --signal=INT --kill-after=15s {seconds}s {shlex.join(words)} > host.log 2>&1; '
            'then code=0; else code=$?; fi; printf \'%s\\n\' "$code" > host-exit.txt')


def downloaded_path(returned, requested, limit):
    requested = Path(requested).absolute()
    require(returned is None or isinstance(returned, str) and returned, 'download return path')
    path = requested if returned is None else Path(returned)
    if not path.is_absolute():
        path = requested.parent / path
    require(not path.is_symlink(), 'download symlink')
    path = path.resolve()
    require(path.parent == requested.parent and path.name not in
            {'request.json', 'run-request.json', 'collection.json', 'client.log', 'launcher-response.txt', 'launcher-error.txt'},
            'download collision/escape')
    digest(path, limit)
    return path


def run_program(args, root=ROOT, launcher_type=None):
    record, artifact = load_record(args.artifact_record, args.backend, root)
    _, bundle = case_record(record['case_label'], root)
    require(args.mode in ('all', 'gcn', 'ah16', 'ah7'), 'invocation mode')
    if record['family'] == 'shah-csr-cora':
        require(args.mode == 'ah' + str(record['case']['params']['AH_WIDTH']), 'CSR mode must match compiled AH width')
    command, out, run_id = host_command(args.backend, args.host_timeout), fresh(args.out), str(uuid.uuid4())
    request = dict(compile_record=record, mode=args.mode, run_id=run_id)
    write(out / 'request.json', dict(command=command, **request))
    write(out / 'run-request.json', request)
    collection = dict(run_id=run_id, downloaded={}, errors={})
    failure = None
    if launcher_type is None:
        launcher_type = sdk_classes()[1]
    with client_log(out), launcher_type(str(artifact), simulator=args.backend == 'simulator',
                                       **sdk_options(record['namespace'])) as launcher:
        try:
            for name, pin in {**record['host_sha256'], **record['source_sha256']}.items():
                require(digest(root / name) == pin, 'staged execution source changed')
                launcher.stage(str(root / name))
            require(digest(bundle) == record['case']['sha256'], 'case changed before staging')
            launcher.stage(str(bundle))
            launcher.stage(str(out / 'run-request.json'))
            reply = launcher.run(command)
            (out / 'launcher-response.txt').write_text('' if reply is None else str(reply))
        except Exception as error:
            failure = error
            (out / 'launcher-error.txt').write_text(repr(error) + '\n')
        finally:
            for remote, limit in (('host.log', 8 << 20), ('host-exit.txt', 128), ('capture', 256 << 20)):
                try:
                    saved = downloaded_path(launcher.download_artifact(remote, str(out / remote)), out / remote, limit)
                    require(saved.name not in {v['path'] for v in collection['downloaded'].values()}, 'download aliases')
                    collection['downloaded'][remote] = dict(path=saved.name, sha256=digest(saved, limit))
                except Exception as error:
                    collection['errors'][remote] = repr(error)
            collection['status'] = 'collection_failed' if collection['errors'] else 'complete'
            write(out / 'collection.json', collection)
    require(not collection['errors'], 'incomplete collection; partial files retained')
    if failure is not None:
        raise RuntimeError('launcher failed; artifacts retained') from failure
    require((out / collection['downloaded']['host-exit.txt']['path']).read_text().strip() == '0', 'worker exit failed')
    return 0


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('compile', 'run'):
        p = sub.add_parser(action)
        p.add_argument('--backend', choices=('hardware', 'simulator'), default='hardware')
        p.add_argument('--out', type=Path, required=True)
        if action == 'compile':
            p.add_argument('--case', required=True)
            p.add_argument('--namespace', default='job-operator')
            p.add_argument('--channels', type=int, choices=(1, 2, 4), default=1)
        else:
            p.add_argument('--artifact-record', type=Path, required=True)
            p.add_argument('--mode', choices=('all', 'gcn', 'ah16', 'ah7'), default='all')
            p.add_argument('--host-timeout', type=int, default=86400)
    p = sub.add_parser('audit')
    p.add_argument('--artifact-record', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        if args.action == 'compile':
            return compile_program(args)
        if args.action == 'run':
            return run_program(args)
        from saved_audit import audit
        require(not args.out.exists(), 'fresh audit output')
        result = audit(args.capture, args.artifact_record)
        write(args.out, result)
        print(args.out, flush=True)
        return 0
    except Exception as error:
        print(f'ERROR status=fail reason={error}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
