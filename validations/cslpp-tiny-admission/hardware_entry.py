"""One tiny target-compiler admission job, or its offline audit; no device run."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import time
import uuid

HERE = Path(__file__).resolve().parent
# The exporter also copies this entrypoint to the family root.
IMPORT_ROOT = HERE.parents[4] if HERE.name == 'cslpp' else HERE
sys.path.insert(0, str(IMPORT_ROOT))
from models.hierarchical_gnn.grouped_csr.native_cora_appliance import appliance, saved_audit as common
from models.hierarchical_gnn.grouped_csr.native_cora_appliance.observation import hardware_entry as helper
try:
    from . import stack_gate
except ImportError:
    import stack_gate

FAMILY = 'cslpp-tiny-admission'
ARMS = ('direct', 'frontend', 'async')
PARAMS = dict(C=2, R=2, K=4, L=3, FW=12, H=16, O=7, T=16, MX=23, MG=5,
              COMPACT=1, U=4, PROFILE=0)
COORDS = {(0, 0), (1, 0), (0, 1), (1, 1)}
require, sha, read, write = helper.require, helper.sha, helper.read, helper.write


def source_inventory(directory):
    """The compiled directory is deliberately flat and contains CSL only."""
    directory = Path(directory).absolute()
    require(directory == directory.resolve() and directory.is_dir(), 'source directory confinement')
    files = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            require(len(files) < 32 and entry.is_file(follow_symlinks=False)
                    and not entry.is_symlink() and re.fullmatch(r'[A-Za-z0-9_]+\.csl', entry.name),
                    'complete flat CSL inventory')
            files[entry.name] = sha(directory/entry.name, 2 << 20)
    require('layout.csl' in files, 'generated layout required')
    return files


def layout_mapping(source, arm):
    text = source.decode('utf-8')
    if arm == 'direct':
        require(len(re.findall(r'@set_rectangle\s*\(', text)) == 1
                and '@set_rectangle(C, R)' in text
                and len(re.findall(r'@set_tile_code\s*\(', text)) == 1
                and '@set_tile_code(x, y, "pe.csl",' in text
                and 'for (@range(u16, C)) |x|' in text
                and 'for (@range(u16, R)) |y|' in text, 'direct tiny parameterized layout')
        return {f'{x},{y}': 'pe.csl' for x, y in sorted(COORDS)}
    require(len(re.findall(r'@set_rectangle\s*\(', text)) == 1
            and re.search(r'@set_rectangle\(2,\s*2\);', text), 'translated tiny rectangle')
    calls = re.findall(r'@set_tile_code\((\d+),\s*(\d+),\s*"([A-Za-z0-9_]+\.csl)",\s*\.\{\}\);', text)
    require(len(calls) == len(re.findall(r'@set_tile_code\s*\(', text)) == 4,
            'four literal generated tile bindings')
    mapping = {f'{int(x)},{int(y)}': name for x, y, name in calls}
    require(set(mapping) == {f'{x},{y}' for x, y in COORDS}
            and len(set(mapping.values())) == 4, 'complete distinct tiny PE mapping')
    return mapping


def identity(root, arm):
    root = Path(root).absolute()
    require(root == root.resolve() and arm in ARMS, 'registered arm and package path')
    package = read(root/'package.json')
    require(package.get('schema') == 1 and package.get('family') == FAMILY
            and package.get('params') == PARAMS and set(package.get('arms', {})) == set(ARMS),
            'fixed tiny package; full Cora is forbidden')
    files = package.get('files')
    require(isinstance(files, dict) and 0 < len(files) <= 256, 'bounded package inventory')
    total = 0
    for name, digest in files.items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts and str(path) == name
                and (root/name).resolve().is_relative_to(root), 'safe package member')
        require(sha(root/name, 16 << 20) == digest, 'frozen package file '+name)
        total += (root/name).stat().st_size
        require(total <= 32 << 20, 'small package byte bound')
    require('case/case.json' in files and sha(root/'case/case.json') == package['case_sha256'],
            'tiny case identity')
    manifest = common.load(root/'case')
    require(manifest['kind'] == 'tiny' and dict(manifest['params'], PROFILE=0) == PARAMS,
            'tiny original case only')
    item = package['arms'][arm]
    directory = root/'generated'/arm
    sources = source_inventory(directory)
    mapping = layout_mapping((directory/'layout.csl').read_bytes(), arm)
    require(sources == item['sources'] and set(sources) == {'layout.csl', *mapping.values()}
            and mapping == item['layout'], 'complete generated source and layout identity')
    require(all(files.get(f'generated/{arm}/{name}') == digest for name, digest in sources.items()),
            'generated sources in package inventory')
    provenance_path = f'references/{arm}/compile.json'
    require(provenance_path in files, 'local reference provenance required')
    provenance = read(root/provenance_path)
    require(provenance['arm'] == arm and provenance['status'] == 'PASS'
            and provenance['backend'] == 'simulator' and provenance['params'] == PARAMS
            and provenance['case_sha256'] == package['case_sha256'], 'local reference arm/case identity')
    require(sources == provenance['source' if arm == 'direct' else 'generated_source'],
            'generated sources differ from established local reference')
    for name, digest in provenance['source'].items():
        require(files.get(f'originals/{arm}/{name}') == digest, 'original source provenance')
    references = {Path(name).name: row['sha256'] for name, row in provenance['inventory']['files'].items()
                  if re.fullmatch(r'bin/[^/]+\.elf', name)}
    require(len(references) == 4 and references == item['reference_files'], 'four original reference programs')
    for name, digest in references.items():
        require(files.get(f'references/{arm}/build/bin/{name}') == digest, 'reference ELF package pin')
    return dict(package_manifest_sha256=sha(root/'package.json'), arm=arm, params=PARAMS,
                case_sha256=package['case_sha256'], source_sha256=sources, layout=mapping,
                reference_sha256=references, provenance_sha256=files[provenance_path])


def program_archive(path, sources):
    """Read every source and generated program from one bounded compiler archive."""
    sha(path, 256 << 20)
    began, total, seen = time.monotonic(), 0, set()
    elfs, metadata, csl, source_roots, payloads = {}, {}, {}, set(), {}
    with tarfile.open(path, 'r:*') as archive:
        for count, member in enumerate(archive):
            name = PurePosixPath(member.name)
            require(name.parts and not name.is_absolute() and '..' not in name.parts
                    and str(name) == member.name.rstrip('/'), 'safe compiler member')
            require(str(name) not in seen and (member.isdir() or member.isfile()),
                    'unique regular compiler member')
            seen.add(str(name))
            total += member.size
            require(count < 20000 and 0 <= member.size and total <= 256 << 20
                    and time.monotonic()-began <= 45, 'compiler archive bound')
            if not member.isfile():
                continue
            in_csl = 'csl' in name.parts[:-1]
            require(name.suffix != '.csl' or in_csl, 'CSL outside registered compiler source tree')
            if in_csl:
                require(name.parent.name == 'csl' and name.name in sources,
                        'unregistered compiler CSL source')
            selected = name.suffix == '.elf' or name.name in ('out.json', 'out_rpc.json')
            if not (selected or in_csl):
                continue
            require(member.size <= (32 << 20 if name.suffix == '.elf' else 2 << 20),
                    'selected compiler member bound')
            with archive.extractfile(member) as stream:
                payload = stream.read(member.size+1)
            require(len(payload) == member.size, 'complete compiler member')
            digest = hashlib.sha256(payload).hexdigest()
            if in_csl:
                require(name.name not in csl and digest == sources[name.name], 'compiled source identity')
                csl[name.name] = digest
                source_roots.add(str(name.parent.parent))
            if name.suffix == '.elf':
                elfs[str(name)] = dict(sha256=digest, bytes=len(payload), **common.p._elf_storage(payload))
                payloads[str(name)] = payload
            elif selected:
                metadata[str(name)] = digest
    roots = {str(PurePosixPath(name).parent.parent) for name in metadata
             if PurePosixPath(name).parts[-2:] == ('west', 'out.json')}
    roots = {root for root in roots if root+'/east/out.json' in metadata and root+'/out.json' in metadata}
    require(len(roots) == 1 and len(source_roots) == 1 and set(csl) == set(sources) and elfs,
            'complete source and unique generated program')
    root = next(iter(roots))
    require(PurePosixPath(root).parent.as_posix() == next(iter(source_roots)),
            'source and generated program share compiler root')
    prefix = root+'/'
    require(all(name.startswith(prefix) for name in (*elfs, *metadata)), 'single generated program tree')
    files = {name[len(prefix):]: value for name, value in elfs.items()}
    inventory = dict(files=files, artifact_bytes=sum(row['bytes'] for row in files.values()),
                     max_static_sram_end=max(row['static_sram_end'] for row in files.values()),
                     scope='worker-observed generated ELF hashes and allocated low-address sections before launch; stack/runtime high-water unmeasured')
    common.admit_compiled_inventory(inventory)
    return dict(elf=inventory, metadata_sha256={name[len(prefix):]: value for name, value in metadata.items()}), {
        name[len(prefix):]: value for name, value in payloads.items()}


def application_map(payloads, fabric_width):
    mapping = {}
    expected = {(y+1)*fabric_width+x+4: (x, y) for x, y in COORDS}
    apps = {name: payload for name, payload in payloads.items() if re.fullmatch(r'bin/[^/]+\.elf', name)}
    require(len(apps) == 4, 'four application programs required')
    for name, payload in apps.items():
        indices = stack_gate.inspect(payload)['tile_indices']
        require(len(indices) == 1 and indices[0] in expected, 'application ELF placement')
        coordinate = expected[indices[0]]
        require(coordinate not in mapping, 'unique application ELF placement')
        mapping[coordinate] = (name, payload)
    require(set(mapping) == COORDS, 'complete application ELF placement')
    return mapping


def verify_stack(root, pinned, payloads):
    arm = pinned['arm']
    reference = {f'bin/{name}': (root/f'references/{arm}/build/bin/{name}').read_bytes()
                 for name in pinned['reference_sha256']}
    references = application_map(reference, 9)
    targets = application_map(payloads, 762)
    results = {}
    for coordinate in sorted(COORDS):
        key = f'{coordinate[0]},{coordinate[1]}'
        reference_name, reference_bytes = references[coordinate]
        target_name, target_bytes = targets[coordinate]
        source = root/'generated'/arm/pinned['layout'][key]
        result = stack_gate.compare(reference_bytes, target_bytes, arm=arm,
            reference_sha256=pinned['reference_sha256'][Path(reference_name).name], source=source.read_bytes())
        require(result['status'] == 'PASS', 'target compiler stack gate')
        results[key] = dict(reference_elf=reference_name, target_elf=target_name, evidence=result)
    return dict(status='PASS', application_programs=results,
                coroutine_capacity_status='NOT_ESTABLISHED' if arm == 'async' else 'NOT_APPLICABLE',
                admission_status='HOLD' if arm == 'async' else 'PASS',
                admission_reason='backend-generated call closure unverified' if arm == 'async' else 'no coroutine; static gates only',
                scope='tiny target program equivalence and reservation evidence; no device execution or independent coroutine capacity proof')


def compile_job(args, root=HERE, compiler_type=None):
    root = Path(root)
    pinned = identity(root, args.arm)
    output = appliance.fresh(args.out)
    record = dict(schema=1, family=FAMILY, action='compile', backend='hardware', status='RUNNING',
                  compiler_status='NOT_RUN', stack_status='NOT_RUN', compile_id=str(uuid.uuid4()),
                  namespace=args.namespace, compiler_args=common.compiler_arguments(PARAMS, 'hardware', 1), **pinned)
    write(output/'request.json', record)
    began = time.monotonic()
    try:
        with appliance.client_log(output) as log:
            cls = compiler_type or appliance.sdk_classes()[0]
            with cls(**appliance.sdk_options(args.namespace, 1)) as compiler:
                returned = Path(compiler.compile(str(root/'generated'/args.arm), 'layout.csl',
                                                record['compiler_args'], str(output)))
                require(not returned.is_symlink(), 'artifact symlink')
                artifact = returned.resolve()
        record['compiler_status'] = 'PASS'
        require(artifact.parent == output and artifact.name not in ('request.json', 'build.json', 'client.log'),
                'compiler artifact confinement')
        record.update(artifact_path=artifact.name, artifact_sha256=sha(artifact, 256 << 20))
        require(not log.truncated and pinned == identity(root, args.arm), 'source unchanged and complete compiler log')
        inventory, payloads = program_archive(artifact, pinned['source_sha256'])
        record['program_inventory'] = inventory
        record['stack_status'] = 'RUNNING'
        record['stack_gate'] = verify_stack(root, pinned, payloads)
        record.update(status='EVIDENCE_COMPLETE', stack_status='REFERENCE_EQUIVALENT',
                      source_identity_status='PASS', static_sram_status='PASS', reference_equivalence_status='PASS',
                      coroutine_capacity_status=record['stack_gate']['coroutine_capacity_status'],
                      admission_status=record['stack_gate']['admission_status'],
                      admission_reason=record['stack_gate']['admission_reason'])
    except BaseException as error:
        record.update(status='FAILED', error=repr(error))
        if record['compiler_status'] == 'NOT_RUN':
            record['compiler_status'] = 'FAILED'
        if record['stack_status'] == 'RUNNING':
            record['stack_status'] = 'FAILED'
        raise
    finally:
        record['compile_wall_seconds'] = time.monotonic()-began
        write(output/'build.json', record)
    return record


def audit_build(root, arm, path):
    root, path = Path(root), Path(path)
    record = read(path)
    pinned = identity(root, arm)
    require(record.get('schema') == 1 and record.get('family') == FAMILY
            and record.get('action') == 'compile' and record.get('backend') == 'hardware'
            and record.get('compiler_status') == 'PASS', 'target compile record')
    require(all(record.get(key) == value for key, value in pinned.items()), 'source/case/reference join')
    require(record['compiler_args'] == common.compiler_arguments(PARAMS, 'hardware', 1), 'exact target compiler flags')
    request = read(path.parent/'request.json')
    initial_keys = {'schema', 'family', 'action', 'backend', 'status', 'compiler_status', 'stack_status',
                    'compile_id', 'namespace', 'compiler_args', *pinned}
    require(set(request) == initial_keys and request['status'] == 'RUNNING'
            and request['compiler_status'] == request['stack_status'] == 'NOT_RUN'
            and all(record.get(key) == value for key, value in request.items()
                    if key not in ('status', 'compiler_status', 'stack_status')), 'initial compile request binding')
    name = record['artifact_path']
    require(isinstance(name, str) and Path(name).name == name and name not in ('', '.', '..'), 'artifact filename')
    artifact = path.parent/name
    require(sha(artifact, 256 << 20) == record['artifact_sha256'], 'target compiler archive identity')
    inventory, payloads = program_archive(artifact, pinned['source_sha256'])
    require(inventory == record['program_inventory'], 'target program inventory')
    gate = verify_stack(root, pinned, payloads)
    require(record['status'] == 'EVIDENCE_COMPLETE' and record['stack_status'] == 'REFERENCE_EQUIVALENT'
            and gate == record['stack_gate']
            and all(record.get(key) == 'PASS' for key in ('source_identity_status', 'static_sram_status', 'reference_equivalence_status'))
            and all(record.get(key) == gate[key] for key in ('coroutine_capacity_status', 'admission_status', 'admission_reason')),
            'replayed target stack admission')
    return dict(status='EVIDENCE_COMPLETE', family=FAMILY, arm=arm, build_record_sha256=sha(path),
                admission_status=gate['admission_status'], coroutine_capacity_status=gate['coroutine_capacity_status'],
                admission_reason=gate['admission_reason'],
                artifact_sha256=record['artifact_sha256'], stack_gate=gate,
                max_static_sram_end=inventory['elf']['max_static_sram_end'],
                scope='offline target compiler admission; no device execution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('compile', 'audit'))
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--namespace', default='job-operator')
    parser.add_argument('--build-record', type=Path)
    args = parser.parse_args()
    if args.action == 'compile':
        result = compile_job(args)
    else:
        require(args.build_record is not None and not args.out.exists(), 'build record and fresh audit output')
        result = audit_build(HERE, args.arm, args.build_record)
        write(args.out, result)
    print(json.dumps({key: result[key] for key in ('status', 'arm', 'compiler_status', 'stack_status', 'admission_status') if key in result}))


if __name__ == '__main__':
    main()
