"""Freeze the E236 tiny compiler-admission family; no SDK calls or submission."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
from models.hierarchical_gnn.grouped_csr.native_cora_appliance import saved_audit as common
from models.hierarchical_gnn.grouped_csr.native_cora_appliance.observation.export import SOURCE_FILES

FAMILY = 'cslpp-tiny-admission'
PREFIX = 'models/hierarchical_gnn/grouped_csr/native_cora_appliance'
EVIDENCE = ROOT/'research/experiments/data/E236'
ARMS = ('direct', 'frontend', 'async')


def export(output, request_id):
    common.require(re.fullmatch('[a-z][a-z0-9-]{0,63}', request_id), 'request prefix')
    output = Path(output).absolute()
    common.require(output == output.resolve(), 'export path symlink')
    output.mkdir(parents=True, exist_ok=False)
    copied = {}

    def copy(source, name):
        source = Path(source)
        digest = common.sha(source)
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        common.require(common.sha(target) == digest, 'export byte identity')
        copied[name] = digest

    # Shared archive, SDK client and ABI/storage helpers remain frozen with the
    # new adapter. Existing registered families and historical inputs stay intact.
    for name in [*SOURCE_FILES, f'{PREFIX}/observation/hardware_entry.py']:
        copy(ROOT/name, name)
    for name in ('hardware_entry.py', 'stack_gate.py'):
        copy(HERE/name, name)
    copy(HERE/'export.py', 'provenance/export.py')
    readme = output/'README.md'
    readme.write_text("""# Tiny CSL++ target-compiler admission

This family compiles the frozen E236 seven-node, 2x2 PE fixture only. It exposes
`hardware_entry.py compile --arm direct|frontend|async --out DIR` and an offline
`audit --arm ARM --build-record DIR/build.json --out FILE`. Each appliance step
submits exactly one compiler operation. There is no device run or full-Cora path.

Every generated layout/PE source is registered and matched to the returned
compiler archive. All application/helper static SRAM endpoints must remain
below 44 KiB. For each application coordinate, target code, function locations,
loaded memory and coroutine storage must match the pinned frame-measured E236
reference. Async has a measured 64 B root frame plus the resolver's 12 B argument area;
backend-generated call closure remains unverified, so its admission is HOLD. Different
code fails this deliberately narrow equivalence gate. Target metadata absence is
accepted only under exact machine-code equivalence; runtime peaks are unmeasured.

The package contains the original/generated sources, local reference ELF files,
generator provenance, frozen tiny input and offline mutation tests. A published
request is not proof of target compiler completion. Compile records preserve the
returned archive even when the subsequent stack gate fails. Source, target
compiler success, admission and numerical device execution are distinct claims.
""")
    copied['README.md'] = common.sha(readme)
    (output/'tests').mkdir()
    for name in ('test_hardware.py', 'test_stack_gate.py'):
        content = (HERE/name).read_text()
        content = content.replace('from . import hardware_entry as hw', 'import hardware_entry as hw')
        content = content.replace('from . import stack_gate', 'import stack_gate')
        content = content.replace('from .stack_gate import ', 'from stack_gate import ')
        target = output/'tests'/name
        target.write_text(content)
        copied['tests/'+name] = common.sha(target)
    case = EVIDENCE/'case'
    manifest = common.load(case)
    common.require(manifest['kind'] == 'tiny' and manifest['params']['C'] == 2
                   and manifest['params']['R'] == 2, 'only frozen tiny fixture')
    for name in ['case.json', *manifest['files']]:
        copy(common.local_file(case, name), 'case/'+name)
    copy(EVIDENCE/'toolchain.json', 'provenance/toolchain.json')
    arms, params = {}, None
    for arm in ARMS:
        old = EVIDENCE/'arms'/arm
        record = common.read_json(old/'compile.json', limit=common.INVENTORY_JSON_LIMIT)
        common.require(record['arm'] == arm and record['status'] == 'PASS', 'reference arm')
        if params is None:
            params = record['params']
        common.require(params == record['params'], 'same tiny parameters')
        copy(old/'compile.json', f'references/{arm}/compile.json')
        for name, digest in record['source'].items():
            common.require(common.sha(old/'source'/name) == digest, 'original source pin')
            copy(old/'source'/name, f'originals/{arm}/{name}')
        source = old/('source' if arm == 'direct' else 'generated')
        pins = record['source'] if arm == 'direct' else record['generated_source']
        common.require({p.name for p in source.iterdir()} == set(pins), 'complete original generated inventory')
        for name, digest in pins.items():
            common.require(common.sha(source/name) == digest, 'generated source pin')
            copy(source/name, f'generated/{arm}/{name}')
        layout = (source/'layout.csl').read_text()
        mapping = {f'{x},{y}': name for x, y, name in re.findall(
            r'@set_tile_code\((\d+),\s*(\d+),\s*"([^"]+)"', layout)}
        if arm == 'direct':
            mapping = {f'{x},{y}': 'pe.csl' for y in range(2) for x in range(2)}
        refs = {}
        for name, entry in record['inventory']['files'].items():
            if not name.startswith('bin/'):
                continue
            common.require(common.sha(old/'build'/name) == entry['sha256'], 'reference ELF identity')
            copy(old/'build'/name, f'references/{arm}/build/{name}')
            refs[Path(name).name] = entry['sha256']
        common.require(len(refs) == 4, 'four reference application programs')
        arms[arm] = dict(sources=pins, layout=mapping, reference_files=refs)
    package = dict(schema=1, family=FAMILY, files=copied, params=params,
                   case_sha256=common.sha(case/'case.json'), arms=arms,
                   scope='compile-only tiny 2x2; no full Cora or device runtime')
    common.write_json(output/'package.json', package)
    common.write_json(output/'validation.json', dict(schema=1, id=FAMILY,
        description='Tiny generated-CSL source, target code/reservation evidence and static SRAM checks; async capacity remains HOLD.',
        entrypoints=['hardware_entry.py'], tests=['tests']))
    script = f'validations/{FAMILY}/hardware_entry.py'
    for arm in ARMS:
        rid = request_id+'-'+arm
        common.require(len(rid) <= 80, 'bounded request ID')
        common.write_json(output/f'request-{arm}.json', dict(schema=1, id=rid,
            description=f'Tiny 2x2 {arm} compiler admission only; no full Cora or runtime. Exact generated sources, code/reservation evidence and static SRAM checks; async capacity remains HOLD.',
            steps=[dict(name='compile', kind='appliance', script=script,
                        args=['compile', '--arm', arm, '--out', '{output}/compile']),
                   dict(name='audit', kind='host', script=script,
                        args=['audit', '--arm', arm, '--build-record', '{output}/compile/build.json',
                              '--out', '{output}/audit.json'])]))
    return package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--request-id', required=True)
    args = parser.parse_args()
    package = export(args.out, args.request_id)
    print(json.dumps(dict(family=FAMILY, files=len(package['files']), arms=list(package['arms']))))


if __name__ == '__main__':
    main()
