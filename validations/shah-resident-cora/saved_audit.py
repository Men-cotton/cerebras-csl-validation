"""Independent saved-output replay; no SDK import or remote invocation."""
import importlib
from pathlib import Path
import sys
import tempfile
import uuid

from archive import check_npz, compare_audits, digest, read, require, unpack, write
from appliance import ROOT, case_record, frozen_driver, host_command, load_record, verify_case


def bind(request, collection, status, execution, record):
    run_id = request['run_id']
    require(str(uuid.UUID(run_id)) == run_id and collection['run_id'] == run_id
            and status.get('run_id') == execution.get('run_id') == run_id, 'capture run identity')
    require(request['compile_record'] == status.get('compile_record') == record,
            'capture compile identity')
    require(status.get('status') == 'PASS' and execution.get('status') == 'CAPTURED', 'complete worker/runtime capture')
    require(status.get('program_inventory') == record['program_inventory']
            and execution.get('compiled_inventory') == record['program_inventory']['elf'], 'actual runtime program identity')
    require(status.get('mode') == execution.get('mode') == request['mode'], 'capture invocation mode')
    require(execution.get('backend') == record['backend'] and execution.get('case_sha256') == record['case']['case_sha256'],
            'capture backend/case identity')


def audit(capture, artifact_record, root=ROOT):
    capture = Path(capture).resolve()
    record, _ = load_record(artifact_record, root=root)
    request, worker_request = read(capture / 'request.json'), read(capture / 'run-request.json')
    require({k: v for k, v in request.items() if k != 'command'} == worker_request, 'staged request differs')
    require(request['compile_record'] == record, 'run consumes a different compile')
    # Exact command is saved; infer only the declared numeric worker bound.
    words = request['command'].split()
    require(len(words) > 4 and words[4].endswith('s') and words[4][:-1].isdigit(), 'worker bound command')
    require(request['command'] == host_command(record['backend'], int(words[4][:-1])), 'worker command contract')
    collection = read(capture / 'collection.json')
    require(collection.get('status') == 'complete' and not collection['errors']
            and set(collection['downloaded']) == {'host.log', 'host-exit.txt', 'capture'}, 'complete collection')
    files = {}
    require(len({v['path'] for v in collection['downloaded'].values()}) == 3, 'download aliases')
    for key, value in collection['downloaded'].items():
        name = value['path']
        require(Path(name).name == name and name not in ('request.json', 'run-request.json', 'collection.json'), 'safe downloaded member')
        path = capture / name
        require(digest(path) == value['sha256'], 'downloaded file hash')
        files[key] = path
    require(files['host-exit.txt'].read_text().strip() == '0', 'worker exit success')
    require(not (capture / 'launcher-error.txt').exists(), 'launcher error despite claimed success')
    item, bundle = case_record(record['case_label'], root)
    with tempfile.TemporaryDirectory(prefix='shah-resident-audit-') as temporary:
        temporary = Path(temporary)
        extracted = unpack(files['capture'], temporary / 'export')
        candidates = [p for p in (extracted, extracted / 'capture') if (p / 'status.json').is_file()]
        require(len(candidates) == 1, 'unique capture root')
        result_root = candidates[0]
        status = read(result_root / 'status.json')
        runtime = result_root / 'runtime'
        execution = read(runtime / 'execution.json')
        check_npz(runtime / 'raw.npz')
        bind(request, collection, status, execution, record)
        case = unpack(bundle, temporary / 'case')
        verify_case(case, item, root)
        for name in read(case / 'case.json')['files']:
            if name.endswith('.npz'):
                check_npz(case / name)
        write(case / 'compile.json', dict(status='PASS', case_sha256=item['case_sha256'], inventory=record['program_inventory']['elf']))
        driver = frozen_driver(root)
        require(Path(driver.__file__).resolve() == (root / 'driver.py').resolve(), 'isolated frozen driver import')
        remote = read(runtime / 'audit.json')
        fresh = driver.audit(case, runtime)
        compare_audits(fresh, remote)
        require(fresh['status'] == 'PASS' and fresh.get('run_id') == request['run_id'], 'numerical/run audit acceptance')
    return dict(schema=1, status='PASS', backend=record['backend'], run_id=request['run_id'], mode=request['mode'],
                case_label=record['case_label'], case_sha256=item['case_sha256'],
                artifact_sha256=record['artifact_sha256'], source_sha256=record['source_sha256'],
                program_inventory=record['program_inventory'], result=fresh,
                scheduler_scope='SDK job terminal states are reconciled separately by the request controller')
