"""Offline integrity, ABI, graph, resident-lifecycle and numerical validation.

The frozen numerical oracle is the one used by the local E189 SDK checks.
This module never imports an SDK or submits a job.
"""
import argparse
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import zipfile
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np

MAX_BYTES = 64 << 20
MODULES = ('data', 'mapping', 'reference', 'resident_validation',
           'selective_schedule', 'colored_schedule', 'colored_validation',
           'selective_validation', 'driver')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 4 << 20, 'bounded JSON')
    return json.loads(path.read_text())


def regular(root, name):
    rel = Path(name)
    require(not rel.is_absolute() and rel.parts and all(p not in ('.', '..') for p in rel.parts), 'relative path')
    path = root / rel
    require(path.resolve(strict=True) == path and path.is_file(), 'regular nonsymlink file: ' + name)
    require(path.stat().st_size <= MAX_BYTES, 'file bound')
    return path


def verify_package(package):
    package = Path(package).resolve()
    m = read_json(package / 'manifest.json')
    require(m.get('schema') == 1 and 1 <= len(m['cases']) <= 16 and len(m['files']) <= 256, 'package bounds')
    python_names = {'models/meshgraphnet/' + n + '.py' for n in MODULES + ('sdk_runner',)} | {'models/hierarchical_gnn/grouped_csr/driver.py'}
    require({n.removeprefix('oracle/') for n in m['files'] if n.startswith('oracle/')} == python_names, 'exact frozen Python inventory')
    total = 0
    for name, value in m['files'].items():
        path = regular(package, name)
        total += path.stat().st_size
        require(total <= MAX_BYTES and digest(path) == value, 'package identity: ' + name)
    for name, record in m['cases'].items():
        require(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', name), 'case name')
        prefix = 'cases/' + name + '/'
        case = read_json(package / (prefix + 'case.json'))
        require(record['case_sha256'] == m['files'][prefix + 'case.json'], 'case identity')
        require(case['params'] == record['params'] and case.get('experiment') == 'E189', 'case params')
        for relative, value in case['files'].items():
            require(m['files'][prefix + relative] == value, 'prepared file pin')
        bindings = {'source/pe.csl':'models/meshgraphnet/device/pe.csl', 'source/layout.csl':'models/meshgraphnet/device/layout.csl', 'source/no_halo.csl':'models/meshgraphnet/device/no_halo.csl', 'source/sparse_halo.csl':'kernels/communication/sparse-halo/pe.csl', 'sdk_runner.py':'models/meshgraphnet/sdk_runner.py'}
        require(set(case['files']) == set(bindings) | {'inputs.npz','original.npz','mapping.npz'}, 'prepared inventory')
        require(all(case['files'][n] == case['sources'][origin] for n,origin in bindings.items()), 'prepared/source binding')
        require({n for n in case['sources'] if n.endswith('.py')} == python_names, 'case Python inventory')
        for relative, value in case['sources'].items():
            if relative.endswith('.py'):
                require(m['files']['oracle/' + relative] == value, 'oracle source pin')
        p = case['params']
        require(p['MODE'] in (0, 1, 2) and case['transport'] == ('broadcast','selective','colored')[p['MODE']], 'mode')
        require(2 <= p['C'] <= 16 and 2 <= p['R'] <= 16 and p['C']*p['R'] <= 128, 'owner bound')
        require(2 <= p['H'] <= 8 and 1 <= p['S'] <= 2 and 1 <= p['STEPS'] <= 4, 'oracle work bound')
        require(1 <= p['L'] <= 2000 and 1 <= p['E'] <= 12000 and 1 <= p['W'] <= 4096, 'capacity bound')
    return m


def arrays(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_BYTES, 'NPZ bound')
    result = {}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require(0 < len(members) <= 512 and len({m.filename for m in members}) == len(members), 'NPZ inventory')
        require(sum(m.file_size for m in members) <= MAX_BYTES, 'NPZ expanded bound')
        for member in members:
            require(Path(member.filename).name == member.filename and member.filename.endswith('.npy'), 'NPZ member')
            payload = archive.read(member)
            stream = io.BytesIO(payload)
            version = np.lib.format.read_magic(stream)
            require(version in ((1, 0), (2, 0)), 'NPY version')
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(stream, max_header_size=16384)
            require(not dtype.hasobject and dtype.kind in 'fiu' and dtype.itemsize in (1,2,4,8), 'NPY dtype')
            require(len(shape) <= 4 and all(type(n) is int and 0 <= n <= MAX_BYTES for n in shape), 'NPY shape')
            count = math.prod(shape)
            require(count*dtype.itemsize == len(payload)-stream.tell(), 'NPY size')
            value = np.frombuffer(payload, dtype=dtype, count=count, offset=stream.tell()).reshape(shape, order='F' if fortran else 'C')
            require(bool(np.isfinite(value).all()), 'nonfinite capture')
            result[member.filename[:-4]] = value
    return result


def oracle(package):
    root = Path(package).resolve() / 'oracle'
    previous = {n: sys.modules.pop(n) for n in MODULES if n in sys.modules}
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / 'models/meshgraphnet'))
    try:
        modules = {n: importlib.import_module(n) for n in MODULES}
        require(all(Path(m.__file__).resolve() == root / 'models/meshgraphnet' / (n+'.py') for n,m in modules.items()), 'oracle import isolation')
        return modules
    finally:
        sys.dont_write_bytecode = previous_bytecode
        sys.path.pop(0)
        for n in MODULES:
            sys.modules.pop(n, None)
        sys.modules.update(previous)


def audit_capture(package, case_name, capture_dir, request):
    from mesh_worker import check_abi, validate_request
    package, capture = Path(package).resolve(), Path(capture_dir).resolve()
    manifest = verify_package(package)
    require(case_name in manifest['cases'], 'case exists')
    case = package / 'cases' / case_name
    m, execution = read_json(case/'case.json'), read_json(capture/'execution.json')
    validate_request(request, request['backend'], (case/'case.json').read_bytes(), Path(__file__).with_name('mesh_worker.py').read_bytes())
    require(request['case_name'] == case_name, 'request case')
    require(request['case_sha256'] == digest(case/'case.json') and
            request['package_manifest_sha256'] == digest(package/'manifest.json'), 'request package identity')
    require(request['backend'] in ('hardware','appliance-sim','local-sim'), 'request backend')
    require(execution['status'] == 'COMPLETED' and execution['raw_sha256'] == digest(capture/'raw.npz'), 'completed raw')
    for field in ('backend','run_id','case_sha256','package_manifest_sha256','artifact_sha256','worker_sha256'):
        require(execution.get(field) == request[field], 'capture binding: '+field)
    # The controller preserves the exact bytes staged as request.json.
    require(execution['request_sha256'] == digest(capture.parent/'request.json'), 'request byte identity')
    require(read_json(capture.parent/'request.json') == request, 'request contents')
    require(execution['worker_sha256'] == digest(Path(__file__).with_name('mesh_worker.py')), 'worker identity')
    check_abi(m, read_json(capture/'out.json'), read_json(capture/'out_rpc.json'))
    require(execution['compiled_sha256'] == {n:digest(capture/n) for n in ('out.json','out_rpc.json')}, 'compiled ABI identity')
    modules = oracle(package)
    inputs, graph, maps, raw = (arrays(path) for path in (case/'inputs.npz',case/'original.npz',case/'mapping.npz',capture/'raw.npz'))
    p = m['params']
    require(0 < len(graph['velocity']) <= 2000 and len(graph['senders']) <= 12000, 'graph work bound')
    if p['MODE']:
        modules['selective_validation'].validate_schedule(graph,maps,inputs,p)
    if p['MODE'] == 2:
        modules['colored_validation'].validate_rounds(inputs,p)
    checkpoints = modules['resident_validation'].validate_capture(m,inputs,raw,execution)
    reports = []
    for checkpoint in checkpoints:
        report = modules['driver'].audit_checkpoint(p,inputs,graph,maps,checkpoint)
        if p['MODE']:
            report['communication'] = modules['selective_validation'].validate_comm(p,inputs,checkpoint['raw'],checkpoint['steps'],checkpoint['cumulative_steps'])
        reports.append(report)
    return dict(status='PASS', backend=request['backend'], case_name=case_name,
                run_id=request['run_id'], case_sha256=request['case_sha256'],
                raw_sha256=execution['raw_sha256'], params=p, checkpoints=reports,
                runtime_wall_seconds=execution['wall_seconds'],
                scope='Full numerical/lifecycle/communication audit; per-PE local timestamp intervals, not global makespan; no ELF SRAM claim')



def compare_runs(package, records):
    """Compare independently audited runs; no SDK call and no implicit speed claim."""
    if len(records) < 2:
        return dict(status='NOT_COMPARABLE', comparisons=[], reason='fewer than two passing full runs')
    package = Path(package).resolve()
    verify_package(package)
    ordered = sorted(records, key=lambda item: (item['case_name'] != 'full-broadcast', item['run_id']))
    baseline = ordered[0]
    left_case = package/'cases'/baseline['case_name']
    left = read_json(left_case/'case.json')
    left_inputs = arrays(left_case/'inputs.npz')
    left_raw = arrays(Path(baseline['capture_dir'])/'raw.npz')
    fields = ('velocity','h','e','aggregate','cached_e0','sender_cache')
    results = []
    for record in ordered[1:]:
        case = package/'cases'/record['case_name']
        right = read_json(case/'case.json')
        require(baseline['report']['status'] == record['report']['status'] == 'PASS', 'prior individual audit required')
        require(baseline['report']['backend'] == record['report']['backend'], 'comparison backend')
        for key in ('residency','partition','model','graph_scope','data_provenance'):
            require(left[key] == right[key], 'matched comparison '+key)
        ignored = {'MODE','T','K','Q'}
        require({k:v for k,v in left['params'].items() if k not in ignored} ==
                {k:v for k,v in right['params'].items() if k not in ignored}, 'matched arithmetic parameters')
        for name in ('original.npz','mapping.npz','source/pe.csl','source/layout.csl','source/sparse_halo.csl','source/no_halo.csl'):
            require(left['files'][name] == right['files'][name], 'matched graph/source: '+name)
        right_inputs = arrays(case/'inputs.npz')
        for name in left_inputs.keys() & right_inputs.keys():
            np.testing.assert_array_equal(left_inputs[name],right_inputs[name],err_msg='matched input '+name)
        right_raw = arrays(Path(record['capture_dir'])/'raw.npz')
        for index in range(len(left['residency']['chunks'])):
            for field in fields:
                name = 'chunk%d_%s' % (index,field)
                np.testing.assert_array_equal(left_raw[name],right_raw[name],err_msg='matched output '+name)
        a = [x['local_cycles_max'] for x in baseline['report']['checkpoints']]
        b = [x['local_cycles_max'] for x in record['report']['checkpoints']]
        results.append(dict(baseline_run_id=baseline['run_id'],run_id=record['run_id'],
                            baseline_case=baseline['case_name'],case_name=record['case_name'],
                            bit_exact_fields=list(fields),baseline_local_maxima=a,local_maxima=b,
                            local_interval_ratio=[x/y if y else None for x,y in zip(a,b)],
                            worker_wall_seconds=[baseline['report']['runtime_wall_seconds'],record['report']['runtime_wall_seconds']]))
    return dict(status='PASS',comparisons=results,
                complete_transport_set={'full-broadcast','full-selective','full-colored'} <= {r['case_name'] for r in records},
                scope='Matched six-array equality of independently audited captures; PE-local interval ratios, not global makespan; worker wall includes host I/O')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path('mesh_cases'))
    parser.add_argument('--case')
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--request', type=Path)
    args = parser.parse_args()
    result = (audit_capture(args.package,args.case,args.capture,read_json(args.request)) if args.capture
              else dict(status='PASS', cases=sorted(verify_package(args.package)['cases'])))
    print(json.dumps(result, indent=2, sort_keys=True))
