"""Portable identical-input AH comparison through the native compact CSR route.

Only preparation imports the repository's native packing modules. Export this
file as driver.py alongside oracle.py, proof.py and csr_compare's CSL sources.
Capture and saved audit then need no repository imports or saved FP64 answers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
try:
    from . import oracle, proof
except ImportError:
    import oracle
    import proof

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ATOL, RTOL = 2e-5, 2e-4
FILES = ('driver.py', 'oracle.py', 'proof.py', 'layout.csl', 'pe.csl')
INPUTS = ('x_ptr', 'x_col', 'x_val', 'w0', 'w1', 'g_ptr', 'g_src',
          'g_val', 'slots', 'sender_slots', 'source_lengths', 'ah_input')
PARAMS = {'C', 'R', 'K', 'L', 'FW', 'H', 'O', 'T', 'MX', 'MG',
          'COMPACT', 'U', 'AH_WIDTH'}


def sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 << 20:
        raise ValueError('bounded regular file required: ' + str(path))
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x:
        x.item() if isinstance(x, np.generic) else x.tolist()) + '\n')


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def load_npz(path):
    sha(path)
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def source_paths():
    """Logical export names are independent of the local comparison directory."""
    if Path(__file__).name == 'csr_driver.py':
        return {name: (Path(__file__) if name == 'driver.py' else
                      HERE / 'csr_compare' / name if name.endswith('.csl') else
                      HERE / name) for name in FILES}
    return {name: HERE / name for name in FILES}


def identities():
    return {name: sha(path) for name, path in source_paths().items()}


def prepare(out, kind, nodes, rows, columns=2, shah_rows=2, width=16, tile=16):
    try:
        from . import csr_compare
    except ImportError:
        import csr_compare
    if width not in (7, 16):
        raise ValueError('comparison width must be 7 or 16')
    original = oracle.load_original(kind=kind, nodes=nodes)
    projected = oracle.project_inputs(original, R=shah_rows)['ah' + str(width)]['h']
    packed = csr_compare.prepare_case(original, projected, columns=columns,
                                      rows=rows, tile=tile)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'original.npz', **{k: v for k, v in original.items()
                                               if isinstance(v, np.ndarray)})
    np.savez_compressed(out / 'inputs.npz', **packed['arrays'])
    np.savez_compressed(out / 'node_ids.npz', node_ids=packed['node_ids'])
    np.savez_compressed(out / 'projected.npz', h=projected)
    write(out / 'original.json', {k: v for k, v in original.items()
                                  if not isinstance(v, np.ndarray)})
    (out / 'source').mkdir()
    for name, path in source_paths().items():
        shutil.copyfile(path, out / 'source' / name)
    raw_paths = []
    if kind == 'cora':
        for name in [*oracle.PINS,
                     'research/experiments/data/E087/captures/checkpoint-original.json']:
            target = out / 'raw' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
            raw_paths.append('raw/' + name)
    names = ['original.npz', 'original.json', 'inputs.npz', 'node_ids.npz',
             'projected.npz', *raw_paths]
    manifest = dict(schema=1, implementation='native-compact-csr-ah', kind=kind,
                    nodes=int(original['x'].shape[0]), shah_rows=shah_rows,
                    params=packed['params'], metadata=packed['metrics'],
                    preparation_source_sha256=packed['source_sha256'],
                    source_sha256=identities(), files={n: sha(out / n) for n in names})
    write(out / 'case.json', manifest)
    print(json.dumps(dict(status='PREPARED', params=manifest['params'],
                          metadata=manifest['metadata'])))
    return manifest


def load_case(case):
    case = Path(case)
    m = read(case / 'case.json')
    if m.get('implementation') != 'native-compact-csr-ah':
        raise ValueError('CSR case implementation')
    for name, digest in m['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('case relative path')
        if sha(case / name) != digest:
            raise ValueError('case drift ' + name)
    if set(m['source_sha256']) != set(FILES):
        raise ValueError('source inventory')
    for name, path in source_paths().items():
        digest = m['source_sha256'][name]
        if sha(case / 'source' / name) != digest or sha(path) != digest:
            raise ValueError('source drift ' + name)
    original = {**read(case / 'original.json'), **load_npz(case / 'original.npz')}
    return m, original, load_npz(case / 'inputs.npz')


def compiler_flags(p, hardware=False, channels=1):
    dims = '762,1172' if hardware else f"{p['C'] + 7},{p['R'] + 2}"
    return ['--arch=wse3', '--fabric-dims=' + dims, '--fabric-offsets=4,1',
            '--params=' + ','.join(f'{k}:{v}' for k, v in p.items()), '--memcpy',
            '--channels=' + str(channels), '--max-parallelism=1', '-o', 'out']


def inventory(build):
    result = proof.compiled_inventory(build)
    proof.admit_compiled_inventory(result)
    return result


def compile_case(case):
    m, _, _ = load_case(case)
    if (case / 'build').exists():
        raise ValueError('fresh build required')
    flags = compiler_flags(m['params'])
    flags[-1] = str(case / 'build')
    command = [str(ROOT / 'cslc'), str(case / 'source/layout.csl'), *flags]
    record = dict(status='FAILED', command=command, case_sha256=sha(case / 'case.json'))
    write(case / 'compile-request.json', record)
    start = time.monotonic()
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        record.update(inventory=inventory(case / 'build'), status='PASS')
    finally:
        record['seconds'] = time.monotonic() - start
        write(case / 'compile.json', record)
    print(json.dumps(dict(status=record['status'], seconds=record['seconds'],
                          max_static_sram_end=record['inventory']['max_static_sram_end'])))


def _equal(actual, expected, name):
    """Exact dtype, shape and bits, including float signed zero and padding."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.dtype != expected.dtype or actual.shape != expected.shape or
            actual.tobytes(order='C') != expected.tobytes(order='C')):
        raise AssertionError('exact input mismatch: ' + name)


def _packed_h(h, ids, p):
    result = np.zeros((*ids.shape, p['H']), np.float32)
    live = ids >= 0
    result[:, :, :h.shape[1]][live] = h[ids[live]]
    return result.reshape(p['C'] * p['R'], p['L'] * p['H'])


def validate_inputs(case, m, original, arrays):
    """Independently decode ownership, CSR occurrences and compact transport.

    This establishes the saved placement's mathematical input. The native
    partition heuristic is preparation provenance, not rerun by this audit.
    """
    p = m['params']
    if set(p) != PARAMS or any(type(v) is not int or not 1 <= v <= 65535 for v in p.values()):
        raise AssertionError('native parameter schema')
    c, r, k, l, fw, h, o, tile, width = (p[n] for n in
        ('C', 'R', 'K', 'L', 'FW', 'H', 'O', 'T', 'AH_WIDTH'))
    if c < 2 or r < 2 or h != 16 or o != 7 or width not in (7, 16) or tile not in (4, 8, 16) or p['COMPACT'] != 1:
        raise AssertionError('native comparison topology')
    canonical = oracle.load_original(kind=m['kind'], nodes=m['nodes'], root=case / 'raw')
    for name in ('x', 'src', 'dst', 'coeff', 'w0', 'w1', 'labels', 'valid'):
        if canonical.get(name) is None:
            if original.get(name) is not None:
                raise AssertionError('canonical optional input ' + name)
        else:
            _equal(original[name], canonical[name], 'original/' + name)
    if original.get('full') != canonical.get('full'):
        raise AssertionError('full input scope')
    projected = oracle.project_inputs(canonical, R=m['shah_rows'])['ah' + str(width)]['h']
    _equal(load_npz(case / 'projected.npz')['h'], projected, 'common projected H')
    ids = load_npz(case / 'node_ids.npz')['node_ids']
    n, fin = canonical['x'].shape
    if ids.dtype != np.int64 or ids.shape != (c*r, l) or np.any(ids < -1):
        raise AssertionError('node ownership shape')
    valid = ids >= 0
    if not np.array_equal(np.sort(ids[valid]), np.arange(n)):
        raise AssertionError('every original node has exactly one owner')
    owner = np.empty(n, np.int64)
    for pe in range(c*r):
        nodes = ids[pe, ids[pe] >= 0]
        if not np.array_equal(ids[pe], np.pad(np.sort(nodes), (0, l-len(nodes)), constant_values=-1)):
            raise AssertionError('native sorted owner slots')
        owner[nodes] = pe
    groups = [np.flatnonzero(owner // c == group) for group in range(r)]
    vertices = [np.flatnonzero(owner == pe) for pe in range(c*r)]
    if k != max(1, max(map(len, groups))) or l != max(1, max(map(len, vertices))) or fw != (fin+c-1)//c:
        raise AssertionError('native geometry extents')
    group_slot = np.empty(n, np.int64)
    local_slot = np.empty(n, np.int64)
    for nodes in groups:
        group_slot[nodes] = np.arange(len(nodes))
    for nodes in vertices:
        local_slot[nodes] = np.arange(len(nodes))
    extents = dict(x_ptr=k+1, x_col=p['MX'], x_val=p['MX'], w0=fw*h, w1=h*o,
                   g_ptr=r*l+1, g_src=p['MG'], g_val=p['MG'], slots=l,
                   sender_slots=k, source_lengths=r, ah_input=l*h)
    floats = {'x_val', 'w0', 'w1', 'g_val', 'ah_input'}
    if set(arrays) != set(INPUTS):
        raise AssertionError('exact native input inventory')
    expected = {name: np.zeros((c*r, extent), np.float32 if name in floats else np.uint16)
                for name, extent in extents.items()}
    expected['slots'].fill(65535)
    # Directory is the sorted source-slot union needed by remote owners in
    # each physical column. Zero-valued and duplicate occurrences remain.
    needed = [[set() for _ in range(r)] for _ in range(c)]
    graph_rows = [[[] for _ in range(r*l)] for _ in range(c*r)]
    for source, destination, coeff in zip(canonical['src'], canonical['dst'], canonical['coeff']):
        pe, group = int(owner[destination]), int(owner[source] // c)
        slot = int(group_slot[source])
        graph_rows[pe][group*l + int(local_slot[destination])].append((slot, coeff))
        if pe // c != group:
            needed[pe % c][group].add(slot)
    remap = {}
    for column in range(c):
        for group in range(r):
            slots = sorted(needed[column][group])
            expected['sender_slots'][group*c+column, :len(slots)] = slots
            expected['source_lengths'][column::c, group] = len(slots)
            remap[column, group] = {slot: index for index, slot in enumerate(slots)}
    if p['U'] != max(1, max(map(len, (slots for column in needed for slots in column)))):
        raise AssertionError('compact receive extent')
    x_counts, graph_counts = [], []
    for pe in range(c*r):
        column, group = pe % c, pe // c
        offset = 0
        for slot in range(k):
            expected['x_ptr'][pe, slot] = offset
            if slot < len(groups[group]):
                node = groups[group][slot]
                features = np.flatnonzero(canonical['x'][node, column::c])
                end = offset + len(features)
                expected['x_col'][pe, offset:end] = features
                expected['x_val'][pe, offset:end] = canonical['x'][node, column + c*features]
                offset = end
        expected['x_ptr'][pe, k] = offset
        x_counts.append(offset)
        shard = canonical['w0'][column::c]
        expected['w0'][pe, :shard.size] = shard.reshape(-1)
        expected['w1'][pe] = canonical['w1'].reshape(-1)
        expected['slots'][pe, :len(vertices[pe])] = group_slot[vertices[pe]]
        offset = 0
        for row, entries in enumerate(graph_rows[pe]):
            expected['g_ptr'][pe, row] = offset
            source_group = row // l
            for slot, coeff in entries:
                expected['g_src'][pe, offset] = (slot if group == source_group else
                                                 remap[column, source_group][slot])
                expected['g_val'][pe, offset] = coeff
                offset += 1
        expected['g_ptr'][pe, r*l] = offset
        graph_counts.append(offset)
    if p['MX'] != max(1, max(x_counts)) or p['MG'] != max(1, max(graph_counts)):
        raise AssertionError('native sparse capacity')
    expected['ah_input'] = _packed_h(projected, ids, p)
    for name in INPUTS:
        _equal(arrays[name], expected[name], name)
    digest = hashlib.sha256(projected.tobytes(order='C')).hexdigest()
    if m['metadata']['projected_sha256'] != digest:
        raise AssertionError('projected input identity')
    return canonical, projected, ids


def capture(case, build, out, cmaddr=None, mode='ah16', run_id=None):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    case, build, out = Path(case), Path(build), Path(out)
    m, original, arrays = load_case(case)
    p = m['params']
    if mode != 'ah' + str(p['AH_WIDTH']):
        raise ValueError('capture mode must match compiled AH width')
    canonical, projected, ids = validate_inputs(case, m, original, arrays)
    compiled = read(case / 'compile.json')
    actual_inventory = inventory(build)
    if (compiled['status'] != 'PASS' or compiled['case_sha256'] != sha(case / 'case.json') or
            actual_inventory != compiled['inventory']):
        raise ValueError('actual runtime program differs from compilation')
    run_id = run_id or str(uuid.uuid4())
    if str(uuid.UUID(run_id)) != run_id:
        raise ValueError('run UUID')
    out.mkdir(parents=True, exist_ok=False)
    raw = {}
    record = dict(run_id=run_id, clock_domain=f'pid:{os.getpid()}:monotonic:{run_id}',
                  status='RUNNING', case_sha256=sha(case / 'case.json'), mode=mode,
                  backend='hardware' if cmaddr else 'simulator', api=[],
                  compiled_inventory=actual_inventory)
    write(out / 'execution.json', record)
    rt = None

    def call(label, fn, *args, **kwargs):
        record['pending'] = label
        write(out / 'execution.json', record)
        start = time.monotonic()
        result = fn(*args, **kwargs)
        record['api'].append(dict(api=label, seconds=time.monotonic()-start))
        record.pop('pending')
        write(out / 'execution.json', record)
        return result

    try:
        rt = SdkRuntime(str(build), cmaddr=cmaddr,
                        **({} if cmaddr else dict(suppress_simfab_trace=True, simfab_numthreads=1)))
        call('load', rt.load)
        call('run', rt.run)
        symbols = {name: rt.get_id(name) for name in (*INPUTS, 'pre', 'logits', 'stats', 'clocks')}

        def opts(dtype):
            return dict(streaming=False, nonblock=False, order=MemcpyOrder.ROW_MAJOR,
                        data_type=MemcpyDataType.MEMCPY_16BIT if dtype == np.uint16 else MemcpyDataType.MEMCPY_32BIT)

        def upload(name, values):
            carrier = values.astype(np.uint32) if values.dtype == np.uint16 else values
            call('h2d:' + name, rt.memcpy_h2d, symbols[name], carrier.ravel(),
                 0, 0, p['C'], p['R'], values.shape[-1], **opts(values.dtype))

        def download(name, extent, dtype):
            values = np.empty((p['C']*p['R'], extent), np.uint32 if dtype == np.uint16 else dtype)
            call('d2h:' + name, rt.memcpy_d2h, values.ravel(), symbols[name],
                 0, 0, p['C'], p['R'], extent, **opts(dtype))
            if dtype == np.uint16:
                if np.any(values > 65535):
                    raise AssertionError('nonzero upper u16 bits')
                values = values.astype(np.uint16)
            return values

        for name in INPUTS:
            upload(name, arrays[name])
        for name in INPUTS:
            raw['before_' + name] = download(name, arrays[name].shape[-1], arrays[name].dtype)
            _equal(raw['before_' + name], arrays[name], 'before/' + name)
        for repeat in range(2):
            key = mode + str(repeat)
            if repeat:
                upload('ah_input', _packed_h((projected*np.float32(-.5)).astype(np.float32), ids, p))
            call('launch:' + key, rt.launch, 'f_ah', nonblock=False)
            for name, extent, dtype in (('pre', p['L']*p['H'], np.float32),
                                         ('logits', p['L']*p['O'], np.float32),
                                         ('stats', 8, np.uint32), ('clocks', 6, np.uint16),
                                         ('ah_input', p['L']*p['H'], np.float32)):
                raw[key + '_' + name] = download(name, extent, dtype)
        for name in INPUTS:
            raw['after_' + name] = download(name, arrays[name].shape[-1], arrays[name].dtype)
        record['status'] = 'CAPTURED'
    except BaseException as error:
        record.update(status='FAILED', error=repr(error))
        raise
    finally:
        primary_error = sys.exc_info()[0] is not None
        if raw:
            np.savez_compressed(out / 'raw.npz', **raw)
        try:
            if rt is not None:
                call('stop', rt.stop)
        except BaseException as error:
            record['stop_error'] = repr(error)
            if not primary_error:
                raise
        finally:
            write(out / 'execution.json', record)
    return audit(case, out)


def _schedule(mode):
    names = ['load', 'run'] + ['h2d:' + name for name in INPUTS]
    names += ['d2h:' + name for name in INPUTS]
    outputs = ('pre', 'logits', 'stats', 'clocks', 'ah_input')
    names += ['launch:' + mode + '0'] + ['d2h:' + name for name in outputs]
    names += ['h2d:ah_input', 'launch:' + mode + '1'] + ['d2h:' + name for name in outputs]
    return names + ['d2h:' + name for name in INPUTS] + ['stop']


def audit(case, out):
    case, out = Path(case), Path(out)
    m, original, arrays = load_case(case)
    p = m['params']
    canonical, projected, ids = validate_inputs(case, m, original, arrays)
    raw, record = load_npz(out / 'raw.npz'), read(out / 'execution.json')
    mode = 'ah' + str(p['AH_WIDTH'])
    if record['status'] != 'CAPTURED' or 'stop_error' in record or record.get('pending'):
        raise AssertionError('incomplete execution')
    if record['mode'] != mode or record['backend'] not in ('hardware', 'simulator'):
        raise AssertionError('capture mode/backend')
    if str(uuid.UUID(record['run_id'])) != record['run_id']:
        raise AssertionError('run UUID')
    if [item['api'] for item in record['api']] != _schedule(mode):
        raise AssertionError('complete invocation and transfer schedule')
    if any(not np.isfinite(item['seconds']) or item['seconds'] < 0 for item in record['api']):
        raise AssertionError('host API durations')
    compiled = read(case / 'compile.json')
    if (compiled['status'] != 'PASS' or record['case_sha256'] != sha(case / 'case.json') or
            compiled['case_sha256'] != record['case_sha256'] or
            record['compiled_inventory'] != compiled['inventory']):
        raise AssertionError('capture/compile/case identity')
    proof.admit_compiled_inventory(record['compiled_inventory'])
    observations = {}
    for repeat in range(2):
        key = mode + str(repeat)
        values = projected if repeat == 0 else (projected*np.float32(-.5)).astype(np.float32)
        reference = oracle.ah_reference(canonical, values)
        pre = raw[key + '_pre']
        if pre.dtype != np.float32 or pre.shape != (p['C']*p['R'], p['L']*p['H']) or not np.isfinite(pre).all():
            raise AssertionError('finite native AH output shape')
        view = pre.reshape(*ids.shape, p['H'])
        if np.any(view[ids < 0] != 0) or np.any(view[:, :, p['AH_WIDTH']:] != 0):
            raise AssertionError('nonzero native owner/feature padding')
        actual = np.empty(reference.shape, np.float32)
        actual[ids[ids >= 0]] = view[:, :, :p['AH_WIDTH']][ids >= 0]
        np.testing.assert_allclose(actual, reference, atol=ATOL, rtol=RTOL)
        _equal(raw[key + '_logits'], np.zeros((p['C']*p['R'], p['L']*p['O']), np.float32), key + '/logits')
        _equal(raw[key + '_ah_input'], _packed_h(values, ids, p), key + '/input')
        tiles = (p['AH_WIDTH'] + p['T'] - 1)//p['T']
        stats = np.zeros((p['C']*p['R'], 8), np.uint32)
        stats[:, 0] = 0x600D
        stats[:, 1:3] = tiles
        stats[:, 3] = p['R']*tiles
        stats[:, 5] = arrays['g_ptr'][:, -1].astype(np.uint32)*p['AH_WIDTH']
        stats[:, 7] = tiles
        _equal(raw[key + '_stats'], stats, key + '/stats')
        clocks = raw[key + '_clocks']
        if clocks.dtype != np.uint16 or clocks.shape != (p['C']*p['R'], 6):
            raise AssertionError('native clock shape')
        clocks = clocks.astype(np.uint64)
        start = clocks[:, 0] + (clocks[:, 1] << 16) + (clocks[:, 2] << 32)
        end = clocks[:, 3] + (clocks[:, 4] << 16) + (clocks[:, 5] << 32)
        spans = (end-start) & ((1 << 48)-1)
        if np.any(spans == 0):
            raise AssertionError('empty PE interval')
        observations[key] = dict(max_abs_error=float(np.max(np.abs(actual.astype(np.float64)-reference))),
                                 max_pe_cycles=int(spans.max()),
                                 projected_sha256=hashlib.sha256(values.tobytes(order='C')).hexdigest(),
                                 logical_fmas=len(canonical['src'])*p['AH_WIDTH'],
                                 stats_unique=np.unique(stats, axis=0).tolist())
    changed = _packed_h((projected*np.float32(-.5)).astype(np.float32), ids, p)
    for name in INPUTS:
        _equal(raw['before_' + name], arrays[name], 'before/' + name)
        _equal(raw['after_' + name], changed if name == 'ah_input' else arrays[name], 'after/' + name)
    result = dict(status='PASS', run_id=record['run_id'], case_sha256=sha(case / 'case.json'),
                  backend=record['backend'], observations=observations,
                  timing_scope='same-PE AH entrypoint cycles and host blocking launch windows; no synchronized makespan or application E2E',
                  raw_sha256=sha(out / 'raw.npz'))
    write(out / 'audit.json', result)
    print(json.dumps(result))
    return result


def simulate(case, out, mode):
    m, _, _ = load_case(case)
    compiled = read(case / 'compile.json')
    if (compiled['status'] != 'PASS' or compiled['case_sha256'] != sha(case / 'case.json') or
            inventory(case / 'build') != compiled['inventory']):
        raise ValueError('build identity')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(case / 'build', out / 'build')
    command = [str(ROOT / 'cs_python'), '-B', str(Path(__file__).resolve()), 'capture',
               '--case', str(case), '--build', str(out / 'build'),
               '--out', str(out / 'capture'), '--mode', mode]
    write(out / 'request.json', dict(command=command, mode=mode))
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'compile', 'simulate', 'capture', 'audit'))
    path = lambda value: Path(value).resolve()
    parser.add_argument('--case', type=path, required=True)
    parser.add_argument('--out', type=path)
    parser.add_argument('--build', type=path)
    parser.add_argument('--kind', choices=('tiny', 'cora'), default='tiny')
    parser.add_argument('--nodes', type=int, default=2708)
    parser.add_argument('--rows', type=int, default=2)
    parser.add_argument('--columns', type=int, default=2)
    parser.add_argument('--shah-rows', type=int, default=2)
    parser.add_argument('--width', type=int, choices=(7, 16), default=16)
    parser.add_argument('--tile', type=int, choices=(4, 8, 16), default=16)
    parser.add_argument('--mode', choices=('ah16', 'ah7'))
    parser.add_argument('--cmaddr')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.case, args.kind, args.nodes, args.rows, args.columns,
                args.shah_rows, args.width, args.tile)
    elif args.action == 'compile':
        compile_case(args.case)
    elif args.action == 'audit':
        audit(args.case, args.out)
    else:
        mode = args.mode or 'ah' + str(load_case(args.case)[0]['params']['AH_WIDTH'])
        if args.action == 'simulate':
            simulate(args.case, args.out, mode)
        else:
            capture(args.case, args.build, args.out, args.cmaddr, mode)


if __name__ == '__main__':
    main()
