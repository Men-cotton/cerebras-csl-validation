"""Portable command-clock records. No SDK imports and no job submission."""
import hashlib
import io
import json
import math
from pathlib import Path
import uuid

SCHEMA = 1
MASK48 = (1 << 48) - 1
REPORT_WORDS = 16
BACKENDS = ('local-sim', 'appliance-sim', 'hardware')
DEFAULT_CONFIG = dict(kind='transfer', start='witness', transfer='queued',
                      arm='rectangle', pause_at='none', pause_seconds=0.0,
                      work=1, steps=3)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 128 << 20,
            'bounded regular file: ' + str(path))
    return digest_bytes(path.read_bytes())


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 4 << 20, 'bounded JSON')
    return json.loads(path.read_text())


def write_json(path, data):
    encoded = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n'
    require(len(encoded.encode()) <= 4 << 20, 'JSON bound')
    Path(path).write_text(encoded)


def source_identity(root):
    root = Path(root)
    return dict(source_sha256={n: digest(root/n) for n in ('clock_source/layout.csl', 'clock_source/pe.csl')},
                host_sha256={n: digest(root/n) for n in ('clock_probe.py', 'clock_common.py', 'clock_pmu.py')})


def validate_params(value):
    require(isinstance(value, dict) and set(value) == {'C', 'R', 'W'}, 'parameter keys')
    require(all(type(x) is int for x in value.values()), 'integer parameters')
    require(1 <= value['C'] <= 8 and 1 <= value['R'] <= 8 and 1 <= value['W'] <= 4096, 'parameter bounds')
    return dict(value)


def validate_config(value):
    require(isinstance(value, dict) and not (set(value)-set(DEFAULT_CONFIG)-{'host_pmu'}), 'configuration keys')
    result = dict(DEFAULT_CONFIG, **value)
    require('host_pmu' not in result or type(result['host_pmu']) is bool, 'host_pmu boolean')
    for key, allowed in (
        ('kind', ('empty', 'transfer', 'internal')),
        ('start', ('blocking', 'queued', 'wait', 'witness')),
        ('transfer', ('blocking', 'queued', 'wait')),
        ('arm', ('rectangle', 'strips', 'tiles')),
        ('pause_at', ('none', 'before_start', 'after_start', 'between_h2d', 'before_end', 'after_end_return', 'after_end_witness'))):
        require(result[key] in allowed, 'configuration ' + key)
    require(type(result['work']) is int and 1 <= result['work'] <= 1024, 'work bound')
    require(type(result['steps']) is int and result['steps'] == 3, 'three A/B/A generations')
    delay = result['pause_seconds']
    require(type(delay) in (int, float) and math.isfinite(delay) and 0 <= delay <= 1, 'pause bound')
    result['pause_seconds'] = float(delay)
    require(result['pause_at'] != 'none' or delay == 0, 'none pause must be zero')
    if result['kind'] == 'internal':
        require(result['pause_at'] in ('none', 'before_start', 'after_end_return', 'after_end_witness'), 'internal window contains no host pause')
        require(result['start'] in ('blocking', 'queued', 'wait'), 'internal has no separate start witness')
    if result['pause_at'] == 'between_h2d':
        require(result['kind'] == 'transfer' and result['arm'] in ('strips', 'tiles'), 'between transfers requires multiple pieces')
    return result


def fixture(params, step):
    import numpy as np
    require(step in (0, 1, 2), 'A/B/A index')
    words = np.arange(params['C']*params['R']*params['W'], dtype=np.uint32)
    words = words * np.uint32(2654435761) + np.uint32(0x12345678)
    if step == 1:
        words ^= np.uint32(0xa5a55a5a)
    return words.reshape(params['R'], params['C'], params['W'])


def pieces(value, arm):
    r, c, w = value.shape
    if arm == 'rectangle':
        return [(0, 0, c, r, value.reshape(-1).copy())]
    if arm == 'strips':
        return [(0, y, c, 1, value[y].reshape(-1).copy()) for y in range(r)]
    return [(x, y, 1, 1, value[y, x].copy()) for y in range(r) for x in range(c)]


def u48(limbs):
    require(len(limbs) == 3 and all(type(int(x)) is int and 0 <= int(x) <= 65535 for x in limbs), 'raw low16 clock limbs')
    return int(limbs[0]) | int(limbs[1]) << 16 | int(limbs[2]) << 32


def delta(end, start):
    result = (end-start) & MASK48
    require(0 < result < (1 << 47), 'positive unambiguous clock interval')
    return result


def decode_report(value, params, config, step):
    import numpy as np
    require(value.dtype == np.uint32 and value.shape == (params['R'], params['C'], REPORT_WORDS), 'report ABI')
    expected_kind = int(config['kind'] == 'internal')
    result = []
    for y in range(params['R']):
        row = []
        for x in range(params['C']):
            p = value[y, x]
            require(list(map(int, p[6:11])) == [step+1, step+1, config['work'], expected_kind, params['W']*config['work']], 'generation/work/kind')
            require(int(p[11]) == 0x5a17c10c and int(p[15]) == 0xc10c5a17, 'report guards')
            start, end, work_start = u48(p[:3]), u48(p[3:6]), u48(p[12:15])
            span, active = delta(end, start), delta(end, work_start)
            require(active <= span, 'work is within command interval')
            row.append(dict(start=start, end=end, work_start=work_start,
                            span_cycles=span, work_cycles=active, pre_work_cycles=span-active))
        result.append(row)
    return result


def validate_arrays(raw, params, config):
    import numpy as np
    expected = {f'{n}_{s}' for n in ('data', 'seen', 'report') for s in range(3)}
    if config['kind'] != 'internal' and config['start'] == 'witness':
        expected |= {f'start_{s}' for s in range(3)}
    if config['pause_at'] == 'after_end_witness':
        expected |= {f'end_before_pause_{s}' for s in range(3)}
    require(set(raw) == expected, 'complete raw array inventory')
    steps, prior = [], None
    for step in range(3):
        wanted = fixture(params, step)
        for name in ('data', 'seen'):
            a = raw[f'{name}_{step}']
            require(a.dtype == np.uint32 and a.shape == wanted.shape and np.array_equal(a, wanted), 'full input/snapshot bits: '+name)
        report = raw[f'report_{step}']
        decoded = decode_report(report, params, config, step)
        if f'start_{step}' in raw:
            start = raw[f'start_{step}']
            require(start.dtype == np.uint32 and start.shape == report.shape, 'start witness ABI')
            require(np.all(start[:, :, 6] == step+1) and np.all(start[:, :, 7] == step), 'start witness generation')
            require(np.array_equal(start[:, :, :3], report[:, :, :3]), 'start witness clock identity')
            require(np.all(start[:, :, 11] == 0x5a17c10c) and np.all(start[:, :, 15] == 0xc10c5a17), 'start witness guards')
        if f'end_before_pause_{step}' in raw:
            a = raw[f'end_before_pause_{step}']
            require(a.dtype == report.dtype and a.shape == report.shape and np.array_equal(a, report), 'frozen end record identity')
        if prior:
            for y in range(params['R']):
                for x in range(params['C']):
                    delta(decoded[y][x]['start'], prior[y][x]['end'])
        prior = decoded
        steps.append(dict(step=step, pe=decoded,
                          max_span_cycles=max(p['span_cycles'] for row in decoded for p in row),
                          max_work_cycles=max(p['work_cycles'] for row in decoded for p in row)))
    return steps


def read_raw(data):
    import numpy as np
    import zipfile
    require(len(data) <= 64 << 20, 'NPZ byte bound')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        info = archive.infolist()
        require(len(info) <= 24 and len({i.filename for i in info}) == len(info)
                and sum(i.file_size for i in info) <= 64 << 20, 'NPZ expanded bound')
        require(all('/' not in i.filename and i.filename.endswith('.npy') for i in info), 'NPZ member names')
        for member in info:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1,0),(2,0)), 'NPY version')
                reader = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(stream)
                require(dtype == np.dtype('uint32') and not fortran and len(shape) == 3
                        and all(type(n) is int and n > 0 for n in shape)
                        and shape[0] <= 8 and shape[1] <= 8 and shape[2] <= 4096, 'bounded NPY uint32 shape')
                require(stream.tell() + math.prod(shape)*4 == member.file_size, 'exact NPY payload size')
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def expected_calls(params, config):
    names = ['construct', 'get_id:data', 'get_id:seen', 'get_id:report', 'load', 'run', 'init']
    if config.get('host_pmu', False):
        names.insert(0, 'pmu-empty')
    for step in range(3):
        def pause(where):
            if config['pause_at'] == where:
                names.append(f'pause:{step}:{where}')
        names.append(f'preload:{step}')
        pause('before_start')
        if config['kind'] == 'internal':
            names.append(f'internal:{step}')
            if config['start'] == 'wait':
                names.append(f'internal:{step}:wait')
        else:
            names.append(f'tic:{step}')
            if config['start'] == 'wait':
                names.append(f'tic:{step}:wait')
            if config['start'] == 'witness':
                names.append(f'd2h:{step}:start')
            pause('after_start')
            if config['kind'] == 'transfer':
                count = 1 if config['arm'] == 'rectangle' else params['R'] if config['arm'] == 'strips' else params['R']*params['C']
                for i in range(count):
                    names.append(f'h2d:{step}:{i}')
                    if config['transfer'] == 'wait':
                        names.append(f'h2d:{step}:{i}:wait')
                    if i == 0:
                        pause('between_h2d')
            pause('before_end')
            names.append(f'toc:{step}')
        pause('after_end_return')
        if config['pause_at'] == 'after_end_witness':
            names.append(f'd2h:{step}:end_before_pause')
            pause('after_end_witness')
        names.extend(f'd2h:{step}:{n}' for n in ('report', 'data', 'seen'))
    return names + ['stop']


def validate_calls(calls, params, config):
    require(isinstance(calls, list) and len(calls) <= 1000, 'API record bound')
    require([x['name'] for x in calls] == expected_calls(params, config), 'exact API sequence')
    ids = {event['args'][0]:event.get('result') for event in calls if event['name'].startswith('get_id:')}
    require(set(ids) == {'data','seen','report'} and all(type(v) is int for v in ids.values())
            and len(set(ids.values())) == 3, 'distinct native symbol IDs')
    previous = 0
    for event in calls:
        require(event['status'] == 'pass', 'API completion')
        require(('pmu' in event) == config.get('host_pmu',False), 'PMU instrumentation matches request')
        if 'pmu' in event:
            validate_pmu(event['pmu'])
        require(all(type(event[k]) is int and event[k] >= 0 for k in ('wall_start_ns','wall_end_ns','cpu_start_ns','cpu_end_ns')), 'nonnegative integer nanoseconds')
        require(type(event['wall_start_ns']) is int and event['wall_start_ns'] >= previous
                and event['wall_end_ns'] >= event['wall_start_ns'], 'ordered host wall record')
        require(event['cpu_end_ns'] >= event['cpu_start_ns'], 'nonnegative host CPU record')
        previous = event['wall_end_ns']
        name, kw = event['name'], event['kwargs']
        if name.startswith('get_id:'):
            require(event['args'] == [name.split(':')[1]] and not kw, 'symbol lookup arguments')
        if name in ('construct','load','run','stop'):
            require(event['args'] == [] and not kw, 'lifecycle arguments')
        if name == 'init':
            require(event['args'] == ['init'] and kw == {'nonblock':False}, 'init arguments')
        if name.startswith('pause:'):
            require(event['args'] == [config['pause_seconds']] and not kw, 'pause call')
            require(event['wall_end_ns']-event['wall_start_ns'] >= int(config['pause_seconds']*1e9), 'observed pause duration')
        if ':wait' in name:
            require(len(event['args']) == 1 and not kw, 'task wait call')
            continue
        if name.startswith(('tic:', 'internal:')):
            require(kw == {'nonblock': config['start'] in ('queued', 'wait')}, 'start nonblock')
            step = int(name.split(':')[1])
            require(event['args'] == (['tic',step+1] if name.startswith('tic:') else ['internal',step+1,config['work']]), 'launch arguments')
        if name.startswith('toc:'):
            require(event['args'] == ['toc',config['work']] and kw == {'nonblock':False}, 'end launch')
        if name.startswith(('preload:', 'h2d:', 'd2h:')):
            require(kw['streaming'] is False and kw['nonblock'] == (name.startswith('h2d:') and config['transfer'] != 'blocking'), 'transfer flags')
            require(len(event['args']) == 7, 'transfer argument count')
            import numpy as np
            require(kw == dict(streaming=False, nonblock=name.startswith('h2d:') and config['transfer'] != 'blocking',
                               data_type='MemcpyDataType.MEMCPY_32BIT', order='MemcpyOrder.ROW_MAJOR'), 'exact native32 row-major flags')
            a = event['args']
            step = int(name.split(':')[1])
            wanted = fixture(params, step)
            if name.startswith('d2h:'):
                n = name.split(':')[2]
                symbol = n if n in ('data','seen') else 'report'
                require(a[1] == ids[symbol], 'D2H symbol ID')
                receive = np.full((params['R'],params['C'],params['W'] if symbol != 'report' else REPORT_WORDS),0xffffffff,dtype=np.uint32)
                descriptor = a[0]
                initial = receive.reshape(-1)
                role = f'{n}:{step}'
            else:
                require(a[0] == ids['data'], 'H2D symbol ID')
                descriptor = a[1]
                if name.startswith('preload:'):
                    initial = (np.bitwise_not(wanted) if config['kind'] == 'transfer' else wanted).reshape(-1)
                else:
                    initial = pieces(wanted,config['arm'])[int(name.split(':')[2])][4]
                role = name
            require(descriptor == dict(role=role,dtype='uint32',shape=list(initial.shape),
                                       initial_sha256=digest_bytes(initial.tobytes())), 'complete payload/receive descriptor')
            shape = a[2:]
            if name.startswith('h2d:'):
                i = int(name.split(':')[2])
                if config['arm'] == 'rectangle':
                    expected = [0,0,params['C'],params['R'],params['W']]
                elif config['arm'] == 'strips':
                    expected = [0,i,params['C'],1,params['W']]
                else:
                    expected = [i % params['C'],i // params['C'],1,1,params['W']]
            else:
                count = params['W'] if not name.startswith('d2h:') or name.split(':')[2] in ('data','seen') else REPORT_WORDS
                expected = [0,0,params['C'],params['R'],count]
            require(shape == expected, 'exact transfer rectangle')


def validate_capture_content(meta, calls, raw_bytes, manifest, request):
    for key, value in dict(schema=SCHEMA, backend=manifest['backend'], params=manifest['params'],
                           config=request['config'], run_id=request['run_id'],
                           manifest_sha256=request['manifest_sha256'],
                           source_sha256=manifest['source_sha256'], host_sha256=manifest['host_sha256']).items():
        require(meta.get(key) == value, 'capture identity: '+key)
    params, config = validate_params(meta['params']), validate_config(meta['config'])
    require(digest_bytes(raw_bytes) == meta['raw_sha256'], 'raw integrity')
    require(meta['status'] in ('pass','observed_numeric_fail','incomplete'), 'capture status')
    if meta['status'] == 'incomplete':
        return dict(status='incomplete', numerical_validation='not complete')
    validate_calls(calls, params, config)
    try:
        steps = validate_arrays(read_raw(raw_bytes), params, config)
    except ValueError as error:
        require(meta['status'] == 'observed_numeric_fail', 'claimed pass has invalid raw arrays: '+str(error))
        return dict(status='observed_numeric_fail', error=str(error))
    require(meta['status'] == 'pass' and meta.get('steps') == steps, 'raw-derived status and cycles')
    return dict(status='pass', steps=steps)


def validate_pmu(record):
    require(isinstance(record,dict) and record.get('scope') == 'thread-self-user', 'PMU scope')
    if record.get('status') == 'unavailable':
        require(set(record) == {'status','scope','error'} and isinstance(record['error'],dict), 'unavailable PMU has no counts')
    else:
        require(record.get('status') == 'ok' and record.get('scaled') is False, 'raw PMU status')
        fields = ('cycles','instructions','time_enabled_ns','time_running_ns')
        require(all(type(record.get(n)) is int and record[n] >= 0 for n in fields), 'raw PMU integers')
        require(0 < record['time_running_ns'] <= record['time_enabled_ns'], 'PMU scheduling times')
