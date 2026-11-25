"""Execute both GraphSAGE hub assignments in one loaded CSL executable with SDK memcpy."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')


def execute(build, casefile, out, cmaddr):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

    doc = json.loads(casefile.read_text())
    params = doc['params']
    cols, rows = params['C'], params['R']
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    report = dict(status='running', backend='hardware' if cmaddr else 'simulator',
                  case_sha256=sha(casefile), startup_count=0, modes={})
    save(out / 'capture.json', report)
    options = dict(cmaddr=cmaddr) if cmaddr else dict(simfab_numthreads=1, suppress_simfab_trace=True)
    runtime = SdkRuntime(str(build), **options)
    symbols = {}
    try:
        runtime.load()
        runtime.run()
        report['startup_count'] += 1
        for key in [*doc['inputs'], *doc['outputs']]:
            symbols[key] = runtime.get_id(key)

        def kwargs(dtype):
            return dict(streaming=False, nonblock=False, order=MemcpyOrder.ROW_MAJOR,
                        data_type=(MemcpyDataType.MEMCPY_16BIT if dtype == 'uint16'
                                   else MemcpyDataType.MEMCPY_32BIT))

        def upload(key, array):
            dtype = str(array.dtype)
            count, declared = doc['inputs'][key]
            if [count, declared] != [array.shape[1], dtype] or len(array) != rows*cols:
                raise ValueError('unexpected type or shape for ' + key)
            values = array.astype(np.uint32) if dtype == 'uint16' else array
            runtime.memcpy_h2d(symbols[key], values.ravel(), 0, 0, cols, rows, count, **kwargs(dtype))

        def read(key):
            count, dtype = (doc['inputs'] if key in doc['inputs'] else doc['outputs'])[key]
            values = np.full((rows*cols, count), np.nan if dtype == 'float32' else 0xDEAD,
                             dtype=np.float32 if dtype == 'float32' else np.uint32)
            runtime.memcpy_d2h(values.ravel(), symbols[key], 0, 0, cols, rows, count, **kwargs(dtype))
            return values.astype(dtype, copy=False)

        modes = doc.get('mode_order', ['control', 'split'])
        if len(modes) < 2 or len(set(modes)) != len(modes) or set(modes) != set(doc['modes']):
            raise ValueError('expected distinct independently pinned input modes')
        for mode in modes:
            file = casefile.parent / doc['modes'][mode].get('file', mode + '.npz')
            if sha(file) != doc['modes'][mode]['inputs_sha256']:
                raise ValueError('changed packed inputs for ' + mode)
            with np.load(file, allow_pickle=False) as packed:
                arrays = dict(packed)
            if set(arrays) != set(doc['inputs']):
                raise ValueError('incomplete staged inputs')
            for key, values in arrays.items():
                upload(key, values)
            initial = {}
            initial_words = {}
            for key, values in arrays.items():
                actual = read(key)
                np.testing.assert_array_equal(actual.view('u1'), values.view('u1'), err_msg=mode+' '+key)
                initial[key] = digest(actual)
                if doc.get('retain_input_words'):
                    initial_words[key] = actual
            mode_report = dict(initial=initial, frames=[])
            if initial_words:
                initial_file = out / f'{mode}-initial.npz'
                np.savez_compressed(initial_file, **initial_words)
                mode_report['initial_file_sha256'] = sha(initial_file)
            report['modes'][mode] = mode_report
            save(out / 'capture.json', report)
            for frame, scale in enumerate((1., 0.5, 1.), start=1):
                feature = arrays['x_val'] * np.float32(scale)
                if frame != 1:
                    upload('x_val', feature)
                np.testing.assert_array_equal(read('x_val').view('u1'), feature.view('u1'))
                runtime.launch('f_run', nonblock=False)
                raw = {key: read(key) for key in doc['outputs']}
                np.testing.assert_array_equal(raw['stats'][:, 0], 0x600D)
                final = {}
                for key, wanted in dict(arrays, x_val=feature).items():
                    actual = read(key)
                    if key == 'packet_template':
                        raw[key] = actual
                    else:
                        np.testing.assert_array_equal(actual.view('u1'), wanted.view('u1'), err_msg=mode+' '+key)
                    if doc.get('retain_input_words'):
                        raw['after_'+key] = actual
                    final[key] = digest(actual)
                file = out / f'{mode}-{frame}.npz'
                np.savez_compressed(file, **raw)
                mode_report['frames'].append(dict(frame=frame, scale=scale, final=final, raw_sha256=sha(file)))
                save(out / 'capture.json', report)
                print(mode, frame, file.name, flush=True)
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        try:
            runtime.stop()
        except BaseException as error:
            report.update(status='failed', stop_error=repr(error))
            raise
        finally:
            save(out / 'capture.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('build', type=Path)
    p.add_argument('case', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--cmaddr')
    a = p.parse_args()
    execute(a.build.resolve(), a.case.resolve(), a.out.resolve(), a.cmaddr)
