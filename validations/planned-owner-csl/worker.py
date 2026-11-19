"""Run three input frames through the resident owner-local aggregation kernel."""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def fixture(owners):
    rows = 64 // owners
    ptr = np.zeros((owners, rows + 1), np.uint32)
    src = np.zeros((owners, rows * 4), np.uint32)
    weight = np.zeros(src.shape, np.float32)
    for owner in range(owners):
        for i in range(rows):
            row = owner * rows + i
            e = ptr[owner, i]
            if row % 8:
                src[owner, e:e+4] = [(3*row + s) % 16 for s in (0, 1, 1, 7)]
                weight[owner, e:e+4] = (0.5, -0.25, 0.75, -1.0)
                e += 4
            ptr[owner, i+1] = e
    return ptr, src, weight


def features(q):
    s, f = np.indices((16, 4))
    if q % 3 == 2:
        return (((s*5 + f*13 + 1) % 29 - 14)/4).astype(np.float32)
    return (((s*11 + f*7 + 3) % 31 - 15)/8).astype(np.float32)


def oracle(x):
    # Independently collapse the duplicate source before multiplying in FP64.
    row = np.arange(64)[:, None]
    col = np.arange(16)[None, :]
    adj = 0.5*((3*row) % 16 == col) + 0.5*((3*row + 1) % 16 == col) - ((3*row + 7) % 16 == col)
    adj[::8] = 0
    return (adj @ x.astype(np.float64)).astype(np.float32)


def execute(build, output, cmaddr=None):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    params = json.loads((build/'out.json').read_text())['params']
    owners, mode = int(params['P']), int(params['MODE'])
    ptr, src, weight = fixture(owners)
    if mode == 2:
        src *= 4
    kwargs = dict(streaming=False, nonblock=False, order=MemcpyOrder.ROW_MAJOR)
    u16 = dict(kwargs, data_type=MemcpyDataType.MEMCPY_16BIT)
    w32 = dict(kwargs, data_type=MemcpyDataType.MEMCPY_32BIT)
    options = dict(cmaddr=cmaddr) if cmaddr else dict(simfab_numthreads=1, suppress_simfab_trace=True)
    runtime = SdkRuntime(str(build), **options)
    runtime.load()
    runtime.run()
    trials = []
    setup_cycles = None
    try:
        for name, data, args in [('ptr', ptr, u16), ('src', src, u16), ('weight', weight, w32)]:
            runtime.memcpy_h2d(runtime.get_id(name), data.ravel(), 0, 0, owners, 1, data.shape[1], **args)
        if mode >= 5:
            runtime.launch('prepare', nonblock=False)
            clock = np.empty((owners,6), np.uint32)
            runtime.memcpy_d2h(clock.ravel(), runtime.get_id('clocks'), 0, 0, owners, 1, 6, **u16)
            ticks = clock.astype(np.int64) @ np.array([-1, -(1<<16), -(1<<32), 1, 1<<16, 1<<32])
            setup_cycles = (ticks % (1<<48)).tolist()
            if min(setup_cycles) <= 0:
                raise ValueError('empty directory setup interval')
        for q in range(1, 12+1):
            x = features(q)
            repeated = np.tile(x.ravel(), owners)
            y = np.full(256, np.nan, np.float32)
            clocks = np.empty((owners, 6), np.uint32)
            start = time.perf_counter_ns()
            runtime.memcpy_h2d(runtime.get_id('features'), repeated, 0, 0, owners, 1, 64, **w32)
            runtime.launch('aggregate', nonblock=False)
            runtime.memcpy_d2h(y, runtime.get_id('out'), 0, 0, owners, 1, 256//owners, **w32)
            host_ns = time.perf_counter_ns() - start
            runtime.memcpy_d2h(clocks.ravel(), runtime.get_id('clocks'), 0, 0, owners, 1, 6, **u16)
            ticks = clocks.astype(np.int64) @ np.array([-1, -(1<<16), -(1<<32), 1, 1<<16, 1<<32])
            ticks %= 1<<48
            y = y.reshape((64,4))
            expected = oracle(x)
            if not np.array_equal(expected.view(np.uint32), y.view(np.uint32)) or not np.all(ticks > 0):
                raise ValueError(f'output or timer mismatch on trial {q}')
            trials.append(dict(trial=q, features=x.tolist(), output=y.tolist(), clocks=clocks.tolist(),
                               cycles=ticks.tolist(), host_input_through_output_ns=host_ns))
    finally:
        runtime.stop()
    output.write_text(json.dumps(dict(owners=owners, mode=mode, backend='hardware' if cmaddr else 'local-sim',
                                      setup_cycles=setup_cycles, trials=trials), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cmaddr')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('capture already exists')
    execute(args.build.resolve(), args.output.resolve(), args.cmaddr)
