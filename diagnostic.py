#!/usr/bin/env python3
"""Bounded diagnostics for the two-PE transfer; original run.py is unchanged."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import run as baseline

VARIANTS = {'control': 0, 'events': 1, 'payload': 2}
MARKER = 0xfefedcba
UNUSED = 0xcafebabe
GUARDS = [0x12345678, 0x23456789, 0x6789abcd, 0x789abcde]
RX_SLOTS = 64
VARS = dict(baseline.VARS, diag_state=('u32', 8), rx_payload=('u32', 68),
            diag_events=('u32', 128))


def expected_stream(panel, off, n, mode, packing):
    """Two whole-stream representations; mode changes storage, not payload."""
    baseline.need(packing in ('scalar16', 'packed16'), 'packing')
    baseline.need(mode in (0, 1) and 0 <= off <= 31 and 0 <= n <= 16 and off+n <= 32, 'case')
    rows, values = baseline.inputs(panel)
    r = [int(x) for x in rows[0, 0, off:off+n]]
    if packing == 'packed16':
        r = [r[i] | ((r[i+1] if i+1 < n else 0) << 16) for i in range(0, n, 2)]
    # Zero length skips rows and values, but still sends the header and marker.
    stream = [n] + r + [int(x) for x in values[0, 0, off:off+n].view(np.uint32)]
    baseline.need(MARKER not in stream, 'marker collision')
    return np.array(stream + [MARKER], np.uint32)


def decode_events(events, state):
    """Return local PE timelines; queue reads and time reads are sequential."""
    events = np.asarray(events).reshape(2, 16, 8)
    state = np.asarray(state).reshape(2, 8)
    output = []
    for pe in range(2):
        count = int(state[pe, 4])
        baseline.need(0 <= count <= 16 and state[pe, 5] == 0, 'event count/overflow')
        records = events[pe, :count]
        baseline.need(not events[pe, count:].any(), 'event unused tail')
        baseline.need(not records[:, 7].any(), 'event reserved word')
        baseline.need(not (records[:, 2:7] > 65535).any(), 'event word bounds')
        start, previous, timeline = None, None, []
        for event in records:
            stamp = int(event[2]) | (int(event[3]) << 16) | (int(event[4]) << 32)
            if start is None:
                start = stamp
            delta = 0 if previous is None else (stamp-previous) & ((1 << 48)-1)
            baseline.need(delta < (1 << 47), 'event timestamp order')
            timeline.append(dict(event=int(event[0]), phase=int(event[1]), timestamp=stamp,
                                 cycles_from_start=(stamp-start) & ((1 << 48)-1),
                                 cycles_from_previous=delta, iq_flags=int(event[5]), oq_flags=int(event[6])))
            previous = stamp
        output.append(timeline)
    return output


def analyze_diagnostics(observed, panel, off, n, mode, variant):
    """Check original arrays, bounded storage, completion and complete stream."""
    baseline.need(variant in VARIANTS, 'diagnostic variant')
    errors, details = {}, {'variant': variant}
    want = baseline.expected(panel, off, n, mode)
    if variant == 'payload':
        empty = baseline.expected(panel, off, 0, mode)
        for key in ('received_rows', 'received_values'):
            want[key] = empty[key]
        want['status'][0, 0, 6] += 1  # marker callback
        want['status'][0, 1, 7] = 0  # data task replaces native receive callbacks
    for key, (dtype, size) in VARS.items():
        got = observed[key]
        expected_type = np.uint16 if dtype == 'u16' else np.float32 if dtype == 'f32' else np.uint32
        baseline.need(got.shape == (1, 2, size) and got.dtype == expected_type, 'array shape/type: '+key)
        if key in want:
            view = np.uint16 if dtype == 'u16' else np.uint32
            bad = np.argwhere(got.view(view) != want[key].view(view))
            if len(bad):
                errors[key] = dict(count=len(bad), indices=bad.tolist())
    state = observed['diag_state'][0]
    payload = observed['rx_payload'][0]
    events = observed['diag_events'][0].reshape(2, 16, 8)
    state_want = np.zeros((2, 8), np.uint32)
    state_want[:, 0] = VARIANTS[variant]
    state_want[:, 6] = 1
    payload_want = np.full((2, 68), UNUSED, np.uint32)
    payload_want[:, [0, 1, 66, 67]] = GUARDS
    if variant == 'payload':
        count = int(state[1, 1])
        state_want[1, 1] = count
        state_want[1, 3] = 1
        if not 2 <= count <= RX_SLOTS:
            errors['payload_count'] = {'observed': count}
        else:
            words = payload[1, 2:2+count]
            payload_want[1, 2:2+count] = words
            candidates = [p for p in ('scalar16', 'packed16')
                          if baseline.bits_equal(words, expected_stream(panel, off, n, mode, p))]
            details.update(payload_words=[f'0x{int(x):08x}' for x in words],
                           matching_representations=candidates,
                           expected_streams={p: [f'0x{int(x):08x}' for x in expected_stream(panel, off, n, mode, p)]
                                             for p in ('scalar16', 'packed16')})
            if not candidates:
                errors['payload_stream'] = {'matching_representations': []}
            if list(words).count(MARKER) != 1 or int(words[-1]) != MARKER:
                errors['payload_marker'] = {'observed': [int(x) for x in words]}
    if variant == 'events':
        ids = [0, 1, 2, 3, 4] if n else [0, 1, 4]
        phases = [1, 1, 2, 3, 4] if n else [1, 1, 4]
        state_want[:, 4] = len(ids)
        if not np.array_equal(events[:, :len(ids), 0], np.tile(ids, (2, 1))):
            errors['event_sequence'] = {'expected': ids}
        if not np.array_equal(events[:, :len(ids), 1], np.tile(phases, (2, 1))):
            errors['event_phases'] = {'expected': phases}
        try:
            details['events'] = decode_events(events, state)
        except ValueError as error:
            errors['event_record'] = str(error)
    elif events.any():
        errors['unexpected_events'] = True
    for key, got, expected in (('diag_state', state, state_want), ('payload_storage', payload, payload_want)):
        bad = np.argwhere(got != expected)
        if len(bad):
            errors[key] = {'count': len(bad), 'indices': bad.tolist()}
    return errors, details


def run_case(args):
    off, n, mode = map(int, args.case.split(':'))
    for packing in ('scalar16', 'packed16'):
        expected_stream(0, off, n, mode, packing)
        expected_stream(1, off, n, mode, packing)
    build, out = args.name.resolve(), args.out.absolute()
    baseline.need(out == out.resolve() and not out.exists(), 'fresh resolved capture directory')
    out.mkdir(parents=True)
    options = baseline.runtime_options(args.backend, args.cmaddr)
    result = dict(status='incomplete', variant=args.variant,
                  config=dict(offset=off, length=n, mode=mode), panels=[], backend=args.backend,
                  host_sha256=baseline.sha(__file__), baseline_host_sha256=baseline.sha(baseline.__file__),
                  runtime_image=os.environ.get('SINGULARITY_CONTAINER') or os.environ.get('APPTAINER_CONTAINER'),
                  simfab_numthreads=options.get('simfab_numthreads'), suppress_simfab_trace=options.get('suppress_simfab_trace'),
                  identity_scope='external compiler/runtime must be recorded by launcher operator')
    baseline.write(out/'capture.json', result)
    io, old, start = None, Path.cwd(), time.monotonic()
    try:
        os.chdir(out)
        from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
        runtime = SdkRuntime(str(build), **options)
        io = baseline.IO(runtime, out, SimpleNamespace(pe_rows=1, pe_cols=2), {}, (MemcpyDataType, MemcpyOrder))
        io.symbols = {key: runtime.get_id(key) for key in VARS}
        io.call('load', runtime.load)
        io.call('run', runtime.run)
        for panel in range(3):
            io.stage = f'panel{panel}'
            rows, values = baseline.inputs(panel)
            io.put('source_rows', rows, word_bits=16)
            io.put('source_values', values)
            baseline.need(baseline.bits_equal(io.get('source_rows', 32, 'u16'), rows), 'source rows readback')
            baseline.need(baseline.bits_equal(io.get('source_values', 32), values), 'source values readback')
            io.launch('configure', np.int16(off), np.int16(n), np.int16(mode))
            io.launch('run_probe')
            observed = {key: io.get(key, size, dtype) for key, (dtype, size) in VARS.items()}
            errors, details = analyze_diagnostics(observed, panel, off, n, mode, args.variant)
            result['panels'].append(dict(panel=panel, errors=errors, diagnostic=details))
            baseline.write(out/'capture.json', result)
            if errors:
                break
        io.stage = 'teardown'
        io.call('stop', runtime.stop)
        if any(p['errors'] for p in result['panels']):
            result['status'] = ('observed_payload_difference'
                                if args.variant == 'payload' and set(result['panels'][-1]['errors']) == {'payload_stream'}
                                else 'observed_numeric_fail')
        else:
            result['status'] = 'pass'
        io = None
    except BaseException as error:
        result.update(status='runtime_error', error=repr(error))
        raise
    finally:
        if io is not None:
            io.stage = 'failed-teardown'
            try:
                io.call('stop', io.runtime.stop)
            except Exception as error:
                result['stop_error'] = repr(error)
        os.chdir(old)
        result['wall_seconds'] = time.monotonic()-start
        for name in ('calls.json', 'raw_device.npz'):
            if (out/name).is_file():
                result[name+'_sha256'] = baseline.sha(out/name)
        baseline.write(out/'capture.json', result)
    print(__import__('json').dumps({'variant': args.variant, 'config': result['config'], 'status': result['status']}), flush=True)
    return 0 if result['status'] == 'pass' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--name', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--backend', choices=('simulator', 'hardware'), default='simulator')
    p.add_argument('--cmaddr')
    return run_case(p.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
