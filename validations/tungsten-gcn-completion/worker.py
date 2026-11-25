"""Measure one Direct Link runtime from dispatch to the last PE witness."""
import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np
from wire import expand, wire_words


CALLS = 6


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def witnesses(words, p, epoch):
    pes = p['C'] * p['R']
    expected = np.arange(epoch * pes, (epoch + 1) * pes, dtype='<u4')
    if words.dtype != expected.dtype or not np.array_equal(words, expected):
        raise ValueError(f'incomplete, reordered, or stale PE witnesses for epoch {epoch}')


def diagnostics(words, p, epoch, reference):
    records = expand(words, p)
    pes = p['R'] * p['C']
    xy = np.arange(pes)
    for i, expected in enumerate((np.full(pes, epoch), xy % p['C'], xy // p['C'])):
        np.testing.assert_array_equal(records[:, i], expected)
    ticks = records[:, 3:9].astype(np.int64)
    intervals = (ticks @ np.array([-1, -65536, -(1 << 32), 1, 65536, 1 << 32])) % (1 << 48)
    if np.any(ticks > 65535) or np.any(intervals <= 0) or np.any(intervals >= 8_000_000):
        raise ValueError('invalid per-PE interval')
    offset = 9
    for key in ('stats_history', 'pre_history', 'logits_history', 'panel_history'):
        expected = np.ascontiguousarray(reference[key]).view('<u4').reshape(pes, -1)
        end = offset + expected.shape[1]
        np.testing.assert_array_equal(records[:, offset:end], expected, err_msg=key)
        offset = end
    if offset != records.shape[1]:
        raise ValueError('diagnostic word count mismatch')
    return int(intervals.max())


def run(root, out, artifact, cmaddr):
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkCompileArtifacts, SdkRuntime, SdkTarget, SimfabConfig, get_platform)
    build = json.loads((root/'build.json').read_text())
    role = 'hardware' if cmaddr else 'simulator'
    if sha(artifact/'program.elf') != build['roles'][role]['elf_sha256'] or sha(
            artifact/'out_port_map.json') != build['ports_sha256']:
        raise ValueError('staged firmware differs from build')
    if sha(root/'reference.npz') != build['comparison_reference_sha256']:
        raise ValueError('reference hash mismatch')
    platform = get_platform(cmaddr, SimfabConfig(num_threads=1, suppress_trace=True), SdkTarget.WSE3)
    if bool(platform.is_system()) != bool(cmaddr) or bool(platform.is_simulation()) == bool(cmaddr):
        raise ValueError('runtime platform mismatch')
    image = SdkCompileArtifacts(str(artifact)).add_port_mapping(str(artifact/'out_port_map.json'))
    dev = SdkRuntime(image, platform, memcpy_required=False, msg_level='INFO')
    p = build['params']
    if any(tuple(dev.coord_logical_to_physical((x, y))) != (x, y)
           for y in range(p['R']) for x in range(p['C'])):
        raise ValueError('endpoints and allocation origins differ')
    out.mkdir(parents=True, exist_ok=False)
    record = dict(role=role, runtime_id=str(uuid.uuid4()),
                  build_sha256=sha(root/'build.json'), worker_sha256=sha(Path(__file__)),
                  warmup=1, repeats=5, samples=[], stopped=False)
    def save():
        tmp = out/'capture.tmp'
        tmp.write_text(json.dumps(record, indent=2)+'\n')
        tmp.replace(out/'capture.json')
    save()
    dev.load()
    dev.run()
    try:
        dev.report_port_infos()
        rx, tx = dev.get_port_id('rx_port'), dev.get_port_id('tx_port')
        pes = p['C'] * p['R']
        initial = np.full(pes, 0xdeadbeef, '<u4')
        dev.receive(tx, initial, pes, nonblock=False)
        np.save(out/'initial.npy', initial)
        witnesses(initial, p, 0)
        with np.load(root/'reference.npz', allow_pickle=False) as reference:
            for epoch in range(1, CALLS + 1):
                command = np.array([epoch], '<f4').view('<u4')
                observed = np.full(pes, 0xdeadbeef, '<u4')
                release = np.array([1], '<f4').view('<u4')
                reply = np.full(wire_words(p), 0xdeadbeef, '<u4')
                begin = time.monotonic_ns()
                dev.send(rx, command, 1, nonblock=False)
                dispatched = time.monotonic_ns()
                dev.receive(tx, observed, pes, nonblock=False)
                completed = time.monotonic_ns()
                witnesses(observed, p, epoch)
                # The device waits for this second command before emitting any diagnostics.
                read = dev.receive(tx, reply, len(reply), nonblock=True)
                dev.send(rx, release, 1, nonblock=False)
                dev.task_wait(read)
                np.savez_compressed(out/f'frame-{epoch}.npz', witness=observed, reply=reply)
                peak = diagnostics(reply, p, epoch, reference)
                record['samples'].append(dict(epoch=epoch, start_ns=begin, dispatched_ns=dispatched,
                    completed_ns=completed, peak_pe_cycles=peak))
                save()
    finally:
        dev.stop()
        record['stopped'] = True
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--cmaddr')
    args = parser.parse_args()
    run(args.root.resolve(), args.out.resolve(), args.artifact.resolve(), args.cmaddr)
