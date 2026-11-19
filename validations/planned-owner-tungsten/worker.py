"""Run the paired Direct Link image, returning every owner output and 48-bit clock."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkCompileArtifacts, SdkRuntime, SdkTarget, SimfabConfig, get_platform

from contract import WIDTHS, TRIALS, digest, input_words, verify


def run(args):
    build = json.loads(args.build.read_text())
    artifact = args.artifact.resolve()
    role = 'hardware' if args.cmaddr else 'simulator'
    width = build['width']
    if width not in WIDTHS:
        raise ValueError('unsupported geometry')
    expected = build['roles'][role]
    elf_hash = digest(np.fromfile(artifact/'program.elf', dtype='u1'))
    ports_hash = digest(np.fromfile(artifact/'out_port_map.json', dtype='u1'))
    if elf_hash != expected['elf_sha256'] or ports_hash != build['ports_sha256']:
        raise ValueError('worker artifact pin mismatch')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    source_hashes = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ('worker.py', 'contract.py')}
    capture = dict(worker_source=source_hashes, backend=role, owners=width, elf_sha256=elf_hash, port_map_sha256=ports_hash,
                   source=build['source'], trials=[], stop_success=False, passed=False)
    def checkpoint(stage):
        capture['stage'] = stage
        tmp = out/'result.tmp'
        tmp.write_text(json.dumps(capture,indent=2)+'\n')
        tmp.replace(out/'result.json')
    checkpoint('platform')
    platform = get_platform(args.cmaddr, SimfabConfig(num_threads=1, suppress_trace=True), SdkTarget.WSE3)
    if bool(platform.is_system()) != bool(args.cmaddr) or bool(platform.is_simulation()) == bool(args.cmaddr):
        raise ValueError('runtime platform mismatch')
    capture['system_target'] = bool(platform.is_system())
    capture['simulator_target'] = bool(platform.is_simulation())
    inputs = SdkCompileArtifacts(str(artifact)).add_port_mapping(str(artifact/'out_port_map.json'))
    checkpoint('constructor')
    device = SdkRuntime(inputs, platform, memcpy_required=False, msg_level='INFO')
    capture['coordinates'] = [dict(logical=[x,y],physical=list(device.coord_logical_to_physical((x,y))))
                              for y in range(2) for x in range(width)]
    checkpoint('load')
    device.load()
    checkpoint('run')
    device.run()
    try:
        device.report_port_infos()
        rx, tx = device.get_port_id('rx_port'), device.get_port_id('tx_port')
        if rx is None or tx is None or rx == tx:
            raise ValueError('invalid Direct Link port IDs')
        for trial in TRIALS:
            source = input_words(trial)
            guarded = np.full(4+256+14*width, 0xbad0c0de, dtype=np.uint32)
            target = guarded[2:-2]
            checkpoint(f'transfer-{trial}')
            start = perf_counter_ns()
            receive = device.receive(tx, target, target.size, nonblock=True)
            send = device.send(rx, source, source.size, nonblock=True)
            device.task_wait(send)
            device.task_wait(receive)
            elapsed = perf_counter_ns()-start
            if not np.all(guarded[[0,1,-2,-1]] == 0xbad0c0de):
                raise ValueError('Direct Link output guard overwritten')
            record = dict(trial=trial, input_sha256=digest(source), reply=target.tolist(),
                          send_receive_wait_ns=elapsed)
            capture['trials'].append(record)
            record.update(verify(width, trial, target))
    finally:
        checkpoint('stop')
        device.stop()
        capture['stop_success'] = True
        checkpoint('stopped')
    capture['passed'] = True
    checkpoint('complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cmaddr')
    run(p.parse_args())
