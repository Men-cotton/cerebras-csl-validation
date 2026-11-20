"""Compile two automatic Direct Link pairs on the active SDK platform and return guarded words."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np


MASKS = (0x61a72d90, 0x16c599af)
INPUTS = ((1, 0x80000000, 19, 0xffffffff), (0x40000000, 7, 0, 31))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix('.pending')
    with tmp.open('w') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def ports(mapping):
    result = {}
    for bus in mapping['buses']:
        leg = mapping['lvds_ports'][bus['lvds_id']]
        name = bus['port_name']
        if name in result:
            raise ValueError('duplicate program port')
        result[name] = dict(io_type=bus['io_type'], side=leg['side'], loc=leg['loc'], color=leg['color'])
    return result


def run(kernels, out, cmaddr):
    from cerebras.sdk.runtime.sdkruntimepybind import (Color,Edge,Route,RoutingPosition,SdkLayout,
                                                      SdkTarget,SimfabConfig,get_platform)
    kernels, out = kernels.resolve(), out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    desc = dict(backend='hardware' if cmaddr else 'simulator', masks=list(MASKS),
                sources={f'echo{i}.csl':sha(kernels/f'echo{i}.csl') for i in range(2)}, transactions=[])
    def mark(name):
        desc['last_phase']=name
        save(out/'trace.json',desc)
    mark('before-platform')
    platform=get_platform(cmaddr, SimfabConfig(num_threads=1,suppress_trace=True), SdkTarget.WSE3)
    if bool(platform.is_system())!=bool(cmaddr) or bool(platform.is_simulation())==bool(cmaddr):
        raise ValueError('different SDK execution platform')
    # The system compiler obtains the available I/O endpoints from the allocated device.
    layout=SdkLayout(platform)
    names=[]
    for i in range(2):
        region=layout.create_code_region(str(kernels/f'echo{i}.csl'),f'echo{i}',1,1)
        rx,tx=Color(f'rx{i}'),Color(f'tx{i}')
        region.set_param_all(rx)
        region.set_param_all(tx)
        inp=region.create_input_port(rx,Edge.RIGHT,[RoutingPosition().set_output([Route.RAMP])],4)
        ret=region.create_output_port(tx,Edge.LEFT,[RoutingPosition().set_input([Route.RAMP])],4)
        region.place(7,4+4*i)
        names.append([layout.create_input_stream(inp,io_buffer_size=32),layout.create_output_stream(ret,io_buffer_size=32)])
    desc['names']=names
    mark('before-system-layout-compile' if cmaddr else 'before-simulator-layout-compile')
    directory=out/'native'
    directory.mkdir()
    os.chdir(directory)
    artifacts=layout.compile(str(directory/'out'),save_port_map=True)
    target=directory/'out_port_map.json'
    shutil.copyfile(target,out/'port-map.json')
    desc['port_map_sha256']=sha(target)
    desc['endpoints']=ports(json.loads(target.read_bytes()))
    desc['native_elf_sha256']={p.name:sha(p) for p in sorted(directory.glob('*.elf'))}
    if not desc['native_elf_sha256'] or set(desc['endpoints'])!={n for pair in names for n in pair}:
        raise ValueError('incomplete native ELF or generated port map')
    fabric = directory/'fabric.json'
    if fabric.is_file():
        state=json.loads(fabric.read_bytes())
        report=dict(fabric_sha256=sha(fabric),dimensions={k:state[k] for k in ('fabric_dimensions','width','height','dim_X','dim_Y') if k in state},
                    ingress_connections=state.get('ingress_connections',[]),egress_connections=state.get('egress_connections',[]))
        save(out/'fabric-connections.json',report)
        desc['fabric_report_sha256']=sha(out/'fabric-connections.json')
    transfer(artifacts, platform, names, desc, mark)


def transfer(artifacts,platform,names,desc,mark):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
    mark('before-runtime-constructor')
    runtime=SdkRuntime(artifacts,platform,memcpy_required=False,msg_level='INFO')
    mark('before-runtime-load')
    runtime.load()
    mark('before-runtime-run')
    runtime.run()
    try:
        for i,(rx,tx) in enumerate(names):
            mark(f'before-pair-{i}-echo')
            source=np.asarray(INPUTS[i],dtype='<u4')
            received=np.empty_like(source)
            pending=runtime.receive(runtime.get_port_id(tx),received,received.size,nonblock=True)
            runtime.send(runtime.get_port_id(rx),source,source.size,nonblock=False)
            runtime.task_wait(pending)
            np.testing.assert_array_equal(received,source^np.uint32(MASKS[i]))
            desc['transactions'].append(dict(pair=i,input=source.tolist(),output=received.tolist()))
            mark(f'completed-pair-{i}-echo')
    finally:
        runtime.stop()
    mark('stopped')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kernels',type=Path);parser.add_argument('out',type=Path);parser.add_argument('--cmaddr')
    a=parser.parse_args()
    run(a.kernels,a.out,a.cmaddr)
