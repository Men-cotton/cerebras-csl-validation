"""Run both expression forms on the same resident Tungsten compute PE."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from contract import PORT, inputs, read, require, sha, verify


def execute(build,fixture,artifact,capture,cmaddr):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkCompileArtifacts,SdkRuntime,SdkTarget,SimfabConfig,get_platform
    m=read(build);n=m['elements'];role='hardware' if cmaddr else 'simulator'
    require(sha(fixture)==m['fixture_sha256'] and sha(artifact/'program.elf')==m['roles'][role]['elf_sha256'] and
            sha(artifact/'out_port_map.json')==m['ports_sha256']==PORT,'physical or simulator artifact/input differs')
    host={name:sha(Path(__file__).with_name(name)) for name in ('worker.py','contract.py')}
    require(host==m['worker_sha256'],'running worker differs from manifest')
    _,upload=inputs(fixture,n)
    guard=np.full(4*4*(n+10)+4,0xdeadc0de,dtype='uint32')
    reply=guard[2:-2]
    capture.mkdir(parents=True,exist_ok=False);os.chdir(capture)
    platform=get_platform(cmaddr,SimfabConfig(num_threads=1,suppress_trace=True),SdkTarget.WSE3)
    require(bool(platform.is_system())==bool(cmaddr) and bool(platform.is_simulation())!=bool(cmaddr),'SDK role mismatch')
    image=SdkCompileArtifacts(str(artifact)).add_port_mapping(str(artifact/'out_port_map.json'))
    runner=SdkRuntime(image,platform,memcpy_required=False,msg_level='INFO')
    coords=[list(runner.coord_logical_to_physical((0,y))) for y in range(2)]
    require(coords==[[0,0],[0,1]],'misplaced compute or relay')
    record=dict(role=role,build_sha256=sha(build),worker_sha256=host,elf_sha256=m['roles'][role]['elf_sha256'],
                ports_sha256=PORT,coordinates=coords,system_target=bool(platform.is_system()),stopped=False)
    def stage(name):
        record['stage']=name
        path=capture/'capture.tmp';path.write_text(json.dumps(record,indent=2)+'\n');path.replace(capture/'capture.json')
    stage('load');runner.load();runner.run()
    try:
        stage('stream');runner.report_port_infos()
        rx=runner.get_port_id('rx_port');tx=runner.get_port_id('tx_port')
        receive=runner.receive(tx,reply,reply.size,nonblock=True)
        send=runner.send(rx,upload.ravel(),upload.size,nonblock=True)
        runner.task_wait(send);runner.task_wait(receive)
    finally:
        runner.stop()
    record['stopped']=True
    np.savez_compressed(capture/'raw.npz',uploads=upload,replies=reply.reshape(4,4,n+10),guard=guard[[0,1,-2,-1]])
    record['raw_sha256']=sha(capture/'raw.npz')
    stage('complete')
    summary=verify(build,fixture,capture,role)
    (capture/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('build','fixture','artifact','capture'):p.add_argument(arg,type=Path)
    p.add_argument('--cmaddr')
    a=p.parse_args();execute(*(x.resolve() for x in (a.build,a.fixture,a.artifact,a.capture)),a.cmaddr)
