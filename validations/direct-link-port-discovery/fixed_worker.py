"""Run the two West connector pairs from the frozen, sparse Tungsten image."""
import argparse
import json
from pathlib import Path
import shutil

from probe import MASKS, ports, save, sha, transfer


def run(artifact, build_path, out, cmaddr, current_fabric):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkCompileArtifacts, SdkTarget, SimfabConfig, get_platform

    artifact, out = artifact.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    role = 'hardware' if cmaddr else 'simulator'
    build = json.loads(build_path.read_bytes())
    image, mapping = artifact/'program.elf', artifact/'out_port_map.json'
    if sha(image) != build['roles'][role]['elf_sha256'] or sha(mapping) != build['ports_sha256']:
        raise ValueError('ELF or port map differs from the requested role')
    endpoint = ports(json.loads(mapping.read_bytes()))
    names = [(f'rx{i}_port', f'tx{i}_port') for i in range(2)]
    for i, (inp, ret) in enumerate(names):
        for name, direction, y, color in [(inp, 'IN', i*18, i*2), (ret, 'OUT', i*18+1, i*2+1)]:
            if endpoint[name] != dict(io_type=direction, side='WEST', loc=y, color=color):
                raise ValueError('wrong fixed delivery endpoint')
    shutil.copyfile(mapping, out/'port-map.json')
    desc = dict(backend=role, names=names, masks=list(MASKS), endpoints=endpoint,
                historical_fabric_sha256=build['historical_fabric_sha256'], physical_or_simulated_elf_sha256=sha(image),
                port_map_sha256=sha(mapping), transactions=[])
    def mark(phase):
        desc['last_phase'] = phase
        save(out/'trace.json', desc)
    mark('before-platform')
    if cmaddr and current_fabric and current_fabric.is_file():
        current = json.loads(current_fabric.read_bytes())
        desc['current_fabric_sha256'] = current['fabric_sha256']
        for key, ys in [('ingress_connections', (0,18)), ('egress_connections', (1,19))]:
            if any({'x':0, 'y':y} not in current[key] for y in ys):
                mark('recorded-connectors-absent-in-current-allocation')
                raise ValueError('fixed connectors unavailable in recorded current allocation')
    platform = get_platform(cmaddr, SimfabConfig(num_threads=1,suppress_trace=True), SdkTarget.WSE3)
    if bool(platform.is_system()) != bool(cmaddr) or bool(platform.is_simulation()) == bool(cmaddr):
        raise ValueError('unexpected SDK platform')
    artifacts = SdkCompileArtifacts(str(artifact)).add_port_mapping(str(mapping))
    transfer(artifacts, platform, names, desc, mark)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('artifact', 'build', 'out'):
        p.add_argument(arg, type=Path)
    p.add_argument('--cmaddr')
    p.add_argument('--current-fabric', type=Path)
    args = p.parse_args()
    run(args.artifact, args.build, args.out, args.cmaddr, args.current_fabric)
