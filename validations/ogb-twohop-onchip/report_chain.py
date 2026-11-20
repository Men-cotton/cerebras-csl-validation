"""Replay on-chip two-hop frames, both complete host echoes and all original logits."""
import argparse
import json
from pathlib import Path

import numpy as np

from chain import TRIALS, audit, device, digest, first_upload, prepare


def replay(first, last, build, capture, expected=None):
    desc = json.loads((build / 'build.json').read_text())
    report = json.loads((capture / 'result.json').read_text())
    role = report['role']
    if not report['stopped'] or len(report['frames']) != len(TRIALS):
        raise ValueError('incomplete two-layer run')
    if (report['build_sha256'] != device.sha(build / 'build.json') or
        report['first_sha256'] != desc['first_sha256'] or desc['first_sha256'] != device.sha(first) or
        report['last_sha256'] != desc['last_sha256'] or desc['last_sha256'] != device.sha(last)):
        raise ValueError('different firmware or original two-hop inputs')
    for name, h in desc['source_sha256'].items():
        if device.sha(build / 'src' / name) != h:
            raise ValueError('changed device source')
    firmware = build / role / 'out'
    if (device.sha(firmware / 'program.elf') != desc['roles'][role]['elf_sha256'] or
        device.sha(firmware / 'out_port_map.json') != desc['port_sha256']):
        raise ValueError('mismatched executable or Direct Link ports')
    data = prepare(first, last, desc['first']['P'], desc['first']['E'])
    if desc['first'] != data[2] or desc['final'] != data[7]:
        raise ValueError('device geometry mismatch')
    states, scores = {}, {}
    frames = []
    for i, ((echo, scale), f) in enumerate(zip(TRIALS, report['frames'])):
        if (f['trial'], f['echo'], f['scale']) != (i+1, echo, scale):
            raise ValueError('echo schedule changed')
        file = capture / f'frame-{i+1}.npz'
        if f['raw_sha256'] != device.sha(file):
            raise ValueError('raw PE reply changed')
        with np.load(file, allow_pickle=False) as raw:
            if ('echo' in raw) != echo:
                raise ValueError('unexpected transfer of hidden states')
            reply = raw['reply']
            hidden = raw['echo'] if echo else None
            logits, metrics = audit(data, i, reply, hidden)
            np.testing.assert_array_equal(raw['logits'].view(np.uint32), logits.view(np.uint32))
            np.testing.assert_array_equal(raw['node'], data[1]['seed'])
            if echo:
                states[scale] = hidden.copy()
            if scale in scores:
                np.testing.assert_array_equal(scores[scale].view(np.uint32), logits.view(np.uint32))
            else:
                scores[scale] = logits
            if expected:
                np.testing.assert_array_equal(logits.view(np.uint32),expected['logits'][scale].view(np.uint32))
                if echo:
                    np.testing.assert_array_equal(hidden, expected['hidden'][scale])
            for key, value in metrics.items():
                if key.endswith('fp64_max_abs') and value is not None:
                    np.testing.assert_allclose(f[key], value, rtol=1e-8, atol=1e-13)
                elif f[key] != value:
                    raise ValueError('different raw value for '+key)
            n = int(data[7]['N']*data[7]['H']) if echo else 0
            if (f['input_words'] != len(first_upload(data,i))+n or f['output_words'] != len(reply)+n or
                f['hidden_to_host_words'] != n or f['hidden_from_host_words'] != n):
                raise ValueError('host transfer counts differ from this graph')
            frames.append(dict(trial=i+1,echo=echo,scale=scale,input_words=f['input_words'],
                               output_words=f['output_words'],hidden_round_trip_words=2*n,**metrics))
    if set(states) != {1.,.5} or np.array_equal(scores[1.].view(np.uint32), scores[.5].view(np.uint32)):
        raise ValueError('both complete hidden states and distinct outputs required')
    for frame in frames:
        if frame['hidden_digest'] != digest(states[frame['scale']]):
            raise ValueError('on-chip hidden digest disagrees with the full echo')
    result = dict(role=role, first_sha256=device.sha(first), last_sha256=device.sha(last), build_sha256=device.sha(build/'build.json'),
                  raw_vertices=int(data[0]['shape'][1]), hidden_vertices=data[7]['N'], final_vertices=data[7]['Z'],
                  arcs=[len(data[0]['src']),len(data[1]['src'])], first_owners=data[7]['P'], rows_per_owner=list(map(len,data[3])),
                  hidden_channels=data[7]['H'], output_classes=int(data[1]['wr'].shape[1]), frames=frames)
    return result, dict(hidden=states,logits=scores)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('first', 'last', 'build', 'capture'):
        p.add_argument(arg,type=Path)
    p.add_argument('--sdk',type=Path,help='compare every FP32 output and both complete hidden echoes')
    p.add_argument('--output',type=Path)
    a = p.parse_args()
    expected = replay(a.first,a.last,a.build,a.sdk)[1] if a.sdk else None
    result,_ = replay(a.first,a.last,a.build,a.capture,expected)
    text = json.dumps(result,indent=2,allow_nan=False)+'\n'
    if a.output:
        a.output.write_text(text)
    else:
        print(text,end='')
