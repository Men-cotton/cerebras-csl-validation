"""Recheck initial and final PE witnesses for one unscaled SDK probe per GraphSAGE assignment."""
import json
from pathlib import Path
import sys

import numpy as np

from paired_audit import check_device_frame, compare_fields, require
from paired_worker import digest, sha


def load(path):
    return json.loads(path.read_text())


def audit_samples(case, original, modes=('control', 'split')):
    sys.path.insert(0, str(original.parents[1]))
    import saved_audit as host
    doc = load(case/'case.json')
    base = host.load(original)
    for name, key in [('case.json', 'original_case_sha256'), ('input_model.npz', 'original_checkpoint_sha256'),
                      ('node_ids.npy', 'original_node_ids_sha256'), ('inputs.npz', 'original_inputs_sha256')]:
        require(sha(original/name) == doc[key], 'different original '+name)
    require(doc['params'] == base['params'], 'original geometry differs')
    old = host.load_npz(original/'inputs.npz')
    ref = host.independent_expected(original, base, old, 1.0)
    p = doc['params']
    report = dict(backend='WSE3 SDK simulator with data-only initialization', nodes=base['nodes'], pes=p['C']*p['R'],
                  original_checkpoint_sha256=doc['original_checkpoint_sha256'], modes={})
    machine = None
    for mode in modes:
        capture = case/'sim'/(mode+'-1.npz')
        evidence = load(case/'sim'/(mode+'-1.json'))
        require(evidence['mode'] == mode and evidence['feature_scale'] == 1 and evidence['frame'] == 1
                and evidence['case_sha256'] == sha(case/'case.json') and evidence['raw_sha256'] == sha(capture)
                and evidence['packed_sha256'] == sha(case/(mode+'.npz'))
                and evidence['linked_source_sha256'] == {n: sha(case/'src'/n) for n in ('layout.csl', 'pe.csl')},
                'different stopped-core assignment, frame, source or input')
        require(evidence['writes_outside_input_arrays'] == 0 and evidence['pe_count'] == p['C']*p['R']
                and all(len(evidence[key]) == 64 for key in ('initial_core_sha256', 'final_core_sha256', 'rpc_sha256')),
                'incomplete stopped-core provenance')
        images = (evidence['original_elf_sha256'], evidence['rpc_sha256'])
        require(machine is None or images == machine, 'the two local probes use different linked CSL programs')
        machine = images
        packed = host.load_npz(case/(mode+'.npz'))
        raw = host.load_npz(capture)
        initial = {k: digest(v) for k, v in packed.items()}
        final = dict(initial, packet_template=digest(raw['packet_template']))
        require(initial == evidence['initial'] and final == evidence['final'], 'initial or final PE input witness')
        frame, hops = check_device_frame(host, base, doc, mode, packed, raw, ref)
        require(frame['per_pe_cycles'] == evidence['per_pe_cycles'] and not frame['predictions']['all_nodes_disagree'],
                'raw numerical result or PE time differs')
        report['modes'][mode] = [dict(frame=1, scale=1.0, raw_sha256=sha(capture), moved_partial_word_hops=hops, **frame)]
    return report


def compare_returned_A(case, capture):
    for mode in ('control', 'split'):
        with np.load(case/'sim'/(mode+'-1.npz'), allow_pickle=False) as archive:
            reference = dict(archive)
        for trial in (1, 3):
            with np.load(capture/f'{mode}-{trial}.npz', allow_pickle=False) as actual:
                compare_fields(dict(actual), reference)
