"""Deterministic payload/guard and bounded queue-observation contract, without SDK."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
path = HERE/'transport_oracle.py'
if not path.is_file():
    path = HERE.parent/'oracle.py'
spec = importlib.util.spec_from_file_location('queue_transport_oracle', path)
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
BASE = ('source', 'received', 'control', 'stats', 'clocks', 'scratch_mirror')
RECORDS, WORDS = 4, 9
DELAYS = (0, 256)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def symbols(n):
    transport.params(0, n, 1)
    return dict(source=6*n+3, received=2*n+3, control=2, stats=32, clocks=12,
                scratch_mirror=2*n+3, observe_control=2, queue_records=RECORDS*WORDS,
                queue_status=4, queue_clock=3, delay_witness=1, completion=1)


def fixture(n, invocation, profile, delay):
    need(type(profile) is int and profile in (0, 1), 'PROFILE 0/1')
    need(type(delay) is int and delay in DELAYS, 'bounded delay 0/256')
    value = transport.fixture(0, n, invocation, 1)
    value.update(observe_control=[[profile, delay] for _ in range(2)],
                 queue_records=[[0]*(RECORDS*WORDS) for _ in range(2)],
                 queue_status=[[0]*4 for _ in range(2)], queue_clock=[[0]*3 for _ in range(2)],
                 delay_witness=[[0], [0]], completion=[[invocation], [invocation]])
    return value


def u48(words):
    need(len(words) == 3 and all(type(v) is int and 0 <= v < 65536 for v in words), 'u48 limbs')
    return sum(v << (16*i) for i, v in enumerate(words))


def audit(value, n, invocation, profile, delay):
    initial = fixture(n, invocation, profile, delay)
    need(set(value) == set(initial), 'complete symbol set')
    for name, width in symbols(n).items():
        need(len(value[name]) == 2 and all(len(row) == width for row in value[name]), 'symbol shape '+name)
        need(all(type(v) is int and 0 <= v < 2**32 for row in value[name] for v in row), 'u32 '+name)
    sender = transport.audit({name: value[name] for name in BASE}, 0, n, invocation, 1)
    epoch = invocation+1
    need(value['observe_control'] == initial['observe_control'], 'immutable observation controls')
    need(value['completion'] == [[epoch], [epoch]], 'fresh all-PE completion')
    need(value['queue_status'] == [[epoch, 4*profile, 0, 0], [epoch, 4*profile, 0, delay]],
         'generation/record count/error/delay steps')
    need(value['delay_witness'] == [[0], [delay*(delay+1)//2]], 'bounded delay witness')
    drain = u48(value['queue_clock'][0])
    need(value['queue_clock'][1] == [0]*3, 'sender-only drain clock')
    need(sender['tx'] <= drain <= sender['end'], 'TX callback precedes queue drain and joined completion')
    snapshots = []
    for role in range(2):
        words = value['queue_records'][role]
        if not profile:
            need(words == [0]*(RECORDS*WORDS), 'disabled snapshot storage stays zero')
            continue
        rows = [words[i*WORDS:(i+1)*WORDS] for i in range(RECORDS)]
        for i, row in enumerate(rows):
            need(row[:2] == [epoch, 1+4*role+i], 'snapshot generation/event identity')
            stamp = u48(row[2:5])
            need(all(0 <= v < 256 for v in row[5:]), 'eight-queue masks')
            empty, full = (row[7], row[8]) if role == 0 else (row[5], row[6])
            need(empty & full & (1 << 2) == 0, 'active DATA queue cannot be both empty and full')
            snapshots.append(dict(role=role, epoch=epoch, event=row[1], local_timestamp=stamp,
                                  input_empty=row[5], input_full=row[6], output_empty=row[7], output_full=row[8]))
        times = [u48(row[2:5]) for row in rows]
        if role == 0:
            # Queue-empty and ACK callbacks can be dispatched in either order.
            need(times[0] <= times[1] <= times[2] and times[0] <= times[3], 'sender causal events')
            # The inherited sender oracle admits nonwrapping windows. Relative
            # u48 offsets also reject a snapshot on the wrong side of that window.
            offsets = [(stamp-sender['start']) & ((1 << 48)-1) for stamp in times]
            tx_offset = sender['tx']-sender['start']
            drain_offset = drain-sender['start']
            ack_offset = sender['terminal_ack']-sender['start']
            need(0 <= offsets[0] <= tx_offset <= offsets[1] <= drain_offset
                 <= offsets[2] <= sender['elapsed'] and
                 ack_offset <= offsets[3] <= sender['elapsed'], 'snapshots within corresponding sender clock boundaries')
            need(rows[2][7] & (1 << 2), 'data OQ2 empty at flush handler')
        else:
            need(all(a <= b for a, b in zip(times, times[1:])), 'receiver causal events')
            need(rows[2][5] & (1 << 2), 'data IQ2 empty after complete receive')
    return dict(epoch=epoch, sender=sender, queue_empty=drain,
                tx_callback_to_queue_empty_cycles=drain-sender['tx'], snapshots=snapshots,
                scope='same-PE elapsed callback intervals and event snapshots; no router occupancy or congestion fraction')
