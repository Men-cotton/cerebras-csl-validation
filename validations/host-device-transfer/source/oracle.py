"""Exact host contract for geometry, buffering, split and broadcast campaigns.

Payload counts describe useful bytes delivered across the complete compute ROI.
Broadcast input bytes, control words, padding and diagnostic readback are reported
separately. None of these counts includes unknown SDK or network framing.
"""

import hashlib
import math

import numpy as np


CAPACITY = 4097
MAX_PAYLOAD_BYTES = 64 << 20
PAYLOAD_BYTES_PER_PE = (CAPACITY - 1) * 4
GENERATION_STRIDE = 16
PE_TAG = 0x85EBCA6B
WORD_TAG = 0x9E3779B9


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rounded_shape(ratio, target=4096):
    """Smallest integer W with W*(ratio*W) >= target; preserve exact W:H."""
    require(type(ratio) is int and ratio in (1, 2, 4, 8, 16), 'supported aspect ratio')
    require(type(target) is int and target > 0, 'positive exact target PE count')
    width = math.isqrt(target // ratio)
    if width * width * ratio < target:
        width += 1
    return width, width * ratio


def _case(width=64, height=64, channels=16, west=0, east=0, chunks=1,
          pattern='unique', mode='unicast', reply='echo', campaign='geometry',
          target_pes=4096):
    divisor = math.gcd(width, height)
    return dict(width=width, height=height, pes=width * height, capacity=CAPACITY,
                channels=channels, west=west, east=east, chunks=chunks,
                pattern=pattern, mode=mode, reply=reply, campaign=campaign,
                target_pes=target_pes, extra_pes=width * height - target_pes,
                aspect_ratio=f'{width // divisor}:{height // divisor}')


CASES = {}
CAMPAIGNS = {name: [] for name in ('geometry', 'buffers', 'split', 'broadcast')}
for ratio in (1, 2, 4, 8, 16):
    width, height = rounded_shape(ratio)
    shapes = [(width, height)] if width == height else [(width, height), (height, width)]
    for width, height in shapes:
        for channels in (1, 16):
            name = f'geometry-w{width}-h{height}-c{channels}'
            CASES[name] = _case(width=width, height=height, channels=channels)
            CAMPAIGNS['geometry'].append(name)
for west, east in ((0, 0), (1, 0), (0, 1), (1, 1), (4, 0), (0, 4),
                   (4, 4), (16, 0), (0, 16), (16, 16)):
    name = f'buffers-w{west}-e{east}'
    CASES[name] = _case(west=west, east=east, campaign='buffers')
    CAMPAIGNS['buffers'].append(name)
for chunks in (1, 2, 4, 8, 16):
    name = f'split-k{chunks}'
    CASES[name] = _case(chunks=chunks, campaign='split')
    CAMPAIGNS['split'].append(name)
for pattern in ('row', 'col'):
    for transfer in ('bcast', 'unicast'):
        name = f'broadcast-{pattern}-{transfer}'
        mode = f'{pattern}-bcast' if transfer == 'bcast' else 'unicast'
        CASES[name] = _case(pattern=pattern, mode=mode, reply='ack', campaign='broadcast')
        CAMPAIGNS['broadcast'].append(name)

SMOKE_CASES = {
    'smoke-echo': _case(width=2, height=2, channels=1, target_pes=4, campaign='smoke'),
    'smoke-buffer': _case(width=2, height=2, channels=1, west=1, east=1,
                         target_pes=4, campaign='smoke'),
    'smoke-split16': _case(width=2, height=2, channels=1, chunks=16,
                          target_pes=4, campaign='smoke'),
}
for pattern in ('row', 'col'):
    for transfer in ('bcast', 'unicast'):
        name = f'smoke-{pattern}-{transfer}'
        mode = f'{pattern}-bcast' if transfer == 'bcast' else 'unicast'
        SMOKE_CASES[name] = _case(width=2, height=2, channels=1, target_pes=4,
                                pattern=pattern, mode=mode, reply='ack', campaign='smoke')


def _config(schedule):
    require(isinstance(schedule, dict), 'plan is an object')
    case = schedule.get('case')
    require(type(case) is str and (case in CASES or case in SMOKE_CASES), 'registered plan case')
    expected = CASES.get(case, SMOKE_CASES.get(case))
    require(all(type(schedule.get(key)) is type(value) and schedule[key] == value
                for key, value in expected.items()), 'unchanged registered configuration')
    require(type(schedule.get('smoke')) is bool and (case not in SMOKE_CASES or schedule['smoke']),
            'local-only smoke configuration')
    return expected


def compiler_params(schedule):
    config = _config(schedule)
    return dict(WIDTH=config['width'], HEIGHT=config['height'], CAPACITY=CAPACITY,
                REPLY_ACK=int(config['reply'] == 'ack'),
                MEMCPYH2D_DATA_1_ID=0, MEMCPYD2H_DATA_1_ID=1)


def plan(case, order=0, smoke=False):
    require(type(case) is str and (case in CASES or case in SMOKE_CASES), 'registered case')
    require(type(order) is int and order in (0, 1, 2), 'size order is 0, 1 or 2')
    require(type(smoke) is bool and (case not in SMOKE_CASES or smoke), 'explicit local smoke')
    config = CASES.get(case, SMOKE_CASES.get(case))
    maximum = min(MAX_PAYLOAD_BYTES, PAYLOAD_BYTES_PER_PE * config['pes'])
    replication = (config['width'] if config['pattern'] == 'row' else
                   config['height'] if config['pattern'] == 'col' else 1)
    sizes = [0] + [1 << exponent for exponent in range(maximum.bit_length())
                   if (1 << exponent) % replication == 0]
    if smoke:
        sizes = sorted({0, replication, 64, 1024})
        sizes = [size for size in sizes if size <= maximum and size % replication == 0]
    if order == 1:
        sizes.reverse()
    elif order == 2:
        sizes.sort(key=lambda size: hashlib.sha256(
            ('host-device-transfer-order-v1:' + str(size)).encode('ascii')).digest())
    return dict(case=case, order=order, smoke=smoke, sizes=sizes,
                warmup=1 if smoke else 10, repeats=3 if smoke else 30,
                generation_stride=GENERATION_STRIDE, **config)


def geometry(size, schedule):
    config = _config(schedule)
    pes, chunks = config['pes'], config['chunks']
    maximum = min(MAX_PAYLOAD_BYTES, PAYLOAD_BYTES_PER_PE * pes)
    require(type(size) is int and 0 <= size <= maximum and
            (size == 0 or size & (size - 1) == 0), 'zero or bounded power-of-two payload')
    replication = (config['width'] if config['pattern'] == 'row' else
                   config['height'] if config['pattern'] == 'col' else 1)
    require(size % replication == 0, 'broadcast-pattern logical bytes divide by replication')
    base_words = (size + 4 * pes - 1) // (4 * pes)
    span = base_words * 4
    chunk_span = (span + chunks - 1) // chunks
    words = (chunk_span + 3) // 4
    # Split the same byte interval on every PE. Useful bytes occupy a prefix
    # of the base image, so tiny tensors can leave later chunks entirely empty.
    full_pes, tail_bytes = divmod(size, span) if span else (0, 0)
    chunk_sizes = []
    for chunk in range(chunks):
        begin, end = min(chunk * chunk_span, span), min((chunk + 1) * chunk_span, span)
        chunk_sizes.append(full_pes * (end - begin) + max(0, min(tail_bytes, end) - begin))
    require(sum(chunk_sizes) == size and all(value % replication == 0 for value in chunk_sizes),
            'exact replicated chunk payload accounting')
    require(words < CAPACITY, 'frame fits PE buffer')
    senders = (config['height'] if config['mode'] == 'row-bcast' else
               config['width'] if config['mode'] == 'col-bcast' else pes)
    actual_replication = pes // senders
    host_payload = size // actual_replication
    padding = chunks * pes * words * 4 - size
    host_padding = chunks * senders * words * 4 - host_payload
    control_in, control_out = chunks * senders * 4, chunks * pes * 4
    ack = config['reply'] == 'ack'
    return dict(payload_bytes=size, delivered_payload_bytes=size,
                active_pes=pes, active_rows=config['height'], active_cols=config['width'],
                chunks=chunks, per_chunk_payload_bytes=chunk_sizes,
                payload_words_per_pe=words, frame_words=words + 1,
                reply_words=1 if ack else words + 1,
                control_h2d_bytes=control_in, control_d2h_bytes=control_out,
                delivered_control_bytes=chunks * pes * 4,
                h2d_payload_bytes=host_payload, h2d_padding_bytes=host_padding,
                h2d_transfer_bytes=control_in + host_payload + host_padding,
                d2h_payload_bytes=0 if ack else size,
                d2h_padding_bytes=0 if ack else padding,
                d2h_transfer_bytes=control_out if ack else control_out + size + padding,
                readback_bytes=chunks * pes * (words + 1) * 4 if ack else 0,
                padding_bytes=padding)


def _source_frame(size, sources, words, generation):
    """Place a deterministic tensor in its fixed padded source-PE slots."""
    result = np.zeros((sources, words + 1), dtype=np.uint32)
    result[:, 0] = generation
    if size:
        used_words = (size + 3) // 4
        values = np.arange(1, used_words + 1, dtype=np.uint32)
        values *= np.uint32(WORD_TAG)
        values ^= np.uint32(generation ^ PE_TAG)
        if size % 4:
            values[-1] &= np.uint32((1 << (8 * (size % 4))) - 1)
        result[:, 1:].flat[:used_words] = values
    return result


def frames(size, generation, schedule):
    """Return (H2D input, full tagged ROI image) for every sequential chunk.

    Chunk headers use generation+chunk. Each PE's K1 padded byte span is split
    into K intervals; chunk word padding does not alter tensor ownership.
    Concatenating those intervals reconstructs the same per-PE image for every
    split count. The driver advances base generation by GENERATION_STRIDE for
    every logical round, independently of the split count.
    """
    shape = geometry(size, schedule)
    require(type(generation) is int and 0 <= generation and
            generation + shape['chunks'] - 1 <= 0xffffffff, 'nonwrapping u32 generation range')
    width, height, pes = schedule['width'], schedule['height'], schedule['pes']
    words, frame_words = shape['payload_words_per_pe'], shape['frame_words']
    pattern, mode = schedule['pattern'], schedule['mode']
    sources = height if pattern == 'row' else width if pattern == 'col' else pes
    replication = pes // sources
    tags = np.arange(pes, dtype=np.uint32) * np.uint32(PE_TAG)
    result = []
    base_words = (size + 4 * pes - 1) // (4 * pes)
    base = _source_frame(size // replication, sources, base_words, generation)
    span = base_words * 4
    chunk_span = (span + shape['chunks'] - 1) // shape['chunks']
    base_bytes = base[:, 1:].view(np.uint8)
    for chunk in range(shape['chunks']):
        begin, end = min(chunk * chunk_span, span), min((chunk + 1) * chunk_span, span)
        source = np.zeros((sources, words + 1), dtype=np.uint32)
        source[:, 0] = generation + chunk
        source[:, 1:].view(np.uint8)[:, :end-begin] = base_bytes[:, begin:end]
        if pattern == 'row':
            expanded = np.broadcast_to(source[:, None, :], (height, width, frame_words)).copy()
        elif pattern == 'col':
            expanded = np.broadcast_to(source[None, :, :], (height, width, frame_words)).copy()
        else:
            expanded = source
        expected = expanded.reshape(pes, frame_words)
        sent = source.reshape(-1) if mode != 'unicast' else expected.copy().reshape(-1)
        expected[:, 0] = tags ^ np.uint32(generation + chunk)
        result.append((sent, expected.reshape(-1)))
    return result


def digest(array):
    require(isinstance(array, np.ndarray) and array.dtype.kind == 'u'
            and array.dtype.itemsize == 4, 'digest requires u32 array')
    return hashlib.sha256(array.astype('<u4', copy=False).tobytes(order='C')).hexdigest()
