"""Fixed, signed FP32 CSR fixture and independent FP64 owner-output verifier."""
import hashlib

import numpy as np

WIDTHS = (1, 2, 4, 8, 16)
TAG = 0x41474752
TRIALS = tuple(range(1,12+1))


def features(trial):
    if trial not in TRIALS:
        raise ValueError('invalid trial')
    s, f = np.indices((16, 4))
    if trial % 3 == 2:
        return (((5*s + 13*f + 1) % 29 - 14) / 4).astype('<f4')
    return (((11*s + 7*f + 3) % 31 - 15) / 8).astype('<f4')


def input_words(trial):
    return np.concatenate((np.array([trial, TAG], dtype='<u4'), features(trial).ravel().view('<u4')))


def expected(trial):
    x = features(trial).astype(np.float64)
    result = np.zeros((64, 4))
    rows = np.arange(64)
    rows = rows[rows % 8 != 0]
    for delta, weight in zip([0, 1, 1, 7], [0.5, -0.25, 0.75, -1.0]):
        result[rows] += x[(3*rows + delta) % 16] * weight
    return result.astype('<f4')


def digest(words):
    return hashlib.sha256(np.asarray(words).tobytes()).hexdigest()


def verify_compute(width, trial, words):
    local_rows = 64 // width
    packets = np.asarray(words, dtype='<u4').reshape(width, 8 + 4*local_rows)
    if not np.all(packets[:, 0] == trial) or packets[:, 1].tolist() != list(range(width)):
        raise ValueError('trial or ordered owner IDs differ')
    clocks = packets[:, 2:8].astype(np.int64)
    if np.any(clocks > 65535):
        raise ValueError('invalid clock word')
    start, stop = [c[:, 0] + (c[:, 1] << 16) + (c[:, 2] << 32) for c in np.hsplit(clocks, 2)]
    duration = (stop - start) % (1 << 48)
    if np.any(duration == 0):
        raise ValueError('empty compute interval')
    got = packets[:, 8:].copy().view('<f4').reshape(64, 4)
    want = expected(trial)
    if not np.array_equal(got.view('<u4'), want.view('<u4')):
        raise ValueError(f'owner output mismatch, maximum absolute error {np.max(np.abs(got-want))}')
    return dict(owners=width, max_compute_cycles=int(duration.max()), cycles=duration.tolist(),
                max_abs_error=0.0, words_matched=got.size, output_sha256=digest(got))


def verify(width, trial, words):
    rows = 64 // width
    packets = np.asarray(words, dtype='<u4').reshape(width, 14 + 4*rows)
    # Six words of immutable graph-setup timestamps follow the six compute words.
    standard = np.concatenate((packets[:, :8], packets[:, 14:]), axis=1)
    result = verify_compute(width, trial, standard)
    stamps = packets[:, 8:14].astype(np.int64)
    if np.any(stamps > 65535):
        raise ValueError('invalid setup timer words')
    times = stamps.reshape(-1, 2, 3) @ np.array([1, 1 << 16, 1 << 32])
    ticks = (times[:, 1] - times[:, 0]) % (1 << 48)
    if np.any((ticks <= 0) | (ticks >= (1 << 47))):
        raise ValueError('invalid setup interval')
    result.update(setup_clocks=stamps.tolist(), setup_cycles=ticks.tolist())
    return result
