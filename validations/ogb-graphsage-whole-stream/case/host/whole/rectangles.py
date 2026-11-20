"""Group consecutive PE transfers while preserving every owner's live prefix."""
import numpy as np


def rectangles(counts, lanes, call_cost_words=256):
    """Minimize transferred words plus a fixed per-call cost over contiguous PE runs."""
    counts = list(map(int, counts))
    p = len(counts)
    if min(counts, default=0) < 0 or lanes <= 0 or call_cost_words < 0:
        raise ValueError("invalid DMA geometry")
    best = [0] * (p + 1)
    next_region = [None] * p
    for begin in range(p - 1, -1, -1):
        if counts[begin] == 0:
            best[begin] = best[begin + 1]
            next_region[begin] = (begin + 1, 0)
            continue
        largest = 0
        for end in range(begin + 1, p + 1):
            largest = max(largest, counts[end - 1])
            if counts[end - 1] == 0:
                continue
            cost = (end - begin)*largest*lanes + call_cost_words + best[end]
            if next_region[begin] is None or cost < best[begin]:
                best[begin] = cost
                next_region[begin] = (end, largest)
    result = []
    begin = 0
    while begin < p:
        end, active = next_region[begin]
        if active:
            result.append((begin, end, active))
        begin = end
    return result


def send_prefixes(dev, name, values, counts, data_type, order):
    lanes = int(np.prod(values.shape[2:]))
    transferred = 0
    symbol = dev.get_id(name)
    for begin, end, length in rectangles(counts, lanes):
        block = np.ascontiguousarray(values[begin:end, :length])
        dev.memcpy_h2d(symbol, block.ravel(), begin, 0, end-begin, 1, length*lanes,
                       streaming=False, nonblock=False, data_type=data_type, order=order)
        transferred += block.size
    return transferred


def broadcast_weights(dev, values, p, data_type, order):
    values = np.ascontiguousarray(values).ravel()
    dev.memcpy_h2d_rowbcast(dev.get_id("weights"), values, 0, 0, p, 1, values.size,
                            streaming=False, nonblock=False, data_type=data_type, order=order)
    return values.size
