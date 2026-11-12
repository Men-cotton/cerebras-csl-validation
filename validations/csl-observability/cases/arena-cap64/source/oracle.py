"""Independent occupancy and payload contract for one owned u16 arena.

This module imports neither NumPy nor SDK/device code. It models live intervals,
not the allocator's search or pointer arithmetic.
"""
CAPACITIES = (64, 128)
FIELDS = ("operation", "request_words", "remaining_words", "peak_words", "query_ok")
RUNS = ((0, 17), (1, 17), (1, 29), (0, 17))
UNOBSERVED = 65535


def require(ok, message):
    if not ok:
        raise ValueError(message)


def plan(capacity):
    require(type(capacity) is int and capacity in CAPACITIES, "arena capacity")
    # operation 1=allocate, 2=capacity query without allocation, 3=free all.
    return ((1, 8), (1, 16), (1, 24), (2, capacity-47), (3, 0),
            (1, capacity//2), (1, capacity//2), (2, 1), (3, 0),
            (1, 4), (3, 0))


def payload(size, tag, seed):
    require(type(seed) is int and seed in (17, 29), "known runtime seed")
    return [0xA000+tag] + [seed+tag*101+i*7 for i in range(1, size-1)] + [0xB000+tag]


def expected(capacity, mode, seed):
    require(type(mode) is int and mode in (0, 1), "observer mode")
    live, peak, rows = [], 0, []
    for operation, amount in plan(capacity):
        query = UNOBSERVED
        if operation == 1:
            live.append(amount)
        elif operation == 2:
            query = int(sum(live)+amount <= capacity)
        else:
            live.clear()
        used = sum(live)
        require(used <= capacity, "oracle allocation extent")
        peak = max(peak, used)
        rows.extend((operation, amount, capacity-used if mode else UNOBSERVED,
                     peak if mode else UNOBSERVED, query))
    sizes = (8, 16, 24, capacity//2, capacity//2, 4)
    values = [word for tag, size in enumerate(sizes, 1) for word in payload(size, tag, seed)]
    return dict(rows=rows, values=values, peak_words=capacity if mode else UNOBSERVED,
                final_remaining_words=capacity if mode else UNOBSERVED)


def u48(words):
    require(type(words) is list and len(words) == 3
            and all(type(v) is int and 0 <= v <= 65535 for v in words), "u48 carriers")
    return sum(v << (16*i) for i, v in enumerate(words))


def validate(raw, capacity, mode, seed, epoch):
    wanted = expected(capacity, mode, seed)
    require(set(raw) == {"control", "rows", "values", "status", "clocks", "completion"}, "raw fields")
    for name, values in raw.items():
        require(type(values) is list and all(type(v) is int and 0 <= v < 2**32 for v in values), "u32 carriers "+name)
    require(raw["control"] == [mode, seed], "immutable control")
    require(raw["rows"] == wanted["rows"], "all occupancy observations and failure queries")
    require(raw["values"] == wanted["values"], "every allocation payload and both guards")
    require(raw["completion"] == [epoch], "fresh completion epoch")
    require(raw["status"] == [0, 6, 2, 4, wanted["peak_words"], wanted["final_remaining_words"]], "complete allocator operations including entry reset")
    require(len(raw["clocks"]) == 6, "start/end clocks")
    cycles = (u48(raw["clocks"][3:])-u48(raw["clocks"][:3])) & ((1 << 48)-1)
    require(0 < cycles < (1 << 47), "positive same-PE interval")
    return dict(status="PASS", mode=mode, seed=seed, epoch=epoch, cycles=cycles,
                observed_peak_words=capacity if mode else None,
                observed_peak_bytes=2*capacity if mode else None,
                scope="owned u16 malloc arena only; excludes compiler/SDK stacks and other SRAM")
