"""Host contract for complete framed echo exchanges.

Application payload, the per-PE tagged generation word, and word padding are counted
separately. None of these counts includes SDK or physical-link framing.
"""

import hashlib
import math

import numpy as np


PAYLOAD_BYTES_PER_PE = 1 << 14
PE_COUNTS = (1, 4, 16, 64, 256, 1024, 4096)
CASES = {
    f"{mode}-p{pes}": {"width": math.isqrt(pes), "height": math.isqrt(pes),
                       "pes": pes, "capacity": 4097, "channels": 1}
    for mode in ("zero", "powers") for pes in PE_COUNTS
}
CASES.update({
    f"powers-p4096-c{channels}": {**CASES["powers-p4096"], "channels": channels}
    for channels in (2, 4, 8, 16)
})


def require(ok, message):
    if not ok:
        raise ValueError(message)


def geometry(size, pes=1):
    """Describe one ROI-wide exchange; byte counts are per direction."""
    require(type(pes) is int and pes in PE_COUNTS, "PE count must be a supported square through 4096")
    require(type(size) is int and 0 <= size <= PAYLOAD_BYTES_PER_PE * pes
            and (size == 0 or size & (size - 1) == 0),
            "payload size must be zero or a power of two through 16 KiB per PE")
    side = math.isqrt(pes)
    words = (size + pes * 4 - 1) // (pes * 4)
    padded_bytes = pes * words * 4
    return {
        "payload_bytes": size,
        "active_pes": pes,
        "active_rows": side,
        "active_cols": side,
        "payload_words_per_pe": words,
        "frame_words": words + 1,
        "control_h2d_bytes": pes * 4,
        "control_d2h_bytes": pes * 4,
        "padding_bytes": padded_bytes - size,
        "transfer_bytes_per_direction": pes * 4 + padded_bytes,
    }


def plan(case, order=0, smoke=False):
    """Return an explicit, JSON-compatible runtime schedule."""
    require(type(case) is str and case in CASES, "known ping-pong case")
    require(type(order) is int and order in (0, 1, 2), "size order must be 0, 1 or 2")
    require(type(smoke) is bool, "smoke must be boolean")
    maximum = PAYLOAD_BYTES_PER_PE * CASES[case]["pes"]
    if case.startswith("zero-"):
        sizes, warmup, repeats = [0], 20, 200
    else:
        sizes = [0] + [1 << exponent for exponent in range(maximum.bit_length())]
        warmup, repeats = 10, 30
        if smoke:
            sizes = sorted({0, 1, 4, 16384, min(32768, maximum), maximum})
    if order == 1:
        sizes.reverse()
    elif order == 2:
        sizes.sort(key=lambda size: hashlib.sha256(
            ("ping-pong-order-v1:" + str(size)).encode("ascii")).digest())
    if smoke:
        warmup, repeats = 1, 3
    return {
        "case": case,
        "order": order,
        "smoke": smoke,
        "sizes": sizes,
        "warmup": warmup,
        "repeats": repeats,
        **CASES[case],
    }


def expected_frame(size, generation, pes=1):
    """Return a flat ROW_MAJOR u32 frame with one tagged generation word per PE.

    Each header is generation XOR (linear PE index * 0x85EBCA6B modulo 2**32),
    so even zero-payload exchanges distinguish every PE's response.
    Payload words vary with generation, linear PE index and word index.
    Application bytes occupy the first ``size`` bytes across row-major payload
    slots. The remaining bytes are padding, including whole padding words on PEs
    with no application data; every PE still carries its own generation word.
"""
    shape = geometry(size, pes)
    require(type(generation) is int and 0 <= generation <= 0xffffffff,
            "generation must be an exact u32 integer")
    words = shape["payload_words_per_pe"]
    frame = np.empty((pes, words + 1), dtype=np.uint32)
    frame[:, 0] = ((np.arange(pes, dtype=np.uint32) * np.uint32(0x85EBCA6B))
                   ^ np.uint32(generation))
    if words:
        indices = np.arange(1, words + 1, dtype=np.uint32) * np.uint32(0x9E3779B9)
        pe_seeds = ((np.arange(1, pes + 1, dtype=np.uint32) * np.uint32(0x85EBCA6B))
                    ^ np.uint32(generation))
        # Write directly into the strided final buffer. This avoids a second
        # full payload allocation at the 4096-PE / 64-MiB endpoint.
        np.bitwise_xor(indices[None, :], pe_seeds[:, None], out=frame[:, 1:])
        if shape["padding_bytes"]:
            whole_words, remaining_bytes = divmod(size, 4)
            full_pes, partial_words = divmod(whole_words, words)
            has_partial = bool(partial_words or remaining_bytes)
            frame[full_pes + has_partial:, 1:] = 0
            if has_partial:
                partial = frame[full_pes, 1:]
                partial[partial_words + bool(remaining_bytes):] = 0
                if remaining_bytes:
                    partial[partial_words] &= np.uint32((1 << (8 * remaining_bytes)) - 1)
    return frame.reshape(-1)


def digest(array):
    """Hash u32 values in C order with an explicit little-endian encoding."""
    require(isinstance(array, np.ndarray) and array.dtype.kind == "u"
            and array.dtype.itemsize == 4, "digest requires a u32 array")
    return hashlib.sha256(array.astype("<u4", copy=False).tobytes(order="C")).hexdigest()
