#!/usr/bin/env python3
"""Independent H055 integer stencil reference and bounded local join model.

Read-first sources: benchmark-libs/stencil_3d_7pts/{layout,wse3/pe}.csl,
especially coefficient order at pe:236-248, length restoration at 258-272,
SEND/RECV join at 384-399, and final callback at 598-627. Source routing, CSL
state machines and simulator output are not used to compute the reference.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np


MAX_N = 32
BLOCK_SIZE = 16
COEFFICIENTS = (1, 2, 3, 4, 5, 6, 7)
# Destination-relative source locations: W,E,S,N,bottom,top,center.
OFFSETS = ((0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0),
           (0, 0, -1), (0, 0, 1), (0, 0, 0))
NUMERICAL_MUTATIONS = ("direction_swap", "periodic_wrap", "stale_length",
                       "missing_epoch", "no_checksum_reset")
EVENT_MUTATIONS = ("early_x_update", "early_halo_reuse", "premature_join",
                   "duplicate_callback", "coalesced_loss")


def dimensions(rows, cols, n, phase=0, epochs=64):
    if any(type(value) is not int for value in (rows, cols, n, phase, epochs)):
        raise ValueError("dimensions, phase and epoch count must be integers")
    if rows < 2 or cols < 2 or not 2 <= n <= MAX_N or phase not in (0, 1) or not 1 <= epochs <= 64:
        raise ValueError("require rows/cols >= 2, 2 <= n <= 32, phase 0/1 and 1..64 epochs")


def input_values(rows, cols, n, phase, epoch):
    """Vectorized input generator used only by the independent control route."""
    dimensions(rows, cols, n, phase)
    if type(epoch) is not int or not 0 <= epoch < 64:
        raise ValueError("epoch must lie in [0, 64)")
    r, c, k = np.indices((rows, cols, n), dtype=np.int64)
    return (1 + (17*c + 31*r + 7*k + 11*phase) % 64 + 64*epoch).astype(np.float32)


def direct_apply(values, coefficients=COEFFICIENTS):
    """Enumerate integer-coordinate neighbors, including every boundary cell."""
    x = np.asarray(values)
    if x.ndim != 3 or len(coefficients) != 7 or not np.all(np.isfinite(x)):
        raise ValueError("require finite R,C,n data and seven integer coefficients")
    if not np.all(x == np.trunc(x)) or any(type(coefficient) is not int for coefficient in coefficients):
        raise ValueError("this reference intentionally admits integer controls only")
    result = np.zeros(x.shape, dtype=np.int64)
    rows, cols, n = x.shape
    for r in range(rows):
        for c in range(cols):
            for k in range(n):
                total = 0
                for coefficient, (dr, dc, dk) in zip(coefficients, OFFSETS):
                    sr, sc, sk = r + dr, c + dc, k + dk
                    if 0 <= sr < rows and 0 <= sc < cols and 0 <= sk < n:
                        total += coefficient * int(x[sr, sc, sk])
                result[r, c, k] = total
    return result


def expected(rows, cols, n, phase, epochs=64):
    """All-PE y/Q/count; y and Q have shape (R,C,32), including zero tails.

    The primary route evaluates the frozen input formula directly at each of
    seven possible source coordinates. Each source is a + 64*epoch, so only
    the epoch sum is closed-form. It does not call input_values(), import the
    CSL state machine, or reuse the sliced control below.
    """
    dimensions(rows, cols, n, phase, epochs)
    y = np.zeros((rows, cols, MAX_N), dtype=np.int64)
    checksum = np.zeros_like(y)
    last_epoch_offset = 64 * (epochs - 1)
    summed_epoch_offset = 32 * epochs * (epochs - 1)
    for r in range(rows):
        for c in range(cols):
            for k in range(n):
                initial_total = 0
                coefficient_sum = 0
                for coefficient, (dr, dc, dk) in zip(COEFFICIENTS, OFFSETS):
                    sr, sc, sk = r + dr, c + dc, k + dk
                    if 0 <= sr < rows and 0 <= sc < cols and 0 <= sk < n:
                        value = 1 + ((17*sc + 31*sr + 7*sk + 11*phase) % 64)
                        initial_total += coefficient * value
                        coefficient_sum += coefficient
                # With E epochs, y=A+64*W*(E-1) and Q=E*A+32*W*E*(E-1).
                # Missing neighbors affect both A and W; zero tails stay zero.
                y[r, c, k] = initial_total + last_epoch_offset * coefficient_sum
                checksum[r, c, k] = epochs * initial_total + summed_epoch_offset * coefficient_sum
    # Nonnegative terms, E<=64 and W<=28 imply Q<=3727360<2**24;
    # the original sequential FP32 sums are therefore exact integers too.
    if np.max(checksum) >= 2**24:
        raise AssertionError("frozen exact-FP32 integer bound exceeded")
    return {"y": y.astype(np.float32), "Q": checksum.astype(np.float32),
            "count": np.full((rows, cols, 1), epochs, dtype=np.uint32)}


def vectorized_apply(values, coefficients=COEFFICIENTS, *, periodic=False):
    """Separate sliced FP32 implementation; no coordinate-offset enumeration."""
    x = np.asarray(values, dtype=np.float32)
    w, e, s, north, bottom, top, center = coefficients
    if periodic:
        y = (w*np.roll(x, 1, axis=1) + e*np.roll(x, -1, axis=1)
             + s*np.roll(x, -1, axis=0) + north*np.roll(x, 1, axis=0))
    else:
        y = np.zeros_like(x)
        y[:, 1:, :] += w*x[:, :-1, :]
        y[:, :-1, :] += e*x[:, 1:, :]
        y[:-1, :, :] += s*x[1:, :, :]
        y[1:, :, :] += north*x[:-1, :, :]
    y[:, :, 1:] += bottom*x[:, :, :-1]
    y[:, :, :-1] += top*x[:, :, 1:]
    y += center*x
    return y


def vectorized_control(rows, cols, n, phase, epochs=64, *, mutation=None,
                       previous_q=None, previous_length=BLOCK_SIZE):
    """Independent full-epoch control plus explicit causal numerical mutants."""
    dimensions(rows, cols, n, phase, epochs)
    if mutation not in (None,) + NUMERICAL_MUTATIONS:
        raise ValueError("unknown numerical mutation")
    y = np.zeros((rows, cols, MAX_N), dtype=np.float32)
    checksum = np.zeros_like(y)
    if mutation == "no_checksum_reset":
        if previous_q is None or np.asarray(previous_q).shape != checksum.shape:
            raise ValueError("reset mutation requires a same-shape prior checksum")
        checksum[:] = previous_q
    iterations = epochs - int(mutation == "missing_epoch")
    coefficients = (2, 1, 4, 3, 5, 6, 7) if mutation == "direction_swap" else COEFFICIENTS
    for epoch in range(iterations):
        x = input_values(rows, cols, n, phase, epoch)
        active = vectorized_apply(x, coefficients, periodic=mutation == "periodic_wrap")
        if mutation == "stale_length":
            # The official next-block offset advances by BLOCK_SIZE even if an
            # un-restored previous tail leaves descriptors at length one.
            xy = vectorized_apply(x, (1, 2, 3, 4, 0, 0, 0))
            valid = np.zeros(n, dtype=np.bool_)
            length = previous_length
            for start in range(0, n, BLOCK_SIZE):
                length = min(length, n-start)
                valid[start:start+length] = True
            active[:, :, ~valid] -= xy[:, :, ~valid]
        y[:, :, :n] = active
        checksum += y
    return {"y": y, "Q": checksum,
            "count": np.full((rows, cols, 1), iterations, dtype=np.uint32)}


def verify_result(actual, reference):
    for name in ("y", "Q", "count"):
        if name not in actual:
            raise AssertionError(f"missing {name}")
        lhs, rhs = np.asarray(actual[name]), reference[name]
        if lhs.dtype != rhs.dtype or lhs.shape != rhs.shape or not np.array_equal(lhs, rhs):
            raise AssertionError(f"all-PE {name} differs in dtype, shape, or values")


def metrics(rows, cols, n=32, epochs=64):
    dimensions(rows, cols, n, epochs=epochs)
    pes = rows*cols
    total_degree = 2*(rows*(cols-1) + cols*(rows-1))
    maximum_degree = (1 if rows == 2 else 2) + (1 if cols == 2 else 2)
    return {"active_n": n,
            "work_per_pe_per_epoch": {"stencil_fmas": 7*n-2, "checksum_adds": n,
                                      "output_zero_elements": n, "generated_input_elements": MAX_N},
            "modeled_payload_word_hops": epochs*n*total_degree,
            "modeled_directed_link_words": epochs*n,
            "maximum_sent_words_per_pe": epochs*n*maximum_degree,
            "mean_sent_words_per_pe": epochs*n*total_degree/pes,
            "maximum_mean_send_ratio": maximum_degree/(total_degree/pes),
            "boundary_pe_fraction": (2*rows+2*cols-4)/pes,
            "missing_direction_slot_fraction": (2*rows+2*cols)/(4*pes),
            "scope": "analytic payload/work only; excludes SDK control and measured traffic"}


@dataclass(frozen=True)
class State:
    epoch: int = 0
    block: int = 0
    send_position: int = 0
    recv_position: int = 0
    send_pending: str = ""
    recv_pending: str = ""
    send_ready: bool = True
    recv_ready: bool = True
    flags: int = 0
    wake: bool = False
    observed: int = 0
    count: int = 0
    ended: bool = False
    callbacks: int = 0


def explore_join(*, epochs=2, blocks=2, active=("W", "E", "S", "N"),
                 mutation=None, state_limit=50000):
    """Finite local directional chains and a coalesced SEND/RECV join.

    One physical operation may be pending per chain; completion is independent
    of the corresponding software callback. Fair delivery/completion is assumed.
    Remote PE state, FIFO/backpressure, route configuration, real UT dispatch and
    whole-mesh termination are unmodeled. Tags identify (epoch,block,direction).
    """
    if (type(epochs) is not int or not 1 <= epochs <= 3 or type(blocks) is not int
            or not 1 <= blocks <= 2 or type(state_limit) is not int or state_limit < 1
            or len(set(active)) != len(active) or set(active) - set("WESN")
            or mutation not in (None,) + EVENT_MUTATIONS):
        raise ValueError("invalid bounded join configuration")
    initial = State()
    pending, seen = deque([(initial, ())]), {initial}
    terminals = 0
    coalesced = False
    completion_orders = set()

    def report(status, **extra):
        return {"status": status, "states": len(seen), "terminal_states": terminals,
                "state_limit": state_limit, "epochs": epochs, "blocks": blocks,
                "active_directions": list(active), "mutation": mutation,
                "both_flags_before_handler_reached": coalesced,
                "first_chain_completion": sorted(completion_orders),
                "scope": "finite local chains/join; no inter-PE FIFO or mesh liveness proof", **extra}

    while pending:
        state, trace = pending.popleft()
        tag = f"epoch{state.epoch}/block{state.block}"
        coalesced |= state.flags == 3 and state.observed == 0
        if state.flags in (1, 2):
            completion_orders.add("SEND" if state.flags == 1 else "RECV")
        if state.callbacks > state.epoch + 1:
            return report("unsafe", reason="duplicate-callback", counterexample=list(trace))
        if state.ended and not state.wake:
            if state.epoch == epochs - 1:
                if state.callbacks != epochs:
                    return report("unsafe", reason="missing-callback", counterexample=list(trace))
                terminals += 1
                continue
            successor = State(epoch=state.epoch+1, callbacks=state.callbacks)
            successors = [(f"{tag}:NEXT-generate-input", successor)]
        else:
            successors = []
            for send, directions in ((True, ("E", "W", "N", "S")),
                                     (False, ("W", "E", "S", "N"))):
                label = "SEND" if send else "RECV"
                pos = state.send_position if send else state.recv_position
                physical = state.send_pending if send else state.recv_pending
                ready = state.send_ready if send else state.recv_ready
                if physical:
                    changes = {"send_pending": "", "send_ready": True} if send else {
                        "recv_pending": "", "recv_ready": True}
                    successors.append((f"{tag}/{physical}:physical-{label}-complete", replace(state, **changes)))
                    bad = "early_x_update" if send else "early_halo_reuse"
                    if mutation == bad:
                        return report("unsafe", reason=bad,
                                      counterexample=list(trace + (f"{tag}/{physical}:{bad}",)))
                if ready:
                    if pos == 4:
                        changes = {"send_ready": False} if send else {"recv_ready": False}
                        bit = 1 if send else 2
                        successors.append((f"{tag}:{label}-callback", replace(
                            state, flags=state.flags | bit, wake=True, **changes)))
                    else:
                        direction = directions[pos]
                        physical = direction if direction in active else ""
                        changes = ({"send_position": pos+1, "send_pending": physical,
                                    "send_ready": not physical} if send else {
                                    "recv_position": pos+1, "recv_pending": physical,
                                    "recv_ready": not physical})
                        successors.append((f"{tag}/{direction}:{label}-" + ("issue" if physical else "boundary-skip"),
                                           replace(state, **changes)))
            if state.wake:
                count = state.flags.bit_count()
                if mutation == "coalesced_loss":
                    count = state.count + int(state.flags != state.observed)
                ready = count == 2
                if mutation == "premature_join":
                    ready = bool(state.flags)
                if ready:
                    if state.flags != 3 or state.send_pending or state.recv_pending:
                        return report("unsafe", reason="compute-before-SEND-RECV-complete",
                                      counterexample=list(trace + (f"{tag}:premature-compute",)))
                    if state.block == blocks - 1:
                        successors.append((f"{tag}:Q-update-terminal-callback", replace(
                            state, ended=True, callbacks=state.callbacks+1,
                            wake=mutation == "duplicate_callback")))
                    else:
                        successors.append((f"{tag}:compute-and-next-block", State(
                            epoch=state.epoch, block=state.block+1, callbacks=state.callbacks)))
                else:
                    successors.append((f"{tag}:join-handler", replace(
                        state, wake=False, observed=state.flags, count=count)))
        if not successors:
            return report("unsafe", reason="quiescent-without-callback", counterexample=list(trace))
        for event, successor in successors:
            if successor in seen:
                continue
            if len(seen) >= state_limit:
                return report("unknown", reason="state-cap", unexplored_states="unknown",
                              unexplored_frontier_lower_bound=len(pending)+1)
            seen.add(successor)
            pending.append((successor, trace + (event,)))
    return report("safe", unexplored_states=0)


def cohort():
    started = time.monotonic()
    cases = []
    for rows, cols in ((2, 2), (2, 3), (3, 2), (3, 3)):
        for n in (2, 15, 16, 17, 32):
            for phase in (0, 1):
                reference = expected(rows, cols, n, phase)
                verify_result(vectorized_control(rows, cols, n, phase), reference)
                cases.append({"rows": rows, "cols": cols, "n": n, "phase": phase,
                              "epochs": 64, "passed": True,
                              "elements_checked": sum(array.size for array in reference.values()),
                              "sha256": {name: hashlib.sha256(array.tobytes()).hexdigest()
                                         for name, array in reference.items()}})
    mutations = []
    reference = expected(3, 3, 32, 1)
    prior = expected(3, 3, 17, 0)
    for mutation in NUMERICAL_MUTATIONS:
        result = vectorized_control(3, 3, 32, 1, mutation=mutation,
                                    previous_q=prior["Q"], previous_length=1)
        try:
            verify_result(result, reference)
        except AssertionError as error:
            mutations.append({"mutation": mutation, "rejected": True, "reason": str(error)})
        else:
            raise AssertionError(f"undetected numerical mutation: {mutation}")
    joins = [explore_join(active=active) for active in (("E", "S"), ("W", "E", "S"), ("W", "E", "S", "N"))]
    event_mutations = [explore_join(mutation=mutation) for mutation in EVENT_MUTATIONS]
    if any(result["status"] != "safe" for result in joins) or any(
            result["status"] != "unsafe" for result in event_mutations):
        raise AssertionError("bounded event cohort did not complete its positive/negative gate")
    return {"schema": "H055-independent-host-v1", "passed": True,
            "cases": cases, "numerical_mutations": mutations,
            "join_controls": joins, "join_mutations": event_mutations,
            "analysis_wall_seconds": time.monotonic()-started,
            "scope": "integer reference and finite local event model; no SDK execution or mesh liveness proof"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = cohort()
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "cases": len(result["cases"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
