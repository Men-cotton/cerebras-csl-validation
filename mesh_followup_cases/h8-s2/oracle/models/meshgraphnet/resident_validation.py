"""Validate resident capture structure and lifecycle without importing the SDK.

Numerical graph semantics remain the independent rollout oracle's responsibility.
The API check proves the recorded call sequence; the frozen runner binds each
record to its SDK invocation. Checkpoint readbacks are intentional host returns.
"""
from collections.abc import Mapping

import numpy as np


U32_MAX = (1 << 32) - 1


def require(value, message):
    if not value:
        raise ValueError(message)


def positive(value, name):
    require(type(value) is int and value > 0, "positive integer " + name)
    return value


def expected_macs(p, counts, steps, initialize):
    """Per-owner MAC count; static geometric encoding occurs only once."""
    positive(steps, "steps")
    require(type(initialize) is bool, "boolean initialization flag")
    h, blocks = positive(p["H"], "H"), positive(p["S"], "S")
    values = []
    for node_count, edge_count in counts:
        n, e = int(node_count), int(edge_count)
        value = (e * (3*h + 2*h*h) if initialize else 0)
        value += n * steps * (4*h*h + 13*h)
        value += blocks * steps * (5*e + 4*n) * h*h
        require(0 <= value <= U32_MAX, "MAC counter bound")
        values.append(value)
    return np.asarray(values, dtype=np.uint32)


def _schema(spec, name):
    require(isinstance(spec, (list, tuple)) and len(spec) == 2, "schema " + name)
    count, dtype = spec
    positive(count, "schema count " + name)
    require(dtype in ("float32", "uint32", "uint16"), "schema dtype " + name)
    return count, dtype


def _array(value, spec, pes, name, *, carrier):
    count, dtype = _schema(spec, name)
    dtype = "uint32" if carrier and dtype == "uint16" else dtype
    require(isinstance(value, np.ndarray), "array required " + name)
    require(value.shape == (pes, count), "raw shape " + name)
    require(value.dtype == np.dtype(dtype), "raw dtype " + name)
    require(bool(np.isfinite(value).all()), "nonfinite array " + name)


def _same_bits(left, right, name):
    require(left.shape == right.shape and left.dtype == right.dtype and
            left.tobytes(order="C") == right.tobytes(order="C"),
            "input bits changed " + name)


def _stats(p, counts, steps, first):
    pes, blocks = p["C"] * p["R"], p["S"]
    rounds = p["Q"] if p.get("MODE") == 2 else pes
    require(type(rounds) is int and 1 <= rounds <= pes, "round count")
    macs = expected_macs(p, counts, steps, first)
    result = []
    for (nodes, edges), mac in zip(counts, macs):
        n, e = int(nodes), int(edges)
        row = [0x600D, steps, blocks*steps, e if first else 0,
               n*steps, e*blocks*steps, n*blocks*steps, n*steps,
               rounds*blocks*steps, steps*blocks*(rounds+1)+1, int(mac),
               blocks-1, rounds-1, n, e, 0]
        require(all(0 <= value <= U32_MAX for value in row), "operation counter bound")
        result.append(row)
    return np.asarray(result, dtype=np.uint32)


def validate_capture(m, arrays, raw, execution):
    """Return validated checkpoint views for independent numerical comparison.

    No graph, weights, velocity, or geometry upload may occur after the initial
    upload sequence. Each launch advances its positive chunk of physical steps;
    only checkpoint readbacks occur between launches. Counters are per launch,
    while lifecycle records retain cumulative launches/steps and one geometry
    initialization. Clock spans are local 48-bit intervals, never a makespan.
    """
    p = m["params"]
    for key in ("C", "R", "H", "S", "STEPS", "L", "E", "W"):
        positive(p[key], key)
    pes = p["C"] * p["R"]
    residency = m["residency"]
    require(isinstance(residency, Mapping) and set(residency) == {"schema", "chunks"} and
            type(residency["schema"]) is int and residency["schema"] == 1,
            "resident schema")
    chunks = residency["chunks"]
    require(isinstance(chunks, list) and bool(chunks), "nonempty resident chunks")
    for chunk in chunks:
        positive(chunk, "chunk")
        require(chunk <= 65535, "u16 launch argument")
    require(sum(chunks) == p["STEPS"], "chunks must sum to total steps")
    require(p["STEPS"] <= U32_MAX and len(chunks) <= U32_MAX, "lifecycle counter bound")
    ins, outs = m["inputs"], m["outputs"]
    require(isinstance(ins, Mapping) and isinstance(outs, Mapping) and not (set(ins) & set(outs)),
            "distinct input/output symbols")
    require(set(arrays) == set(ins), "input inventory")
    require(tuple(ins["counts"]) == (2, "uint32") and
            tuple(ins["velocity"]) == (2*p["L"], "float32"), "resident input ABI")
    require(tuple(outs["stats"]) == (16, "uint32") and
            tuple(outs["lifecycle"]) == (4, "uint32") and
            tuple(outs["clocks"]) == (6, "uint16"), "resident diagnostic ABI")
    for name, spec in ins.items():
        _array(arrays[name], spec, pes, name, carrier=False)
    counts = arrays["counts"]
    require(bool((counts[:, 0] <= p["L"]).all()) and bool((counts[:, 1] <= p["E"]).all()),
            "owner count exceeds capacity")

    schema = {"initial_stats": outs["stats"], "initial_lifecycle": outs["lifecycle"]}
    schema.update({"before_" + name: spec for name, spec in ins.items()})
    for index in range(len(chunks)):
        schema.update({f"chunk{index}_" + name: spec for name, spec in outs.items()})
        schema[f"chunk{index}_velocity"] = ins["velocity"]
    schema.update({"after_" + name: spec for name, spec in ins.items() if name != "velocity"})
    require(set(raw) == set(schema), "raw inventory")
    for name, spec in schema.items():
        _array(raw[name], spec, pes, name, carrier=True)
    require(bool((raw["initial_stats"] == 0).all()), "initial stats must be zero")
    require(bool((raw["initial_lifecycle"] == 0).all()), "initial lifecycle must be zero")
    for name, value in arrays.items():
        _same_bits(raw["before_" + name], value, "before_" + name)
        if name != "velocity":
            _same_bits(raw["after_" + name], value, "after_" + name)

    api = ["construct"] + ["get_id:" + name for name in dict(ins, **outs)] + ["load", "run"]
    api += ["h2d:" + name for name in arrays]
    api += ["d2h:initial_stats", "d2h:initial_lifecycle"]
    api += ["d2h:before_" + name for name in arrays]
    for index in range(len(chunks)):
        api += ["launch:run"] + [f"d2h:chunk{index}_" + name for name in outs]
        api += [f"d2h:chunk{index}_velocity"]
    api += ["d2h:after_" + name for name in arrays if name != "velocity"] + ["stop"]
    require(execution.get("status") == "COMPLETED", "capture did not complete")
    require(execution.get("api") == api, "exact resident API sequence")
    launches = execution.get("launches")
    require(isinstance(launches, list) and len(launches) == len(chunks), "launch argument inventory")
    for record, steps in zip(launches, chunks):
        require(isinstance(record, dict) and set(record) == {"entrypoint", "steps"} and
                record["entrypoint"] == "run" and type(record["steps"]) is int and record["steps"] == steps,
                "launch argument differs from chunk")

    checkpoints, cumulative = [], 0
    for index, steps in enumerate(chunks):
        cumulative += steps
        values = {name: raw[f"chunk{index}_" + name] for name in outs}
        values["velocity"] = raw[f"chunk{index}_velocity"]
        lifecycle = np.tile(np.array([1, index+1, cumulative, 1], np.uint32), (pes, 1))
        require(np.array_equal(values["lifecycle"], lifecycle), "resident lifecycle reset or reinitialization")
        require(np.array_equal(values["stats"], _stats(p, counts, steps, index == 0)),
                "per-launch operation counters")
        clocks = values["clocks"].astype(np.uint64)
        require(bool((clocks <= 65535).all()), "u16 clock carrier corruption")
        begin = clocks[:, 0] + (clocks[:, 1] << 16) + (clocks[:, 2] << 32)
        end = clocks[:, 3] + (clocks[:, 4] << 16) + (clocks[:, 5] << 32)
        spans = (end - begin) & ((1 << 48) - 1)
        require(bool(((spans > 0) & (spans < (1 << 47))).all()), "invalid local 48-bit clock span")
        checkpoints.append(dict(steps=steps, cumulative_steps=cumulative, first=index == 0, raw=values))
    return checkpoints
