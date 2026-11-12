"""Repeated complete native GCN observation, separate from frozen E217 captures."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.hierarchical_gnn.grouped_csr.native_cora_appliance import run as base
from models.hierarchical_gnn.grouped_csr.native_cora_appliance import saved_audit as audit

STAGES = ("projection", "row_reduce_wait", "row_broadcast_wait", "packing",
          "column_delivery_wait", "aggregation", "local_control")
EXTRA = dict(stage_cycles=[14, "uint32"], stage_calls=[14, "uint32"],
             stage_invalid=[1, "uint32"], completion=[1, "uint32"])
SCOPE = {
    "host_completed": "host launch entry through blocking fresh completion epoch read from every application PE; includes control and completion-read transfer",
    "local_cycles": "same-PE TSC after reset through final output stores; no common start or global makespan",
    "stages": "PROFILE=1 cumulative same-PE, same-layer intervals; collective waits include transport, synchronization and scheduling; timestamp/accounting overhead is charged to following interval",
    "composition": "do not add per-PE maxima or compare local cycles to CUDA whole time; only stages from the same PE can be stacked",
    "repetition": "same runtime/input; output diagnostics between launches are outside completed window",
    "control": "PROFILE=0 has epoch instrumentation and differs from frozen E226 source",
}


def dedicated(path, fresh=False):
    path = Path(path).absolute()
    root = ROOT/"research/runs"
    audit.require(path == path.resolve() and path != root and path.is_relative_to(root), "dedicated research/runs descendant")
    if fresh:
        path.mkdir(parents=True, exist_ok=False)
    return path


def sources():
    return {name: audit.sha(HERE/name) for name in ("run.py", "pe.csl", "layout.csl")}


def manifest_for(case, profile):
    manifest = audit.load(case)
    audit.require(manifest["params"].get("COMPACT") == 1 and "UNICAST" not in manifest["params"], "compact grouped CSR case")
    audit.require(profile in (0, 1), "profile flag")
    return dict(manifest, params=dict(manifest["params"], PROFILE=profile),
                outputs=dict(manifest["outputs"], **EXTRA))


def compile_case(case, profile):
    case = dedicated(case)
    manifest = manifest_for(case, profile)
    audit.require(manifest["admission"]["admitted"], "base storage forecast admitted")
    audit.require(manifest["admission"]["forecast_bytes"] + 132 < 44*1024, "observer storage forecast admitted")
    destination = case/f"observation-{profile}"
    destination.mkdir(exist_ok=False)
    (destination/"source").mkdir()
    for name in ("pe.csl", "layout.csl"):
        shutil.copyfile(HERE/name, destination/"source"/name)
    flags = shlex.split(audit.compiler_arguments(manifest["params"], "simulator", 1))
    flags[-1] = str(destination/"build")
    command = [str(ROOT/"cslc"), str(destination/"source/layout.csl"), *flags]
    record = dict(status="FAILED", backend="simulator", profile=profile,
                  case_sha256=audit.sha(case/"case.json"), sources=sources(), command=command,
                  compiler_wrapper_sha256=audit.sha(ROOT/"cslc"), runtime_wrapper_sha256=audit.sha(ROOT/"cs_python"))
    started = time.monotonic()
    try:
        record["returncode"] = subprocess.run(command, cwd=ROOT).returncode
        audit.require(record["returncode"] == 0, "observer compiler failed")
        audit.check_abi(destination/"build", manifest)
        record["inventory"] = audit.compiled_inventory(destination/"build")
        audit.admit_compiled_inventory(record["inventory"])
        audit.require(record["sources"] == sources(), "observer source changed")
        record["status"] = "PASS"
    except BaseException as error:
        record["error"] = str(error)
        raise
    finally:
        record["wall_seconds"] = time.monotonic()-started
        audit.write_json(destination/"compile.json", record, limit=audit.INVENTORY_JSON_LIMIT)
    return record


def compiled(case, profile):
    case = dedicated(case)
    destination = case/f"observation-{profile}"
    record = audit.read_json(destination/"compile.json", limit=audit.INVENTORY_JSON_LIMIT)
    audit.require(record.get("status") == "PASS" and record.get("returncode") == 0 and "error" not in record, "successful observer compile")
    audit.require(record["case_sha256"] == audit.sha(case/"case.json") and record["sources"] == sources(), "compiled observer identity")
    audit.require(audit.compiled_inventory(destination/"build") == record["inventory"], "observer ELF identity")
    audit.check_abi(destination/"build", manifest_for(case, profile))
    audit.admit_compiled_inventory(record["inventory"])
    return record


def summary(values):
    value = np.asarray(values)
    return dict(min=int(value.min()), mean=float(value.mean()), max=int(value.max()),
                per_pe=value.astype(np.uint64).tolist())


def decode_profile(raw, profile):
    clocks = np.asarray(raw["clocks"])
    audit.require(clocks.ndim == 2 and clocks.shape[1] == 6 and clocks.dtype == np.uint32
                  and np.all(clocks <= 65535), "six u16 clock carriers")
    w = clocks.astype(np.uint64)
    ticks = w[:, 0::3] | (w[:, 1::3] << np.uint64(16)) | (w[:, 2::3] << np.uint64(32))
    elapsed = (ticks[:, 1]-ticks[:, 0]) & np.uint64((1 << 48)-1)
    audit.require(np.all((elapsed > 0) & (elapsed < (1 << 47))), "positive unambiguous same-PE interval")
    cycles = np.asarray(raw["stage_cycles"])
    calls = np.asarray(raw["stage_calls"])
    audit.require(cycles.shape == calls.shape == (len(w), 14) and cycles.dtype == calls.dtype == np.uint32,
                  "stage counter ABI")
    audit.require(np.array_equal(raw["stage_invalid"], np.zeros((len(w), 1), np.uint32)), "no interval or accumulator overflow")
    if profile == 0:
        audit.require(not cycles.any() and not calls.any(), "profile-disabled counters")
        stages = {}
    else:
        audit.require(np.all(calls[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13]] > 0), "each layer phase observed")
        audit.require(np.all(cycles.sum(axis=1, dtype=np.uint64) <= elapsed), "same-PE stage sum inside local interval")
        stages = {f"layer{layer+1}.{name}": summary(cycles[:, layer*7+i])
                  for layer in range(2) for i, name in enumerate(STAGES)}
    return dict(local_interval_cycles=summary(elapsed), stages=stages,
                stage_calls_per_pe=calls.tolist())


def numerical(raw, expected, context):
    errors = {}
    for name, wanted in expected.items():
        actual = raw[name]
        audit.require(np.isfinite(actual).all(), "finite "+name)
        if wanted.dtype.kind in "ui":
            audit.require(np.array_equal(actual, wanted), "exact "+name)
        else:
            delta = np.abs(actual.astype(np.float64)-wanted)
            audit.require(np.all(delta <= audit.ATOL+audit.RTOL*np.abs(wanted)), "independent FP64 "+name)
            errors[name] = dict(max_abs=float(delta.max()), max_relative_floor_1e_12=float((delta/np.maximum(np.abs(wanted), 1e-12)).max()),
                                relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(wanted), 1e-12)))
    nodes = context["nodes"]
    for name, width in (("pre", 16), ("logits", 7)):
        packed = raw[name].reshape(*nodes.shape, width)
        audit.require(np.all(packed[nodes < 0] == 0), "zero owner padding "+name)
    restored = np.empty_like(context["logits"])
    restored[nodes[nodes >= 0]] = raw["logits"].reshape(*nodes.shape, 7)[nodes >= 0]
    predicted, reference = restored.argmax(axis=1), context["logits"].argmax(axis=1)
    mismatch = int(np.count_nonzero(predicted != reference))
    original = context["original"]
    # Native full-Cora requires every prediction, not merely validation accuracy.
    if original["full"]:
        audit.require(mismatch == 0, "all native predictions match FP64")
    valid = original["valid"] if original["full"] else None
    accuracy = None if valid is None else dict(correct=int(np.sum(predicted[valid] == original["labels"][valid])), count=len(valid),
                                               fraction=float(np.mean(predicted[valid] == original["labels"][valid])), split="custom validation; not Planetoid test")
    return dict(status="PASS", errors=errors, argmax_mismatches=mismatch, accuracy=accuracy)


def completion_check(values, epoch, pes):
    audit.require(values.dtype == np.uint32 and values.shape == (pes, 1)
                  and np.all(values == epoch), "fresh completed epoch from every application PE")


def capture(case, build, output, profile, warmup=1, repeats=3, backend="simulator", cmaddr=None, sdk=None):
    """Registered hardware adapters may call this with their staged actual build."""
    audit.require(backend in ("simulator", "hardware") and (backend == "hardware") == bool(cmaddr), "explicit backend/endpoint")
    audit.require(type(warmup) is int and type(repeats) is int and 0 <= warmup <= 10 and 1 <= repeats <= 30 and warmup+repeats <= 25, "bounded repetition count and raw archive inventory")
    if sdk is None:
        from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    else:
        SdkRuntime, MemcpyDataType, MemcpyOrder = sdk
    case, build, output = Path(case).resolve(), Path(build).resolve(), Path(output).absolute()
    manifest = manifest_for(case, profile)
    audit.check_abi(build, manifest)
    inventory = audit.compiled_inventory(build)
    audit.admit_compiled_inventory(inventory)
    expected, context = audit.independent_expected(case, audit.load(case))
    arrays = audit.load_npz(case/"inputs.npz")
    output.mkdir(parents=True, exist_ok=False)
    fields = dict(manifest["inputs"], **manifest["outputs"])
    p = manifest["params"]
    pes, raw, runtime = p["C"]*p["R"], {}, None
    record = dict(schema=1, family="native-cora-observation", status="RUNNING", backend=backend,
                  profile=profile, params=p, case_sha256=audit.sha(case/"case.json"), sources=sources(),
                  input_sha256={name: audit.array_sha(context["original"][name]) for name in ("x", "src", "dst", "coeff", "w0", "w1")},
                  warmup=warmup, repeats=repeats, iterations=[], api_timings=[], timing_scope=SCOPE,
                  memory=dict(named_baseline_bytes_per_pe=audit.named_bytes(audit.load(case)["params"]),
                              observer_added_named_bytes_per_pe=132, static_inventory=inventory,
                              runtime_stack_peak_bytes=None, scope="compiled static endpoints; runtime stack high water unmeasured"))
    start_session = time.monotonic_ns()

    def call(label, fn, *args, **kwargs):
        record["pending_api"] = label
        start = time.monotonic_ns()
        result = fn(*args, **kwargs)
        end = time.monotonic_ns()
        record["api_timings"].append(dict(api=label, start_ns=start, end_ns=end, seconds=(end-start)/1e9))
        record.pop("pending_api", None)
        return result

    def opts(dtype):
        return dict(streaming=False, nonblock=False, order=MemcpyOrder.ROW_MAJOR,
                    data_type=MemcpyDataType.MEMCPY_16BIT if dtype == "uint16" else MemcpyDataType.MEMCPY_32BIT)

    def read(name, label):
        count, dtype = fields[name]
        value = np.full((pes, count), np.nan if dtype == "float32" else 0xDEAD,
                        np.float32 if dtype == "float32" else np.uint32)
        try:
            call("d2h:"+label, runtime.memcpy_d2h, value.ravel(), ids[name], 0, 0, p["C"], p["R"], count, **opts(dtype))
        finally:
            raw[label] = value.copy()
        audit.require(np.isfinite(value).all(), "finite capture "+label)
        if dtype == "uint16":
            audit.require(np.all(value <= 65535), "u16 carrier "+label)
        return value

    def immutable(prefix):
        for name, wanted in arrays.items():
            got = read(name, prefix+name)
            carrier = wanted.astype(np.uint32) if wanted.dtype == np.uint16 else wanted
            audit.require(np.array_equal(got.view(np.uint32), carrier.view(np.uint32)), "immutable "+name)

    def persist():
        np.savez_compressed(output/"raw.npz", **raw)
        record["raw_sha256"] = audit.sha(output/"raw.npz", audit.CAPTURE_LIMIT)
        audit.write_json(output/"execution.json", record, limit=audit.INVENTORY_JSON_LIMIT)

    try:
        options = dict(cmaddr=cmaddr)
        if backend == "simulator":
            options.update(simfab_numthreads=1, suppress_simfab_trace=True)
        # Retain native ownership before any fallible persistence.
        runtime = call("construct", SdkRuntime, str(build), **options)
        ids = {name: call("get_id:"+name, runtime.get_id, name) for name in fields}
        call("load", runtime.load)
        call("run", runtime.run)
        for name, values in arrays.items():
            carrier = values.astype(np.uint32) if values.dtype == np.uint16 else values
            call("h2d:"+name, runtime.memcpy_h2d, ids[name], carrier.ravel(), 0, 0, p["C"], p["R"], values.shape[1], **opts(str(values.dtype)))
        immutable("before_")
        completion_check(read("completion", "initial_completion"), 0, pes)
        previous = None
        for epoch in range(1, warmup+repeats+1):
            # Allocate the completion destination before the interval. No JSON,
            # oracle work, tensor diagnostics or disk writes occur in this window.
            complete = np.full((pes, 1), 0xDEAD, np.uint32)
            record["pending_api"] = "launch-and-completion"
            begin = time.monotonic_ns()
            runtime.launch("f_run", nonblock=False)
            launch_return = time.monotonic_ns()
            runtime.memcpy_d2h(complete.ravel(), ids["completion"], 0, 0, p["C"], p["R"], 1, **opts("uint32"))
            complete_return = time.monotonic_ns()
            record.pop("pending_api", None)
            raw[f"e{epoch}_completion"] = complete
            completion_check(complete, epoch, pes)
            record["api_timings"].extend([
                dict(api="launch:f_run", start_ns=begin, end_ns=launch_return, seconds=(launch_return-begin)/1e9),
                dict(api="d2h:completion", start_ns=launch_return, end_ns=complete_return, seconds=(complete_return-launch_return)/1e9)])
            observed = {name: read(name, f"e{epoch}_"+name) for name in manifest["outputs"] if name != "completion"}
            observed["completion"] = complete
            numerics = numerical(observed, expected, context)
            if previous is not None:
                for name in ("pre", "logits", "panel_sum"):
                    audit.require(np.array_equal(observed[name].view(np.uint32), previous[name].view(np.uint32)), "same-input repeated bits "+name)
            previous = observed
            record["iterations"].append(dict(epoch=epoch, warmup=epoch <= warmup,
                launch_to_completion_seconds=(complete_return-begin)/1e9,
                launch_call_seconds=(launch_return-begin)/1e9, completion_read_seconds=(complete_return-launch_return)/1e9,
                numerics=numerics, **decode_profile(observed, profile)))
            persist()
        immutable("after_")
        record["status"] = "PASS"
    except BaseException as error:
        record.update(status="FAILED", error=type(error).__name__+": "+str(error))
    finally:
        if runtime is not None:
            try:
                call("stop", runtime.stop)
            except BaseException as error:
                record.update(status="FAILED", stop_error=str(error))
        record["session_seconds"] = (time.monotonic_ns()-start_session)/1e9
        record["host_api_seconds"] = {label: sum(row["seconds"] for row in record["api_timings"] if row["api"] == label)
                                     for label in ("construct", "load", "run", "stop")}
        record["host_api_seconds"]["initial_upload"] = sum(row["seconds"] for row in record["api_timings"] if row["api"].startswith("h2d:"))
        record["host_api_seconds"]["diagnostic_readback"] = sum(row["seconds"] for row in record["api_timings"] if row["api"].startswith("d2h:") and row["api"] != "d2h:completion")
        persist()
    audit.require(record["status"] == "PASS", "observer capture failed; partial evidence retained: "+str(output))
    return record


def simulate(case, output, profile, warmup=1, repeats=3):
    case, output = dedicated(case), dedicated(output, fresh=True)
    build_record = compiled(case, profile)
    work = output/"sdk-work"
    work.mkdir()
    shutil.copytree(case/f"observation-{profile}/build", work/"build")
    audit.require(audit.compiled_inventory(work/"build") == build_record["inventory"], "copied build identity")
    command = [str(ROOT/"cs_python"), "-B", str(HERE/"run.py"), "capture", "--case", str(case),
               "--build", str(work/"build"), "--out", str(output/"runtime"), "--profile", str(profile),
               "--warmup", str(warmup), "--repeats", str(repeats)]
    result = dict(status="FAILED", backend="simulator", command=command, compile_sha256=audit.sha(case/f"observation-{profile}/compile.json", audit.INVENTORY_JSON_LIMIT))
    began = time.monotonic()
    try:
        result["returncode"] = subprocess.run(command, cwd=ROOT).returncode
        audit.require(result["returncode"] == 0, "observer simulator failed")
        result.update(audit.read_json(output/"runtime/execution.json", limit=audit.INVENTORY_JSON_LIMIT))
        compiled(case, profile)
    except BaseException as error:
        result.update(status="FAILED", error=type(error).__name__+": "+str(error))
        raise
    finally:
        result["simulator_process_wall_seconds"] = time.monotonic()-began
        audit.write_json(output/"result.json", result, limit=audit.INVENTORY_JSON_LIMIT)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "compile", "simulate", "capture"))
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--profile", type=int, choices=(0, 1), default=1)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--kind", choices=("tiny", "cora"), default="tiny")
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--tile", type=int, choices=(4, 8, 16), default=16)
    parser.add_argument("--backend", choices=("simulator", "hardware"), default="simulator")
    parser.add_argument("--cmaddr")
    args = parser.parse_args()
    if args.action == "prepare":
        result = base.prepare(args.case, kind=args.kind, columns=args.columns, rows=args.rows, tile=args.tile,
                              transport="compact", tiny_demand=args.kind == "tiny")
    elif args.action == "compile":
        audit.require(args.backend == "simulator", "local compiler entrypoint is simulator only; hardware uses registered compile workflow")
        result = compile_case(args.case, args.profile)
    elif args.action == "simulate":
        audit.require(args.out is not None, "output required")
        result = simulate(args.case, args.out, args.profile, args.warmup, args.repeats)
    else:
        audit.require(args.build is not None and args.out is not None, "build/output required")
        audit.require(args.backend == "simulator" and args.cmaddr is None, "hardware capture runs only through registered hardware_entry worker")
        # SDK side effects stay in its private staged working directory.
        os.chdir(args.build.resolve().parent)
        result = capture(args.case, args.build, args.out, args.profile, args.warmup, args.repeats, args.backend, args.cmaddr)
    print(json.dumps({k: result[k] for k in ("status", "backend", "profile", "case_sha256") if k in result}, sort_keys=True))


if __name__ == "__main__":
    main()
