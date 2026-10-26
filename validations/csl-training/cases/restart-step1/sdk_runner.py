"""Capture complete CSL supervised updates; no host gradient or optimizer."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(case, build, output, backend="simulator", cmaddr=None):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    case, output = Path(case), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((case / "case.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(case / name) != expected:
            raise ValueError("frozen case changed: " + name)
    p = manifest["params"]
    static = dict(np.load(case / "inputs.npz", allow_pickle=False))
    samples = dict(np.load(case / "samples.npz", allow_pickle=False))
    schema = dict(manifest["inputs"], **manifest["outputs"])
    raw, calls, runtime = {}, [], None
    report = dict(status="FAILED", backend=backend, case_sha256=sha(case / "case.json"),
                  local_build_sha256=sha(case / "build.json") if (case / "build.json").is_file() else None)
    started = time.monotonic()

    def call(name, function, *args, **kwargs):
        begin = time.monotonic()
        print(json.dumps(dict(api=name, event="begin")), flush=True)
        result = function(*args, **kwargs)
        calls.append(dict(api=name, seconds=time.monotonic()-begin))
        print(json.dumps(dict(api=name, event="end")), flush=True)
        return result

    def options(dtype):
        return dict(streaming=False, nonblock=False, order=MemcpyOrder.ROW_MAJOR,
            data_type=MemcpyDataType.MEMCPY_16BIT if dtype == "uint16" else MemcpyDataType.MEMCPY_32BIT)

    try:
        extra = {} if backend == "hardware" else dict(simfab_numthreads=1, suppress_simfab_trace=True)
        runtime = call("construct", SdkRuntime, str(build), cmaddr=cmaddr, **extra)
        ids = {name: call("get_id:" + name, runtime.get_id, name) for name in schema}
        call("load", runtime.load)
        call("run", runtime.run)

        def read(label, name):
            count, dtype = schema[name]
            value = np.full((p["C"] * p["R"], count),
                            np.nan if dtype == "float32" else 0xdead,
                            np.float32 if dtype == "float32" else np.uint32)
            call("d2h:" + label, runtime.memcpy_d2h, value.ravel(), ids[name],
                 0, 0, p["C"], p["R"], count, **options(dtype))
            raw[label] = value

        def write(name, value):
            count, dtype = schema[name]
            call("h2d:" + name, runtime.memcpy_h2d, ids[name], value.ravel(),
                 0, 0, p["C"], p["R"], count, **options(dtype))

        for name, value in static.items():
            write(name, value)
        read("initial_lifecycle", "lifecycle")
        read("initial_weights", "weights")
        for step in range(manifest["steps"]):
            for name, value in samples.items():
                write(name, value[step])
                read(f"step{step}_input_{name}", name)
            call("launch:run", runtime.launch, "run", nonblock=False)
            for name in manifest["outputs"]:
                read(f"step{step}_{name}", name)
            read(f"step{step}_weights", "weights")
        for name in static:
            if name != "weights":
                read("after_" + name, name)
    except BaseException as error:
        report["error"] = repr(error)
    finally:
        if runtime is not None:
            try:
                call("stop", runtime.stop)
            except BaseException as error:
                report["stop_error"] = repr(error)
        report.update(calls=calls, seconds=time.monotonic()-started)
        if "error" not in report and "stop_error" not in report:
            report["status"] = "COMPLETED"
        np.savez_compressed(output / "raw.npz", **raw)
        report["raw_sha256"] = sha(output / "raw.npz")
        (output / "execution.json").write_text(json.dumps(report, indent=2)+"\n")
    if report["status"] != "COMPLETED":
        raise RuntimeError(report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--backend", choices=("simulator", "hardware"), default="simulator")
    ap.add_argument("--cmaddr")
    args = ap.parse_args()
    main(args.case, args.build, args.output, args.backend, args.cmaddr)
