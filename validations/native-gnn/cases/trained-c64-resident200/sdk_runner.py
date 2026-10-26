"""SDK raw capture only; one H2D upload, resident chunks, final immutable readback."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def main(case):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    m = json.loads((case/"case.json").read_text())
    p = m["params"]
    arrays = dict(np.load(case/"inputs.npz", allow_pickle=False))
    raw, api, runner, timings, launches = {}, [], None, [], []
    result = dict(status="FAILED", case_sha256=hashlib.sha256((case/"case.json").read_bytes()).hexdigest())
    start = time.monotonic()

    def call(label, fn, *args, **kwargs):
        began = time.monotonic()
        print(json.dumps(dict(api=label, event="begin", elapsed_seconds=began-start)), flush=True)
        out = fn(*args, **kwargs)
        api.append(label)
        timings.append(dict(api=label, wall_seconds=time.monotonic()-began))
        print(json.dumps(dict(api=label, event="end", wall_seconds=timings[-1]["wall_seconds"])), flush=True)
        return out

    try:
        runner = call("construct", SdkRuntime, str(case/"build"), cmaddr=None,
                      simfab_numthreads=1, suppress_simfab_trace=True)
        schema = dict(m["inputs"], **m["outputs"])
        ids = {n: call("get_id:"+n, runner.get_id, n) for n in schema}
        call("load", runner.load)
        call("run", runner.run)

        def opts(dtype):
            return dict(streaming=False, nonblock=False, order=MemcpyOrder.ROW_MAJOR,
                        data_type=MemcpyDataType.MEMCPY_16BIT if dtype == "uint16" else MemcpyDataType.MEMCPY_32BIT)

        def read(label, name):
            count, dtype = schema[name]
            a = np.full(p["C"]*p["R"]*count, np.nan if dtype == "float32" else 0xdead,
                        np.float32 if dtype == "float32" else np.uint32)
            try:
                call("d2h:"+label, runner.memcpy_d2h, a, ids[name], 0, 0, p["C"], p["R"], count, **opts(dtype))
            finally:
                raw[label] = a.reshape(p["C"]*p["R"], count).copy()

        for n in m["inputs"]:
            a = arrays[n]
            call("h2d:"+n, runner.memcpy_h2d, ids[n], a.ravel(), 0, 0, p["C"], p["R"], a.shape[1], **opts(str(a.dtype)))
        read("initial_stats", "stats")
        read("initial_lifecycle", "lifecycle")
        for index, steps in enumerate(m["residency"]["chunks"]):
            call("launch:run", runner.launch, "run", np.uint32(steps), nonblock=False)
            launches.append(dict(entrypoint="run", steps=steps))
            for n in m["outputs"]:
                read(f"launch{index}_"+n, n)
        for n in m["inputs"]:
            read("after_"+n, n)
    except Exception as error:
        result["error"] = type(error).__name__+": "+str(error)
    finally:
        try:
            if runner is not None:
                call("stop", runner.stop)
        except Exception as error:
            result["stop_error"] = str(error)
        result.update(api=api, api_timings=timings, launches=launches, wall_seconds=time.monotonic()-start)
        if "error" not in result and "stop_error" not in result:
            result["status"] = "COMPLETED"
        np.savez_compressed(case/"raw.npz", **raw)
        result["raw_sha256"] = hashlib.sha256((case/"raw.npz").read_bytes()).hexdigest()
        (case/"execution.json").write_text(json.dumps(result, indent=2)+"\n")
    if result["status"] != "COMPLETED":
        raise RuntimeError(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    main(parser.parse_args().case.resolve())
