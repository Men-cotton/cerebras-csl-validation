"""Raw SDK capture with a frozen launch schedule; no numerical audit."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def main(case, build, output, backend, cmaddr=None):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    if backend not in ("hardware", "simulator") or (backend == "hardware" and not cmaddr):
        raise ValueError("explicit backend and hardware endpoint required")
    if backend == "simulator" and cmaddr is not None:
        raise ValueError("simulator must not use hardware endpoint")
    case, build, output = Path(case), Path(build), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((case / "case.json").read_text())
    params = manifest["params"]
    arrays = dict(np.load(case / "inputs.npz", allow_pickle=False))
    raw, api, timings, runtime = {}, [], [], None
    launches = []
    result = dict(status="FAILED", backend=backend,
                  case_sha256=hashlib.sha256((case / "case.json").read_bytes()).hexdigest())
    started = time.monotonic()

    def call(label, fn, *args, **kwargs):
        began = time.monotonic()
        print(json.dumps(dict(api=label, event="begin", seconds=began - started)), flush=True)
        value = fn(*args, **kwargs)
        api.append(label)
        timings.append(dict(api=label, wall_seconds=time.monotonic() - began))
        print(json.dumps(dict(api=label, event="end", seconds=time.monotonic() - started)), flush=True)
        return value

    try:
        options = dict(cmaddr=cmaddr)
        if backend == "simulator":
            options.update(simfab_numthreads=1, suppress_simfab_trace=True)
        runtime = call("construct", SdkRuntime, str(build), **options)
        schema = dict(manifest["inputs"], **manifest["outputs"])
        ids = {name: call("get_id:" + name, runtime.get_id, name) for name in schema}
        call("load", runtime.load)
        call("run", runtime.run)

        def opts(dtype):
            return dict(streaming=False, nonblock=False, order=MemcpyOrder.ROW_MAJOR,
                        data_type=MemcpyDataType.MEMCPY_16BIT if dtype == "uint16" else MemcpyDataType.MEMCPY_32BIT)

        def read(label, name):
            count, dtype = schema[name]
            values = np.full(params["C"] * params["R"] * count,
                             np.nan if dtype == "float32" else 0xDEAD,
                             np.float32 if dtype == "float32" else np.uint32)
            try:
                call("d2h:" + label, runtime.memcpy_d2h, values, ids[name],
                     0, 0, params["C"], params["R"], count, **opts(dtype))
            finally:
                raw[label] = values.reshape(params["C"] * params["R"], count).copy()

        for name in manifest["inputs"]:
            values = arrays[name]
            call("h2d:" + name, runtime.memcpy_h2d, ids[name], values.ravel(),
                 0, 0, params["C"], params["R"], values.shape[1], **opts(str(values.dtype)))
        for name in manifest["inputs"]:
            read("before_" + name, name)
        read("initial_stats", "stats")
        if manifest["schema"] == 2:
            read("initial_lifecycle", "lifecycle")
            for index, launch in enumerate(manifest["launches"]):
                call("launch:run", runtime.launch, "run", np.uint32(launch["steps"]), nonblock=False)
                launches.append(dict(entrypoint="run", steps=int(launch["steps"])))
                for name in manifest["outputs"]:
                    read(f"launch{index}_{name}", name)
        else:
            call("launch:run", runtime.launch, "run", nonblock=False)
            for name in manifest["outputs"]:
                read(name, name)
        for name in manifest["inputs"]:
            read("after_" + name, name)
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    finally:
        try:
            if runtime is not None:
                call("stop", runtime.stop)
        except Exception as error:
            result["stop_error"] = repr(error)
        result.update(api=api, api_timings=timings, wall_seconds=time.monotonic() - started)
        if manifest["schema"] == 2:
            result["launches"] = launches
        if "error" not in result and "stop_error" not in result:
            result["status"] = "COMPLETED"
        save = np.savez_compressed
        save(output / "raw.npz", **raw)
        result["raw_sha256"] = hashlib.sha256((output / "raw.npz").read_bytes()).hexdigest()
        (output / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "COMPLETED":
        raise RuntimeError("SDK capture incomplete; partial arrays retained: " + str(output))
