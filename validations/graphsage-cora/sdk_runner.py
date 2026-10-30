# Transport provenance: adapted from models/hierarchical_gnn/grouped_csr/native_cora_appliance/sdk_runner.py (E217).
"""Native-feature Cora GraphSAGE inference capture using the ordinary explicit-lifecycle runtime."""
from pathlib import Path
import os
import time
import uuid
import numpy as np
try:
    from . import saved_audit as common
except ImportError:
    import saved_audit as common


class Journal:
    """Keep the native handle even when constructor-exit persistence fails."""
    def __init__(self, record, persist):
        self.record, self.persist = record, persist
        self.runtime, self.stop_invoked = None, False
        record.update(api=[], api_timings=[], launch_arguments=[], launch_typed_arguments=[],
                      clock_domain=f"pid:{os.getpid()}:monotonic_ns:{uuid.uuid4().hex}")

    def call(self, label, fn, *args, **kwargs):
        self.record["pending_api"] = label
        self.record["last_api"] = label
        entry_error = None
        try:
            self.persist()
        except BaseException as error:
            if label != "stop":
                raise
            entry_error = error
        began = time.monotonic_ns()
        if label == "stop":
            common.require(not self.stop_invoked, "native stop exactly once")
            self.stop_invoked = True
        value = fn(*args, **kwargs)
        if label == "construct":
            # Ownership must precede every fallible journal operation.
            self.runtime = value
        ended = time.monotonic_ns()
        self.record["api"].append(label)
        self.record["api_timings"].append(dict(api=label, method=fn.__name__, start_ns=began, end_ns=ended, wall_seconds=(ended-began)/1e9,
                                                **({"symbol": args[0], "nonblock": kwargs.get("nonblock")} if label.startswith("launch:") else {}),
                                                **({"nonblock": kwargs.get("nonblock")} if label.startswith(("h2d:", "d2h:")) else {})))
        self.record.pop("pending_api", None)
        if label.startswith("launch:"):
            self.record["launch_arguments"].append([int(value) for value in args[1:]])
            self.record["launch_typed_arguments"].append([dict(dtype=str(np.asarray(value).dtype), value=int(value)) for value in args[1:]])
        try:
            self.persist()
        except BaseException:
            if entry_error is None:
                raise
        if entry_error is not None:
            raise entry_error
        return value

    def stop(self):
        if self.runtime is not None and not self.stop_invoked:
            self.call("stop", self.runtime.stop)


def main(case, build, output, backend, cmaddr=None, *, sdk=None, persist_hook=None):
    common.require(backend in ("hardware", "simulator") and (backend != "hardware" or bool(cmaddr)), "explicit backend/endpoint")
    common.require(backend != "simulator" or cmaddr is None, "simulator cannot use hardware endpoint")
    if sdk is None:
        from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    else:
        SdkRuntime, MemcpyDataType, MemcpyOrder = sdk
    case, build, output = Path(case).absolute(), Path(build).absolute(), Path(output).absolute()
    manifest = common.load(case)
    common.require(manifest["admission"]["admitted"], "case static forecast rejected")
    common.check_abi(build, manifest)
    output.mkdir(parents=True, exist_ok=False)
    params = manifest["params"]
    arrays, raw = common.load_npz(case/"inputs.npz"), {}
    record = dict(status="RUNNING", backend=backend, case_sha256=common.sha(case/"case.json"))
    began = time.monotonic()

    def persist():
        if persist_hook is not None:
            persist_hook(record)
        common.write_json(output/"execution.json", record)

    journal = Journal(record, persist)
    try:
        options = dict(cmaddr=cmaddr)
        if backend == "simulator":
            options.update(simfab_numthreads=1, suppress_simfab_trace=True)
        runtime = journal.call("construct", SdkRuntime, str(build), **options)
        inputs, outputs = common.schema(params), common.schema(params, True)
        combined = dict(inputs, **outputs)
        ids = {name: journal.call("get_id:"+name, runtime.get_id, name) for name in combined}
        journal.call("load", runtime.load)
        journal.call("run", runtime.run)

        def opts(dtype):
            return dict(streaming=False, nonblock=False, order=MemcpyOrder.ROW_MAJOR,
                        data_type=MemcpyDataType.MEMCPY_16BIT if dtype == "uint16" else MemcpyDataType.MEMCPY_32BIT)

        def read(label, symbol):
            count, dtype = combined[symbol]
            values = np.full(params["C"]*params["R"]*count, np.nan if dtype == "float32" else 0xDEAD,
                             np.float32 if dtype == "float32" else np.uint32)
            try:
                journal.call("d2h:"+label, runtime.memcpy_d2h, values, ids[symbol],
                             0, 0, params["C"], params["R"], count, **opts(dtype))
            finally:
                raw[label] = values.reshape(params["C"]*params["R"], count).copy()

        for name in inputs:
            values = arrays[name]
            dtype = str(values.dtype)
            carrier = values.astype(np.uint32) if dtype == "uint16" else values
            journal.call("h2d:"+name, runtime.memcpy_h2d, ids[name], carrier.ravel(),
                         0, 0, params["C"], params["R"], values.shape[1], **opts(dtype))
        read("upload_tail", "slots")
        read("initial_stats", "stats")
        journal.call("launch:f_run", runtime.launch, "f_run", nonblock=False)
        for name in outputs:
            read(name, name)
    except BaseException as error:
        record["error"] = type(error).__name__+": "+str(error)
        record["failed_api"] = record.get("pending_api", record.get("last_api"))
    finally:
        try:
            journal.stop()
        except BaseException as error:
            record["stop_error"] = type(error).__name__+": "+str(error)
        record["wall_seconds"] = time.monotonic()-began
        record["status"] = "FAILED" if "error" in record or "stop_error" in record else "COMPLETED"
        np.savez_compressed(output/"raw.npz", **raw)
        record["raw_sha256"] = common.sha(output/"raw.npz", common.CAPTURE_LIMIT)
        common.write_json(output/"execution.json", record)
    if record["status"] != "COMPLETED":
        raise RuntimeError("incomplete SDK capture; partial arrays retained: "+str(output))
    common.admit_execution(case, output)
