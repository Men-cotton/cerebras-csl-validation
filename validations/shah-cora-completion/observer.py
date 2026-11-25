"""Measure resident Shah GCN with the completed-host boundary of the CSR observer."""
from pathlib import Path
import time
import uuid

import numpy as np

import driver
import oracle
from archive import require

INPUTS = (*driver.ROUTERS, *driver.WORKERS, *driver.ACCUMULATORS)
VALUES = ("pre", "projected", "logits", "h")
STAGES = ("input_projection", "first_fences_and_aggregation", "relu_and_projection",
          "redistribution", "second_fences_and_aggregation")
WARMUP, REPEATS = 1, 5


def rectangle(name, p):
    if name in ("completion", "stages"):
        return 0, 0, p["F"] + 1, 2*p["R"]
    return driver.role(name, p)


def epoch_check(words, p, epoch):
    require(words.shape == (2*p["R"], p["F"] + 1, 1) and words.dtype == np.uint32
            and np.all(words == epoch), "every application PE must report its fresh completed epoch")


def capture(case, build, out, cmaddr=None, mode="gcn", run_id=None):
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    manifest, original, arrays = driver.load_case(case)
    reference = oracle.independent_reference(original)
    p = manifest["params"]
    run_id = run_id or str(uuid.uuid4())
    require(mode == "gcn" and str(uuid.UUID(run_id)) == run_id, "whole-model invocation identity")
    sha = driver.sha(case/"case.json")
    compiled = driver.read(case/"compile.json")
    actual = driver.inventory(build)
    parameters = driver.read(Path(build)/"out.json")["params"]
    require({k: int(v) for k, v in parameters.items()} == p, "actual compiled Shah geometry and profile")
    require(compiled == dict(status="PASS", case_sha256=sha, inventory=actual), "runtime compilation identity")
    expected = {name: driver.region(arrays[name], name, p) for name in INPUTS}
    fields = {name: (p["V"], np.float32) for name in VALUES}
    fields.update(stats=(8, np.uint32), clocks=(6, np.uint16), stages=(12, np.uint16))
    record = dict(run_id=run_id, status="RUNNING", case_sha256=sha, compiled_inventory=actual,
                  mode=mode, profile=p["PROFILE"], backend="hardware" if cmaddr else "simulator",
                  warmup=WARMUP, repeats=REPEATS, iterations=[], api=[])
    raw = {}
    out.mkdir(parents=True, exist_ok=False)
    options = dict(cmaddr=cmaddr) if cmaddr else dict(suppress_simfab_trace=True, simfab_numthreads=1)
    rt = None

    def call(label, fn, *args, **kwargs):
        record["pending"] = label
        begin = time.monotonic_ns()
        value = fn(*args, **kwargs)
        record["api"].append(dict(api=label, start_ns=begin, end_ns=time.monotonic_ns()))
        del record["pending"]
        return value

    def opts(dtype):
        kind = MemcpyDataType.MEMCPY_16BIT if dtype == np.uint16 else MemcpyDataType.MEMCPY_32BIT
        return dict(data_type=kind, streaming=False, nonblock=False, order=MemcpyOrder.ROW_MAJOR)

    def download(key, name, count, dtype):
        x, y, w, h = rectangle(name, p)
        buf = np.full((h, w, count), np.nan if dtype == np.float32 else 0xdead,
                      dtype=np.uint32 if dtype == np.uint16 else dtype)
        call("d2h:"+key, rt.memcpy_d2h, buf.ravel(), ids[name], x, y, w, h, count, **opts(dtype))
        if dtype == np.uint16:
            require(np.all(buf <= 65535), "uint16 carrier upper bits")
        raw[key] = buf.astype(dtype, copy=False)
        return raw[key]

    def snapshot(prefix):
        for name, want in expected.items():
            got = download(prefix+name, name, want.shape[-1], want.dtype)
            np.testing.assert_array_equal(got.view(np.uint8), want.view(np.uint8))

    try:
        rt = call("construct", SdkRuntime, str(build), **options)
        ids = {name: rt.get_id(name) for name in (*INPUTS, *fields, "completion")}
        call("load", rt.load)
        call("run", rt.run)
        for name, a in expected.items():
            carrier = a.astype(np.uint32) if a.dtype == np.uint16 else a
            call("h2d:"+name, rt.memcpy_h2d, ids[name], carrier.ravel(), *rectangle(name, p),
                 a.shape[-1], **opts(a.dtype))
        snapshot("before_")
        epoch_check(download("initial_completion", "completion", 1, np.uint32), p, 0)
        x, y, w, h = rectangle("completion", p)
        completion_id, completion_opts = ids["completion"], opts(np.uint32)
        for epoch in range(1, WARMUP + REPEATS + 1):
            key = f"e{epoch}_"
            words = np.full((h, w, 1), 0xdead, np.uint32)
            dest = words.ravel()
            record["pending"] = "launch-and-completion"
            begin = time.monotonic_ns()
            rt.launch("f_run", nonblock=False)
            returned = time.monotonic_ns()
            rt.memcpy_d2h(dest, completion_id, x, y, w, h, 1, **completion_opts)
            completed = time.monotonic_ns()
            del record["pending"]
            raw[key+"completion"] = words
            epoch_check(words, p, epoch)
            record["iterations"].append(dict(epoch=epoch, warmup=epoch <= WARMUP,
                                              begin_ns=begin, returned_ns=returned, completed_ns=completed))
            record["api"].extend([dict(api="launch:f_run", start_ns=begin, end_ns=returned),
                                  dict(api="d2h:completion", start_ns=returned, end_ns=completed)])
            for name, (count, dtype) in fields.items():
                download(key+name, name, count, dtype)
            # CSR also performs diagnostics and persists each invocation outside its timed window.
            numerical(raw, epoch, p, original, reference)
            np.savez_compressed(out/"raw.npz", **raw)
            driver.write(out/"execution.json", record)
        snapshot("after_")
        record["status"] = "CAPTURED"
    except BaseException as exc:
        record.update(status="FAILED", error=repr(exc))
        raise
    finally:
        try:
            if rt is not None:
                call("stop", rt.stop)
        except BaseException as exc:
            record.update(status="FAILED", stop_error=repr(exc))
            raise
        finally:
            np.savez_compressed(out/"raw.npz", **raw)
            driver.write(out/"execution.json", record)
    return audit(case, out)


def clocks(words):
    require(words.dtype == np.uint16 and words.shape[-1] % 3 == 0, "48-bit PE samples")
    a = words.astype(np.uint64).reshape(*words.shape[:-1], -1, 3)
    return a[..., 0] | (a[..., 1] << 16) | (a[..., 2] << 32)


def timeline(record):
    expected = ["construct", "load", "run"]
    expected += ["h2d:"+n for n in INPUTS] + ["d2h:before_"+n for n in INPUTS] + ["d2h:initial_completion"]
    previous = 0
    api = record["api"]
    require(record["warmup"] == WARMUP and record["repeats"] == REPEATS
            and len(record["iterations"]) == WARMUP + REPEATS, "one warmup and five same-input iterations")
    for epoch, item in enumerate(record["iterations"], 1):
        require(item["epoch"] == epoch and item["warmup"] is (epoch <= WARMUP), "ordered epoch")
        t0, t1, t2 = (item[k] for k in ("begin_ns", "returned_ns", "completed_ns"))
        require(previous < t0 < t1 < t2, "monotonic completed-host window")
        start = len(expected)
        require(api[start:start+2] == [dict(api="launch:f_run", start_ns=t0, end_ns=t1),
                                       dict(api="d2h:completion", start_ns=t1, end_ns=t2)],
                "adjacent launch and all-PE read")
        previous = t2
        expected += ["launch:f_run", "d2h:completion"]
        expected += [f"d2h:e{epoch}_{n}" for n in (*VALUES, "stats", "clocks", "stages")]
    expected += ["d2h:after_"+n for n in INPUTS] + ["stop"]
    require([v["api"] for v in api] == expected, "exact lifecycle, resident inputs and readback schedule")
    previous = 0
    for v in api:
        require(previous <= v["start_ns"] <= v["end_ns"], "monotonic SDK events")
        previous = v["end_ns"]


def numerical(raw, epoch, p, original, reference):
    n = len(original["x"])
    key = f"e{epoch}_"
    result = dict(max_abs_errors={})
    for name, ref in (("pre", "pre"), ("projected", "projected1"), ("logits", "logits")):
        value = raw[key+name]
        require(value.shape == (p["R"], p["F"], p["V"]) and value.dtype == np.float32, name+" rectangle")
        if epoch > 1:
            np.testing.assert_array_equal(value.view(np.uint32), raw["e1_"+name].view(np.uint32))
        matrix = value.transpose(0, 2, 1).reshape(-1, p["F"])
        wanted = reference[ref]
        active = matrix[:n, :wanted.shape[1]]
        np.testing.assert_allclose(active, wanted, rtol=driver.RTOL, atol=driver.ATOL)
        np.testing.assert_array_equal(matrix[n:], 0)
        np.testing.assert_array_equal(matrix[:n, wanted.shape[1]:], 0)
        result["max_abs_errors"][name] = float(np.max(np.abs(active - wanted)))
        if name == "logits" and original["full"]:
            pred = active.argmax(axis=1)
            np.testing.assert_array_equal(pred, wanted.argmax(axis=1))
            valid = original["valid"]
            result.update(prediction_disagreements=0, validation_correct=int(np.sum(pred[valid] == original["labels"][valid])))
    return result


def audit(case, out):
    m, original, arrays = driver.load_case(case)
    p, n = m["params"], m["nodes"]
    r = driver.read(out/"execution.json")
    raw = driver.load_npz(out/"raw.npz")
    require(r["status"] == "CAPTURED" and "pending" not in r and "stop_error" not in r
            and r["case_sha256"] == driver.sha(case/"case.json") and r["mode"] == "gcn"
            and r["profile"] == p["PROFILE"] in (0, 1), "completed bound whole-model observation")
    require(driver.read(case/"compile.json") == dict(status="PASS", case_sha256=r["case_sha256"],
                                                     inventory=r["compiled_inventory"]), "actual compiled image")
    timeline(r)
    canonical = oracle.load_original(kind=m["kind"], nodes=n, root=case/"raw")
    for key in ("x", "src", "dst", "coeff", "w0", "w1", "labels", "valid"):
        np.testing.assert_array_equal(original.get(key), canonical.get(key))
    rebuilt = oracle.build_case(canonical, p["R"])
    require(rebuilt["metadata"] == m["metadata"] and dict(rebuilt["params"], PROFILE=p["PROFILE"]) == p,
            "packed inputs and parameters from original corpus")
    require(set(rebuilt["arrays"]) == set(arrays), "packed array inventory")
    for name, a in arrays.items():
        np.testing.assert_array_equal(rebuilt["arrays"][name], a)
    for name in INPUTS:
        wanted = driver.region(arrays[name], name, p)
        for prefix in ("before_", "after_"):
            np.testing.assert_array_equal(raw[prefix+name].view(np.uint8), wanted.view(np.uint8))
    epoch_check(raw["initial_completion"], p, 0)
    reference = rebuilt["reference"]
    summary = []
    for epoch, row in enumerate(r["iterations"], 1):
        key = f"e{epoch}_"
        epoch_check(raw[key+"completion"], p, epoch)
        np.testing.assert_array_equal(raw[key+"stats"], oracle.expected_stats(rebuilt, epoch=epoch))
        np.testing.assert_array_equal(raw[key+"h"], raw[key+"projected"])
        item = dict(epoch=epoch, warmup=row["warmup"], host_completed_ns=row["completed_ns"]-row["begin_ns"],
                    host_launch_ns=row["returned_ns"]-row["begin_ns"],
                    host_completion_read_ns=row["completed_ns"]-row["returned_ns"], max_abs_errors={})
        item.update(numerical(raw, epoch, p, canonical, reference))
        outer, middle = clocks(raw[key+"clocks"]), clocks(raw[key+"stages"])
        require(outer.shape == (2*p["R"], p["F"]+1, 2) and middle.shape == (*outer.shape[:-1], 4),
                "timestamps from all application PEs")
        spans = (outer[..., 1] - outer[..., 0]) & ((1 << 48)-1)
        require(np.all(spans > 0) and np.all(spans < 1 << 47), "positive unambiguous same-PE intervals")
        y, x = np.unravel_index(spans.argmax(), spans.shape)
        item.update(max_pe_cycles=int(spans[y, x]), critical_pe_xy=[int(x), int(y)])
        if p["PROFILE"]:
            all_times = np.concatenate((outer[..., :1], middle, outer[..., 1:]), axis=-1)
            parts = (all_times[..., 1:] - all_times[..., :-1]) & ((1 << 48)-1)
            require(np.all(parts <= spans[..., None]) and np.all(parts.sum(axis=-1) == spans),
                    "contiguous stage intervals on each PE")
            item["critical_pe_stages"] = {k: int(v) for k, v in zip(STAGES, parts[y, x])}
        else:
            np.testing.assert_array_equal(middle, 0)
        summary.append(item)
    measured = summary[WARMUP:]
    result = dict(status="PASS", run_id=r["run_id"], case_sha256=r["case_sha256"], backend=r["backend"],
                  profile=p["PROFILE"], application_pes=(p["F"]+1)*2*p["R"], raw_sha256=driver.sha(out/"raw.npz"),
                  host_completed_median_ns=int(np.median([i["host_completed_ns"] for i in measured])),
                  max_pe_median_cycles=int(np.median([i["max_pe_cycles"] for i in measured])),
                  observations=summary)
    driver.write(out/"audit.json", result)
    return result
