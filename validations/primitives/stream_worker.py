#!/usr/bin/env python3
"""Execute an immutable input-free halo request with direct rectangular D2H."""
import argparse
import json
import os
from pathlib import Path
import shutil

if __package__:
    from . import chunks, halo, stream_contract as contract
else:
    import chunks
    import halo
    import stream_contract as contract


def sdk_classes():
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
    return SdkRuntime, MemcpyDataType, MemcpyOrder


def capture(args, sdk_factory=None):
    case_path, request_path, build = Path(args.case).resolve(), Path(args.request).resolve(), Path(args.name).resolve()
    case_bytes, request_bytes = halo.bounded(case_path), halo.bounded(request_path)
    case, request = json.loads(case_bytes), json.loads(request_bytes)
    contract.validate_request(request, case_bytes, args.backend)
    contract.require((args.backend == "hardware") == bool(args.cmaddr), "hardware cmaddr required only on hardware")
    build_record = halo.bounded(args.build_record)
    contract.require(chunks.digest(build_record) == request["build_record_sha256"], "compiled record identity")
    record = json.loads(build_record)
    contract.require(record["backend"] == args.backend and all(record[k] == request[k] for k in
                     ("case_name", "case_sha256", "package_manifest_sha256", "artifact_sha256")), "compiled/request binding")
    contract.require(record["worker_sha256"] == request["worker_sha256"], "compiled worker digest")
    out_bytes, rpc_bytes = halo.bounded(build/"out.json"), halo.bounded(build/"bin/out_rpc.json")
    contract.require(record["compiled_metadata_sha256"] ==
                     {"out.json": chunks.digest(out_bytes), "out_rpc.json": chunks.digest(rpc_bytes)},
                     "runtime ABI metadata differs from compiled artifact")
    ids = contract.check_abi(case, json.loads(out_bytes), json.loads(rpc_bytes))
    identity = contract.plan_identity(request_bytes)
    plan = contract.make_plan(case, identity)
    destination = contract.new_directory(args.out)
    runtime = contract.new_directory(destination.with_name(destination.name+"-runtime"))
    for name, value in {"case.json": case_bytes, "request.json": request_bytes, "build.json": build_record,
                        "out.json": out_bytes, "out_rpc.json": rpc_bytes, "plan.json": chunks.encoded(plan)}.items():
        (destination/name).write_bytes(value)
    result = dict(execution_status="FAILED", numerical_status="NOT_RUN", identity=identity, events=[], lifecycle=[],
                  compiled_metadata_sha256={"out.json": chunks.digest(out_bytes), "out_rpc.json": chunks.digest(rpc_bytes)})
    runner, previous = None, Path.cwd()
    try:
        sdk_build = build
        if args.backend == "local-sim":
            # Runtime-generated ELF files stay in this disposable copy. Inventory
            # hashes bind the unchanged compiler output used to construct it.
            halo.verify_build(build, record["inventory"])
            sdk_build = runtime/"build"
            shutil.copytree(build, sdk_build)
            halo.verify_build(sdk_build, record["inventory"])
        os.chdir(runtime)
        runtime_type, datatype, order = (sdk_factory or sdk_classes)()
        options = dict(cmaddr=args.cmaddr)
        if args.backend != "hardware":
            options.update(simfab_numthreads=1, suppress_simfab_trace=True)
        runner = runtime_type(str(sdk_build), **options)
        result["lifecycle"].append("construct")
        observed = {}
        for name in case["symbols"]:
            observed[name] = runner.get_id(name)
            result["lifecycle"].append("get_id:"+name)
        contract.require(observed == ids, "runtime/compiled symbol IDs")
        runner.load()
        result["lifecycle"].append("load")
        runner.run()
        result["lifecycle"].append("run")
        result["capture"] = halo.program(case, plan, runner, ids, datatype, order, destination/"chunks", result["events"])
    except BaseException as error:
        result["error"] = repr(error)[:1024]
    finally:
        try:
            if runner is not None:
                runner.stop()
                result["lifecycle"].append("stop")
        except BaseException as error:
            result["stop_error"] = repr(error)[:1024]
        finally:
            os.chdir(previous)
    try:
        contract.validate_request(request, halo.bounded(case_path), args.backend)
        contract.require(halo.bounded(request_path) == request_bytes and halo.bounded(args.build_record) == build_record,
                         "request/build record changed during execution")
        if args.backend == "local-sim":
            halo.verify_build(build, record["inventory"])
        contract.require(not any(k in result for k in ("error", "stop_error")), "SDK execution or stop failed")
        contract.require(result["events"] == list(halo.expected_events(case, plan, ids)), "complete SDK generation/chunk order")
        contract.require(result["lifecycle"] == ["construct"]+["get_id:"+n for n in case["symbols"]]+["load", "run", "stop"],
                         "complete SDK lifecycle")
        chunks.audit(destination/"chunks", plan, result["capture"]["manifest_sha256"])
        result["execution_status"] = "PASS"
    except BaseException as error:
        result["admission_error"] = repr(error)[:1024]
    contract.write_json(destination/"execution.json", result)
    return result


def audit_capture(folder, case_bytes, request_bytes, build_bytes):
    folder = chunks.regular_root(folder)
    case, request = json.loads(case_bytes), json.loads(request_bytes)
    contract.validate_request(request, case_bytes, request["backend"])
    contract.require(chunks.digest(build_bytes) == request["build_record_sha256"], "audit compiled record hash")
    record = json.loads(build_bytes)
    contract.require(record["backend"] == request["backend"] and record["worker_sha256"] == request["worker_sha256"]
                     and all(record[k] == request[k] for k in
                             ("case_name", "case_sha256", "package_manifest_sha256", "artifact_sha256")), "audit build/request binding")
    for name, expected in (("case.json", case_bytes), ("request.json", request_bytes), ("build.json", build_bytes)):
        contract.require(halo.bounded(folder/name) == expected, "capture identity: "+name)
    identity = contract.plan_identity(request_bytes)
    plan = contract.make_plan(case, identity)
    contract.require(contract.read_json(folder/"plan.json") == plan, "capture plan differs")
    out_bytes, rpc_bytes = halo.bounded(folder/"out.json"), halo.bounded(folder/"out_rpc.json")
    contract.require(record["compiled_metadata_sha256"] ==
                     {"out.json": chunks.digest(out_bytes), "out_rpc.json": chunks.digest(rpc_bytes)},
                     "runtime ABI metadata differs from compiled artifact")
    ids = contract.check_abi(case, json.loads(out_bytes), json.loads(rpc_bytes))
    execution = contract.read_json(folder/"execution.json")
    contract.require(execution.get("execution_status") == "PASS" and execution.get("numerical_status") == "NOT_RUN"
                     and execution.get("identity") == identity, "complete capture execution identity")
    contract.require(execution.get("compiled_metadata_sha256") ==
                     {"out.json": chunks.digest(out_bytes), "out_rpc.json": chunks.digest(rpc_bytes)}, "captured compiled ABI identity")
    if request["backend"] == "local-sim":
        contract.require(all(record["inventory"]["files"][key] == chunks.digest(value)
                             for key, value in (("out.json", out_bytes), ("bin/out_rpc.json", rpc_bytes))), "local compiled metadata binding")
    contract.require(execution.get("events") == list(halo.expected_events(case, plan, ids)) and
                     execution.get("lifecycle") == ["construct"]+["get_id:"+n for n in case["symbols"]]+["load", "run", "stop"],
                     "captured SDK lifecycle/generation/chunk order")
    report = halo.audit(folder/"chunks", case_bytes, plan, execution["capture"]["manifest_sha256"], contract,
                        total_bytes=case["capture"]["total_bytes"], max_chunks=case["capture"]["max_chunks"])
    admission = chunks.validate_plan(plan)
    contract.require(all(execution["capture"][k] == admission[k] for k in ("payload_bytes", "chunks")) and
                     execution["capture"]["peak_chunk_bytes"] == report["peak_chunk_bytes"], "capture census")
    return dict(report, execution_status="PASS", status="PASS", identity=identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--build-record", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("local-sim", "hardware", "appliance-sim"), required=True)
    parser.add_argument("--cmaddr")
    args = parser.parse_args(argv)
    result = capture(args)
    print(json.dumps({k: result[k] for k in ("execution_status", "numerical_status")}))
    return 0 if result["execution_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
