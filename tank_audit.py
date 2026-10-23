"""Portable full-array and SDK-contract audit for frozen TANK capsules."""
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import tank_worker as worker

require = worker.require
MAX_BYTES = 256 << 20
VENDOR = "csl-extras-202505230211-4-d9070058/examples/benchmarks/benchmark-libs/stencil_3d_7pts"
ACTIVE_VENDOR = "sdk-2.10.2/csl-extras-202604101435-6-d2f7d96e/examples/benchmarks/benchmark-libs/stencil_3d_7pts"
SDK_TRANSFORMS = {
    VENDOR + "/layout.csl": ("sdk-2.10.2-stencil-layout-types", ACTIVE_VENDOR + "/layout.csl"),
    VENDOR + "/wse3/pe.csl": ("sdk-2.10.2-stencil-queue-dsds", ACTIVE_VENDOR + "/wse3/pe.csl"),
}
LOCAL_IMPORTS = {
    "kernels/stencil/independent-halo/layout.csl": ("../../../" + VENDOR + "/layout.csl", "vendor_layout.csl"),
    "kernels/stencil/independent-halo/pe.csl": ("../../../" + VENDOR + "/wse3/pe.csl", "vendor_pe.csl"),
}


def stencil_compatibility(original_name, payload):
    """Reconstruct the published syntax migration without importing exporter code."""
    require(original_name in SDK_TRANSFORMS, "known SDK stencil source")
    colors = ("c_recv_west", "c_send_east", "c_recv_east", "c_send_west",
              "c_recv_south", "c_send_north", "c_recv_north", "c_send_south")
    if original_name == VENDOR + "/layout.csl":
        fields = [(name, "color") for name in colors]
        fields += [(name, "local_task_id") for name in ("SEND", "RECV", "COMM")]
        fields += [(name, "bool") for name in ("first_px", "last_px", "first_py", "last_py")]
        declaration = "const Params = struct {\n" + "".join("    %s: %s,\n" % pair for pair in fields) + "};\n\n"
        before = b"fn get_params(px:i16, py:i16) comptime_struct {"
        require(payload.count(before) == 1 and payload.count(b"    return .{") == 1, "historical stencil layout syntax")
        payload = payload.replace(before, declaration.encode() + b"fn get_params(px:i16, py:i16) Params {")
        return payload.replace(b"    return .{", b"    return Params{")
    for color in colors:
        field = (("   " if color.startswith("c_recv_") else "    ") + ".fabric_color = " + color + ",\n").encode()
        require(payload.count(field) == 1, "historical queue DSD color field")
        payload = payload.replace(field, b"")
    return payload


def forward_stencil_params(payload):
    """Reconstruct only explicit forwarding of the original 15 layout fields."""
    fields = ("c_recv_west", "c_send_east", "c_recv_east", "c_send_west",
              "c_recv_south", "c_send_north", "c_recv_north", "c_send_south",
              "COMM", "SEND", "RECV", "first_px", "last_px", "first_py", "last_py")
    before = b'const stencil = @import_module("vendor_pe.csl", @concat_structs(stencil_params, .{\n'
    declaration = 'const stencil = @import_module("vendor_pe.csl", .{\n'
    declaration += "".join("  .%s=stencil_params.%s,\n" % (name, name) for name in fields)
    closing = b"}));\n\nvar x ="
    require(payload.count(before) == 1 and payload.count(closing) == 1, "historical wrapper merge syntax")
    return payload.replace(before, declaration.encode()).replace(closing, b"});\n\nvar x =")


def digest(path):
    path = Path(path)
    return hashlib.sha256(worker.bounded_bytes(path, worker.ARRAY_LIMIT)).hexdigest()


def read_json(path):
    value = json.loads(worker.bounded_bytes(path, worker.JSON_LIMIT))
    require(isinstance(value, dict), "JSON object")
    return value


def regular(root, relative):
    name = Path(relative)
    require(not name.is_absolute() and name.parts and all(p not in (".", "..") for p in name.parts), "relative capsule filename")
    path = root / name
    require(path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(), "regular capsule file")
    return path


def verify_package(package):
    package = Path(package).absolute()
    require(package == package.resolve() and package.is_dir(), "regular capsule directory")
    manifest = read_json(package / "manifest.json")
    require(manifest.get("schema") == 1 and isinstance(manifest["files"], dict) and
            1 <= len(manifest["files"]) <= 512 and 1 <= len(manifest["cases"]) <= 16, "capsule inventory bound")
    total = 0
    for name, value in manifest["files"].items():
        path = regular(package, name)
        total += path.stat().st_size
        require(total <= MAX_BYTES and re.fullmatch(r"[0-9a-f]{64}", value) and digest(path) == value, "capsule file identity: " + name)
    require(set(manifest["oracle_sources"]) == {name for name in manifest["files"] if name.startswith("oracle/")} and
            set(manifest["oracle_sources"]) and
            all(name.startswith("oracle/") and manifest["files"].get(name) == value
                for name, value in manifest["oracle_sources"].items()), "frozen oracle inventory")
    require(manifest["files"].get("provenance/export_tank_cases.py") == manifest["exporter_sha256"], "frozen exporter identity")
    for name, entry in manifest["cases"].items():
        require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", name), "capsule case name")
        prefix = "cases/" + name + "/"
        case = read_json(package / (prefix + "case.json"))
        worker.validate_manifest(case)
        require(entry["case_sha256"] == manifest["files"][prefix + "case.json"] and
                entry["params"] == case["params"] and entry["grid"] == case["grid"] and
                entry["family"] == entry["kind"] == case["kind"], "case descriptor identity")
        expected_files = {"inputs.npz", "expected.npz", "source/layout.csl", "source/pe.csl"}
        if case["kind"] == "halo":
            expected_files |= {"source/vendor_layout.csl", "source/vendor_pe.csl"}
        require(set(case["files"]) == expected_files and
                all(manifest["files"].get(prefix + relative) == value for relative, value in case["files"].items()), "case file inventory")
        require(case["oracle_sources"] == manifest["oracle_sources"], "case oracle identity")
        require(set(case["source_origin"]) == {n for n in expected_files if n.startswith("source/")}, "CSL source provenance inventory")
        for relative, origin in case["source_origin"].items():
            require(origin["exported_sha256"] == case["files"][relative] and
                    origin["transformation"] in ("none", "localize-stencil-import", "localize-stencil-import-and-forward-params",
                                                 "sdk-2.10.2-stencil-layout-types", "sdk-2.10.2-stencil-queue-dsds") and
                    re.fullmatch(r"[0-9a-f]{64}", origin["original_sha256"]), "CSL transformation identity")
            original_name = "provenance/originals/" + origin["original_path"]
            require(manifest["files"].get(original_name) == origin["original_sha256"], "frozen original CSL identity")
            original = worker.bounded_bytes(regular(package, original_name), worker.ARRAY_LIMIT)
            if origin["transformation"] in ("localize-stencil-import", "localize-stencil-import-and-forward-params"):
                require(case["kind"] == "halo" and origin["original_path"] in LOCAL_IMPORTS, "known import localization")
                old, new = ('"' + value + '"' for value in LOCAL_IMPORTS[origin["original_path"]])
                require(original.count(old.encode()) == 1, "one original import")
                original = original.replace(old.encode(), new.encode())
                if origin["transformation"] == "localize-stencil-import-and-forward-params":
                    require(origin["original_path"] == "kernels/stencil/independent-halo/pe.csl", "known wrapper parameter forwarding")
                    reference = origin["pattern_reference"]
                    reference_name = ACTIVE_VENDOR + "/pe.csl"
                    require(reference["path"] == reference_name and
                            manifest["files"].get("provenance/originals/" + reference_name) == reference["sha256"],
                            "active official parameter-forwarding reference")
                    original = forward_stencil_params(original)
            elif origin["transformation"].startswith("sdk-"):
                require(case["kind"] == "halo" and origin["original_path"] in SDK_TRANSFORMS, "known SDK compatibility source")
                wanted_transform, reference_name = SDK_TRANSFORMS[origin["original_path"]]
                require(origin["transformation"] == wanted_transform, "SDK compatibility transformation label")
                reference = origin["compatibility_reference"]
                reference_file = "provenance/originals/" + reference_name
                require(set(reference) == {"path", "sha256"} and reference["path"] == reference_name and
                        manifest["files"].get(reference_file) == reference["sha256"], "active official reference identity")
                original = stencil_compatibility(origin["original_path"], original)
                require(original == worker.bounded_bytes(regular(package, reference_file), worker.ARRAY_LIMIT),
                        "adaptation must exactly equal active official stencil source")
            require(worker.sha(original) == origin["exported_sha256"], "only the declared CSL transformation")
    return manifest


def same(actual, expected, label):
    require(actual.dtype == expected.dtype and actual.shape == expected.shape and actual.tobytes(order="C") == expected.tobytes(order="C"),
            "exact array bytes differ: " + label)


def audit_capture(package, case_name, capture_dir, request):
    package, capture = Path(package).absolute(), Path(capture_dir).absolute()
    require(capture == capture.resolve() and capture.is_dir(), "regular capture directory")
    manifest = verify_package(package)
    require(case_name in manifest["cases"], "known capsule case")
    folder = package / "cases" / case_name
    case_bytes = worker.bounded_bytes(folder / "case.json", worker.JSON_LIMIT)
    case, execution = json.loads(case_bytes), read_json(capture / "execution.json")
    worker.validate_request(request, request["backend"], case_bytes,
                            worker.bounded_bytes(Path(__file__).with_name("tank_worker.py"), worker.JSON_LIMIT))
    require(request["case_name"] == case_name and request["package_manifest_sha256"] == digest(package / "manifest.json"), "request package/case")
    require(read_json(capture.parent / "request.json") == request and execution["request_sha256"] == digest(capture.parent / "request.json"), "staged request bytes")
    require(execution.get("status") == "COMPLETED" and not any(key in execution for key in ("error", "stop_error", "integrity_error", "capture_error")), "closed successful capture")
    for key in worker.REQUEST_KEYS - {"schema"}:
        require(execution.get(key) == request[key], "capture/request identity: " + key)
    out, rpc = read_json(capture / "out.json"), read_json(capture / "out_rpc.json")
    ids = worker.check_abi(case, out, rpc)
    require(execution["symbol_ids"] == ids and execution["compiled_sha256"] ==
            {name: digest(capture / name) for name in ("out.json", "out_rpc.json")}, "captured ABI/ID identity")
    require(execution["raw_sha256"] == digest(capture / "raw.npz"), "raw snapshot identity")
    runtime_options = dict(cmaddr="<provided>" if request["backend"] == "hardware" else None)
    if request["backend"] != "hardware":
        runtime_options.update(simfab_numthreads=1, suppress_simfab_trace=True)
    require(execution["runtime_options"] == runtime_options, "runtime backend options")
    expected_api = ["construct"] + ["get_id:" + name for name in case["symbols"]] + ["load", "run"]
    expected_api += ["%03d:%s:%s" % (index, op["op"], op["symbol"]) for index, op in enumerate(case["operations"])]
    expected_api += ["stop"]
    require(execution["api"] == expected_api and execution["operations"] ==
            [worker.operation_record(op, case["symbols"], ids) for op in case["operations"]], "exact SDK operation sequence")
    require([item["api"] for item in execution["api_timings"]] == expected_api and
            all(type(item["wall_seconds"]) in (int, float) and math.isfinite(item["wall_seconds"]) and item["wall_seconds"] >= 0
                for item in execution["api_timings"]), "API timing inventory")
    require(type(execution["wall_seconds"]) in (int, float) and math.isfinite(execution["wall_seconds"]) and execution["wall_seconds"] >= 0,
            "worker wall time")
    input_schema, output_schema = worker.validate_manifest(case)
    inputs, expected, raw = (worker.arrays(path) for path in (folder / "inputs.npz", folder / "expected.npz", capture / "raw.npz"))
    clock_names = {op["array"] for op in case["operations"] if op.get("clock")}
    require(set(inputs) == set(input_schema) and set(raw) == set(output_schema) and set(expected) == set(output_schema) - clock_names,
            "complete input/expected/raw inventory")
    for name, (shape, dtype) in input_schema.items():
        require(inputs[name].shape == shape and inputs[name].dtype == np.dtype(dtype), "input schema")
    clock_reports = []
    for name, (shape, dtype) in output_schema.items():
        value = raw[name]
        require(value.shape == shape and value.dtype == np.dtype(dtype), "raw schema: " + name)
        if name not in clock_names:
            same(value, expected[name], name)
            continue
        require(bool((value <= 65535).all()), "u16 timestamp carrier")
        wide = value.astype(np.uint64)
        start = wide[:, 0] + (wide[:, 1] << 16) + (wide[:, 2] << 32)
        end = wide[:, 3] + (wide[:, 4] << 16) + (wide[:, 5] << 32)
        intervals = (end - start) & np.uint64((1 << 48) - 1)
        require(bool(((intervals > 0) & (intervals < 1 << 47)).all()), "positive unambiguous local timestamps")
        clock_reports.append(dict(array=name, pe_count=len(intervals), local_cycles_min=int(intervals.min()),
                                  local_cycles_max=int(intervals.max()), local_cycles=intervals.tolist(),
                                  epochs=case["census"]["epochs_per_launch"],
                                  max_local_cycles_per_epoch=int(intervals.max()) / case["census"]["epochs_per_launch"]))
    for left, right in case["validations"]["restore"]:
        same(raw[left], raw[right], "same-runtime restoration " + left + "/" + right)
    numerical_words = sum(raw[name].size for name in expected)
    return dict(status="PASS", backend=request["backend"], case_name=case_name, kind=case["kind"], params=case["params"], grid=case["grid"],
                run_id=request["run_id"], raw_sha256=execution["raw_sha256"], expected_sha256=case["files"]["expected.npz"],
                numerical_arrays=len(expected), numerical_words=numerical_words, total_captured_arrays=len(raw),
                native_calls=len(expected_api), restoration_pairs=case["validations"]["restore"], clocks=clock_reports,
                worker_wall_seconds=execution["wall_seconds"], census=case["census"],
                scope="All captured arrays compared byte-for-byte with frozen independent-oracle outputs; captured ABI/request/SDK sequence checked; halo clocks are PE-local, Top-K has no device clock; no dynamic SRAM proof.")
