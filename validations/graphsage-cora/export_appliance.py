"""Export prepared cora_device cases as a standalone graphsage-cora validation.

Request/job layout adapts native_cora_appliance/export_appliance.py (E217).
Export prepares files only; it neither submits nor executes an appliance job.
"""
import argparse
from pathlib import Path
import re
import shutil
import numpy as np
try:
    from . import saved_audit as common
except ImportError:
    import saved_audit as common

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def prepare_case(source, destination, device_source, training=None, weights=None):
    """Convert a prepared case while retaining its original file bytes."""
    source, destination, device_source = (Path(path).absolute() for path in (source, destination, device_source))
    upstream = common.read_json(source/"case.json")
    common.require(set(upstream.get("files", {})) == {"input_model.npz", "arrays.npz", "node_ids.npy"}, "prepared case files")
    for name, digest in upstream["files"].items():
        common.require(common.sha(common.local_file(source, name)) == digest, "prepared file identity: "+name)
    params = upstream["params"]
    common.validate_params(params)
    common.require(common.named_bytes(params) <= 44*1024, "prepared declared-state forecast rejected")
    common.require(upstream.get("input_kind") in ("synthetic", "native-cora"), "prepared input kind")
    kind = "tiny" if upstream["input_kind"] == "synthetic" else "cora"
    if kind == "cora":
        common.require(training is not None and weights is not None, "native case requires training.json and retained weights.npz")
    common.require(destination == destination.resolve(), "case destination symlink")
    destination.mkdir(parents=True, exist_ok=False)
    (destination/"source").mkdir()
    for name in ("layout.csl", "pe.csl"):
        source_name = "models/graphsage/cora_device/"+name
        common.require(common.sha(device_source/name) == upstream.get("source_sha256", {}).get(source_name), "prepared device source pin: "+name)
        shutil.copyfile(device_source/name, destination/"source"/name)
    for old, new in (("case.json", "prepared-case.json"), ("input_model.npz", "input_model.npz"),
                     ("arrays.npz", "inputs.npz"), ("node_ids.npy", "node_ids.npy")):
        shutil.copyfile(source/old, destination/new)
    np.savez(destination/"node_ids.npz", nodes=common.prepared_nodes(source/"node_ids.npy"))
    if kind == "cora":
        shutil.copyfile(training, destination/"training.json")
        shutil.copyfile(weights, destination/"weights.npz")
    common.write_json(destination/"provenance.json", dict(schema=1,
        prepared_case_sha256=common.sha(source/"case.json"), source_sha256=upstream["source_sha256"],
        origin="models/graphsage/cora_device prepared case",
        exact_preserved_files={"input_model.npz": "input_model.npz", "arrays.npz": "inputs.npz", "node_ids.npy": "node_ids.npy"},
        model_audit="independent FP64 mean over original input_model.npz occurrences before projection",
        dataset_provenance="native training.json retains raw corpus hashes and splits; raw corpus bytes are not bundled",
        transport_origin="models/hierarchical_gnn/grouped_csr/native_cora_appliance (E217)"))
    manifest = dict(schema=1, family="graphsage-cora", kind=kind, params=params,
        nodes=upstream["selected_nodes"], original_nodes=upstream["original_nodes"],
        features=upstream["metadata"]["features"], dataset_scope=upstream["dataset_scope"],
        original_checkpoint_sha256=upstream["original_checkpoint_sha256"],
        inputs=common.schema(params), outputs=common.schema(params, True),
        launch=dict(symbol="f_run", arguments=[]), tolerance=dict(atol=common.ATOL, rtol=common.RTOL),
        model_contract=common.MODEL_CONTRACT, capture_contract=common.CAPTURE_CONTRACT,
        admission=dict(named_bytes=common.named_bytes(params), limit_bytes=44*1024, admitted=True,
            scope="declared arrays and state only; generated ELF allocation checked before launch; stack high-water unmeasured"),
        scope="complete frozen GraphSAGE inference; custom native splits only for full Cora; PE-local clocks; no device training or global makespan")
    arrays = common.load_npz(destination/"inputs.npz")
    common.require(set(arrays) == set(common.schema(params)), "prepared input inventory")
    for name, (size, dtype) in common.schema(params).items():
        common.require(arrays[name].shape == (params["C"]*params["R"], size) and arrays[name].dtype == dtype,
                       "prepared input shape/dtype: "+name)
    expected, _ = common.independent_expected(destination, manifest, arrays)
    np.savez(destination/"expected.npz", **expected)
    required = common.CASE_REQUIRED | ({"training.json", "weights.npz"} if kind == "cora" else set())
    manifest["files"] = {name: common.sha(destination/name) for name in sorted(required)}
    common.write_json(destination/"case.json", manifest)
    common.load(destination)
    # Preserve the exact prepared source and all three payload identities even
    # if the source tree is modified while the export is in progress.
    common.require(common.sha(source/"case.json") == common.sha(destination/"prepared-case.json"), "prepared case changed during export")
    for name, digest in upstream["files"].items():
        common.require(common.sha(source/name) == digest, "prepared input changed during export")
    return manifest


def export(cases, output, request_id, *, device_source=None, training=None, weights=None,
           compile_workers=1, host_timeout=600):
    common.require(isinstance(cases, dict) and 1 <= len(cases) <= 8, "one to eight named cases")
    common.require(isinstance(request_id, str) and re.fullmatch("[a-z][a-z0-9_-]{0,95}", request_id), "request ID")
    workers = common.compile_workers(compile_workers)
    common.require(type(host_timeout) is int and 1 <= host_timeout <= 1800, "bounded hardware worker timeout")
    output = Path(output).absolute()
    common.require(output == output.resolve(), "export symlink")
    output.mkdir(parents=True, exist_ok=False)
    device_source = Path(device_source) if device_source is not None else HERE.parent/"cora_device"
    for name in (*common.HOST_FILES, "export_appliance.py", "README.md"):
        shutil.copyfile(HERE/name, output/name)
    (output/"tests").mkdir()
    shutil.copyfile(HERE/"test_appliance.py", output/"tests/test_appliance.py")
    package = dict(schema=1, family="graphsage-cora", cases={}, compile_workers=workers,
        host_sha256={name: common.sha(output/name) for name in common.HOST_FILES},
        exporter_sha256=common.sha(output/"export_appliance.py"),
        scope="resident full-neighbor GraphSAGE; native1433 features; hidden ReLU/L2 and raw logits; original-occurrence FP64 audit",
        evidence="package preparation only; no remote execution claim")
    steps = []
    for name, source in cases.items():
        common.require(isinstance(name, str) and re.fullmatch("[a-z][a-z0-9_-]{0,63}", name), "case name")
        destination = output/"cases"/name
        prepare_case(source, destination, device_source, training, weights)
        package["cases"][name] = common.sha(destination/"case.json")
        script = "validations/graphsage-cora/appliance.py"
        steps.extend([
            dict(name="compile-"+name, kind="appliance", script=script, timeout_seconds=1200,
                 args=["compile", "--case", name, "--backend", "hardware", "--namespace", "job-operator",
                       "--compile-workers", str(workers), "--out", "{output}/compile-"+name]),
            dict(name="run-"+name, kind="appliance", script=script, timeout_seconds=min(3300, host_timeout+600),
                 args=["run", "--case", name, "--backend", "hardware", "--artifact-record",
                       "{output}/compile-"+name+"/native-build.json", "--host-timeout", str(host_timeout), "--out", "{output}/run-"+name]),
            dict(name="audit-"+name, kind="host", script="validations/graphsage-cora/saved_audit.py", timeout_seconds=120,
                 args=["{output}/run-"+name+"/capture-data", "--out", "{output}/numerical-"+name+".json"])])
    common.write_json(output/"package.json", package)
    common.write_json(output/"validation.json", dict(schema=1, id="graphsage-cora",
        description="Resident GraphSAGE classifier with native Cora features and complete independent portable inference audit.",
        entrypoints=["appliance.py", "saved_audit.py"], tests=["tests"]))
    common.write_json(output/"request-template.json", dict(schema=1, id=request_id,
        description="Compile, capture and audit each GraphSAGE case before advancing; prefix cases report no classification accuracy.", steps=steps))
    return package


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--device-source", type=Path, default=HERE.parent/"cora_device")
    parser.add_argument("--training", type=Path, default=REPO/"research/experiments/data/E219/training.json")
    parser.add_argument("--weights", type=Path, default=REPO/"research/experiments/data/E219/weights.npz")
    parser.add_argument("--compile-workers", type=int, choices=(1, 4, 8), default=1)
    parser.add_argument("--host-timeout", type=int, default=600)
    args = parser.parse_args(argv)
    cases = {}
    for item in args.case:
        if "=" not in item:
            parser.error("--case needs NAME=PATH")
        name, path = item.split("=", 1)
        if name in cases:
            parser.error("duplicate case name")
        cases[name] = Path(path)
    export(cases, args.output, args.request_id, device_source=args.device_source,
           training=args.training, weights=args.weights, compile_workers=args.compile_workers, host_timeout=args.host_timeout)


if __name__ == "__main__":
    main()
