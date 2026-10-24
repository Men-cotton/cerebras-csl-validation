"""Export frozen NumPy expectations and one-action hardware request templates."""
import argparse
import importlib
from pathlib import Path
import re
import shutil
import sys
import numpy as np

try:
    from . import saved_audit as common
except ImportError:
    import saved_audit as common

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
ORACLE_MODULES = {"models.meshgraphnet.native_mesh.driver", "models.meshgraphnet.native_full.driver"}


def same_arrays(path, expected):
    """Compare logical arrays bitwise while retaining the original ZIP bytes."""
    original = common.load_npz(path)
    return (set(original) == set(expected) and
            all(original[name].shape == value.shape and original[name].dtype == value.dtype
                and original[name].tobytes(order="C") == value.tobytes(order="C")
                for name, value in expected.items()))


def native_block_expectations(case):
    if str(REPOSITORY) not in sys.path:
        sys.path.insert(0, str(REPOSITORY))
    from models.meshgraphnet.native_block import driver, model
    manifest = driver.load(case)
    inputs = common.load_npz(case / "inputs.npz")
    full = common.load_npz(case / "original.npz")
    params = manifest["params"]
    expected = model.packed_outputs(params, full, model.dense_oracle(full))
    expected["stats"] = model.expected_stats(params, inputs)
    exact = ["sender_cache", "stats"]
    if params["MODE"] == 1:
        expected["comm_stats"] = model.checked_halo_schedule(params, full, inputs)[2]
        exact.append("comm_stats")
    masks = {}
    for name in ("h_out", "delta_h", "aggregate", "e_out", "delta_e", "sender_cache"):
        mask = np.zeros_like(expected[name], dtype=np.uint32)
        count_column = 1 if name in ("e_out", "delta_e", "sender_cache") else 0
        for pe, counts in enumerate(inputs["counts"]):
            mask[pe, int(counts[count_column]) * params["T"]:] = 1
        masks[name] = mask
    return dict(expectedarrays=expected, exactfields=exact, zeromasks=masks,
                rtol=3e-4, atol=3e-4, immutable_inputs=list(manifest["inputs"]),
                scope="H128 encoded-state processor block, edge/node MLPs, delta aggregation and residuals; no encoders, decoder, physical rollout or trained accuracy")


def freeze(case, output, oracle=None):
    case, output = Path(case).absolute(), Path(output).absolute()
    common.require(output == output.resolve(), "freeze destination symlink")
    original_bytes = common.bounded(common.local_file(case, "case.json"), common.JSON_LIMIT)
    original = common.read_json(case / "case.json")
    if original.get("family") == "native-gnn":
        manifest = common.load(case)
        output.mkdir(parents=True, exist_ok=False)
        for name in set(manifest["files"]) | {"case.json"}:
            target = common.local_file(output, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(common.local_file(case, name), target)
        common.require(common.load(output) == manifest, "portable copy identity")
        return manifest
    common.require(isinstance(original.get("files"), dict) and len(original["files"]) <= 120, "source case inventory")
    for name, digest in original["files"].items():
        common.require(common.sha(common.local_file(case, name)) == digest, "original case file changed")
    if oracle is None:
        module_name = original.get("portable_oracle_module")
        if original.get("experiment") == "E199-native-processor-block":
            oracle = native_block_expectations
        else:
            common.require(module_name in ORACLE_MODULES, "explicit supported native oracle module")
            if str(REPOSITORY) not in sys.path:
                sys.path.insert(0, str(REPOSITORY))
            oracle = importlib.import_module(module_name).portable_expectations
    contract = oracle(case)
    common.require(common.bounded(case / "case.json", common.JSON_LIMIT) == original_bytes, "source case changed during oracle")
    for name, digest in original["files"].items():
        common.require(common.sha(common.local_file(case, name)) == digest, "source file changed during oracle")
    expected = contract["expectedarrays"]
    common.require(set(expected) == set(original["outputs"]) - {"clocks"}, "oracle must cover every output")
    masks = common.mask_arrays(contract.get("zeromasks", {}), expected, "zero-mask")
    bit_masks = common.mask_arrays(contract.get("exactmasks", {}), expected, "exact-mask", float_outputs=True)
    exact = set(contract["exactfields"])
    common.require(exact <= set(expected), "oracle exact fields")
    overrides = contract.get("tolerances", {})
    approximate = {name for name, (_, dtype) in original["outputs"].items()
                   if dtype == "float32" and name not in exact}
    common.require(isinstance(overrides, dict) and set(overrides) <= approximate, "per-output tolerance fields")
    checks = {}
    for name, (_, dtype) in original["outputs"].items():
        if name == "clocks":
            checks[name] = dict(kind="clock")
        elif name in exact or dtype != "float32":
            checks[name] = dict(kind="exact")
        else:
            values = overrides[name] if name in overrides else {n: contract[n] for n in ("rtol", "atol")}
            checks[name] = dict(kind="approx", **common.tolerance(values))
    output.mkdir(parents=True, exist_ok=False)
    origin_mapping = {}
    generated = {"expected.npz", "zero_masks.npz", "exact_masks.npz", "origin-case.json"}
    archives = {"expected.npz": expected, "zero_masks.npz": masks}
    if "exactmasks" in contract:
        archives["exact_masks.npz"] = bit_masks
    reused = {name for name, values in archives.items()
              if name in original["files"] and same_arrays(case / name, values)}
    for name in original["files"]:
        target_name = name
        if name in generated and name not in reused:
            target_name = "origin-files/" + name
            while target_name in original["files"] or target_name in origin_mapping.values():
                target_name = "origin-files/" + target_name
        target = common.local_file(output, target_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(common.local_file(case, name), target)
        origin_mapping[name] = target_name
    (output / "origin-case.json").write_bytes(original_bytes)
    for name, values in archives.items():
        if name not in reused:
            np.savez(output / name, **values)
    names = set(origin_mapping.values()) | {"origin-case.json"} | set(archives)
    manifest = dict(schema=1, family="native-gnn", experiment=original["experiment"],
                    params=original["params"], inputs=original["inputs"], outputs=original["outputs"],
                    launch=dict(entrypoint="run", arguments=[]), checks=checks,
                    immutable_inputs=contract.get("immutable_inputs", list(original["inputs"])),
                    scope=contract["scope"], origin_files=origin_mapping,
                    origin_case_sha256=common.sha(output / "origin-case.json"),
                    files={name: common.sha(output / name) for name in names},
                    oracle_export=dict(exporter_sha256=common.sha(Path(__file__)),
                        implementation=oracle.__module__ + ":" + oracle.__qualname__,
                        source_hashes=original.get("sources", {})),
                    evidence="frozen host oracle only; no device execution")
    if "exactmasks" in contract:
        manifest["exact_mask_counts"] = {name: int(value.sum(dtype=np.uint64)) for name, value in bit_masks.items()}
    common.write_json(output / "case.json", manifest)
    common.load(output)
    return manifest


def export(cases, output, request_id="tank28-native-gnn-20261023", root=ROOT, compile_workers=1):
    common.require(isinstance(cases, dict) and 1 <= len(cases) <= 16, "one to sixteen named cases")
    compile_workers = common.compile_workers(compile_workers)
    common.require(re.fullmatch("[a-z][a-z0-9_-]{0,95}", request_id), "request ID")
    output = Path(output).absolute()
    common.require(output == output.resolve(), "export symlink")
    output.mkdir(parents=True, exist_ok=False)
    for name in (*common.HOST_FILES, "export_appliance.py"):
        shutil.copyfile(Path(root) / name, output / name)
    (output / "tests").mkdir()
    shutil.copyfile(Path(root) / "test_appliance.py", output / "tests/test_appliance.py")
    package = dict(schema=1, family="native-gnn", cases={},
                   compile_workers=compile_workers,
                   host_sha256={name: common.sha(output / name) for name in common.HOST_FILES},
                   exporter_sha256=common.sha(output / "export_appliance.py"),
                   scope="frozen native-width GNN cases with numerical checks separated from execution admission",
                   evidence="export only; no compile or hardware execution")
    steps = []
    for name, case in cases.items():
        common.require(re.fullmatch("[a-z][a-z0-9_-]{0,63}", name), "case name")
        destination = output / "cases" / name
        freeze(case, destination)
        package["cases"][name] = common.sha(destination / "case.json")
        script = "validations/native-gnn/appliance.py"
        steps.extend([
            dict(name="compile-" + name, kind="appliance", script=script, timeout_seconds=1200,
                 args=["compile", "--case", name, "--backend", "hardware", "--namespace", "job-operator",
                       "--compile-workers", str(compile_workers),
                       "--out", "{output}/compile-" + name]),
            dict(name="run-" + name, kind="appliance", script=script, timeout_seconds=1200,
                 args=["run", "--case", name, "--backend", "hardware", "--artifact-record",
                       "{output}/compile-" + name + "/native-build.json", "--host-timeout", "300",
                       "--out", "{output}/run-" + name]),
            dict(name="audit-" + name, kind="host", script="validations/native-gnn/saved_audit.py",
                 timeout_seconds=120,
                 args=["{output}/run-" + name + "/capture-data", "--out",
                       "{output}/numerical-" + name + ".json"])])
    common.write_json(output / "package.json", package)
    common.write_json(output / "validation.json", dict(schema=1, id="native-gnn",
        description="Native H128 GNN execution capture and separate strict local numerical audit.",
        entrypoints=["appliance.py", "saved_audit.py"], tests=["tests"]))
    common.write_json(output / "request-template.json", dict(schema=1, id=request_id,
        description="Compile, run and separately audit each frozen native GNN case; numerical PASS gates the next case.", steps=steps))
    shutil.copyfile(Path(root) / "README.md", output / "README.md")
    return package


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-id", default="tank28-native-gnn-20261023")
    parser.add_argument("--compile-workers", type=int, choices=(1, 4, 8), default=1)
    args = parser.parse_args(argv)
    cases = {}
    for text in args.case:
        if "=" not in text:
            parser.error("each case requires NAME=PATH")
        name, path = text.split("=", 1)
        if name in cases:
            parser.error("duplicate case name")
        cases[name] = Path(path)
    export(cases, args.output, args.request_id, compile_workers=args.compile_workers)


if __name__ == "__main__":
    main()
