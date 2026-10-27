#!/usr/bin/env python3
"""Freeze Top-K and SDK-compatible halo CSL with independent oracles.

Export and verification are host-only preparation. Neither imports the SDK nor
constructs a device runtime. Run preparation under the repository's host budget.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np

TOPK = "kernels/sparse/exact-knn/mesh"
HALO = "kernels/stencil/independent-halo"
HALO_RUNTIME = HALO + "/runtime_coords"
VENDOR = "csl-extras-202505230211-4-d9070058/examples/benchmarks/benchmark-libs/stencil_3d_7pts"
ACTIVE_VENDOR = "sdk-2.10.2/csl-extras-202604101435-6-d2f7d96e/examples/benchmarks/benchmark-libs/stencil_3d_7pts"
ORACLES = (TOPK + "/model.py", "kernels/sparse/exact-knn/model.py",
           "kernels/sparse/exact-knn/reference.py",
           "kernels/sparse/resident-2d-spmm/physical_schedule/armed_clock/storage.py",
           HALO + "/oracle.py")
ORIGINALS = (TOPK + "/layout.csl", TOPK + "/pe.csl", HALO + "/layout.csl",
             HALO + "/pe.csl", VENDOR + "/layout.csl", VENDOR + "/wse3/pe.csl",
             ACTIVE_VENDOR + "/layout.csl", ACTIVE_VENDOR + "/wse3/pe.csl", ACTIVE_VENDOR + "/pe.csl",
             HALO_RUNTIME + "/layout.csl", HALO_RUNTIME + "/pe.csl")
FORWARDED_STENCIL_PARAMS = ("c_recv_west", "c_send_east", "c_recv_east", "c_send_west",
    "c_recv_south", "c_send_north", "c_recv_north", "c_send_south",
    "COMM", "SEND", "RECV", "first_px", "last_px", "first_py", "last_py")
SDK_TRANSFORMS = {
    VENDOR + "/layout.csl": ("sdk-2.10.2-stencil-layout-types", ACTIVE_VENDOR + "/layout.csl"),
    VENDOR + "/wse3/pe.csl": ("sdk-2.10.2-stencil-queue-dsds", ACTIVE_VENDOR + "/wse3/pe.csl"),
}
PARAMS_DECLARATION = b"""const Params = struct {
    c_recv_west: color,
    c_send_east: color,
    c_recv_east: color,
    c_send_west: color,
    c_recv_south: color,
    c_send_north: color,
    c_recv_north: color,
    c_send_south: color,
    SEND: local_task_id,
    RECV: local_task_id,
    COMM: local_task_id,
    first_px: bool,
    last_px: bool,
    first_py: bool,
    last_py: bool,
};

"""
CASES = {
    "topk-small": ("topk", 3, 2, "uneven"),
    "topk24": ("topk", 24, 24, "balanced"),
    "topk32": ("topk", 32, 32, "balanced"),
    "topk64": ("topk", 64, 64, "balanced"),
    "halo-small": ("halo", 2, 2, ((17, 0), (32, 1), (32, 0))),
    "halo16": ("halo", 16, 16, ((32, 0),)),
    "halo32": ("halo", 32, 32, ((32, 0),)),
    "halo64": ("halo", 64, 64, ((32, 0),)),
}
# Existing recipes remain unchanged; independent variants use fresh capsules.
CASES["halo-uneven"] = ("halo", 3, 2, ((17, 0), (32, 1), (32, 0)))
CASES["halo128"] = ("halo", 128, 128, ((32, 0),))
RUNTIME_CASES = {
    "halo-rc-small": CASES["halo-small"],
    "halo-rc-uneven": CASES["halo-uneven"],
    "halo-rc16": CASES["halo16"],
    "halo-rc32": CASES["halo32"],
    "halo-rc64": CASES["halo64"],
    "halo-rc128": CASES["halo128"],
}
CASES.update(RUNTIME_CASES)
# Keep the existing sixteen recipes as the default capsule. Larger halos are
# opt-in so the unchanged sixteen-case package bound remains meaningful.
DEFAULT_CASES = tuple(CASES)
RUNTIME_CASES.update({
    "halo-rc256": ("halo", 256, 256, ((32, 0),)),
    "halo-rc512": ("halo", 512, 512, ((32, 0),)),
    "halo-rc-odd": ("halo", 3, 3, ((17, 0), (32, 1), (32, 0))),
    "halo-rc512x663": ("halo", 512, 663, ((32, 0),)),
    "halo-rc663x512": ("halo", 663, 512, ((32, 0),)),
})
CASES.update(RUNTIME_CASES)
FILE_LIMIT = 96 << 20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= FILE_LIMIT, "bounded regular source")
    with path.open("rb") as stream:
        value = stream.read(FILE_LIMIT + 1)
    require(len(value) <= FILE_LIMIT, "source grew beyond bound")
    return value


def sha(value):
    return hashlib.sha256(value).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def stencil_compatibility(original_name, payload):
    """Apply only the exact syntax changes present in the active official source."""
    require(original_name in SDK_TRANSFORMS, "known SDK stencil source")
    if original_name == VENDOR + "/layout.csl":
        before = b"fn get_params(px:i16, py:i16) comptime_struct {"
        after = PARAMS_DECLARATION + b"fn get_params(px:i16, py:i16) Params {"
        require(payload.count(before) == 1 and payload.count(b"    return .{") == 1, "historical stencil layout syntax")
        return payload.replace(before, after).replace(b"    return .{", b"    return Params{")
    for color in ("c_recv_west", "c_recv_east", "c_recv_south", "c_recv_north",
                  "c_send_east", "c_send_west", "c_send_north", "c_send_south"):
        line = (("   " if color.startswith("c_recv_") else "    ") + ".fabric_color = " + color + ",\n").encode()
        require(payload.count(line) == 1, "historical queue DSD color field")
        payload = payload.replace(line, b"")
    return payload


def forward_stencil_params(payload):
    """Replace the retired merge builtin with the active official import pattern.

    Input already has its import localized. The original callback, queues,
    DSR arguments, and the complete executable body are retained verbatim.
    """
    before = b'const stencil = @import_module("vendor_pe.csl", @concat_structs(stencil_params, .{\n'
    after = b'const stencil = @import_module("vendor_pe.csl", .{\n' + b"".join(
        ("  .%s=stencil_params.%s,\n" % (name, name)).encode() for name in FORWARDED_STENCIL_PARAMS)
    closing = b"}));\n\nvar x ="
    require(payload.count(before) == 1 and payload.count(closing) == 1, "historical wrapper merge syntax")
    return payload.replace(before, after).replace(closing, b"});\n\nvar x =")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


@contextmanager
def frozen_oracles(package):
    """Preserve the original import-relative hierarchy; import frozen files only."""
    old_path, old_bytecode = sys.path[:], sys.dont_write_bytecode
    names = ("e192_topk", "e192_halo", "e176_original_topk", "e176_independent_topk", "e176_bounded_storage")
    saved = {name: sys.modules.get(name) for name in names}
    try:
        sys.dont_write_bytecode = True
        yield (module("e192_topk", package / "oracle" / (TOPK + "/model.py")),
               module("e192_halo", package / "oracle" / (HALO + "/oracle.py")))
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def topk_case(model, width, height, profile):
    p = model.parameters(width, height, 16, 8, 4)
    fixture = model.make_case(p, profile)
    packed = model.pack(fixture)
    symbols = {name: list(spec) for name, spec in model.spec(p).items()}
    operations, inputs, expected = [], {}, {}
    rectangle = [0, 0, width, height]

    def transfer(op, symbol, key, value, rect=None):
        operations.append(dict(op=op, symbol=symbol, array=key, rect=rect or rectangle))
        (inputs if op == "h2d" else expected)[key] = np.ascontiguousarray(value)

    for name in model.INPUT_NAMES:
        transfer("h2d", name, "setup_" + name, packed[name])
    operations.append(dict(op="launch", symbol="f_load", args=[]))
    for name in model.INPUT_NAMES:
        transfer("d2h", name, "before_" + name, packed[name])
    transfer("d2h", "stats", "loaded_stats", model.loaded(fixture))
    winners = []
    for generation in (1, 2, 3):
        prefix = "q%d_" % generation
        seed = model.query_seed(fixture, generation)
        transfer("h2d", "query_seed", prefix + "seed", seed[:1], [0, 0, 1, 1])
        transfer("d2h", "query_seed", prefix + "before_seed", seed)
        operations.append(dict(op="launch", symbol="f_query", args=[generation]))
        output = model.expected(fixture, generation)
        winners.append(output["best"][0, 1::2].tolist())
        for name in model.OUTPUT_NAMES:
            transfer("d2h", name, prefix + name, output[name])
    for name in ("database", "identities", "meta"):
        transfer("d2h", name, "after_" + name, packed[name])
    require(winners[0] == winners[2] and winners[0] != winners[1], "A/B/A distinct-query witness")
    return dict(schema=1, kind="topk", grid=[width, height], params=p, symbols=symbols,
                functions={"f_load": [], "f_query": [dict(name="generation", type="i16")]},
                operations=operations,
                validations=dict(restore=[["q1_" + name, "q3_" + name] for name in ("query", "distances", "best")]),
                census=dict(model.cost(fixture), profile=profile, sequence=["A", "B", "A"],
                            query_winners=winners, timing_scope="No device clock; worker and client turnaround only.")), inputs, expected


def halo_case(oracle, width, height, sequence, *, runtime_coords=False):
    symbols = {"y": [32, "float32"], "Q": [32, "float32"], "count": [1, "uint32"], "clocks": [6, "uint16"]}
    operations, expected, launches = [], {}, []
    for index, (n, phase) in enumerate(sequence):
        operations.append(dict(op="launch", symbol="apply_epochs", args=[n, phase]))
        reference = oracle.expected(height, width, n, phase)
        # The frozen oracle has a separate sliced FP32 route. Both must agree.
        oracle.verify_result(oracle.vectorized_control(height, width, n, phase), reference)
        for name in ("y", "Q", "count"):
            key = "s%d_%s" % (index, name)
            operations.append(dict(op="d2h", symbol=name, array=key, rect=[0, 0, width, height]))
            expected[key] = np.ascontiguousarray(reference[name].reshape(width * height, symbols[name][0]))
        operations.append(dict(op="d2h", symbol="clocks", array="s%d_clocks" % index,
                               rect=[0, 0, width, height], clock=True))
        launches.append(dict(length=n, phase=phase, **oracle.metrics(height, width, n)))
    return dict(schema=1, kind="halo", grid=[width, height], params=dict(width=width, height=height), symbols=symbols,
                functions={"apply_epochs": [dict(name="length", type="i16"), dict(name="input_phase", type="i16")]},
                operations=operations, validations=dict(restore=[]),
                census=dict(epochs_per_launch=64, max_n=32, block_size=16, sequence=[list(s) for s in sequence], launches=launches,
                            timing_scope="PE-local timestamp interval over 64 independent epochs; not global makespan.",
                            placement=("Runtime coordinates computed once per launch inside the PE-local clock; boundary/parity routing and active SDK stencil unchanged." if runtime_coords else
                                       "Original compile-time X/Y specialization; wrapper imports localized and parameters explicitly forwarded; vendor layout/DSD syntax matches SDK 2.10.2 official source."))), {}, expected


def source_map(kind, *, runtime_coords=False):
    if runtime_coords:
        require(kind == "halo", "runtime coordinates apply only to halo")
        return {"source/layout.csl": (HALO_RUNTIME + "/layout.csl", None),
                "source/pe.csl": (HALO_RUNTIME + "/pe.csl", None),
                "source/vendor_layout.csl": (ACTIVE_VENDOR + "/layout.csl", None),
                "source/vendor_pe.csl": (ACTIVE_VENDOR + "/wse3/pe.csl", None)}
    base = TOPK if kind == "topk" else HALO
    result = {"source/layout.csl": (base + "/layout.csl", None), "source/pe.csl": (base + "/pe.csl", None)}
    if kind == "halo":
        result.update({"source/vendor_layout.csl": (VENDOR + "/layout.csl", None),
                       "source/vendor_pe.csl": (VENDOR + "/wse3/pe.csl", None)})
        result["source/layout.csl"] = (base + "/layout.csl", ("../../../" + VENDOR + "/layout.csl", "vendor_layout.csl"))
        result["source/pe.csl"] = (base + "/pe.csl", ("../../../" + VENDOR + "/wse3/pe.csl", "vendor_pe.csl"))
    return result


def inventory(package):
    # Only enumerate this fresh small capsule; never scan SDK or the workspace.
    files, total = {}, 0
    for path in sorted(package.rglob("*")):
        require(not path.is_symlink(), "capsule symlink")
        if path.is_dir():
            continue
        require(path.is_file() and len(files) < 512, "capsule file bound")
        total += path.stat().st_size
        require(total <= 256 << 20, "capsule byte bound")
        files[path.relative_to(package).as_posix()] = sha(read(path))
    return files


def export(repo, output, source_commit, names=None):
    repo, output = Path(repo).resolve(), Path(output).absolute()
    require(output == output.resolve() and not output.exists(), "fresh canonical output required")
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit), "full source commit")
    names = list(DEFAULT_CASES) if names is None else list(names)
    require(1 <= len(names) <= 16 and len(names) == len(set(names)) and
            all(n in CASES for n in names), "1..16 known unique cases")
    original = {name: read(repo / name) for name in ORACLES + ORIGINALS}
    exporter_bytes = read(Path(__file__).resolve())
    output.mkdir(parents=True)
    for name in ORACLES:
        put(output / "oracle" / name, original[name])
    for name in ORIGINALS:
        put(output / "provenance/originals" / name, original[name])
    put(output / "provenance/export_tank_cases.py", exporter_bytes)
    oracle_sources = {"oracle/" + name: sha(original[name]) for name in ORACLES}
    entries = {}
    with frozen_oracles(output) as (topk, halo):
        for name in names:
            kind, width, height, argument = CASES[name]
            case, inputs, expected = (topk_case(topk, width, height, argument) if kind == "topk"
                                      else halo_case(halo, width, height, argument, runtime_coords=name in RUNTIME_CASES))
            directory = output / "cases" / name
            directory.mkdir(parents=True)
            np.savez(directory / "inputs.npz", **inputs)
            np.savez(directory / "expected.npz", **expected)
            origins = {}
            for relative, (original_name, change) in source_map(kind, runtime_coords=name in RUNTIME_CASES).items():
                payload = original[original_name]
                transformation = "none"
                compatibility_reference = None
                if change is not None:
                    old, new = ('"' + value + '"' for value in change)
                    require(payload.count(old.encode()) == 1, "one immutable import localization")
                    payload = payload.replace(old.encode(), new.encode())
                    transformation = "localize-stencil-import"
                    if original_name == HALO + "/pe.csl":
                        payload = forward_stencil_params(payload)
                        transformation = "localize-stencil-import-and-forward-params"
                elif original_name in SDK_TRANSFORMS:
                    transformation, reference_name = SDK_TRANSFORMS[original_name]
                    payload = stencil_compatibility(original_name, payload)
                    require(payload == original[reference_name], "adaptation must exactly equal active official stencil source")
                    compatibility_reference = dict(path=reference_name, sha256=sha(original[reference_name]))
                put(directory / relative, payload)
                origins[relative] = dict(original_path=original_name, original_sha256=sha(original[original_name]),
                                         exported_sha256=sha(payload), transformation=transformation)
                if compatibility_reference is not None:
                    origins[relative]["compatibility_reference"] = compatibility_reference
                if transformation == "localize-stencil-import-and-forward-params":
                    reference_name = ACTIVE_VENDOR + "/pe.csl"
                    origins[relative]["pattern_reference"] = dict(path=reference_name, sha256=sha(original[reference_name]),
                        scope="Explicit forwarding of the 15 layout fields; callback, queues, DSR arguments and executable body remain original.")
            case.update(files=inventory(directory), source_origin=origins, oracle_sources=oracle_sources)
            payload = json_bytes(case)
            put(directory / "case.json", payload)
            entries[name] = dict(case_sha256=sha(payload), params=case["params"], grid=case["grid"], kind=kind, family=kind)
            print(json.dumps(dict(case=name, status="EXPORTED", numerical_arrays=len(expected))), flush=True)
    require(all(read(repo / name) == value for name, value in original.items()) and
            read(Path(__file__).resolve()) == exporter_bytes, "source changed during export")
    manifest = dict(schema=1, files=inventory(output), cases=entries, oracle_sources=oracle_sources,
                    source_commit=source_commit, exporter_sha256=sha(exporter_bytes))
    put(output / "manifest.json", json_bytes(manifest))
    return manifest


def verify(package):
    """Regenerate all fixture arrays from the package's frozen independent source."""
    import tank_audit
    import tank_worker
    package = Path(package).resolve()
    manifest = tank_audit.verify_package(package)
    require(sha(read(Path(__file__).resolve())) == manifest["exporter_sha256"], "run the package's exact frozen exporter revision")
    with frozen_oracles(package) as (topk, halo):
        for name in manifest["cases"]:
            require(name in CASES, "known frozen case recipe")
            kind, width, height, argument = CASES[name]
            wanted, inputs, expected = (topk_case(topk, width, height, argument) if kind == "topk"
                                        else halo_case(halo, width, height, argument, runtime_coords=name in RUNTIME_CASES))
            directory = package / "cases" / name
            saved = json.loads(read(directory / "case.json"))
            require(all(saved[key] == value for key, value in wanted.items()), "regenerated case recipe")
            for filename, regenerated in (("inputs.npz", inputs), ("expected.npz", expected)):
                actual = tank_worker.arrays(directory / filename)
                require(set(actual) == set(regenerated), "regenerated array inventory")
                for key, value in regenerated.items():
                    tank_audit.same(actual[key], value, name + "/" + key)
            print(json.dumps(dict(case=name, status="ORACLE_REGENERATED")), flush=True)
    return dict(status="PASS", cases=list(manifest["cases"]), package_manifest_sha256=sha(read(package / "manifest.json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--cases", help="Up to 16 comma-separated recipes; default is the existing cases through halo128; runtime cases beyond halo-rc128 and the odd pilot require explicit selection")
    parser.add_argument("--verify", type=Path, help="Regenerate every expected array from a frozen package")
    args = parser.parse_args()
    if args.verify is not None:
        require(args.repo is None and args.out is None and args.source_commit is None and args.cases is None, "verification has no export arguments")
        print(json.dumps(verify(args.verify)))
    else:
        require(args.repo is not None and args.out is not None and args.source_commit is not None, "export requires repo/out/source-commit")
        export(args.repo, args.out, args.source_commit, None if args.cases is None else args.cases.split(","))


if __name__ == "__main__":
    main()
