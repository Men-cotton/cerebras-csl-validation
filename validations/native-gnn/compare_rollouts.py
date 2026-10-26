"""Compare complete resident rollout captures without importing the SDK."""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

try:
    from . import saved_audit as common
except ImportError:
    import saved_audit as common


NUMERICAL = ("h_out", "e_out", "delta_h", "delta_e", "aggregate", "sender_cache",
             "encoded_h", "cached_e0", "decoder_output", "velocity_out")


def same_arrays(left, right, label):
    common.require(set(left) == set(right), label + " array inventory")
    for name, value in left.items():
        other = right[name]
        common.require(value.shape == other.shape and value.dtype == other.dtype
                       and value.tobytes(order="C") == other.tobytes(order="C"), label + " array bits: " + name)


def origin_arrays(folder, manifest, name):
    mapping = manifest["origin_files"]
    common.require(name in mapping, "original physical arrays required: " + name)
    return common.load_npz(common.local_file(folder / "case", mapping[name]))


def local_intervals(manifest, raw):
    totals = np.zeros(manifest["params"]["C"] * manifest["params"]["R"], np.uint64)
    chunks = []
    for index, launch in enumerate(manifest["launches"]):
        clock = raw[f"launch{index}_clocks"].astype(np.uint64)
        begin = clock[:, 0] + (clock[:, 1] << 16) + (clock[:, 2] << 32)
        end = clock[:, 3] + (clock[:, 4] << 16) + (clock[:, 5] << 32)
        span = (end - begin) & ((1 << 48) - 1)
        # Both captures have already passed the strict clock auditor.
        totals += span
        chunks.append(dict(steps=launch["steps"], local_cycles_min=int(span.min()), local_cycles_max=int(span.max())))
    return dict(chunks=chunks, per_pe_sum_cycles=totals.tolist(),
                per_pe_sum_min=int(totals.min()), per_pe_sum_max=int(totals.max()),
                scope="sum of intervals on each same PE; host checkpoint gaps excluded; not a global makespan")


def compare(first, second):
    folders = [Path(first).absolute(), Path(second).absolute()]
    reports = [common.audit_report(folder) for folder in folders]
    common.require(all(row.get("execution_status") == row.get("numerical_status") == "PASS" for row in reports),
                   "both captures must pass strict numerical audit before comparison")
    common.require(reports[0]["backend"] == reports[1]["backend"], "same execution backend required")
    manifests = [common.load(folder / "case") for folder in folders]
    common.require(all(m["schema"] == 2 for m in manifests), "two resident schema2 captures required")
    left, right = manifests
    common.require(left["params"] == right["params"] and left["inputs"] == right["inputs"]
                   and left["outputs"] == right["outputs"], "same parameters and device ABI required")
    common.require(set(NUMERICAL) <= set(left["outputs"]), "complete physical numerical output inventory")
    common.require(left["checks"] == right["checks"] and left["immutable_inputs"] == right["immutable_inputs"],
                   "same numerical and input mutation contracts required")
    chunks = [[row["steps"] for row in manifest["launches"]] for manifest in manifests]
    common.require(chunks[0] != chunks[1] and sum(chunks[0]) == sum(chunks[1]),
                   "different chunks with equal cumulative physical steps required")
    csl = [{name: digest for name, digest in manifest["files"].items()
            if name.startswith("source/") and name.endswith(".csl")} for manifest in manifests]
    common.require(csl[0] and csl[0] == csl[1], "same frozen CSL source required")
    origins = [common.read_json(folder / "case/origin-case.json") for folder in folders]
    for name in ("provenance", "sources", "transport", "schedule"):
        common.require(name in origins[0] and name in origins[1] and origins[0][name] == origins[1][name],
                       "same originating " + name + " required")
    same_arrays(*[common.load_npz(folder / "case/inputs.npz") for folder in folders], "uploaded physical inputs")
    same_arrays(*[origin_arrays(folder, manifest, "original.npz") for folder, manifest in zip(folders, manifests)],
                "original graph, parameters and normalization")
    public = [{name: digest for name, digest in manifest["files"].items()
               if name in ("public-model-import.json", "PUBLIC_MODEL_LICENSE")} for manifest in manifests]
    common.require(public[0] == public[1], "same public checkpoint provenance required")
    raw = [common.load_npz(folder / "runtime/raw.npz", common.CAPTURE_LIMIT) for folder in folders]
    prefixes = [f"launch{len(manifest['launches'])-1}_" for manifest in manifests]
    same_arrays(*[{name: arrays[prefix + name] for name in NUMERICAL} for arrays, prefix in zip(raw, prefixes)],
                "final numerical state")
    lifecycle = [arrays[prefix + "lifecycle"] for arrays, prefix in zip(raw, prefixes)]
    for values, plan in zip(lifecycle, chunks):
        wanted = np.array([1, len(plan), sum(plan), 1], np.uint32)
        common.require(np.array_equal(values, np.broadcast_to(wanted, values.shape)), "completed resident lifecycle")
    common.require(np.array_equal(lifecycle[0][:, [0, 2, 3]], lifecycle[1][:, [0, 2, 3]]),
                   "same initialization, physical steps and geometry count")
    inventories = [common.read_json(folder / "build/inventory.json", limit=common.INVENTORY_JSON_LIMIT) for folder in folders]
    kernel_hashes = [{name: row["sha256"] for name, row in inventory["files"].items()} for inventory in inventories]
    return dict(status="PASS", backend=reports[0]["backend"], bit_equal_arrays=list(NUMERICAL),
                physical_steps=sum(chunks[0]), chunks=chunks,
                identical_generated_elf_hashes=kernel_hashes[0] == kernel_hashes[1],
                identical_artifact_hashes=reports[0]["artifact_sha256"] == reports[1]["artifact_sha256"],
                generated_elf_sha256=kernel_hashes,
                captures=[dict(path=str(folder), case_sha256=report["case_sha256"], raw_sha256=report["raw_sha256"],
                               manifest_sha256=report["manifest_sha256"], request_sha256=report["request_sha256"],
                               numerical_status=report["numerical_status"]) for folder, report in zip(folders, reports)],
                intervals=[local_intervals(manifest, arrays) for manifest, arrays in zip(manifests, raw)],
                comparator_sha256=common.sha(Path(__file__)), auditor_sha256=common.sha(Path(common.__file__)),
                scope="equal final physical state across different resident chunk schedules; independent numerical audits pass; executable identity reported separately")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    common.require(not args.out.exists() and not args.out.is_symlink(), "fresh comparison report required")
    try:
        result = compare(args.first, args.second)
    except (AssertionError, ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        result = dict(status="FAIL", error=str(error)[:65536],
                      scope="resident comparison did not establish equal final states")
    common.write_json(args.out, result)
    print(json.dumps(dict(status=result["status"], report=str(args.out))))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
