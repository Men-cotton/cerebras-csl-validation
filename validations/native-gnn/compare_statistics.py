"""Compare matched resident captures differing only in ROOT_STATS."""
import argparse
import json
from pathlib import Path
import zipfile

import saved_audit as common
import compare_rollouts as resident


def compare(first, second):
    folders = [Path(first).absolute(), Path(second).absolute()]
    reports = [common.audit_report(folder) for folder in folders]
    common.require(all(row.get("execution_status") == row.get("numerical_status") == "PASS" for row in reports),
                   "both complete numerical audits must pass")
    common.require(reports[0]["backend"] == reports[1]["backend"], "same execution backend")
    manifests = [common.load(folder / "case") for folder in folders]
    left, right = manifests
    common.require(all(m["schema"] == 2 for m in manifests), "resident captures")
    common.require([m["params"].get("ROOT_STATS") for m in manifests] == [0, 1], "baseline then root statistics")
    common.require({k: v for k, v in left["params"].items() if k != "ROOT_STATS"} ==
                   {k: v for k, v in right["params"].items() if k != "ROOT_STATS"}, "matched parameters")
    for name in ("launches", "inputs", "outputs", "checks", "immutable_inputs", "exact_mask_counts"):
        common.require(left[name] == right[name], "matched " + name)
    csl = [{name: digest for name, digest in m["files"].items() if name.startswith("source/") and name.endswith(".csl")}
           for m in manifests]
    common.require(csl[0] and csl[0] == csl[1], "same frozen device source")
    origins = [common.read_json(folder / "case/origin-case.json") for folder in folders]
    for name in ("provenance", "sources", "transport", "schedule"):
        common.require(origins[0][name] == origins[1][name], "same originating " + name)
    for filename in ("inputs.npz", "expected.npz", "zero_masks.npz", "exact_masks.npz"):
        common.require(left["files"][filename] == right["files"][filename], "bit-identical frozen " + filename)
    resident.same_arrays(*[resident.origin_arrays(folder, m, "original.npz")
                          for folder, m in zip(folders, manifests)], "original model and graph")
    raw = [common.load_npz(folder / "runtime/raw.npz", common.CAPTURE_LIMIT) for folder in folders]
    # Check all checkpoints, including the integer controls; clocks alone differ.
    fields = [name for name in common.checkpoint_fields(left) if not name.endswith("_clocks")]
    resident.same_arrays(*[{name: arrays[name] for name in fields} for arrays in raw], "all checkpoint outputs")
    return dict(status="PASS", backend=reports[0]["backend"], bit_equal_checkpoint_arrays=fields,
                physical_steps=sum(step["steps"] for step in left["launches"]),
                captures=[dict(path=str(folder), case_sha256=report["case_sha256"],
                               raw_sha256=report["raw_sha256"], artifact_sha256=report["artifact_sha256"])
                          for folder, report in zip(folders, reports)],
                intervals=[resident.local_intervals(m, arrays) for m, arrays in zip(manifests, raw)],
                comparator_sha256=common.sha(Path(__file__)),
                scope="bit-identical resident outputs at matched statistics modes; PE-local intervals exclude host gaps and are not global makespan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    common.require(not args.out.exists() and not args.out.is_symlink(), "fresh comparison report")
    try:
        result = compare(args.first, args.second)
    except (AssertionError, ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        result = dict(status="FAIL", error=str(error)[:65536])
    common.write_json(args.out, result)
    print(json.dumps(dict(status=result["status"], report=str(args.out))))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
