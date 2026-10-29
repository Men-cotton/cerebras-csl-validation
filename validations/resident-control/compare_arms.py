"""Audit three E215 arms, then require bit-identical final arithmetic outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from . import saved_audit as common
except ImportError:
    import saved_audit as common

EXCLUDED = ("control_status", "step_times")


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def local_build(case, rows):
    """Bind available local run receipts; an absent receipt is never invented."""
    requests = [row["runtime"].parent/"request.json" for row in rows]
    compile_path = case/"compile.json"
    present = [path.exists() for path in requests]
    if not any(present) and not compile_path.exists() and not (case/"build").exists():
        return dict(status="UNAVAILABLE", scope="raw runtime captures have no local compile receipts; no shared-artifact claim")
    common.require(all(present) and compile_path.exists(), "incomplete local compile provenance")
    compiled = common.read_json(compile_path, limit=common.INVENTORY_JSON_LIMIT)
    digest = common.sha(compile_path, common.INVENTORY_JSON_LIMIT)
    common.require(compiled.get("status") == "PASS" and compiled.get("returncode") == 0
                   and "error" not in compiled and compiled.get("backend") == "simulator", "successful local compile receipt")
    common.require(compiled.get("case_sha256") == common.sha(case/"case.json"), "compiled case identity")
    inventory = compiled.get("inventory")
    common.admit_compiled_inventory(inventory)
    for row, request_path in zip(rows, requests):
        request = common.read_json(request_path)
        common.require(request.get("backend") == row["execution"]["backend"] == "simulator"
                       and request.get("arm") == row["arm"]
                       and request.get("case_sha256") == row["execution"]["case_sha256"]
                       and request.get("compile_sha256") == digest, "same local compile receipt for all arms")
    actual = (case/"build").exists()
    if actual:
        common.check_abi(case/"build", rows[0]["manifest"])
        common.require(common.compiled_inventory(case/"build") == inventory, "current generated ELF inventory")
    return dict(status="BOUND", compile_sha256=digest, inventory_sha256=object_sha(inventory),
                host_sha256=compiled.get("host_sha256", {}), current_generated_inventory_checked=actual,
                scope="same local compile receipt and generated-ELF inventory; hardware artifact absent")


def appliance_build(rows):
    identities = []
    for row in rows:
        folder = row["capture"]
        build = common.read_json(folder/"manifest.json")
        record = common.read_json(folder/"capture.json")
        # Each field was already admitted by common.admit_capture.
        identities.append(dict(manifest_sha256=common.sha(folder/"manifest.json"),
                               artifact_sha256=build["artifact_sha256"], host_sha256=build["host_sha256"],
                               compiled_metadata_sha256=record["compiled_metadata_sha256"]))
    common.require(all(identity == identities[0] for identity in identities[1:]), "same appliance artifact and compiled inventory for all arms")
    return dict(status="BOUND", **identities[0], scope="same admitted appliance compile manifest, artifact hash and generated-ELF inventory")


def compare(*, case=None, runtimes=None, captures=None):
    local = runtimes is not None
    common.require(local != (captures is not None), "choose local runtimes or appliance captures")
    common.require((case is not None) == local, "local runtimes require exactly one case")
    locations = [Path(path).absolute() for path in (runtimes if local else captures)]
    common.require(len(locations) == 3 and len(set(locations)) == 3
                   and all(path == path.resolve() and path.is_dir() for path in locations), "three distinct canonical capture directories")
    case = Path(case).absolute() if local else None
    rows = []
    for location in locations:
        if local:
            runtime, current_case = location, case
            admission = None
        else:
            admission = common.admit_capture(location)
            runtime, current_case = location/"runtime", location/"case"
        # A comparator cannot rescue a failed individual numerical audit.
        audited = common.audit(current_case, runtime)
        common.require(audited.get("status") == audited.get("numerical_status") == "PASS", "individual numerical audit")
        manifest, execution, raw = common.execution_arrays(current_case, runtime)
        if local:
            common.require(execution["backend"] == "simulator", "local comparison requires simulator captures")
        rows.append(dict(arm=execution["arm"], runtime=runtime, capture=None if local else location,
                         manifest=manifest, execution=execution, raw=raw, audited=audited, admission=admission))
    common.require({row["arm"] for row in rows} == set(common.ARMS), "exactly one resident, staged and download arm")
    common.require(len({row["execution"]["case_sha256"] for row in rows}) == 1, "same frozen case for all arms")
    common.require(len({row["execution"]["backend"] for row in rows}) == 1, "same backend for all arms")
    build = local_build(case, rows) if local else appliance_build(rows)
    by_arm = {row["arm"]: row for row in rows}
    reference = by_arm["resident"]
    compared = sorted(set(reference["manifest"]["outputs"]) - set(EXCLUDED))
    common.require(compared == sorted(("history", "stats", "release", "w0", "w1", "checkpoint_step", "run_summary")), "complete final comparison contract")
    words, digests = 0, {}
    for name in compared:
        wanted = reference["raw"][name]
        for selected in ("staged", "download"):
            common.equal_bits(by_arm[selected]["raw"][name], wanted, "cross-arm final bits: "+name)
        words += wanted.size
        digests[name] = hashlib.sha256(wanted.tobytes(order="C")).hexdigest()
    # Individual auditors permit only the intended per-arm control difference.
    # All other immutable readbacks must also agree bit-for-bit across arms.
    immutable = sorted(set(reference["manifest"]["immutable_inputs"]) - {"control"})
    for name in immutable:
        for selected in ("staged", "download"):
            common.equal_bits(by_arm[selected]["raw"]["after_"+name], reference["raw"]["after_"+name], "cross-arm immutable input: "+name)
    return dict(status="PASS", numerical_status="PASS", cross_arm_final_bits="PASS",
                backend=reference["execution"]["backend"], case_sha256=reference["execution"]["case_sha256"],
                build_identity=build, arms=list(common.ARMS), compared_final_outputs=compared,
                compared_32bit_words_per_arm=int(words), final_output_sha256=digests,
                excluded_final_outputs=list(EXCLUDED), compared_immutable_inputs=immutable,
                individual={selected: dict(runtime=str(by_arm[selected]["runtime"]),
                    raw_sha256=by_arm[selected]["execution"]["raw_sha256"],
                    execution_sha256=common.sha(by_arm[selected]["runtime"]/"execution.json"),
                    max_absolute_error=by_arm[selected]["audited"]["max_absolute_error"],
                    host_timing=by_arm[selected]["audited"]["host_timing"],
                    active_cycles=by_arm[selected]["audited"]["active_cycles"],
                    gap_cycles=by_arm[selected]["audited"]["gap_cycles"],
                    span_cycles=by_arm[selected]["audited"]["span_cycles"]) for selected in common.ARMS},
                claim="three individually audited arms have identical final arithmetic/protocol bits",
                timing_scope="one unbalanced pilot per arm; host and same-PE clocks reported separately; no speedup or global makespan claim")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--runtime", nargs=3, type=Path, metavar=("RESIDENT", "STAGED", "DOWNLOAD"))
    mode.add_argument("--capture", nargs=3, type=Path, metavar=("RESIDENT", "STAGED", "DOWNLOAD"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    common.require(not args.out.exists() and not args.out.is_symlink(), "fresh comparator report")
    try:
        result = compare(case=args.case, runtimes=args.runtime, captures=args.capture)
    except Exception as error:
        common.write_json(args.out, dict(status="FAIL", cross_arm_final_bits="FAIL", error=str(error)))
        return 2
    common.write_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
