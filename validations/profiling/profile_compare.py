"""Compare four admitted off/on/on/off TANK29 captures without submitting jobs."""
import argparse
from pathlib import Path
import statistics

from profile_common import audit_capture, require, write_json


def compare(captures):
    require(len(captures) == 4, "exactly four ABBA captures")
    rows = [audit_capture(path) for path in captures]
    require([row["flag"] for row in rows] == [0, 1, 1, 0], "runtime order must be off/on/on/off")
    require(len({row["run_id"] for row in rows}) == 4, "four distinct fresh runtimes")
    for field in ("backend", "manifest_sha256", "host_sha256", "source_sha256", "n"):
        require(all(row[field] == rows[0][field] for row in rows), "matched build/source: " + field)
    comparisons = []
    for index in range(3):
        off = [rows[j]["invocations"][index]["sender"]["elapsed"] for j in (0, 3)]
        on = [rows[j]["invocations"][index]["sender"]["elapsed"] for j in (1, 2)]
        baseline, observed = statistics.mean(off), statistics.mean(on)
        comparisons.append(dict(invocation=index+1, input_case="ABA"[index],
                                off_sender_cycles=off, on_sender_cycles=on,
                                off_mean_cycles=baseline, on_mean_cycles=observed,
                                difference_cycles=observed-baseline,
                                relative_change=(observed-baseline)/baseline))
    return dict(schema=1, status="pass", backend=rows[0]["backend"], n=4,
                manifest_sha256=rows[0]["manifest_sha256"], order=[0, 1, 1, 0],
                independent_runtimes=4, independent_runtimes_per_arm=2,
                nested_invocations_per_runtime=3, comparisons=comparisons, runtimes=rows,
                interpretation="Marginal effect of enabling five receiver records in one fixed ELF. Off retains buffers, branches and TSC controls. Nested A/B/A invocations are not independent repeats. Same-PE intervals include observer work and callback scheduling; no cross-PE subtraction, pure timer-cost subtraction or speed threshold.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", nargs=4, type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists() and not args.out.is_symlink(), "fresh comparison output required")
    result = compare(args.captures)
    write_json(args.out, result)
    print("PASS: four distinct runtimes; off/on/on/off; " + str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
