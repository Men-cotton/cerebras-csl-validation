"""Combine the two independently replayed fresh runtimes of one Shah profile."""
import argparse
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audits", type=Path, nargs=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audits = [json.loads(p.read_text()) for p in args.audits]
    keys = ("case_sha256", "case_label", "artifact_sha256", "source_sha256")
    if (any(a["status"] != "PASS" or a["backend"] != "hardware" or a["mode"] != "gcn" for a in audits)
            or any(audits[0][k] != audits[1][k] for k in keys) or audits[0]["run_id"] == audits[1]["run_id"]):
        raise ValueError("two distinct audited runtimes of one physical compile required")
    result = {k: audits[0][k] for k in keys}
    runs = []
    for a in audits:
        r = a["result"]
        if (r["status"] != "PASS" or r["backend"] != "hardware" or r["run_id"] != a["run_id"]
                or r["case_sha256"] != a["case_sha256"] or len(r["observations"]) != 6
                or [i["warmup"] for i in r["observations"]] != [True]+[False]*5):
            raise ValueError("complete, independently replayed one-warmup/five-measurement sequence")
        runs.append({k: r[k] for k in ("run_id", "profile", "application_pes", "host_completed_median_ns",
                                        "max_pe_median_cycles", "observations")})
    result.update(status="PASS", runtimes=runs, median_of_runtime_medians_ns=median(
        run["host_completed_median_ns"] for run in runs))
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
