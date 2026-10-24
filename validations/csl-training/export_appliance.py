"""Export frozen CSL training cases to a standalone validation family.

Exporting a request template does not submit work. Publishing and request
trailers are owned by the caller and the validation repository controller.
"""
import argparse
from pathlib import Path
import re
import shutil

try:
    from . import saved_audit as common
except ImportError:
    import saved_audit as common

ROOT = Path(__file__).resolve().parent


def export(cases, output, request_id="tank28-csl-training-20261023", root=ROOT):
    common.require(bool(cases) and len(cases) <= 16, "one to sixteen named cases required")
    common.require(re.fullmatch("[a-z][a-z0-9_-]{0,95}", request_id), "request ID")
    loaded = {}
    for name, path in cases.items():
        common.require(re.fullmatch("[a-z][a-z0-9_-]{0,63}", name), "case name")
        path = Path(path).absolute()
        loaded[name] = (path, common.load(path))
    output = Path(output).absolute()
    common.require(output == output.resolve(), "export path symlink")
    output.mkdir(parents=True, exist_ok=False)
    for name in (*common.HOST_FILES, "export_appliance.py"):
        shutil.copyfile(Path(root)/name, output/name)
    (output/"tests").mkdir()
    shutil.copyfile(Path(root)/"test_appliance.py", output/"tests/test_appliance.py")
    package = dict(schema=1, family="csl-training", cases={},
                   host_sha256={name: common.sha(output/name) for name in common.HOST_FILES},
                   exporter_sha256=common.sha(output/"export_appliance.py"),
                   scope="frozen complete CSL forward/backward/SGD; no host gradient or optimizer during runtime",
                   evidence="export only; no compile, simulation or hardware execution")
    steps = []
    for name, (source, manifest) in loaded.items():
        destination = output/"cases"/name
        destination.mkdir(parents=True)
        for relative in sorted(common.CASE_FILES | {"case.json"}):
            target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(common.local_file(source, relative), target)
        common.require(common.load(destination) == manifest, "exported frozen case differs")
        package["cases"][name] = common.sha(destination/"case.json")
        script = "validations/csl-training/appliance.py"
        steps += [dict(name="compile-"+name, kind="appliance", script=script, timeout_seconds=1200,
                       args=["compile", "--case", name, "--backend", "hardware", "--namespace", "job-operator",
                             "--out", "{output}/compile-"+name]),
                  dict(name="run-"+name, kind="appliance", script=script, timeout_seconds=1200,
                       args=["run", "--case", name, "--backend", "hardware", "--artifact-record",
                             "{output}/compile-"+name+"/training-build.json", "--host-timeout", "300",
                             "--out", "{output}/run-"+name])]
    common.write_json(output/"package.json", package)
    common.write_json(output/"validation.json", dict(schema=1, id="csl-training",
        description="Frozen full-graph CSL GNN forward, backward and SGD correctness with exact capture checks.",
        entrypoints=["appliance.py"], tests=["tests"]))
    common.write_json(output/"request-template.json", dict(schema=1, id=request_id,
        description="One compile and one runtime per frozen case; each launch performs one complete CSL SGD update.", steps=steps))
    (output/"README.md").write_text(
        "# Full CSL GNN training validation\n\n"
        "Each named case fixes normalized inputs, supervised targets, initial weights, graph ownership, "
        "CSL sources and independently computed expected predictions, gradients and SGD updates. "
        "The appliance worker supplies frames and invokes the compiled kernel; it computes no gradients or optimizer updates. "
        "The portable NumPy auditor compares every prediction, parameter gradient and updated parameter, "
        "plus input readbacks, exact SDK calls, all operation counters, lifecycle and local clocks.\n\n"
        "`appliance.py compile --case NAME --backend hardware --namespace job-operator --out OUT` "
        "submits one compiler operation. `appliance.py run --case NAME --backend hardware "
        "--artifact-record BUILD/training-build.json --host-timeout 300 --out OUT` submits one runtime. "
        "Outputs must be fresh. Both actions record job IDs through SDK logging and retain failed states. "
        "The request template bounds each client action to 1200 seconds; the worker has a separate 300-second timeout "
        "and 15-second termination grace. A client timeout does not establish remote termination.\n\n"
        "Case admission is bounded to 128 MiB; capture admission to 256 MiB and 256 regular entries. "
        "Download sizes are checked after transfer; these are admission limits, not live remote disk quotas. "
        "Captures include frozen case/host source, compiled ABI metadata and all raw arrays. "
        "Compiler artifacts and caller outputs stay outside the source tree. No retry or campaign loop exists here.\n\n"
        "H4/S2 and a few SGD steps establish a numerical training primitive only. "
        "They do not establish converged fluid accuracy, Adam training, native H128/S15 admission, "
        "or an end-to-end speedup. `request-template.json` is a draft: follow the parent request contract "
        "to publish it with a new execution ID and explicit request trailers. No job was submitted by export.\n")
    return package


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--request-id", default="tank28-csl-training-20261023")
    args = ap.parse_args(argv)
    cases = {}
    for value in args.case:
        if "=" not in value:
            ap.error("each case must use NAME=PATH")
        name, path = value.split("=", 1)
        if name in cases:
            ap.error("duplicate case name")
        cases[name] = Path(path)
    export(cases, args.output, args.request_id)


if __name__ == "__main__":
    main()
