"""Export frozen full-feature Cora cases through the registered appliance workflow."""
import argparse
from pathlib import Path
import re
import shutil
try:
    from . import saved_audit as common
except ImportError:
    import saved_audit as common
HERE = Path(__file__).resolve().parent


def export(cases, output, request_id, *, compile_workers=1, host_timeout=600):
    common.require(isinstance(cases, dict) and 1 <= len(cases) <= 8, "one to eight named cases")
    common.require(re.fullmatch("[a-z][a-z0-9_-]{0,95}", request_id), "request ID")
    workers = common.compile_workers(compile_workers)
    common.require(type(host_timeout) is int and 1 <= host_timeout <= 1800, "bounded hardware worker timeout")
    output = Path(output).absolute()
    common.require(output == output.resolve(), "export symlink")
    output.mkdir(parents=True, exist_ok=False)
    for name in (*common.HOST_FILES, "export_appliance.py", "README.md"):
        shutil.copyfile(HERE/name, output/name)
    (output/"tests").mkdir()
    shutil.copyfile(HERE/"test_appliance.py", output/"tests/test_appliance.py")
    package = dict(schema=1, family="native-cora", cases={}, compile_workers=workers,
                   host_sha256={name: common.sha(output/name) for name in common.HOST_FILES},
                   exporter_sha256=common.sha(output/"export_appliance.py"),
                   scope="native1433-feature original-Cora learned1433-16-7 inference; complete portable numerical audits",
                   evidence="package preparation only; no remote execution claim")
    steps = []
    for name, path in cases.items():
        common.require(re.fullmatch("[a-z][a-z0-9_-]{0,63}", name), "case name")
        source = Path(path).absolute()
        manifest = common.load(source)
        common.require(manifest["admission"]["admitted"], "cannot export forecast-rejected case")
        destination = output/"cases"/name
        destination.mkdir(parents=True)
        for relative in sorted(set(manifest["files"]) | {"case.json"}):
            target = common.local_file(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(common.local_file(source, relative), target)
        common.load(destination)
        package["cases"][name] = common.sha(destination/"case.json")
        script = "validations/native-cora/appliance.py"
        steps.extend([
            dict(name="compile-"+name, kind="appliance", script=script, timeout_seconds=1200,
                 args=["compile", "--case", name, "--backend", "hardware", "--namespace", "job-operator",
                       "--compile-workers", str(workers), "--out", "{output}/compile-"+name]),
            dict(name="run-"+name, kind="appliance", script=script, timeout_seconds=min(3300, host_timeout+600),
                 args=["run", "--case", name, "--backend", "hardware", "--artifact-record",
                       "{output}/compile-"+name+"/native-build.json", "--host-timeout", str(host_timeout), "--out", "{output}/run-"+name]),
            dict(name="audit-"+name, kind="host", script="validations/native-cora/saved_audit.py", timeout_seconds=120,
                 args=["{output}/run-"+name+"/capture-data", "--out", "{output}/numerical-"+name+".json"])])
    common.write_json(output/"package.json", package)
    common.write_json(output/"validation.json", dict(schema=1, id="native-cora",
                      description="Complete native-feature Cora GCN inference with retained trained weights and independent portable audit.",
                      entrypoints=["appliance.py", "saved_audit.py"], tests=["tests"]))
    common.write_json(output/"request-template.json", dict(schema=1, id=request_id,
                      description="Compile, capture and numerically audit each complete GCN case before advancing; calibration subsets make no accuracy claim.", steps=steps))
    return package


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
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
    export(cases, args.output, args.request_id, compile_workers=args.compile_workers, host_timeout=args.host_timeout)


if __name__ == "__main__":
    main()
