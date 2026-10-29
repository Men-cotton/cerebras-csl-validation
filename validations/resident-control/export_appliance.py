"""Export one frozen E166 case with one compile and three separately audited arms."""
import argparse
from pathlib import Path
import re
import shutil
try:
    from . import saved_audit as common
except ImportError:
    import saved_audit as common

HERE = Path(__file__).resolve().parent


def export(case, output, request_id, *, compile_workers=1):
    case, output = Path(case).absolute(), Path(output).absolute()
    manifest = common.load(case)
    workers = common.compile_workers(compile_workers)
    common.require(re.fullmatch("[a-z][a-z0-9_-]{0,95}", request_id), "request ID")
    common.require(output == output.resolve(), "export path symlink")
    output.mkdir(parents=True, exist_ok=False)
    for name in (*common.HOST_FILES, "export_appliance.py", "compare_arms.py", "README.md"):
        shutil.copyfile(HERE/name, output/name)
    (output/"tests").mkdir()
    shutil.copyfile(HERE/"test_appliance.py", output/"tests/test_appliance.py")
    shutil.copyfile(HERE/"test_compare.py", output/"tests/test_compare.py")
    name = manifest["kind"]
    frozen = output/"cases"/name
    frozen.mkdir(parents=True)
    for relative in sorted(set(manifest["files"]) | {"case.json"}):
        destination = common.local_file(frozen, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(common.local_file(case, relative), destination)
    common.load(frozen)
    package = dict(schema=1, family="resident-control", cases={name: common.sha(frozen/"case.json")},
                   host_sha256={name: common.sha(output/name) for name in common.HOST_FILES},
                   exporter_sha256=common.sha(output/"export_appliance.py"), compile_workers=workers,
                   scope=manifest["scope"], evidence="frozen host oracle only; compile and remote execution unrun")
    common.write_json(output/"package.json", package)
    script = "validations/resident-control/appliance.py"
    steps = [dict(name="compile-"+name, kind="appliance", script=script, timeout_seconds=1200,
                  args=["compile", "--case", name, "--backend", "hardware", "--namespace", "job-operator",
                        "--compile-workers", str(workers), "--out", "{output}/compile-"+name])]
    for selected in common.ARMS:
        suffix = name+"-"+selected
        steps += [dict(name="run-"+suffix, kind="appliance", script=script, timeout_seconds=1200,
                       args=["run", "--case", name, "--backend", "hardware", "--arm", selected,
                             "--artifact-record", "{output}/compile-"+name+"/native-build.json",
                             "--host-timeout", "300", "--out", "{output}/run-"+suffix]),
                  dict(name="audit-"+suffix, kind="host", script="validations/resident-control/saved_audit.py",
                       timeout_seconds=120, args=["{output}/run-"+suffix+"/capture-data", "--out",
                                                "{output}/numerical-"+suffix+".json"])]
    steps.append(dict(name="compare-"+name, kind="host",
                      script="validations/resident-control/compare_arms.py", timeout_seconds=120,
                      args=["--capture", *["{output}/run-"+name+"-"+selected+"/capture-data" for selected in common.ARMS],
                            "--out", "{output}/comparison-"+name+".json"]))
    common.write_json(output/"validation.json", dict(schema=1, id="resident-control",
                      description="Finite eight-step f_run/f_resume control ABI with separate numerical audit.",
                      entrypoints=["appliance.py", "saved_audit.py", "compare_arms.py"], tests=["tests"]))
    common.write_json(output/"request-template.json", dict(schema=1, id=request_id,
                      description="One artifact, three finite continuation arms; each numerical audit gates the next arm. Pilot only, no speedup claim.", steps=steps))
    return package


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--compile-workers", type=int, choices=(1, 4, 8), default=1)
    args = parser.parse_args(argv)
    export(args.case, args.output, args.request_id, compile_workers=args.compile_workers)


if __name__ == "__main__":
    main()
