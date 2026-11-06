"""Convert already prepared E214 cases into a standalone bounded-adjoint family.

The exporter validates the live E214 source/SDK identity once. Exported workers
and auditors use only frozen files, with no repository or SDK-image dependency.
"""
import argparse
import importlib.util
from pathlib import Path
import re
import shutil
try:
    from . import saved_audit as common
except ImportError:
    import saved_audit as common

HERE = Path(__file__).resolve().parent


def original_driver():
    spec = importlib.util.spec_from_file_location("bounded_export_modules", HERE.parent/"modules.py")
    modules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modules)
    fixture = modules.load("bounded_export_fixture", HERE.parent/"fixture.py", {"modules": modules})
    return modules.load("bounded_export_driver", HERE.parent/"driver.py", {"modules": modules, "fixture": fixture})


def convert(source, destination, validator=None):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    driver = validator or original_driver()
    parent, _ = driver.load_case(source)
    common.require(parent["metadata"]["declared_admission"] == "ADMITTED", "parent forecast rejected")
    destination.mkdir(parents=True, exist_ok=False)
    for name in parent["files"]:
        target = common.local_file(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(common.local_file(source, name), target)
    shutil.copyfile(source/"case.json", destination/"source-manifest.json")
    params = parent["params"]
    named = common.named_bytes(params)
    reserve = parent["metadata"]["code_sdk_reserve_bytes"]
    common.require(named+reserve == parent["metadata"]["forecast_static_bytes"], "complete source census unchanged")
    manifest = dict(schema=1, family="bounded-adjoint", params=params,
                    inputs=common.schema(params), outputs=common.schema(params, True),
                    launches=[dict(entrypoint="run", arguments=[], update=None),
                              dict(entrypoint="run", arguments=[], update="derivative")],
                    source_case_sha256=common.sha(destination/"source-manifest.json"),
                    files={name: common.sha(destination/name) for name in common.CASE_REQUIRED},
                    admission=dict(named_bytes=named, code_reserve_bytes=reserve, forecast_bytes=named+reserve,
                                   admitted=named+reserve <= 44*1024),
                    atol=common.ATOL, rtol=common.RTOL, clock_scope=common.CLOCK_SCOPE,
                    metadata=parent["metadata"],
                    scope="linear AZ and A-transpose-D, two resident launches with only D updated; no projection, loss or optimizer")
    common.write_json(destination/"case.json", manifest)
    common.load(destination)
    expected, _ = common.independent_expected(destination, manifest)
    reference = common.load_npz(destination/"reference.npz", common.RAW_LIMIT)
    for name, value in expected.items():
        common.exact(reference[name], value, "export independent reference: "+name)
    return manifest


def export(cases, output, request_id, *, host_timeout=600):
    common.require(isinstance(cases, dict) and 1 <= len(cases) <= 8, "one to eight named cases")
    common.require(re.fullmatch("[a-z][a-z0-9_-]{0,95}", request_id), "request ID")
    common.require(type(host_timeout) is int and 1 <= host_timeout <= 1800, "bounded worker timeout")
    output = Path(output).absolute()
    common.require(output == output.resolve(), "export symlink")
    output.mkdir(parents=True, exist_ok=False)
    for name in (*common.HOST_FILES, "export_appliance.py", "README.md", "source-origin.json"):
        shutil.copyfile(HERE/name, output/name)
    (output/"tests").mkdir()
    shutil.copyfile(HERE/"test_appliance.py", output/"tests/test_appliance.py")
    package = dict(schema=1, family="bounded-adjoint", cases={}, compile_workers=1,
                   host_sha256={name: common.sha(output/name) for name in common.HOST_FILES},
                   exporter_sha256=common.sha(output/"export_appliance.py"),
                   scope="two resident forward/adjoint launches; independent original-occurrence audit",
                   evidence="package preparation only; no hardware execution claim")
    steps = []
    driver = original_driver()
    for name, path in cases.items():
        common.require(re.fullmatch("[a-z][a-z0-9_-]{0,63}", name), "case name")
        destination = output/"cases"/name
        convert(path, destination, driver)
        package["cases"][name] = common.sha(destination/"case.json")
        script = "validations/bounded-adjoint/appliance.py"
        steps.extend([
            dict(name="compile-"+name, kind="appliance", script=script, timeout_seconds=1200,
                 args=["compile", "--case", name, "--backend", "hardware", "--namespace", "job-operator",
                       "--compile-workers", "1", "--out", "{output}/compile-"+name]),
            dict(name="run-"+name, kind="appliance", script=script, timeout_seconds=min(3300, host_timeout+600),
                 args=["run", "--case", name, "--backend", "hardware", "--artifact-record",
                       "{output}/compile-"+name+"/native-build.json", "--host-timeout", str(host_timeout), "--out", "{output}/run-"+name]),
            dict(name="audit-"+name, kind="host", script="validations/bounded-adjoint/saved_audit.py", timeout_seconds=120,
                 args=["{output}/run-"+name+"/capture-data", "--out", "{output}/numerical-"+name+".json"])])
    common.write_json(output/"package.json", package)
    common.write_json(output/"validation.json", dict(schema=1, id="bounded-adjoint",
                      description="Bounded linear forward/adjoint with a D-only second resident launch and independent audit.",
                      entrypoints=["appliance.py", "saved_audit.py"], tests=["tests"]))
    common.write_json(output/"request-template.json", dict(schema=1, id=request_id,
                      description="Each case compiles, captures two resident launches and passes a separate numerical audit before continuing.", steps=steps))
    return package


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
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
    export(cases, args.output, args.request_id, host_timeout=args.host_timeout)


if __name__ == "__main__":
    main()
