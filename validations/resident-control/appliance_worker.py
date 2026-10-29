"""One finite resident-control capture; structural execution admission only."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import sys
import time

try:
    from . import saved_audit as common
except ImportError:
    import saved_audit as common

ROOT = Path(__file__).resolve().parent


def capture(args, root=ROOT, runner=None):
    root = Path(root).absolute()
    build = common.read_json(args.manifest)
    request = common.read_json(args.request)
    common.require(args.backend in ("hardware", "simulator") and build.get("schema") == request.get("schema") == 1
                   and build.get("action") == "compile" and build.get("backend") == args.backend, "worker compile/backend contract")
    common.require(request.get("case_name") == build.get("case_name")
                   and request.get("manifest_sha256") == common.sha(args.manifest), "worker request binding")
    common.arm(request.get("arm"))
    common.compile_binding(build)
    common.require(set(build.get("host_sha256", {})) == set(common.HOST_FILES), "worker host inventory")
    for name, digest in build["host_sha256"].items():
        common.require(common.sha(root / name) == digest, "staged host identity")
    common.require(common.sha(args.case_archive) == build.get("case_archive_sha256"), "staged case archive")
    out = Path(args.out).absolute()
    common.require(out == out.resolve(), "worker output symlink")
    out.mkdir(parents=True, exist_ok=False)
    record = dict(schema=1, status="incomplete", backend=args.backend, numerical_status="NOT_RUN",
                  run_id=request["run_id"], manifest_sha256=common.sha(args.manifest),
                  request_sha256=common.sha(args.request))
    common.write_json(out / "capture.json", record)
    began = time.monotonic()
    try:
        case = common.unpack_archive(args.case_archive, out / "case", "case", common.CASE_LIMIT,
                                     common.CASE_REQUIRED | {"case.json"})
        manifest = common.load(case)
        common.require(common.sha(case / "case.json") == build["case_sha256"]
                       and manifest["params"] == build["params"] and manifest["files"] == build["case_files"], "frozen case identity")
        common.check_abi(args.build, manifest)
        (out / "host").mkdir()
        for name in common.HOST_FILES:
            shutil.copyfile(root / name, out / "host" / name)
        shutil.copyfile(args.manifest, out / "manifest.json")
        shutil.copyfile(args.request, out / "request.json")
        (out / "build/bin").mkdir(parents=True)
        for name in ("out.json", "bin/out_rpc.json"):
            shutil.copyfile(Path(args.build) / name, out / "build" / name)
        inventory = common.compiled_inventory(args.build)
        common.write_json(out / "build/inventory.json", inventory, limit=common.INVENTORY_JSON_LIMIT)
        record["compiled_metadata_sha256"] = {
            name: common.sha(out / "build" / name) for name in ("out.json", "bin/out_rpc.json", "inventory.json")}
        common.admit_compiled_inventory(inventory)
        # Runtime cwd stays outside capture: SDK-generated configuration files
        # and simulator traces must not enter the bounded exported archive.
        if runner is None:
            spec = importlib.util.spec_from_file_location("native_frozen_runner", root / "sdk_runner.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            runner = module.main
        runner(case, Path(args.build).absolute(), out / "runtime", args.backend, args.cmaddr, request["arm"])
        admitted = common.admit_execution(case, out / "runtime")
        common.require(admitted["backend"] == args.backend and admitted["arm"] == request["arm"], "captured backend")
        common.write_json(out / "execution-admission.json", admitted)
        record["status"] = "COMPLETED"
    except BaseException as error:
        record["error"] = repr(error)
        raise
    finally:
        record["seconds"] = time.monotonic() - began
        common.write_json(out / "capture.json", record)
    common.admit_capture(out)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--backend", choices=("hardware", "simulator"), required=True)
    parser.add_argument("--cmaddr")
    args = parser.parse_args(argv)
    try:
        return capture(args)
    except Exception as error:
        print("error: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
