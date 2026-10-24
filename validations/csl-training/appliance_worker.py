"""Single staged CSL training capture with frozen expected outputs; no autograd."""
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
    manifest = common.read_json(args.manifest)
    request = common.read_json(args.request)
    common.require(manifest.get("schema") == request.get("schema") == 1 and
                   manifest.get("action") == "compile" and manifest.get("backend") == args.backend,
                   "worker compile/backend contract")
    common.require(request.get("case_name") == manifest.get("case_name") and
                   request.get("manifest_sha256") == common.sha(args.manifest), "worker request binding")
    common.require(set(manifest["host_sha256"]) == set(common.HOST_FILES), "worker host inventory")
    for name, expected in manifest["host_sha256"].items():
        common.require(common.sha(root/name) == expected, "staged host identity: "+name)
    common.require(common.sha(args.case_archive) == manifest["case_archive_sha256"], "staged case archive identity")
    out = Path(args.out).absolute()
    common.require(out == out.resolve(), "worker output symlink")
    out.mkdir(parents=True, exist_ok=False)
    record = dict(schema=1, status="incomplete", backend=args.backend, run_id=request["run_id"],
                  manifest_sha256=common.sha(args.manifest), request_sha256=common.sha(args.request))
    common.write_json(out/"capture.json", record)
    started = time.monotonic()
    try:
        case = common.unpack_archive(args.case_archive, out/"case", "case", common.CASE_LIMIT,
                                     common.CASE_FILES | {"case.json"})
        case_manifest = common.load(case)
        common.require(common.sha(case/"case.json") == manifest["case_sha256"] and
                       case_manifest["params"] == manifest["params"], "staged frozen case")
        common.check_abi(args.build, case_manifest)
        (out/"host").mkdir()
        for name in common.HOST_FILES:
            shutil.copyfile(root/name, out/"host"/name)
        shutil.copyfile(args.manifest, out/"manifest.json")
        shutil.copyfile(args.request, out/"request.json")
        (out/"build/bin").mkdir(parents=True)
        record["compiled_metadata_sha256"] = {}
        for name in ("out.json", "bin/out_rpc.json"):
            shutil.copyfile(Path(args.build)/name, out/"build"/name)
            record["compiled_metadata_sha256"][name] = common.sha(out/"build"/name)
        if runner is None:
            spec = importlib.util.spec_from_file_location("frozen_training_runner", case/"sdk_runner.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            runner = module.main
        runner(case, Path(args.build).absolute(), out/"runtime", args.backend, args.cmaddr)
        result = common.audit(case, out/"runtime")
        common.require(result["backend"] == args.backend, "runtime backend")
        common.write_json(out/"audit.json", result)
        record["status"] = "PASS"
    except BaseException as error:
        record["error"] = repr(error)
        raise
    finally:
        record["seconds"] = time.monotonic()-started
        common.write_json(out/"capture.json", record)
    common.audit_capture(out)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--case-archive", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--request", type=Path, required=True)
    ap.add_argument("--backend", choices=("hardware", "simulator"), required=True)
    ap.add_argument("--cmaddr")
    args = ap.parse_args(argv)
    try:
        return capture(args)
    except Exception as error:
        print("error: "+str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
