"""Prepare, admit, locally compile and simulate complete native-feature GCN inference."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import numpy as np
try:
    from . import saved_audit as common
except ImportError:
    import saved_audit as common
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUNS = ROOT/"research/runs/E217"
SCOPE = ("One complete sparse-input GCN forward pass: Ahat ReLU(Ahat X W0) W1, H16/O7. "
         "Cora uses every native1433 feature and the retained unpruned seed11 final200epoch checkpoint. "
         "Full2708 cases report custom500-node validation; induced subsets report no accuracy. "
         "Inference with host-trained weights; no resident training or global makespan claim.")


def dedicated(path, fresh=False):
    path = Path(path).absolute()
    roots = (RUNS, ROOT/"research/runs/E224", ROOT/"research/runs/E225", ROOT/"research/runs/E226", ROOT/"research/runs/E229")
    common.require(path == path.resolve() and any(path != root and path.is_relative_to(root) for root in roots), "dedicated E217/E224/E225/E226/E229 descendant")
    if fresh:
        path.mkdir(parents=True, exist_ok=False)
    return path


def host_sources():
    return {name: common.sha(HERE/name) for name in common.HOST_FILES}


def prepare(case, kind="tiny", nodes=2708, columns=2, rows=2, tile=4, code_reserve_bytes=18*1024,
            transport="full", tiny_demand=False):
    case = dedicated(case, fresh=True)
    common.require(kind in ("tiny", "cora") and type(nodes) is int and 1 <= nodes <= 2708, "finite dataset and node bound")
    common.require(type(code_reserve_bytes) is int and 0 <= code_reserve_bytes <= 44*1024, "explicit code reserve")
    common.require(transport in ("full", "compact", "selective") and (not tiny_demand or (kind == "tiny" and columns == rows == 2)), "transport and demand fixture")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from models.hierarchical_gnn.grouped_csr import oracle
    from models.hierarchical_gnn.grouped_csr.native_cora_appliance import demand
    from models.hierarchical_gnn.partition import place
    if kind == "tiny":
        nodes = 7
        rng = np.random.default_rng(712)
        x = rng.normal(size=(7, 23)).astype(np.float32)
        x[rng.random(x.shape) < .6] = 0
        x[:, 17], x[:, 22] = np.float32(.75), np.float32(-.5)
        original = dict(x=x, src=np.array([0, 0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 0], np.int64),
                        dst=np.array([1, 1, 4, 0, 2, 5, 6, 4, 3, 6, 0, 6], np.int64),
                        coeff=np.array([.5, -.25, 1, .7, 1, 1, -.3, 1, .2, 1, .4, .1], np.float32),
                        w0=(rng.normal(size=(23, 16))*.1).astype(np.float32),
                        w1=(rng.normal(size=(16, 7))*.1).astype(np.float32))
        if tiny_demand:
            # Row 1 -> column 0 uses all four source slots (dense fallback).
            # Row 0 uses partial panels in both columns. Row 1 -> column 1 is
            # empty. Node 2 is local-only and PE 1 owns no destination.
            original.update(src=np.array([3, 4, 5, 6, 0, 0, 0, 1, 2], np.int64),
                            dst=np.array([0, 0, 0, 0, 3, 3, 3, 5, 0], np.int64),
                            coeff=np.array([.5, -.25, .75, 1., .5, -.125, 0., -.3, .4], np.float32))
        np.savez_compressed(case/"original.npz", **original)
        owner = np.array([0, 0, 0, columns*(rows-1), columns*(rows-1), columns*rows-1, columns*rows-1], np.int64)
    else:
        (case/"dataset").mkdir()
        for name in ("cora.content", "cora.cites"):
            shutil.copyfile(ROOT/"models/gcn/data/cora"/name, case/"dataset"/name)
        checkpoint = ROOT/"research/experiments/data/E087/captures/checkpoint-original"
        for extension in (".npz", ".json"):
            shutil.copyfile(checkpoint.with_suffix(extension), case/("checkpoint"+extension))
        original = common.native_inputs(case, dict(kind=kind, nodes=nodes))
        edge_count = (len(original["src"])-nodes)//2
        edges = np.column_stack((original["src"][:edge_count], original["dst"][:edge_count]))
        owner = place(nodes, edges, columns, rows)["owner"]
    packed = oracle.build_case(original["x"], original["src"], original["dst"], original["coeff"], owner,
                               columns, rows, [original["w0"], original["w1"]], feature_tile=tile)
    plan = demand.build_plan(packed)
    demand_report = demand.describe(packed, plan)
    emulated = oracle.emulate(packed)
    candidate = demand.execute(packed, plan)
    for name in ("pre", "logits"):
        np.testing.assert_array_equal(candidate[name].view(np.uint32), emulated[name].view(np.uint32))
    params = dict(packed["params"], COMPACT=int(transport != "full"),
                  U=max(1, demand_report["max_compact_slots"]) if transport != "full" else packed["params"]["K"])
    arrays = dict(packed["arrays"], sender_slots=plan["source_slots"], source_lengths=plan["lengths"])
    if transport != "full":
        arrays["g_src"] = plan["remote_index"]
    if transport == "selective":
        from models.hierarchical_gnn.grouped_csr.native_cora_appliance import unicast
        packets = unicast.build_plan(packed)
        params.update(UNICAST=1, MS=packets["MS"])
        arrays.update({name: packets[name] for name in ("send_ptr", "send_slots", "recv_lengths")})
        arrays["g_src"] = packets["remote_index"]
        demand_report["selective_protocol"] = packets["report"]
    common.validate_params(params)
    np.savez_compressed(case/"inputs.npz", **arrays)
    np.savez_compressed(case/"node_ids.npz", nodes=packed["node_ids"])
    stub = dict(kind=kind, nodes=nodes, params=params)
    expected, context = common.independent_expected(case, stub, arrays)
    errors = {}
    for name in ("pre", "logits"):
        want = packed["reference"][name]
        np.testing.assert_allclose(emulated[name], want, atol=common.ATOL, rtol=common.RTOL)
        restored = np.empty_like(want)
        active = packed["node_ids"] >= 0
        restored[packed["node_ids"][active]] = expected[name].reshape(*packed["node_ids"].shape, want.shape[1])[active]
        np.testing.assert_allclose(restored, want, atol=2e-12, rtol=2e-12)
        errors[name] = float(np.max(np.abs(emulated[name]-want)))
    witness = None
    if kind == "tiny":
        limited = original["x"].copy()
        limited[:, 16:] = 0
        alternative = oracle.reference(limited, original["src"], original["dst"], original["coeff"], [original["w0"], original["w1"]])
        witness = float(np.max(np.abs(alternative["logits"]-packed["reference"]["logits"])))
        common.require(witness > 1e-3, "full projection differs from a first16-feature fallback")
    np.savez_compressed(case/"expected.npz", **expected)
    (case/"source").mkdir()
    for name in ("layout.csl", "pe.csl"):
        shutil.copyfile((HERE/"selective" if transport == "selective" else HERE)/name, case/"source"/name)
    files = set(common.CASE_REQUIRED) | ({"original.npz"} if kind == "tiny" else set(common.PINS) | {"checkpoint.json"})
    sources = [HERE.parent/"oracle.py", HERE.parent/"column_demand.py", HERE.parent/"pe.csl", HERE.parent/"layout.csl", HERE.parent.parent/"partition.py",
               ROOT/"csl-extras-202505230211-4-d9070058/examples/benchmarks/gemv-collectives_2d/pe.csl",
               ROOT/"sdk-2.10.2/csl-extras-202604101435-6-d2f7d96e/examples/benchmarks/gemv-collectives_2d/pe.csl", HERE/"demand.py", HERE/"unicast.py"]
    common.write_json(case/"provenance.json", dict(origin="E087", source_sha256={str(path.relative_to(ROOT)): common.sha(path) for path in sources},
                      adapter_source_sha256=host_sources(), preparer_sha256=common.sha(HERE/"run.py"),
                      source_change=("finite source-group MP inputs plus ACK/RELEASE; unchanged source-group/occurrence arithmetic" if transport == "selective" else "matched full/compact column panels; unchanged source-group/occurrence arithmetic; immutable demand directories in both arms"),
                      host_max_abs_error=errors, beyond16_feature_logit_difference=witness))
    tiles = (16+tile-1)//tile+(7+tile-1)//tile
    admission = dict(named_bytes=common.named_bytes(params), code_reserve_bytes=code_reserve_bytes,
                     forecast_bytes=common.named_bytes(params)+code_reserve_bytes,
                     admitted=common.named_bytes(params)+code_reserve_bytes < 44*1024,
                     scope="uniform arrays plus eight scalar bytes and explicit code/SDK reserve; actual ELF gate required, stack unmeasured")
    manifest = dict(schema=1, family="native-cora", kind=kind, nodes=nodes, features=original["x"].shape[1],
                    profile="adversarial" if kind == "tiny" else "original", params=params,
                    inputs=common.schema(params), outputs=common.schema(params, True), immutable_inputs=sorted(common.schema(params)),
                    launch=dict(entrypoint="f_run", arguments=[]), atol=common.ATOL, rtol=common.RTOL, admission=admission,
                    metadata=packed["metadata"], transport=transport, tiny_demand=tiny_demand, demand_model=demand_report,
                    collective_model=dict(tiles=tiles, received_callbacks_per_pe=tiles*rows,
                        row_calls_per_pe=tiles*2, column_calls_per_column=(demand_report["compact_column_broadcasts_per_column"] if transport != "full" else [tiles*rows]*columns),
                        payload_word_hops=demand_report[("compact" if transport != "full" else "baseline")+"_payload_word_hops"]),
                    scope=SCOPE, files={name: common.sha(case/name) for name in sorted(files)})
    if transport == "selective":
        manifest["collective_model"]["scope"] = "compact reference model; selected transport census is demand_model.selective_protocol"
    common.write_json(case/"case.json", manifest)
    common.load(case)
    return dict(status="PREPARED" if admission["admitted"] else "FORECAST_REJECTED", case=str(case), params=params,
                admission=admission, metadata=packed["metadata"], collective_model=manifest["collective_model"],
                evidence="independent full-feature host oracle only", host_max_abs_error=errors)


def compile_case(case):
    case = dedicated(case)
    manifest = common.load(case)
    common.require(manifest["admission"]["admitted"], "static forecast rejected")
    common.require(not (case/"build").exists(), "fresh compiled tree")
    flags = shlex.split(common.compiler_arguments(manifest["params"], "simulator", 1))
    flags[-1] = str(case/"build")
    command = [str(ROOT/"cslc"), str(case/"source/layout.csl"), *flags]
    record = dict(status="FAILED", backend="simulator", case_sha256=common.sha(case/"case.json"), host_sha256=host_sources(),
                  compiler_wrapper_sha256=common.sha(ROOT/"cslc"), runtime_wrapper_sha256=common.sha(ROOT/"cs_python"), command=command, cwd=str(ROOT))
    common.write_json(case/"compile-request.json", record)
    began = time.monotonic()
    try:
        record["returncode"] = subprocess.run(command, cwd=ROOT).returncode
        common.require(record["returncode"] == 0, "compiler failed")
        common.check_abi(case/"build", manifest)
        record["inventory"] = common.compiled_inventory(case/"build")
        common.admit_compiled_inventory(record["inventory"])
        common.require(common.sha(case/"case.json") == record["case_sha256"] and host_sources() == record["host_sha256"], "source changed during compile")
        record["status"] = "PASS"
    except BaseException as error:
        record["error"] = str(error)
        raise
    finally:
        record["wall_seconds"] = time.monotonic()-began
        common.write_json(case/"compile.json", record, limit=common.INVENTORY_JSON_LIMIT)
    return record


def compiled(case):
    case = dedicated(case)
    manifest = common.load(case)
    record = common.read_json(case/"compile.json", limit=common.INVENTORY_JSON_LIMIT)
    common.require(record.get("status") == "PASS" and record.get("returncode") == 0 and "error" not in record
                   and record.get("backend") == "simulator", "successful local compile")
    common.require(record.get("case_sha256") == common.sha(case/"case.json") and record.get("host_sha256") == host_sources(), "compiled source identity")
    common.require(common.compiled_inventory(case/"build") == record["inventory"], "canonical compiled inventory")
    common.admit_compiled_inventory(record["inventory"])
    common.check_abi(case/"build", manifest)
    return record


def simulation_request(case, out):
    case, out = dedicated(case), dedicated(out)
    return dict(backend="simulator", command=[str(ROOT/"cs_python"), "-B", str(HERE/"run.py"), "capture", "--case", str(case), "--out", str(out/"runtime")],
                wrapper_cwd=str(ROOT), runtime_cwd=str(out/"sdk-work"))


def stage_runtime_build(case, out, record):
    case, out = dedicated(case), dedicated(out)
    work = out.parent/"sdk-work"
    common.require(work.is_dir() and work == work.resolve(), "prepared private runtime directory")
    build = work/"build"
    shutil.copytree(case/"build", build)
    common.require(common.compiled_inventory(build) == record["inventory"], "private copied ELF inventory")
    common.check_abi(build, common.load(case))
    return work, build


def simulate(case, out):
    case, out = dedicated(case), dedicated(out, fresh=True)
    record = compiled(case)
    (out/"sdk-work").mkdir()
    request = dict(simulation_request(case, out), case_sha256=record["case_sha256"],
                   compile_sha256=common.sha(case/"compile.json", common.INVENTORY_JSON_LIMIT))
    common.write_json(out/"request.json", request)
    result, began = dict(status="FAILED", **request), time.monotonic()
    try:
        result["returncode"] = subprocess.run(request["command"], cwd=request["wrapper_cwd"]).returncode
        common.require(result["returncode"] == 0, "simulator capture failed")
        result.update(common.audit(case, out/"runtime"))
    except BaseException as error:
        result["error"] = str(error)
        raise
    finally:
        result["wall_seconds"] = time.monotonic()-began
        common.write_json(out/"result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "compile", "simulate", "capture", "audit"):
        child = sub.add_parser(action)
        child.add_argument("--case", type=Path, required=True)
        if action == "prepare":
            child.add_argument("--kind", choices=("tiny", "cora"), default="tiny")
            child.add_argument("--nodes", type=int, default=2708)
            child.add_argument("--columns", type=int, default=2)
            child.add_argument("--rows", type=int, default=2)
            child.add_argument("--tile", type=int, choices=(4, 8, 16), default=4)
            child.add_argument("--code-reserve-bytes", type=int, default=18*1024)
            child.add_argument("--transport", choices=("full", "compact", "selective"), default="full")
            child.add_argument("--tiny-demand", action="store_true")
        elif action in ("simulate", "capture", "audit"):
            child.add_argument("--out", type=Path, required=True)
            if action == "audit":
                child.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        result = prepare(args.case, args.kind, args.nodes, args.columns, args.rows, args.tile, args.code_reserve_bytes,
                         args.transport, args.tiny_demand)
    elif args.action == "compile":
        result = compile_case(args.case)
    elif args.action == "simulate":
        result = simulate(args.case, args.out)
    elif args.action == "capture":
        try:
            from . import sdk_runner
        except ImportError:
            import sdk_runner
        case, out = dedicated(args.case), dedicated(args.out)
        record = compiled(case)
        work, build = stage_runtime_build(case, out, record)
        os.chdir(work)
        sdk_runner.main(case, build, out, "simulator")
        result = dict(status="COMPLETED", numerical_status="NOT_RUN")
    else:
        common.require(not args.out.exists(), "fresh audit report")
        result = common.audit(args.case, args.runtime)
        common.write_json(args.out, result)
    print(__import__("json").dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
