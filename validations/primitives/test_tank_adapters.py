"""SDK-free protocol, failure cleanup, and bitwise-oracle mutation tests.

The compact fixtures below test the adapter contract, not kernel mathematics.
The exporter separately regenerates real capsules from frozen original oracles.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import zipfile

import numpy as np

import export_tank_cases as exporter
import tank_audit as audit
import tank_worker as worker


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def fixture(root, kind="topk"):
    package, run, build = root / "package", root / "run", root / "build"
    name = kind + "-test"
    directory = package / "cases" / name
    directory.mkdir(parents=True)
    run.mkdir()
    (build / "bin").mkdir(parents=True)
    rectangle = [0, 0, 2, 2]
    operations, inputs, expected = [], {}, {}
    if kind == "topk":
        symbols = {"database": [1, "float32"], "query_seed": [1, "float32"], "best": [2, "uint32"]}
        functions = {"f_load": [], "f_query": [dict(name="generation", type="i16")]}
        params = dict(Pw=2, Ph=2, CAP=1, D=1, K=1)
        # Exact NaN payload, negative zero, infinity, and a finite value.
        inputs["setup"] = np.array([[0x7fc01234], [0x80000000], [0x7f800000], [0x3f800000]], np.uint32).view(np.float32)
        inputs["seed"] = np.array([[2]], np.float32)
        operations += [dict(op="h2d", symbol="database", array="setup", rect=rectangle),
                       dict(op="h2d", symbol="query_seed", array="seed", rect=[0, 0, 1, 1]),
                       dict(op="launch", symbol="f_load", args=[]),
                       dict(op="d2h", symbol="database", array="before", rect=rectangle)]
        expected["before"] = inputs["setup"].copy()
        for generation in (1, 2, 3):
            key = "q%d_best" % generation
            operations += [dict(op="launch", symbol="f_query", args=[generation]),
                           dict(op="d2h", symbol="best", array=key, rect=rectangle)]
            expected[key] = np.tile(np.array([[0x3f800000, 17 if generation != 2 else 18]], np.uint32), (4, 1))
        restore = [["q1_best", "q3_best"]]
    else:
        symbols = {"y": [2, "float32"], "Q": [2, "float32"], "count": [1, "uint32"], "clocks": [6, "uint16"]}
        functions = {"apply_epochs": [dict(name="length", type="i16"), dict(name="input_phase", type="i16")]}
        params = dict(width=2, height=2)
        for index, args in enumerate(([17, 0], [32, 1], [32, 0])):
            operations.append(dict(op="launch", symbol="apply_epochs", args=args))
            for symbol in ("y", "Q", "count", "clocks"):
                key = "s%d_%s" % (index, symbol)
                operation = dict(op="d2h", symbol=symbol, array=key, rect=rectangle)
                if symbol == "clocks":
                    operation["clock"] = True
                else:
                    expected[key] = np.full((4, symbols[symbol][0]), 64 if symbol == "count" else index + 1,
                                            np.uint32 if symbol == "count" else np.float32)
                operations.append(operation)
        restore = []
    case = dict(schema=1, kind=kind, grid=[2, 2], params=params, symbols=symbols, functions=functions,
                operations=operations, validations=dict(restore=restore), census=dict(epochs_per_launch=64))
    np.savez(directory / "inputs.npz", **inputs)
    np.savez(directory / "expected.npz", **expected)
    source_names = ["layout.csl", "pe.csl"] + (["vendor_layout.csl", "vendor_pe.csl"] if kind == "halo" else [])
    origins = {}
    for source in source_names:
        payload = ("// isolated adapter fixture " + source + "\n").encode()
        original_name = "fixture/" + kind + "/" + source
        write(directory / "source" / source, payload)
        write(package / "provenance/originals" / original_name, payload)
        origins["source/" + source] = dict(original_path=original_name, original_sha256=worker.sha(payload),
                                           exported_sha256=worker.sha(payload), transformation="none")
    oracle_name = "oracle/fixture.py"
    write(package / oracle_name, b"# Frozen adapter-only fixture.\n")
    write(package / "provenance/export_tank_cases.py", b"# Frozen fixture recipe.\n")
    oracle_sources = {oracle_name: audit.digest(package / oracle_name)}
    case.update(files=exporter.inventory(directory), source_origin=origins, oracle_sources=oracle_sources)
    worker.write_json(directory / "case.json", case)
    # Read canonical serialized order, as the worker and auditor do.
    case = audit.read_json(directory / "case.json")
    entry = dict(case_sha256=audit.digest(directory / "case.json"), params=params, grid=[2, 2], kind=kind, family=kind)
    manifest = dict(schema=1, files=exporter.inventory(package), cases={name: entry}, oracle_sources=oracle_sources,
                    source_commit="1" * 40, exporter_sha256=audit.digest(package / "provenance/export_tank_cases.py"))
    worker.write_json(package / "manifest.json", manifest)
    types = {"float32": "[*]f32", "uint32": "[*]u32", "uint16": "[*]u16"}
    ids = {symbol: index + 10 for index, symbol in enumerate(case["symbols"])}
    rpc = [dict(name=symbol, kind="Var", type=types[spec[1]], id=ids[symbol]) for symbol, spec in case["symbols"].items()]
    rpc += [dict(name=symbol, kind="Func", type="void", inputs=args) for symbol, args in case["functions"].items()]
    worker.write_json(build / "out.json", dict(params={key: str(value) for key, value in params.items()}))
    worker.write_json(build / "bin/out_rpc.json", dict(rpc_symbols=rpc))
    request = dict(schema=1, run_id=str(uuid.uuid4()), backend="hardware", case_name=name,
                   case_sha256=entry["case_sha256"], package_manifest_sha256=audit.digest(package / "manifest.json"),
                   artifact_sha256="2" * 64, worker_sha256=audit.digest(Path(worker.__file__)))
    worker.write_json(run / "request.json", request)
    args = SimpleNamespace(case=directory, name=build, out=run / "capture", request=run / "request.json", backend="hardware", cmaddr="fixture-address")
    return SimpleNamespace(package=package, run=run, build=build, name=name, case=case, inputs=inputs, expected=expected,
                           ids=ids, args=args, request=request)


def runtime_factory(f, *, launch_error=False):
    downloads = iter(op for op in f.case["operations"] if op["op"] == "d2h")
    instances = []

    class Runtime:
        def __init__(self, name, **options):
            self.stops, self.launches, self.uploads, self.downloads = 0, [], [], []
            instances.append(self)

        def get_id(self, name):
            return f.ids[name]

        def load(self):
            pass

        def run(self):
            pass

        def launch(self, name, *args, **options):
            self.launches.append((name, args, options))
            if launch_error:
                raise RuntimeError("fixture launch failure")

        def memcpy_h2d(self, symbol, value, *rectangle, **options):
            self.uploads.append((symbol, value.copy(), rectangle, options))

        def memcpy_d2h(self, destination, symbol, *rectangle, **options):
            operation = next(downloads)
            self.downloads.append((symbol, rectangle, options))
            if operation.get("clock"):
                # One wrap of the 48-bit counter: (12 - (2^48-8)) mod 2^48 = 20.
                value = np.tile(np.array([[65528, 65535, 65535, 12, 0, 0]], np.uint32), (4, 1))
            else:
                value = f.expected[operation["array"]]
            destination[:] = value.ravel()

        def stop(self):
            self.stops += 1

    return lambda: (Runtime, SimpleNamespace(MEMCPY_16BIT="u16", MEMCPY_32BIT="u32"), SimpleNamespace(ROW_MAJOR="row")), instances


def add_sdk_compatibility_fixture(f):
    """Small source-only migration fixtures; not compilable kernel fixtures."""
    layout = b"// Preserve routing here.\nfn get_params(px:i16, py:i16) comptime_struct {\n    return .{\n    };\n}\n"
    pe = b"// Preserve the queue and compute implementation.\n"
    for direction in ("west", "east", "south", "north"):
        pe += ("   .fabric_color = c_recv_" + direction + ",\n").encode()
    for direction in ("east", "west", "north", "south"):
        pe += ("    .fabric_color = c_send_" + direction + ",\n").encode()
    pe += b"    @initialize_queue(iq_recv_west, .{.color = c_recv_west});\n"
    case = audit.read_json(f.args.case / "case.json")
    for relative, original_name, payload in (("source/vendor_layout.csl", exporter.VENDOR + "/layout.csl", layout),
                                            ("source/vendor_pe.csl", exporter.VENDOR + "/wse3/pe.csl", pe)):
        transformation, reference_name = exporter.SDK_TRANSFORMS[original_name]
        transformed = exporter.stencil_compatibility(original_name, payload)
        write(f.package / "provenance/originals" / original_name, payload)
        write(f.package / "provenance/originals" / reference_name, transformed)
        write(f.args.case / relative, transformed)
        case["files"][relative] = worker.sha(transformed)
        case["source_origin"][relative] = dict(original_path=original_name, original_sha256=worker.sha(payload),
            exported_sha256=worker.sha(transformed), transformation=transformation,
            compatibility_reference=dict(path=reference_name, sha256=worker.sha(transformed)))
    worker.write_json(f.args.case / "case.json", case)
    manifest = audit.read_json(f.package / "manifest.json")
    manifest["files"] = {name: value for name, value in exporter.inventory(f.package).items() if name != "manifest.json"}
    manifest["cases"][f.name]["case_sha256"] = audit.digest(f.args.case / "case.json")
    worker.write_json(f.package / "manifest.json", manifest)
    return case


def add_wrapper_forwarding_fixture(f):
    original_name = exporter.HALO + "/pe.csl"
    original = ("param X: i16;\nparam Y: i16;\n"
        'const stencil = @import_module("../../../' + exporter.VENDOR + '/wse3/pe.csl", @concat_structs(stencil_params, .{\n'
        "  .f_callback=completed,\n"
        "  .input_queues=[4]u16{4,5,6,7}, .output_queues=[4]u16{4,5,6,7},\n"
        "  .output_ut_id=3, .BLOCK_SIZE=16,\n"
        "  .dest_dsr_ids=[2]u16{2,3}, .src0_dsr_ids=[1]u16{2}, .src1_dsr_ids=[2]u16{2,3}\n"
        "}));\n\nvar x = @zeros([32]f32);\n"
        "fn completed() void { count[0] += 1; }\n").encode()
    old, new = audit.LOCAL_IMPORTS[original_name]
    localized = original.replace(('"' + old + '"').encode(), ('"' + new + '"').encode())
    transformed = exporter.forward_stencil_params(localized)
    reference_name = exporter.ACTIVE_VENDOR + "/pe.csl"
    reference = b"// Adapter fixture for the pinned official explicit-field import pattern.\n"
    write(f.package / "provenance/originals" / original_name, original)
    write(f.package / "provenance/originals" / reference_name, reference)
    write(f.args.case / "source/pe.csl", transformed)
    case = audit.read_json(f.args.case / "case.json")
    case["files"]["source/pe.csl"] = worker.sha(transformed)
    case["source_origin"]["source/pe.csl"] = dict(original_path=original_name, original_sha256=worker.sha(original),
        exported_sha256=worker.sha(transformed), transformation="localize-stencil-import-and-forward-params",
        pattern_reference=dict(path=reference_name, sha256=worker.sha(reference), scope="Adapter-only fixture"))
    worker.write_json(f.args.case / "case.json", case)
    manifest = audit.read_json(f.package / "manifest.json")
    manifest["files"] = {name: value for name, value in exporter.inventory(f.package).items() if name != "manifest.json"}
    manifest["cases"][f.name]["case_sha256"] = audit.digest(f.args.case / "case.json")
    worker.write_json(f.package / "manifest.json", manifest)
    return original, transformed


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def capture(self, kind="topk", **kwargs):
        f = fixture(self.root, kind)
        factory, instances = runtime_factory(f, **kwargs)
        with redirect_stdout(io.StringIO()):
            result = worker.capture(f.args, sdk_factory=factory)
        return f, result, instances

    def replay(self, f):
        return audit.audit_capture(f.package, f.name, f.args.out, f.request)

    def mutate_raw(self, f, change):
        raw = {name: value.copy() for name, value in worker.arrays(f.args.out / "raw.npz").items()}
        change(raw)
        np.savez(f.args.out / "raw.npz", **raw)
        execution = audit.read_json(f.args.out / "execution.json")
        execution["raw_sha256"] = audit.digest(f.args.out / "raw.npz")
        worker.write_json(f.args.out / "execution.json", execution)

    def test_topk_complete_capture_typed_rpc_root_upload_and_exact_special_bits(self):
        f, result, instances = self.capture()
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(instances[0].stops, 1)
        self.assertEqual(instances[0].launches[0], ("f_load", (), {"nonblock": False}))
        self.assertTrue(all(type(values[0]) is np.int16 for _, values, _ in instances[0].launches[1:]))
        self.assertEqual(instances[0].uploads[1][2], (0, 0, 1, 1, 1))
        report = self.replay(f)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["clocks"], [])
        self.assertEqual(report["numerical_arrays"], len(f.expected))

    def test_halo_full_state_u16_carriers_and_local_wrap_interval(self):
        f, result, instances = self.capture("halo")
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(instances[0].uploads, [])
        self.assertEqual(sum(item[2]["data_type"] == "u16" for item in instances[0].downloads), 3)
        report = self.replay(f)
        self.assertEqual(len(report["clocks"]), 3)
        self.assertEqual(report["clocks"][0]["local_cycles"], [20] * 4)
        self.assertEqual(report["clocks"][0]["max_local_cycles_per_epoch"], 20 / 64)

    def test_nan_payload_mutation_is_rejected_even_with_updated_raw_hash(self):
        f, _, _ = self.capture()
        self.mutate_raw(f, lambda raw: raw["before"].view(np.uint32).__setitem__((0, 0), 0x7fc01235))
        with self.assertRaisesRegex(ValueError, "exact array bytes"):
            self.replay(f)

    def test_missing_full_array_is_rejected(self):
        f, _, _ = self.capture()
        self.mutate_raw(f, lambda raw: raw.pop("q2_best"))
        with self.assertRaisesRegex(ValueError, "complete input/expected/raw"):
            self.replay(f)

    def test_clock_upper_carrier_bits_are_rejected(self):
        f, _, _ = self.capture("halo")
        self.mutate_raw(f, lambda raw: raw["s0_clocks"].__setitem__((0, 0), 65536))
        with self.assertRaisesRegex(ValueError, "u16 timestamp carrier"):
            self.replay(f)

    def test_changed_rpc_argument_type_rejected_before_runtime(self):
        f = fixture(self.root)
        rpc = audit.read_json(f.build / "bin/out_rpc.json")
        next(item for item in rpc["rpc_symbols"] if item["name"] == "f_query")["inputs"][0]["type"] = "u16"
        worker.write_json(f.build / "bin/out_rpc.json", rpc)
        factory, instances = runtime_factory(f)
        with redirect_stdout(io.StringIO()):
            result = worker.capture(f.args, sdk_factory=factory)
        self.assertEqual(instances, [])
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("compiled RPC argument ABI", result["error"])

    def test_failed_launch_stops_and_cannot_audit_as_success(self):
        f, result, instances = self.capture(launch_error=True)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(instances[0].stops, 1)
        self.assertTrue((f.args.out / "raw.npz").is_file())
        with self.assertRaisesRegex(ValueError, "closed successful capture"):
            self.replay(f)

    def test_construct_end_log_failure_still_stops_owned_runtime(self):
        f = fixture(self.root)
        factory, instances = runtime_factory(f)

        def output(message, **kwargs):
            event = json.loads(message)
            if event["api"] == "construct" and event["event"] == "end":
                raise BrokenPipeError("fixture end log")

        with patch("builtins.print", side_effect=output):
            result = worker.capture(f.args, sdk_factory=factory)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(instances[0].stops, 1)

    def test_stop_begin_log_failure_does_not_skip_stop(self):
        f = fixture(self.root)
        factory, instances = runtime_factory(f)

        def output(message, **kwargs):
            event = json.loads(message)
            if event["api"] == "stop" and event["event"] == "begin":
                raise BrokenPipeError("fixture stop log")

        with patch("builtins.print", side_effect=output):
            result = worker.capture(f.args, sdk_factory=factory)
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("stop_error", result)
        self.assertEqual(instances[0].stops, 1)

    def test_relabeling_csl_change_as_unchanged_does_not_pass_provenance(self):
        f = fixture(self.root)
        directory = f.args.case
        (directory / "source/pe.csl").write_bytes(b"// undeclared implementation change\n")
        case = audit.read_json(directory / "case.json")
        changed = audit.digest(directory / "source/pe.csl")
        case["files"]["source/pe.csl"] = changed
        case["source_origin"]["source/pe.csl"]["exported_sha256"] = changed
        worker.write_json(directory / "case.json", case)
        manifest = audit.read_json(f.package / "manifest.json")
        prefix = "cases/" + f.name + "/"
        manifest["files"][prefix + "source/pe.csl"] = changed
        manifest["files"][prefix + "case.json"] = audit.digest(directory / "case.json")
        manifest["cases"][f.name]["case_sha256"] = manifest["files"][prefix + "case.json"]
        worker.write_json(f.package / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "only the declared CSL transformation"):
            audit.verify_package(f.package)

    def test_npz_huge_shape_is_rejected_before_array_allocation(self):
        data = io.BytesIO()
        np.lib.format.write_array_header_1_0(data, dict(descr="<u4", fortran_order=False, shape=(1 << 35, 1)))
        path = self.root / "hostile.npz"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("values.npy", data.getvalue())
        with self.assertRaisesRegex(ValueError, "NPY schema"):
            worker.arrays(path)

    def test_export_recipes_have_two_independent_families_and_explicit_adaptations(self):
        self.assertEqual(list(exporter.CASES)[:8], ["topk-small", "topk24", "topk32", "topk64", "halo-small", "halo16", "halo32", "halo64"])
        self.assertTrue(all(change is None for _, change in exporter.source_map("topk").values()))
        changes = [(original, change) for original, change in exporter.source_map("halo").values() if change is not None]
        self.assertEqual(dict(changes), audit.LOCAL_IMPORTS)
        self.assertEqual(exporter.SDK_TRANSFORMS, audit.SDK_TRANSFORMS)
        self.assertEqual(exporter.CASES["halo-small"][3], ((17, 0), (32, 1), (32, 0)))

    def test_runtime_coordinate_recipes_keep_legacy_work_and_source_bytes(self):
        pairs = (("halo-rc-small", "halo-small"), ("halo-rc-uneven", "halo-uneven"),
                 ("halo-rc16", "halo16"), ("halo-rc32", "halo32"), ("halo-rc64", "halo64"),
                 ("halo-rc128", "halo128"))
        self.assertEqual(set(exporter.RUNTIME_CASES),
                         {candidate for candidate, _ in pairs} | {"halo-rc256", "halo-rc512", "halo-rc-odd", "halo-rc512x663", "halo-rc663x512"})
        for candidate, control in pairs:
            self.assertEqual(exporter.CASES[candidate], exporter.CASES[control])
        sources = exporter.source_map("halo", runtime_coords=True)
        self.assertEqual(set(sources), set(exporter.source_map("halo")))
        self.assertTrue(all(change is None for _, change in sources.values()))
        self.assertEqual(sources["source/vendor_pe.csl"][0], exporter.ACTIVE_VENDOR + "/wse3/pe.csl")
        self.assertEqual(sources["source/pe.csl"][0], exporter.HALO_RUNTIME + "/pe.csl")
        with self.assertRaisesRegex(ValueError, "only to halo"):
            exporter.source_map("topk", runtime_coords=True)

    def test_large_halo_recipes_are_opt_in_and_preserve_defaults(self):
        wanted_defaults = (
            "topk-small", "topk24", "topk32", "topk64",
            "halo-small", "halo16", "halo32", "halo64", "halo-uneven", "halo128",
            "halo-rc-small", "halo-rc-uneven", "halo-rc16", "halo-rc32", "halo-rc64", "halo-rc128",
        )
        self.assertEqual(exporter.DEFAULT_CASES, wanted_defaults)
        self.assertEqual(set(exporter.CASES) - set(exporter.DEFAULT_CASES),
                         {"halo-rc256", "halo-rc512", "halo-rc-odd", "halo-rc512x663", "halo-rc663x512"})
        old_recipes = {
            "topk-small": ("topk", 3, 2, "uneven"),
            "topk24": ("topk", 24, 24, "balanced"),
            "topk32": ("topk", 32, 32, "balanced"),
            "topk64": ("topk", 64, 64, "balanced"),
            "halo-small": ("halo", 2, 2, ((17, 0), (32, 1), (32, 0))),
            "halo16": ("halo", 16, 16, ((32, 0),)),
            "halo32": ("halo", 32, 32, ((32, 0),)),
            "halo64": ("halo", 64, 64, ((32, 0),)),
            "halo-uneven": ("halo", 3, 2, ((17, 0), (32, 1), (32, 0))),
            "halo128": ("halo", 128, 128, ((32, 0),)),
        }
        for name, recipe in old_recipes.items():
            self.assertEqual(exporter.CASES[name], recipe)
        for side in (256, 512):
            name = "halo-rc" + str(side)
            self.assertEqual(exporter.CASES[name], ("halo", side, side, ((32, 0),)))
            self.assertIn(name, exporter.RUNTIME_CASES)
            self.assertNotIn("halo" + str(side), exporter.CASES)

    def test_export_rejects_more_than_sixteen_cases_before_reading_sources(self):
        with patch.object(exporter, "read", side_effect=AssertionError("source read before admission")):
            with self.assertRaisesRegex(ValueError, "1..16 known unique cases"):
                exporter.export(self.root / "source", self.root / "too-many-cases",
                                "1" * 40, list(exporter.CASES))

    def test_halo_128_admission_keeps_allocation_bound_and_topk_limit(self):
        f = fixture(self.root, "halo")
        case = json.loads(json.dumps(f.case))
        case["grid"] = [128, 128]
        case["params"] = dict(width=128, height=128)
        case["symbols"]["y"][0] = case["symbols"]["Q"][0] = 32
        for operation in case["operations"]:
            if "rect" in operation:
                operation["rect"] = [0, 0, 128, 128]
        _, outputs = worker.validate_manifest(case)
        self.assertEqual(outputs["s0_y"][0], (16384, 32))
        too_wide = dict(case, grid=[756, 128])
        with self.assertRaisesRegex(ValueError, "PE axes"):
            worker.validate_manifest(too_wide)
        with self.assertRaisesRegex(ValueError, "PE axes"):
            worker.validate_manifest(dict(case, kind="topk", grid=[65, 2]))
        case["symbols"]["y"][0] = 16384
        with self.assertRaisesRegex(ValueError, "allocation bound"):
            worker.validate_manifest(case)

    def test_halo_512_single_launch_fits_but_two_launches_exceed_total_bound(self):
        f = fixture(self.root, "halo")
        case = json.loads(json.dumps(f.case))
        case["grid"] = [512, 512]
        case["params"] = dict(width=512, height=512)
        case["symbols"]["y"][0] = case["symbols"]["Q"][0] = 32
        # One launch followed by y, Q, count and clock readbacks.
        case["operations"] = case["operations"][:5]
        for operation in case["operations"]:
            if "rect" in operation:
                operation["rect"] = [0, 0, 512, 512]
        _, outputs = worker.validate_manifest(case)
        sizes = [shape[0] * shape[1] * 4 for shape, _ in outputs.values()]
        self.assertEqual(max(sizes), 32 << 20)
        self.assertEqual(sum(sizes), 74448896)
        self.assertEqual(outputs["s0_y"][0], (262144, 32))
        self.assertEqual(worker.ARRAY_LIMIT, 96 << 20)
        self.assertEqual(worker.ARRAY_LIMIT - worker.JSON_LIMIT, 92 << 20)

        two_launches = json.loads(json.dumps(case))
        for operation in json.loads(json.dumps(case["operations"])):
            if "array" in operation:
                operation["array"] = operation["array"].replace("s0_", "s1_", 1)
            two_launches["operations"].append(operation)
        with self.assertRaisesRegex(ValueError, "allocation bound"):
            worker.validate_manifest(two_launches)

    def test_shape_capture_frontier_and_coordinate_guard(self):
        f = fixture(self.root, "halo")
        def shaped(width, height):
            case = json.loads(json.dumps(f.case))
            case["grid"] = [width, height]
            case["params"] = dict(width=width, height=height)
            case["symbols"]["y"][0] = case["symbols"]["Q"][0] = 32
            case["operations"] = case["operations"][:5]
            for op in case["operations"]:
                if "rect" in op: op["rect"] = [0, 0, width, height]
            return case
        for width, height in ((3,3), (513,128), (512,640), (640,512), (512,663), (663,512), (582,582)):
            with self.subTest(shape=(width,height)):
                _, outputs = worker.validate_manifest(shaped(width,height))
                self.assertEqual(sum(shape[0]*shape[1]*4 for shape,_ in outputs.values()), width*height*284)
        for width,height in ((512,664), (664,512), (583,583), (640,640)):
            with self.subTest(shape=(width,height)), self.assertRaisesRegex(ValueError, "allocation bound"):
                worker.validate_manifest(shaped(width,height))
        # Small area can still overflow the old i16 coordinate expression.
        with self.assertRaisesRegex(ValueError, "coordinate sum"):
            worker.validate_manifest(shaped(2,1051))
        worker.validate_manifest(shaped(2,1050))
        with self.assertRaisesRegex(ValueError, "grid/parameter identity"):
            worker.validate_manifest(dict(shaped(3,3), params=dict(width=3,height=2)))
        with self.assertRaisesRegex(ValueError, "PE axes"):
            worker.validate_manifest(dict(shaped(3,3), grid=[True,3]))

    def test_runtime_coordinate_metadata_preserves_operations_and_oracle(self):
        # The variant changes source placement, never numerical fixture inputs.
        class Oracle:
            @staticmethod
            def expected(height, width, length, phase):
                return {name: np.full((height, width, size), length + phase, dtype)
                        for name, size, dtype in (("y", 32, np.float32), ("Q", 32, np.float32),
                                                  ("count", 1, np.uint32))}
            vectorized_control = expected

            @staticmethod
            def verify_result(actual, expected):
                for name in expected:
                    np.testing.assert_array_equal(actual[name], expected[name])

            @staticmethod
            def metrics(height, width, length):
                return dict(work=height * width * length)

        seq = ((17, 0), (32, 1), (32, 0))
        direct, inputs, expected = exporter.halo_case(Oracle, 3, 2, seq)
        candidate, other_inputs, other_expected = exporter.halo_case(Oracle, 3, 2, seq, runtime_coords=True)
        self.assertEqual(inputs, other_inputs)
        self.assertNotEqual(direct["census"]["placement"], candidate["census"]["placement"])
        candidate["census"]["placement"] = direct["census"]["placement"]
        self.assertEqual(candidate, direct)
        for name in expected:
            np.testing.assert_array_equal(expected[name], other_expected[name])

    def test_sdk_stencil_compatibility_has_original_and_active_reference_provenance(self):
        f = fixture(self.root, "halo")
        case = add_sdk_compatibility_fixture(f)
        self.assertEqual(audit.verify_package(f.package)["cases"][f.name]["kind"], "halo")
        layout = (f.args.case / "source/vendor_layout.csl").read_bytes()
        pe = (f.args.case / "source/vendor_pe.csl").read_bytes()
        self.assertNotIn(b"comptime_struct", layout)
        self.assertIn(b"fn get_params(px:i16, py:i16) Params", layout)
        self.assertNotIn(b".fabric_color", pe)
        self.assertIn(b"@initialize_queue(iq_recv_west, .{.color = c_recv_west})", pe)
        self.assertEqual(case["source_origin"]["source/vendor_pe.csl"]["transformation"], "sdk-2.10.2-stencil-queue-dsds")

    def test_compatibility_label_and_repinned_active_reference_cannot_hide_semantic_change(self):
        f = fixture(self.root, "halo")
        case = add_sdk_compatibility_fixture(f)
        relative = "source/vendor_pe.csl"
        origin = case["source_origin"][relative]
        reference_file = "provenance/originals/" + origin["compatibility_reference"]["path"]
        changed = (f.args.case / relative).read_bytes().replace(b".color = c_recv_west", b".color = c_recv_east")
        write(f.args.case / relative, changed)
        write(f.package / reference_file, changed)
        origin["exported_sha256"] = origin["compatibility_reference"]["sha256"] = worker.sha(changed)
        case["files"][relative] = worker.sha(changed)
        worker.write_json(f.args.case / "case.json", case)
        manifest = audit.read_json(f.package / "manifest.json")
        manifest["files"] = {name: value for name, value in exporter.inventory(f.package).items() if name != "manifest.json"}
        manifest["cases"][f.name]["case_sha256"] = audit.digest(f.args.case / "case.json")
        worker.write_json(f.package / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "exactly equal active official stencil source"):
            audit.verify_package(f.package)

    def test_backport_matches_published_active_sdk_files_when_repository_is_available(self):
        repo = Path(__file__).resolve().parents[3]
        if not all((repo / name).is_file() for name in exporter.ORIGINALS):
            self.skipTest("Original SDK trees are absent from this standalone installation")
        for original_name, (_, reference_name) in exporter.SDK_TRANSFORMS.items():
            with self.subTest(source=original_name):
                original, current = (repo / original_name).read_bytes(), (repo / reference_name).read_bytes()
                self.assertEqual(exporter.stencil_compatibility(original_name, original), current)
                self.assertEqual(audit.stencil_compatibility(original_name, original), current)
                with self.assertRaises(ValueError):
                    exporter.stencil_compatibility(original_name, current)

    def test_explicit_wrapper_forwarding_preserves_coordinates_callback_and_body(self):
        f = fixture(self.root, "halo")
        original, transformed = add_wrapper_forwarding_fixture(f)
        self.assertEqual(audit.verify_package(f.package)["cases"][f.name]["kind"], "halo")
        self.assertTrue(transformed.startswith(b"param X: i16;\nparam Y: i16;\n"))
        self.assertNotIn(b"@concat_structs", transformed)
        self.assertEqual(original.split(b"\nvar x =", 1)[1], transformed.split(b"\nvar x =", 1)[1])
        self.assertEqual(original.split(b"  .f_callback=", 1)[1].split(b"\n}));", 1)[0],
                         transformed.split(b"  .f_callback=", 1)[1].split(b"\n});", 1)[0])
        for field in exporter.FORWARDED_STENCIL_PARAMS:
            self.assertEqual(transformed.count((".%s=stencil_params.%s," % (field, field)).encode()), 1)

    def test_repinned_wrong_forwarded_direction_is_rejected_by_reconstruction(self):
        f = fixture(self.root, "halo")
        _, transformed = add_wrapper_forwarding_fixture(f)
        changed = transformed.replace(b".c_recv_west=stencil_params.c_recv_west,", b".c_recv_west=stencil_params.c_recv_east,")
        write(f.args.case / "source/pe.csl", changed)
        case = audit.read_json(f.args.case / "case.json")
        case["files"]["source/pe.csl"] = worker.sha(changed)
        case["source_origin"]["source/pe.csl"]["exported_sha256"] = worker.sha(changed)
        worker.write_json(f.args.case / "case.json", case)
        manifest = audit.read_json(f.package / "manifest.json")
        manifest["files"] = {name: value for name, value in exporter.inventory(f.package).items() if name != "manifest.json"}
        manifest["cases"][f.name]["case_sha256"] = audit.digest(f.args.case / "case.json")
        worker.write_json(f.package / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "only the declared CSL transformation"):
            audit.verify_package(f.package)


if __name__ == "__main__":
    unittest.main()
