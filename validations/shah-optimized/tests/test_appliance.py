"""SDK-free failures that must not admit a mismatched or incomplete experiment."""
import importlib.util
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import appliance
import saved_audit
import compiled


def synthetic_artifact(path):
    """Match the cs_<hash>/out tree observed in returned E219 compiler archives."""
    files = {"out.json": b'{"params":{"R":4}}', "bin/out_0_0.elf": b"synthetic application ELF",
             "west/bin/out_0_0.elf": b"synthetic west ELF", "east/bin/out_0_0.elf": b"synthetic east ELF"}
    for side, mux in (("west", "demux"), ("east", "mux")):
        params = {"height": 8, "num_" + mux: 4, **{f"height_{mux}_{i}": 2 for i in range(4)}}
        files[side + "/out.json"] = json.dumps(dict(params=params)).encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo("cs_test/out/" + name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return files


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "source").mkdir()
        for name in appliance.SOURCE_FILES:
            (self.root / "source" / name).write_text("// test " + name)
        for name in (*appliance.HOST_FILES, "saved_audit.py"):
            (self.root / name).write_text("# test " + name)

    def compile_record(self):
        class Compiler:
            def __init__(self, **options):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def compile(self, source, layout, flags, output):
                self.flags = flags
                path = Path(output) / "artifact.tar.gz"
                synthetic_artifact(path)
                return str(path)
        args = SimpleNamespace(R=4, F=4, V=8, Q=4, B=4, channels=4, empty=False,
                               workload="adversary", seed=7, degree=8, backend="hardware",
                               namespace="job-operator", out=self.root / "build")
        appliance.compile_program(args, self.root, Compiler)
        return args.out / "build.json"

    def test_artifact_and_source_drift_rejected_before_sdk(self):
        path = self.compile_record()
        record, artifact = appliance.load_record(path, "hardware", self.root)
        self.assertIn("--channels=4", record["compiler_args"])
        self.assertIn("MEMCPYH2D_DATA_1_ID:0", record["compiler_args"])
        original = artifact.read_bytes()
        artifact.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            appliance.load_record(path, "hardware", self.root)
        artifact.write_bytes(original)
        (self.root / "oracle.py").write_text("# changed source")
        with self.assertRaisesRegex(ValueError, "host_sha256 changed"):
            appliance.load_record(path, "hardware", self.root)

    def test_compiled_archive_binds_mapping_and_elf_inventory(self):
        archive = self.root / "program.tar.gz"
        files = synthetic_artifact(archive)
        build = self.root / "unpacked"
        for name, data in files.items():
            path = build / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        expected = compiled.archive_inventory(archive)
        self.assertEqual(expected, compiled.tree_inventory(build))
        run_id = str(uuid.uuid4())
        request = dict(run_id=run_id, compile_record=dict(compiled_inventory=expected))
        collection = dict(run_id=run_id)
        status = dict(run_id=run_id)
        saved_audit.verify_capture_binding(request, collection, status, expected, expected["io_layout"])
        wrong = copy.deepcopy(expected)
        wrong["files"]["bin/out_0_0.elf"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "pinned compiler archive"):
            saved_audit.verify_capture_binding(request, collection, status, wrong, wrong["io_layout"])
        wrong_map = copy.deepcopy(expected["io_layout"])
        wrong_map["west"]["partitions"] = [1, 3, 2, 2]
        with self.assertRaisesRegex(ValueError, "compiled metadata"):
            saved_audit.verify_capture_binding(request, collection, status, expected, wrong_map)
        wrong_map = copy.deepcopy(expected["io_layout"])
        wrong_map["east"]["source_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "compiled metadata"):
            saved_audit.verify_capture_binding(request, collection, status, expected, wrong_map)
        with self.assertRaisesRegex(ValueError, "another run"):
            saved_audit.verify_capture_binding(request, collection, dict(run_id=str(uuid.uuid4())), expected, expected["io_layout"])
        (build / "west/bin/out_0_0.elf").write_bytes(b"changed")
        self.assertNotEqual(expected, compiled.tree_inventory(build))

    def test_returned_download_path_is_checked(self):
        requested = self.root / "capture"
        actual = self.root / "capture.tar.gz"
        actual.write_bytes(b"archive")
        self.assertEqual(appliance.downloaded_path(str(actual), requested, 128), actual)
        with self.assertRaisesRegex(ValueError, "escapes or collides"):
            appliance.downloaded_path(str(self.root / "request.json"), requested, 128)
        with self.assertRaisesRegex(ValueError, "escapes or collides"):
            appliance.downloaded_path(str(self.root.parent / "outside"), requested, 128)

    def test_launcher_failure_collects_every_artifact(self):
        path = self.compile_record()
        attempted = []
        class Launcher:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def stage(self, path):
                pass
            def run(self, command):
                raise RuntimeError("synthetic launcher failure")
            def download_artifact(self, remote, saved):
                attempted.append(remote)
                if remote == "host.log":
                    raise RuntimeError("synthetic missing host log")
                Path(saved).write_bytes(b"partial failure evidence")
        args = SimpleNamespace(artifact_record=path, backend="hardware", host_timeout=30,
                               output_mode="buffered", out=self.root / "run")
        with self.assertRaisesRegex(ValueError, "collection failed"):
            appliance.run_program(args, self.root, Launcher)
        self.assertEqual(attempted, ["host.log", "host-exit.txt", "capture"])
        self.assertTrue((args.out / "launcher-error.txt").is_file())
        self.assertEqual(appliance.read_json(args.out / "collection.json")["status"], "collection_failed")

    def test_channels_do_not_imply_router_channel_coverage(self):
        record = dict(params=dict(R=2, B=3), channels=2)
        layout = {side: dict(height=5, partitions=[2, 3], source_sha256="a" * 64)
                  for side in ("west", "east")}
        result = saved_audit.verify_io_layout(record, layout)
        self.assertEqual(result["west"]["active_channels"], [0])
        self.assertEqual(result["east"]["active_channels"], [1])
        record = dict(params=dict(R=4, B=4), channels=4)
        layout = {side: dict(height=8, partitions=[2, 2, 2, 2], source_sha256="b" * 64)
                  for side in ("west", "east")}
        result = saved_audit.verify_io_layout(record, layout)
        self.assertEqual(result["west"]["active_channels"], [0, 1])
        self.assertEqual(result["east"]["active_channels"], [2, 3])
        layout["west"]["partitions"] = [2, 2, 2, 3]
        with self.assertRaisesRegex(ValueError, "geometry"):
            saved_audit.verify_io_layout(record, layout)

    def test_numerical_audit_rejects_counter_and_schedule_corruption(self):
        for workload in ("adversary", "balanced", "paired-skew"):
            with self.subTest(workload=workload):
                self._check_numerical_audit(workload)

    def _check_numerical_audit(self, workload):
        import numpy as np
        oracle_path = ROOT / "oracle.py"
        if not oracle_path.exists():
            oracle_path = ROOT.parent / "oracle.py"  # pre-export local preparation
        spec = importlib.util.spec_from_file_location("test_shah_oracle", oracle_path)
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        params = dict(R=2, F=2, V=3, Q=2, B=3)
        record = dict(params=params, workload=workload, empty=False, seed=7, degree=8)
        if workload in ("balanced", "paired-skew"):
            case = oracle.controlled_fixture(**params, seed=record["seed"], degree=record["degree"],
                                             skew=workload == "paired-skew")
        else:
            case = oracle.fixture(**params)
        R, F, V, Q, B = [params[k] for k in ("R", "F", "V", "Q", "B")]
        stats = np.zeros((R + B, F + 1, 4), np.uint32)
        for r in range(R):
            nz = sum(1 for _, c, _ in case["edges"] if r * V <= c < (r + 1) * V)
            pairs = sum(len(stream[r]) // 2 for stream in case["streams"])
            padding = sum(np.count_nonzero(stream[r, ::2] == oracle.EMPTY) for stream in case["streams"])
            stats[r, 0] = [pairs, nz, B * Q, padding]
            stats[r, 1:] = [nz, B * Q, B, B]
        for b in range(B):
            stats[R + b, 1:] = [(B - b) * Q, (B - b - 1) * Q, Q, 1]
        raw = dict(actual=case["reference"].astype(np.float32), reference=case["reference"],
                   h=case["h"], edges=np.array(case["edges"]), stats=stats)
        sequence = [("memcpy_h2d", False, False)] + [("memcpy_h2d", True, True)] * B
        sequence += [("memcpy_d2h", True, False), ("memcpy_d2h", False, False)]
        calls = [dict(api=api, streaming=streaming, nonblock=nonblock, seconds=0.1)
                 for api, streaming, nonblock in sequence]
        error = float(np.max(np.abs(raw["actual"] - raw["reference"])))
        report = dict(calls=calls, host_stream_completion_seconds=1.0,
                      max_abs_error=error, metrics=case["metrics"])
        saved_audit.verify_numerics(record, "buffered", report, raw, oracle)
        # A numerically plausible result must still bind to the exact inputs.
        for array, index in (("h", (0, 0)), ("edges", (0, 2))):
            corrupted = copy.deepcopy(raw)
            corrupted[array][index] += 0.5
            with self.subTest(corruption=array), self.assertRaises(AssertionError):
                saved_audit.verify_numerics(record, "buffered", report, corrupted, oracle)
        raw["stats"][0, 0, 0] += 1
        with self.assertRaises(AssertionError):
            saved_audit.verify_numerics(record, "buffered", report, raw, oracle)
        raw["stats"][0, 0, 0] -= 1
        report["calls"][1], report["calls"][-2] = report["calls"][-2], report["calls"][1]
        with self.assertRaisesRegex(ValueError, "transfer sequence"):
            saved_audit.verify_numerics(record, "buffered", report, raw, oracle)
        sequence = [("memcpy_h2d", False, False)]
        for _ in range(B):
            sequence += [("memcpy_h2d", True, True), ("memcpy_d2h", True, False)]
        sequence += [("memcpy_d2h", False, False)]
        report["calls"] = [dict(api=api, streaming=streaming, nonblock=nonblock, seconds=0.1)
                           for api, streaming, nonblock in sequence]
        report["metrics"] = {**case["metrics"], "d2h_output_calls": B}
        saved_audit.verify_numerics(record, "per-chunk", report, raw, oracle)


if __name__ == "__main__":
    unittest.main()
