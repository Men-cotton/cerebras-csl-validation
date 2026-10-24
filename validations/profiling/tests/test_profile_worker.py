"""Worker contracts using a fake runtime; never import or construct the SDK."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import tarfile
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

import numpy as np

import oracle
import profile_common as common
import profile_appliance as appliance
import profile_worker as worker


HERE = Path(__file__).resolve().parents[1]
OPTIONS = dict(streaming=False, data_type="u32", order="row-major", nonblock=False)


def limbs(value):
    return [(value >> shift) & 0xFFFF for shift in (0, 16, 32)]


def completed_outputs(invocation, flag):
    """Independent synthetic completed payload/protocol/phase witnesses."""
    tag, selected_case = invocation + 1, int(invocation == 1)
    values = oracle.fixture(0, 4, invocation, 1)
    low = [0x3400 + 256 * selected_case + index for index in range(8)]
    carriers = [((index % 2) << 16) | value for index, value in enumerate(low)]
    terminal = 0x6000 + tag
    carriers.append((1 << 16) | terminal)
    values["received"][1][1:-1] = low + [terminal]
    values["scratch_mirror"] = [oracle.guarded(11), [oracle.LEFT] + carriers + [oracle.RIGHT]]
    values["stats"] = [oracle.expected_stats(0, 4, invocation, role, 1) for role in (0, 1)]
    values["clocks"] = [limbs(10) + limbs(30) + limbs(40) + limbs(50), [0] * 12]
    values.update(profile_control=[[flag], [flag]], profile_records=[[0] * 25 for _ in range(2)],
                  profile_status=[[tag, 0, 0, 0], [tag, 5 if flag else 0, 0, 0]],
                  perf_samples=[[0] * 12 for _ in range(2)])
    if flag:
        values["profile_records"][1] = [word for event in range(1, 6)
                                       for word in (tag, event, *limbs(100 + 10 * event))]
    return {name: np.asarray(value, np.uint32) for name, value in values.items()}


class WorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "staged"
        self.root.mkdir()
        (self.root / "source").mkdir()
        for name in ("profile_worker.py", "profile_common.py", "oracle.py"):
            shutil.copyfile(HERE / name, self.root / name)
        for name in ("layout.csl", "pe.csl"):
            shutil.copyfile(HERE / "source" / name, self.root / "source" / name)
            shutil.copyfile(HERE / "source" / name, self.root / name)
        self.build = self.directory / "build"
        (self.build / "bin").mkdir(parents=True)
        self.ids = {name: number for number, name in enumerate(common.shape(4), 1)}
        self.symbols = [dict(name=name, id=number, kind="Var", type="[*]u32")
                        for name, number in self.ids.items()]
        self.symbols += [dict(name=name, id=number, kind="Func", type="void", inputs=[])
                         for number, name in ((11, "prepare"), (12, "run_probe"))]
        common.write_json(self.build / "out.json", {"params": {"N": 4}})
        self.write_abi(self.symbols)
        self.manifest = dict(schema=1, backend="hardware", params={"N": 4},
                             host_sha256={name: common.digest(self.root / name)
                                          for name in common.HOST_NAMES},
                             source_sha256={name: common.digest(self.root / name)
                                            for name in common.SOURCE_NAMES})
        self.request = dict(schema=1, run_id=str(uuid.uuid4()), flag=1)
        self.args = SimpleNamespace(name=str(self.build), out=self.directory / "capture",
                                    manifest=self.directory / "manifest.json",
                                    request=self.directory / "request.json",
                                    backend="hardware", cmaddr="test-cm-address")
        self.write_requests()
        self.runtime = None
        self.fail_at = None
        self.corrupt = None
        self.constructor_calls = []
        self.runtime_directories = []
        self.config_bytes = 0
        self.factory = mock.Mock(side_effect=lambda: (self.construct, OPTIONS))

    def write_abi(self, symbols):
        common.write_json(self.build / "bin/out_rpc.json", {"rpc_symbols": symbols})

    def write_requests(self):
        common.write_json(self.args.manifest, self.manifest)
        self.request["manifest_sha256"] = common.digest(self.args.manifest)
        common.write_json(self.args.request, self.request)

    def construct(self, name, **options):
        self.constructor_calls.append((name, options))
        self.runtime_directories.append(Path.cwd())
        if self.config_bytes:
            config = Path("cs_hio_config_files/hw_chain_cm.conf")
            config.parent.mkdir()
            with config.open("wb") as stream:
                stream.truncate(self.config_bytes)
        if self.fail_at == "construct":
            raise RuntimeError("constructor failure")
        test = self

        class Runtime:
            def __init__(self):
                self.stopped = False
                self.loaded = False
                self.invocation = -1
                self.uploads = []
                self.downloads = []
                self.launches = []

            def get_id(self, symbol):
                if test.fail_at == "ids":
                    return test.ids[symbol] + 100
                return test.ids[symbol]

            def load(self):
                self.loaded = True
                if test.fail_at == "load":
                    raise RuntimeError("load failure")

            def run(self):
                if test.fail_at == "run":
                    raise RuntimeError("run failure")

            def memcpy_h2d(self, symbol, data, *geometry, **options):
                test.assertEqual(geometry[:4], (0, 0, 2, 1))
                test.assertEqual(options, OPTIONS)
                test.assertEqual(data.dtype, np.uint32)
                self.uploads.append((symbol, data.copy(), geometry))

            def launch(self, name, **options):
                test.assertEqual(options, {"nonblock": False})
                self.launches.append(name)
                if test.fail_at == "launch":
                    raise RuntimeError("launch failure")
                if name == "run_probe":
                    self.invocation += 1

            def memcpy_d2h(self, data, symbol, *geometry, **options):
                test.assertEqual(geometry[:4], (0, 0, 2, 1))
                test.assertEqual(options, OPTIONS)
                if test.fail_at == "read":
                    raise RuntimeError("read failure")
                name = next(name for name, value in test.ids.items() if value == symbol)
                expected = completed_outputs(self.invocation, test.request["flag"])[name].copy()
                if test.corrupt == name:
                    if name == "profile_records":
                        expected[1, 1] = 99  # Invalid event identity, not a valid alternative time.
                    else:
                        expected.flat[-2] ^= np.uint32(1)
                test.assertEqual(geometry[4], expected.shape[1])
                data[:] = expected.ravel()
                self.downloads.append((symbol, geometry))

            def stop(self):
                test.assertEqual(Path.cwd(), test.runtime_directories[-1])
                if test.config_bytes:
                    Path("runtime-stop.log").write_text("stopped")
                self.stopped = True
                if test.fail_at == "stop":
                    raise RuntimeError("stop failure")

        self.runtime = Runtime()
        return self.runtime

    def run_capture(self):
        return worker.capture(self.args, sdk_factory=self.factory, root=self.root)

    def assert_preflight_rejects(self):
        with self.assertRaises((ValueError, KeyError, TypeError)):
            self.run_capture()
        self.factory.assert_not_called()
        self.assertFalse(self.args.out.exists())

    def test_compiled_parameters_and_complete_typed_abi_checked_before_sdk(self):
        for value in (32, 4.5, 4.0, True, "4.0", "04"):
            with self.subTest(compiled_n=value):
                common.write_json(self.build / "out.json", {"params": {"N": value}})
                self.assert_preflight_rejects()
        common.write_json(self.build / "out.json", {"params": {"N": 4}})
        bad = []
        candidate = copy.deepcopy(self.symbols)
        candidate[0]["type"] = "[*]u16"
        bad.append(candidate)
        candidate = copy.deepcopy(self.symbols)
        candidate[1]["id"] = candidate[0]["id"]
        bad.append(candidate)
        candidate = copy.deepcopy(self.symbols)
        candidate[-1]["inputs"] = [{"type": "u32"}]
        bad.append(candidate)
        bad.append(self.symbols[:-1])
        for symbols in bad:
            with self.subTest(symbols=symbols):
                self.write_abi(symbols)
                self.assert_preflight_rejects()

    def test_host_and_csl_source_pins_checked_before_sdk(self):
        for name in ("profile_worker.py", "profile_common.py", "oracle.py", "layout.csl", "pe.csl"):
            with self.subTest(source=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b"\n# changed\n")
                try:
                    self.assert_preflight_rejects()
                finally:
                    path.write_bytes(original)

    def test_request_identity_and_flags_checked_before_sdk(self):
        original = copy.deepcopy(self.request)
        changes = ({"flag": 2}, {"flag": True}, {"run_id": "not-a-uuid"},
                   {"manifest_sha256": "0" * 64}, {"unexpected": 1})
        for change in changes:
            with self.subTest(change=change):
                common.write_json(self.args.request, dict(original, **change))
                self.assert_preflight_rejects()

    def test_backend_address_mismatches_checked_before_sdk(self):
        self.args.cmaddr = None
        self.assert_preflight_rejects()
        self.args.backend = "simulator"
        self.args.cmaddr = "unexpected-address"
        self.assert_preflight_rejects()
        self.manifest["backend"] = "simulator"
        self.write_requests()
        self.assert_preflight_rejects()

    def test_hardware_complete_arrays_api_sequence_and_three_invocations(self):
        self.assertEqual(self.run_capture(), 0)
        self.assertEqual(self.constructor_calls, [(str(self.build), {"cmaddr": "test-cm-address"})])
        self.assertTrue(self.runtime.stopped)
        self.assertEqual(self.runtime.launches, ["prepare", "run_probe"] * 3)
        self.assertEqual(len(self.runtime.uploads), 30)
        self.assertEqual(len(self.runtime.downloads), 30)
        capture = common.read_json(self.args.out / "capture.json")
        self.assertEqual(capture["status"], "pass")
        self.assertEqual([row["name"] for row in capture["api"]], common.expected_calls(4))
        self.assertEqual(len(capture["api"]), 80)
        raw = common.load_raw(self.args.out / "raw.npz", 4)
        self.assertEqual(len(raw), 60)
        for invocation in range(3):
            inputs = common.fixture(4, 1, invocation)
            outputs = completed_outputs(invocation, 1)
            for name in common.shape(4):
                np.testing.assert_array_equal(raw[f"input{invocation}_{name}"], inputs[name])
                np.testing.assert_array_equal(raw[f"output{invocation}_{name}"], outputs[name])
        self.assertEqual(common.audit_capture(self.args.out)["status"], "pass")

    def test_runtime_files_stay_outside_downloadable_capture(self):
        # The hardware failure contained this 50 MB SDK-generated file.
        self.config_bytes = 50_250_000
        previous = Path.cwd()
        self.assertEqual(self.run_capture(), 0)
        self.assertEqual(Path.cwd(), previous)
        runtime_directory = self.runtime_directories[-1]
        self.assertFalse(runtime_directory.is_relative_to(self.args.out))
        self.assertEqual((runtime_directory / "cs_hio_config_files/hw_chain_cm.conf").stat().st_size,
                         self.config_bytes)
        self.assertTrue((runtime_directory / "runtime-stop.log").is_file())
        archive = self.directory / "capture.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(self.args.out, arcname="capture")
        extracted = appliance.unpack_capture(archive, self.directory / "downloaded")
        self.assertFalse((extracted / "cs_hio_config_files").exists())
        self.assertFalse((extracted / "runtime-stop.log").exists())
        self.assertEqual(common.audit_capture(extracted)["status"], "pass")

    def test_existing_runtime_directory_is_not_reused(self):
        runtime_directory = self.args.out.with_name(self.args.out.name + "-runtime")
        runtime_directory.mkdir()
        marker = runtime_directory / "marker"
        marker.write_text("previous runtime")
        with self.assertRaises(FileExistsError):
            self.run_capture()
        self.factory.assert_not_called()
        self.assertEqual(marker.read_text(), "previous runtime")

    def test_relative_paths_remain_valid_after_runtime_cwd_change(self):
        previous = Path.cwd()
        os.chdir(self.directory)
        try:
            self.args.name = "build"
            self.args.out = Path("capture")
            self.args.manifest = Path("manifest.json")
            self.args.request = Path("request.json")
            self.assertEqual(self.run_capture(), 0)
            self.assertEqual(Path.cwd(), self.directory)
            self.assertEqual(common.audit_capture(self.args.out)["status"], "pass")
        finally:
            os.chdir(previous)

    def test_sdk_factory_failure_restores_cwd_and_preserves_capture(self):
        previous = Path.cwd()
        self.factory.side_effect = RuntimeError("SDK factory failure")
        with self.assertRaisesRegex(ValueError, "SDK factory failure"):
            self.run_capture()
        self.assertEqual(Path.cwd(), previous)
        self.assertIsNone(self.runtime)
        self.assertEqual(common.read_json(self.args.out / "capture.json")["status"], "incomplete")
        self.assertTrue((self.args.out / "raw.npz").is_file())

    def test_symlinked_runtime_directory_is_not_followed(self):
        runtime_directory = self.args.out.with_name(self.args.out.name + "-runtime")
        target = self.directory / "other-runtime"
        target.mkdir()
        runtime_directory.symlink_to(target, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            self.run_capture()
        self.factory.assert_not_called()
        self.assertEqual(list(target.iterdir()), [])

    def test_simulator_modes_have_trace_off_and_no_hardware_address(self):
        for backend, flag in (("simulator", 0), ("local-sim", 1)):
            with self.subTest(backend=backend):
                self.args.backend = self.manifest["backend"] = backend
                self.args.cmaddr = None
                self.args.out = self.directory / backend
                self.request["flag"] = flag
                self.write_requests()
                self.assertEqual(self.run_capture(), 0)
                self.assertEqual(self.constructor_calls[-1],
                                 (str(self.build), {"simfab_numthreads": 1, "suppress_simfab_trace": True}))
                self.assertTrue(self.runtime.stopped)

    def test_runtime_symbol_disagreement_stops_before_load(self):
        self.fail_at = "ids"
        with self.assertRaisesRegex(ValueError, "runtime symbol IDs"):
            self.run_capture()
        self.assertTrue(self.runtime.stopped)
        self.assertFalse(self.runtime.loaded)
        self.assertEqual(common.read_json(self.args.out / "capture.json")["status"], "incomplete")

    def test_runtime_failures_stop_and_preserve_incomplete_capture(self):
        original_cwd = Path.cwd()
        self.config_bytes = 64
        for stage in ("load", "run", "launch", "read", "stop"):
            with self.subTest(stage=stage):
                self.fail_at = stage
                self.args.out = self.directory / ("failure-" + stage)
                with self.assertRaisesRegex(ValueError, "failure"):
                    self.run_capture()
                self.assertTrue(self.runtime.stopped)
                self.assertEqual(Path.cwd(), original_cwd)
                self.assertFalse((self.args.out / "cs_hio_config_files").exists())
                capture = common.read_json(self.args.out / "capture.json")
                self.assertEqual(capture["status"], "incomplete")
                self.assertIsNotNone(capture["error"])
                self.assertTrue((self.args.out / "raw.npz").is_file())
                self.assertFalse((self.args.out / "audit.json").exists())

    def test_constructor_failure_keeps_record_without_stop_on_unowned_runtime(self):
        self.fail_at = "construct"
        self.config_bytes = 64
        original_cwd = Path.cwd()
        with self.assertRaisesRegex(ValueError, "constructor failure"):
            self.run_capture()
        self.assertIsNone(self.runtime)
        self.assertEqual(Path.cwd(), original_cwd)
        self.assertFalse((self.args.out / "cs_hio_config_files").exists())
        capture = common.read_json(self.args.out / "capture.json")
        self.assertEqual(capture["status"], "incomplete")
        self.assertEqual([row["name"] for row in capture["api"]], ["construct"])

    def test_invalid_completed_arrays_are_not_reported_as_pass(self):
        for symbol in ("received", "profile_records", "profile_status"):
            with self.subTest(symbol=symbol):
                self.corrupt = symbol
                self.args.out = self.directory / ("invalid-" + symbol)
                with self.assertRaises(ValueError):
                    self.run_capture()
                self.assertTrue(self.runtime.stopped)
                self.assertEqual(common.read_json(self.args.out / "capture.json")["status"], "invalid")
                self.assertFalse((self.args.out / "audit.json").exists())

    def test_existing_or_symlinked_capture_destination_is_preserved(self):
        self.args.out.mkdir()
        marker = self.args.out / "marker"
        marker.write_bytes(b"existing result")
        with self.assertRaises(FileExistsError):
            self.run_capture()
        self.factory.assert_not_called()
        self.assertEqual(marker.read_bytes(), b"existing result")
        alias = self.directory / "alias"
        alias.symlink_to(self.args.out, target_is_directory=True)
        self.args.out = alias
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.run_capture()
        self.factory.assert_not_called()
        self.assertEqual(marker.read_bytes(), b"existing result")


if __name__ == "__main__":
    unittest.main()
