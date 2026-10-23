"""Synthetic capture admission and comparison tests; no SDK or timing evidence."""
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import uuid
import zipfile

import numpy as np
import oracle
import profile_common as common
import profile_compare as comparison

ROOT = Path(__file__).resolve().parents[1]
MODULUS = 1 << 48


def limbs(value):
    return [(value >> shift) & 65535 for shift in (0, 16, 32)]


def events(raw, invocation, timestamps):
    raw[f"output{invocation}_profile_records"][1] = np.asarray(
        [[invocation+1, event_id, *limbs(timestamp)] for event_id, timestamp in enumerate(timestamps, 1)],
        np.uint32).reshape(25)


def completed_arrays(flag=1, elapsed=40):
    raw = {}
    for invocation in range(3):
        tag, case = invocation+1, int(invocation == 1)
        inputs = common.fixture(4, flag, invocation)
        outputs = copy.deepcopy(inputs)
        low = [0x3400+256*case+i for i in range(8)] + [0x6000+tag]
        words = [((i % 2) << 16) | value for i, value in enumerate(low[:-1])]
        words.append((1 << 16) | low[-1])
        outputs["received"][1][1:-1] = low
        outputs["scratch_mirror"] = [oracle.guarded(11), [oracle.LEFT] + words + [oracle.RIGHT]]
        outputs["stats"] = [oracle.expected_stats(0, 4, invocation, role, 1) for role in (0, 1)]
        outputs["clocks"] = [limbs(10)+limbs(20)+limbs(30)+limbs(10+elapsed), [0]*12]
        outputs["profile_status"] = [[tag, 0, 0, 0], [tag, 5 if flag else 0, 0, 0]]
        for prefix, values in (("input", inputs), ("output", outputs)):
            raw.update({f"{prefix}{invocation}_{name}": np.asarray(value, np.uint32)
                        for name, value in values.items()})
        if flag:
            events(raw, invocation, [MODULUS-30, MODULUS-10, 10, 20, 50])
    return raw


def make_capture(path, flag=1, run_id=None, elapsed=40):
    path.mkdir()
    (path / "sources/source").mkdir(parents=True)
    pins = {}
    for key, names in (("host_sha256", common.HOST_NAMES), ("source_sha256", common.SOURCE_NAMES)):
        pins[key] = {}
        for name in names:
            shutil.copyfile(ROOT / name, path / "sources" / name)
            pins[key][name] = common.digest(path / "sources" / name)
    manifest = dict(schema=1, backend="hardware", params={"N": 4},
                    artifact_path="fixture-artifact", artifact_sha256="0"*64, **pins)
    common.write_json(path / "manifest.json", manifest)
    run_id = str(uuid.uuid4()) if run_id is None else run_id
    request = dict(schema=1, run_id=run_id, flag=flag, manifest_sha256=common.digest(path / "manifest.json"))
    common.write_json(path / "request.json", request)
    np.savez(path / "raw.npz", **completed_arrays(flag, elapsed))
    common.write_json(path / "out.json", {"params": {"N": "4"}})
    rpc = [dict(name=name, id=i, kind="Var", type="[*]u32") for i, name in enumerate(common.shape(4))]
    rpc += [dict(name=name, id=10+i, kind="Func", type="void", inputs=[]) for i, name in enumerate(("prepare", "run_probe"))]
    common.write_json(path / "out_rpc.json", {"rpc_symbols": rpc})
    record = dict(schema=1, status="pass", n=4, flag=flag, backend="hardware", run_id=run_id,
                  manifest_sha256=request["manifest_sha256"], request_sha256=common.digest(path / "request.json"),
                  raw_sha256=common.digest(path / "raw.npz"), **pins,
                  compiled_metadata_sha256={name: common.digest(path / name) for name in ("out.json", "out_rpc.json")},
                  api=[dict(name=name, status="complete") for name in common.expected_calls(4)])
    common.write_json(path / "capture.json", record)
    return path


class RawTests(unittest.TestCase):
    def test_both_arms_three_generations_and_interval_matched_work(self):
        for flag in (0, 1):
            with self.subTest(flag=flag):
                result = common.validate_raw(completed_arrays(flag), 4, flag)
                self.assertEqual(result["arrays"], 60)
                self.assertEqual(result["independent_runtimes"], 1)
                self.assertEqual(result["work"]["total_directed_link_bytes"], 44)
                self.assertEqual(result["work"]["measured_interval_link_bytes"], 40)
                self.assertEqual(result["work"]["profiling_export_bytes_per_pe"], 168)
                self.assertEqual(result["work"]["exported_array_bytes_per_pe"], 548)
                self.assertEqual(result["work"]["h2d_bytes_per_invocation"], 1096)
                for row in result["invocations"]:
                    self.assertEqual(row["sender"]["elapsed"], 40)
                    self.assertEqual(list(row["receiver_phase_cycles"].values()), [20, 20, 10, 30] if flag else [])

    def test_unsupported_shape_flag_and_missing_arrays_rejected(self):
        for n, flag in ((32, 1), (4, 2), (4, True)):
            with self.subTest(n=n, flag=flag), self.assertRaises(ValueError):
                common.validate_raw(completed_arrays(), n, flag)
        raw = completed_arrays()
        del raw["input2_source"]
        with self.assertRaises(ValueError):
            common.validate_raw(raw, 4, 1)

    def test_full_oracle_and_input_identity_survive_profile_port(self):
        for name, role, column in (("input0_source", 0, 1), ("output0_received", 1, 1),
                                   ("output1_scratch_mirror", 1, 2), ("output2_stats", 0, 7),
                                   ("output0_received", 1, 0)):
            raw = completed_arrays()
            raw[name][role, column] ^= np.uint32(1)
            with self.subTest(name=name), self.assertRaises(ValueError):
                common.validate_raw(raw, 4, 1)

    def test_generation_id_count_overflow_error_and_disabled_counters(self):
        for name, role, column in (("profile_records", 1, 0), ("profile_records", 1, 1),
                                   ("profile_records", 0, 0), ("profile_status", 1, 1),
                                   ("profile_status", 1, 2), ("profile_status", 1, 3),
                                   ("profile_status", 0, 0), ("perf_samples", 0, 0),
                                   ("perf_samples", 1, 0)):
            raw = completed_arrays()
            raw["output1_" + name][role, column] += np.uint32(1)
            with self.subTest(name=name, column=column), self.assertRaises(ValueError):
                common.validate_raw(raw, 4, 1)

    def test_disabled_records_and_wrong_array_type_rejected(self):
        raw = completed_arrays(0)
        raw["output0_profile_records"][1, 0] = 1
        with self.assertRaises(ValueError):
            common.validate_raw(raw, 4, 0)
        raw = completed_arrays()
        raw["output0_source"] = raw["output0_source"].astype(np.int64)
        with self.assertRaises(ValueError):
            common.validate_raw(raw, 4, 1)

    def test_ambiguous_reversed_zero_and_high_limb_stamps_rejected(self):
        for timestamps in ([1, 1, 3, 4, 5], [1, 4, 3, 5, 6],
                           [0, 1 << 47, (1 << 47)+1, (1 << 47)+2, (1 << 47)+3],
                           [0, (1 << 47)-1, (1 << 47)+1, (1 << 47)+2, (1 << 47)+3]):
            raw = completed_arrays()
            events(raw, 0, timestamps)
            with self.subTest(timestamps=timestamps), self.assertRaises(ValueError):
                common.validate_raw(raw, 4, 1)
        raw = completed_arrays()
        raw["output0_profile_records"][1, 2] = 65536
        with self.assertRaises(ValueError):
            common.validate_raw(raw, 4, 1)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.capture = make_capture(self.root / "capture")

    def record(self):
        return common.read_json(self.capture / "capture.json")

    def save_record(self, value):
        common.write_json(self.capture / "capture.json", value)

    def test_complete_capture_replays_with_source_and_build_binding(self):
        result = common.audit_capture(self.capture)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["native_calls"], 80)
        self.assertEqual(result["manifest_sha256"], common.digest(self.capture / "manifest.json"))

    def test_rehashed_numerical_tamper_does_not_pass(self):
        raw = completed_arrays()
        raw["output0_received"][1, 1] ^= np.uint32(1)
        np.savez(self.capture / "raw.npz", **raw)
        record = self.record()
        record["raw_sha256"] = common.digest(self.capture / "raw.npz")
        self.save_record(record)
        with self.assertRaises(ValueError):
            common.audit_capture(self.capture)

    def test_runtime_request_and_backend_mismatches_rejected(self):
        original = self.record()
        for field, value in (("run_id", str(uuid.uuid4())), ("flag", 0),
                             ("backend", "simulator"), ("status", "incomplete"),
                             ("request_sha256", "0"*64), ("manifest_sha256", "0"*64)):
            self.save_record(dict(original, **{field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                common.audit_capture(self.capture)

    def test_source_tamper_symlink_and_unclosed_api_rejected(self):
        path = self.capture / "sources/oracle.py"
        original = path.read_bytes()
        path.write_bytes(original + b"\n# changed\n")
        with self.assertRaises(ValueError):
            common.audit_capture(self.capture)
        path.unlink()
        path.symlink_to(ROOT / "oracle.py")
        with self.assertRaisesRegex(ValueError, "symlink"):
            common.audit_capture(self.capture)
        path.unlink()
        path.write_bytes(original)
        record = self.record()
        record["api"][-1]["status"] = "started"
        self.save_record(record)
        with self.assertRaisesRegex(ValueError, "API"):
            common.audit_capture(self.capture)

    def test_compiled_metadata_rehash_cannot_hide_wrong_params_or_abi(self):
        for name, value in (("out.json", {"params": {"N": 32}}),
                             ("out_rpc.json", {"rpc_symbols": []})):
            original = (self.capture / name).read_bytes()
            original_record = self.record()
            common.write_json(self.capture / name, value)
            record = self.record()
            record["compiled_metadata_sha256"][name] = common.digest(self.capture / name)
            self.save_record(record)
            with self.subTest(name=name), self.assertRaises(ValueError):
                common.audit_capture(self.capture)
            (self.capture / name).write_bytes(original)
            self.save_record(original_record)

    def test_npy_shape_bomb_rejected_before_numpy_allocation(self):
        with zipfile.ZipFile(self.capture / "raw.npz") as archive:
            entries = {name: archive.read(name) for name in archive.namelist()}
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, dict(descr="<u4", fortran_order=False, shape=(2, 1 << 60)))
        entries["output0_source.npy"] = stream.getvalue()
        with zipfile.ZipFile(self.capture / "raw.npz", "w") as archive:
            for name, value in entries.items():
                archive.writestr(name, value)
        with self.assertRaisesRegex(ValueError, "NPY shape"):
            common.load_raw(self.capture / "raw.npz", 4)

    def test_atomic_json_replaces_regular_file_but_rejects_symlink(self):
        path = self.root / "state.json"
        common.write_json(path, {"status": "started"})
        common.write_json(path, {"status": "pass"})
        self.assertEqual(common.read_json(path), {"status": "pass"})
        link = self.root / "link.json"
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            common.write_json(link, {"changed": True})
        self.assertEqual(common.read_json(path), {"status": "pass"})


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.captures = [make_capture(self.root / str(i), flag, elapsed=40+10*flag)
                         for i, flag in enumerate((0, 1, 1, 0))]

    def test_abba_preserves_nested_samples_and_marginal_effect(self):
        result = comparison.compare(self.captures)
        self.assertEqual(result["independent_runtimes_per_arm"], 2)
        self.assertEqual(result["nested_invocations_per_runtime"], 3)
        for row in result["comparisons"]:
            self.assertEqual(row["off_sender_cycles"], [40, 40])
            self.assertEqual(row["on_sender_cycles"], [50, 50])
            self.assertEqual(row["difference_cycles"], 10)
            self.assertEqual(row["relative_change"], 0.25)

    def test_wrong_order_or_reused_runtime_rejected(self):
        for captures in (self.captures[:2],
                          [self.captures[0], self.captures[3], self.captures[1], self.captures[2]],
                          [self.captures[0], self.captures[1], self.captures[1], self.captures[3]]):
            with self.subTest(captures=captures), self.assertRaises(ValueError):
                comparison.compare(captures)

    def test_different_compile_manifest_rejected_even_when_each_capture_valid(self):
        path = self.captures[2]
        manifest = common.read_json(path / "manifest.json")
        manifest["artifact_path"] = "another-artifact"
        common.write_json(path / "manifest.json", manifest)
        pin = common.digest(path / "manifest.json")
        request = common.read_json(path / "request.json")
        request["manifest_sha256"] = pin
        common.write_json(path / "request.json", request)
        record = common.read_json(path / "capture.json")
        record.update(manifest_sha256=pin, request_sha256=common.digest(path / "request.json"))
        common.write_json(path / "capture.json", record)
        with self.assertRaisesRegex(ValueError, "matched build"):
            comparison.compare(self.captures)


if __name__ == "__main__":
    unittest.main()
