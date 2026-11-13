import copy
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import driver


class MemcpyDataType:
    MEMCPY_32BIT = "u32"


class MemcpyOrder:
    ROW_MAJOR = "row-major"


def fake_runtime(*, corrupt_reply=False, permute_reply=False, fail_h2d=None, fail_stop=False):
    """Model pending input and completion, never synthesize the expected reply."""
    instances = []

    class Runtime:
        def __init__(self, build, **kwargs):
            params = json.loads((Path(build)/"out.json").read_text())["params"]
            self.width, self.height = int(params["WIDTH"]), int(params["HEIGHT"])
            self.pes = self.width * self.height
            self.events = []
            self.pending = None
            self.config = None
            self.loaded = False
            self.started = False
            self.stopped = False
            self.h2d_count = 0
            self.cwd = Path.cwd()
            self.options = kwargs
            instances.append(self)

        def load(self):
            self.loaded = True
            self.events.append(("load",))

        def run(self):
            if not self.loaded:
                raise AssertionError("run before load")
            self.started = True
            self.events.append(("run",))

        def get_id(self, name):
            return name

        def launch(self, name, words, *, nonblock):
            if name != "configure" or nonblock is not True or self.pending is not None:
                raise AssertionError("configuration protocol")
            self.config = int(words)
            self.events.append(("configure", self.config))

        def memcpy_h2d(self, symbol, source, x, y, width, height, count, **kwargs):
            self.h2d_count += 1
            if self.h2d_count == fail_h2d:
                raise RuntimeError("injected H2D failure")
            if (not self.started or self.pending is not None or symbol != 0
                    or (x, y, width, height) != (0, 0, self.width, self.height)
                    or count != self.config + 1 or kwargs != {
                        "streaming": True, "nonblock": True,
                        "data_type": MemcpyDataType.MEMCPY_32BIT,
                        "order": MemcpyOrder.ROW_MAJOR}):
                raise AssertionError("one outstanding configured streaming H2D")
            if source.dtype != np.uint32 or source.shape != (width * height * count,):
                raise AssertionError("input frame representation")
            self.pending = (source.copy(), width, height, count)
            self.events.append(("h2d", width, height, count))

        def memcpy_d2h(self, destination, symbol, x, y, width, height, count, **kwargs):
            if (kwargs.get("nonblock") is not False
                    or (x, y, width, height) != (0, 0, self.width, self.height)):
                raise AssertionError("blocking D2H boundary")
            if kwargs.get("streaming"):
                if symbol != 1 or self.pending is None:
                    raise AssertionError("reply without a request")
                values, pending_width, pending_height, pending_count = self.pending
                if (width, height, count) != (pending_width, pending_height, pending_count):
                    raise AssertionError("reply extent")
                np.copyto(destination, values)
                if permute_reply:
                    destination[:] = destination[::-1]
                if corrupt_reply:
                    destination[-1] ^= np.uint32(1)
                self.pending = None
                self.events.append(("d2h", width, height, count))
            else:
                if (width, height, count) != (self.width, self.height, 1) or symbol not in ("config_words", "status"):
                    raise AssertionError("configuration confirmation extent")
                destination.fill(self.config if symbol == "config_words" else 0)
                self.events.append(("configuration-read", symbol))

        def stop(self):
            self.stopped = True
            self.events.append(("stop",))
            if fail_stop:
                raise RuntimeError("injected stop failure")

    return (Runtime, MemcpyDataType, MemcpyOrder), instances


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.serial = 0

    def build(self, case):
        self.serial += 1
        folder = self.root/f"build-{self.serial}"
        (folder/"bin").mkdir(parents=True)
        parameters = driver.oracle.CASES[case]
        (folder/"out.json").write_text(json.dumps({"params": {
            "WIDTH": str(parameters["width"]),
            "HEIGHT": str(parameters["height"]), "CAPACITY": str(parameters["capacity"]),
            "MEMCPYH2D_DATA_1_ID": "0", "MEMCPYD2H_DATA_1_ID": "1",
        }}))
        (folder/"bin/out_0_0.elf").write_bytes(b"mock compiler artifact, never executed")
        (folder/"bin/out_rpc.json").write_text("{}")
        self.write_channel_metadata(folder, parameters)
        return folder

    def write_channel_metadata(self, folder, parameters, heights=None):
        if heights is None:
            heights = [parameters["height"] // parameters["channels"]] * parameters["channels"]
        for side, count_name, height_prefix, color_prefix in (
                ("west", "num_demux", "height_demux_", "FABRIC_H2D_DATA_ID_"),
                ("east", "num_mux", "height_mux_", "FABRIC_D2H_DATA_ID_")):
            params = {"height": str(parameters["height"]), count_name: str(len(heights))}
            if side == "west":
                params["num_inputs"] = "1"
            for index, height in enumerate(heights):
                params[height_prefix + str(index)] = str(height)
                params[color_prefix + str(index)] = str(5 + index)
            (folder/side).mkdir(exist_ok=True)
            (folder/side/"out.json").write_text(json.dumps({"params": params, "colors": {}}))

    def capture(self, case="zero-p1", order=0, smoke=True, **faults):
        build = self.build(case)
        output = self.root/f"capture-{self.serial}"
        api, instances = fake_runtime(**faults)
        with mock.patch.object(driver.time, "perf_counter_ns", side_effect=itertools.count(10000, 100)):
            result = driver.capture(driver.HERE, build, output, case, order,
                                    smoke=smoke, runtime_api=api)
        return output, result, instances[0]

    def test_full_zero_schedule_completes_each_reply_before_next_send(self):
        previous = Path.cwd()
        output, record, runtime = self.capture(order=2, smoke=False)
        audit = driver.audit(output)
        self.assertEqual(Path.cwd(), previous)
        self.assertTrue(runtime.stopped)
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["exchanges"], 220)
        self.assertEqual(record["rows"][0]["generation"], 200001)
        self.assertEqual(record["rows"][-1]["generation"], 200220)
        self.assertEqual(sum(row["phase"] == "warmup" for row in record["rows"]), 20)
        self.assertEqual(audit["summary"][0]["samples"], 200)
        self.assertEqual(audit["summary"][0]["median_rtt_ns"], 100)
        self.assertEqual(audit["summary"][0]["payload_bytes"], 0)
        self.assertEqual(audit["summary"][0]["control_h2d_bytes"], 4)
        exchanges = [event for event in runtime.events if event[0] in ("h2d", "d2h")]
        self.assertEqual(exchanges, [("h2d", 1, 1, 1), ("d2h", 1, 1, 1)] * 220)
        self.assertEqual(sum(event[0] == "configure" for event in runtime.events), 1)
        self.assertEqual(runtime.cwd, output.parent/(output.name + "-sdk-work"))

    def test_powers_smoke_replays_padding_topology_and_large_raw_witness(self):
        output, record, runtime = self.capture("powers-p64", order=1)
        audit = driver.audit(output)
        self.assertEqual(audit["exchanges"], 24)
        self.assertEqual([row["payload_bytes"] for row in audit["summary"]],
                         [1048576, 32768, 16384, 4, 1, 0])
        self.assertEqual([row["active_pes"] for row in audit["summary"]], [64] * 6)
        self.assertEqual([(row["active_cols"], row["active_rows"]) for row in audit["summary"]], [(8, 8)] * 6)
        self.assertEqual([row["control_h2d_bytes"] for row in audit["summary"]], [256] * 6)
        self.assertEqual({event[1:3] for event in runtime.events if event[0] == "h2d"}, {(8, 8)})
        self.assertEqual(sum(event[0] == "configure" for event in runtime.events), 6)
        self.assertEqual(len(record["witnesses"]), 6)
        witness = record["witnesses"]["1048576"]
        values = np.load(output/witness["path"], allow_pickle=False)
        self.assertEqual(values.shape, (64 * 4097,))
        self.assertEqual(values.nbytes, 1048576 + 256)
        self.assertEqual(audit["summary"][-2]["padding_bytes"], 255)

    def test_each_case_keeps_all_fixed_pes_in_small_and_zero_exchanges(self):
        for pes in (1, 4, 16, 64, 256, 1024, 4096):
            with self.subTest(pes=pes):
                output, record, runtime = self.capture(f"zero-p{pes}")
                audit = driver.audit(output)
                self.assertEqual(audit["summary"][0]["active_pes"], pes)
                self.assertEqual(audit["summary"][0]["control_h2d_bytes"], 4 * pes)
                self.assertEqual({event[1:3] for event in runtime.events if event[0] == "h2d"},
                                 {(math.isqrt(pes), math.isqrt(pes))})
                witness = record["witnesses"]["0"]
                raw = np.load(output/witness["path"], allow_pickle=False)
                self.assertEqual(raw.shape, (pes,))
                expected = [witness["generation"] ^ ((pe * 0x85ebca6b) & 0xffffffff)
                            for pe in range(pes)]
                np.testing.assert_array_equal(raw, expected)
        output, _, runtime = self.capture("powers-p4")
        audit = driver.audit(output)
        self.assertEqual([row["active_pes"] for row in audit["summary"]], [4] * 6)
        self.assertEqual({event[1:3] for event in runtime.events if event[0] == "h2d"}, {(2, 2)})
        self.assertEqual(audit["summary"][-1]["payload_bytes"], 65536)

    def test_zero_payload_rejects_permuted_pe_responses(self):
        with self.assertRaisesRegex(ValueError, "echo reply differs"):
            self.capture("zero-p4", permute_reply=True)
        record = driver.read(self.root/"capture-1/capture.json")
        self.assertEqual(record["status"], "FAILED")
        self.assertIs(record["rows"][0]["equal"], False)

    def test_channel_mapping_accepts_complete_unequal_row_partitions(self):
        build = self.build("powers-p4096-c4")
        plan = driver.oracle.plan("powers-p4096-c4")
        self.write_channel_metadata(build, plan, heights=[1, 3, 20, 40])
        expected = [
            {"row_begin": 0, "row_end": 1, "routing_color": 5},
            {"row_begin": 1, "row_end": 4, "routing_color": 6},
            {"row_begin": 4, "row_end": 24, "routing_color": 7},
            {"row_begin": 24, "row_end": 64, "routing_color": 8},
        ]
        self.assertEqual(driver.channel_mapping(build, plan), {"h2d": expected, "d2h": expected})

    def test_channel_mapping_rejects_wrong_counts_extents_and_unregistered_input(self):
        build = self.build("powers-p4096-c4")
        plan = driver.oracle.plan("powers-p4096-c4")
        for side, field, value in (
                ("west", "num_demux", "3"), ("east", "num_mux", "2"),
                ("west", "height_demux_0", "0"), ("east", "height_mux_3", "-1"),
                ("west", "height_demux_0", "17"), ("east", "height", "65"),
                ("west", "num_inputs", "0")):
            with self.subTest(side=side, field=field, value=value):
                self.write_channel_metadata(build, plan)
                path = build/side/"out.json"
                metadata = json.loads(path.read_text())
                metadata["params"][field] = value
                path.write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    driver.channel_mapping(build, plan)

    def test_wrong_streaming_color_parameters_fail_before_runtime_start(self):
        for field, value in (("MEMCPYH2D_DATA_1_ID", "1"), ("MEMCPYD2H_DATA_1_ID", "0")):
            with self.subTest(field=field):
                build = self.build("zero-p1")
                metadata = json.loads((build/"out.json").read_text())
                metadata["params"][field] = value
                (build/"out.json").write_text(json.dumps(metadata))
                api, instances = fake_runtime()
                with self.assertRaises(ValueError):
                    driver.capture(driver.HERE, build, self.root/f"wrong-color-{self.serial}",
                                   "zero-p1", 0, smoke=True, runtime_api=api)
                self.assertEqual(instances, [])

    def test_audit_rejects_timing_schedule_digest_and_scope_mutations(self):
        output, original, _ = self.capture()
        changes = [
            ("zero RTT", lambda r: r["rows"][0].update(elapsed_ns=0)),
            ("negative RTT", lambda r: r["rows"][0].update(elapsed_ns=-1)),
            ("boolean RTT", lambda r: r["rows"][0].update(elapsed_ns=True)),
            ("floating RTT", lambda r: r["rows"][0].update(elapsed_ns=100.0)),
            ("unbounded RTT", lambda r: r["rows"][0].update(elapsed_ns=600 * 10**9)),
            ("stale generation", lambda r: r["rows"][0].update(generation=0)),
            ("wrong phase", lambda r: r["rows"][0].update(phase="measured")),
            ("wrong payload size", lambda r: r["rows"][0].update(payload_bytes=4)),
            ("wrong iteration", lambda r: r["rows"][0].update(iteration=1)),
            ("missing exchange", lambda r: r["rows"].pop()),
            ("unequal echo", lambda r: r["rows"][0].update(equal=False)),
            ("wrong digest", lambda r: r["rows"][0].update(received_sha256="0" * 64)),
            ("changed schedule", lambda r: r["plan"].update(warmup=2)),
            ("hardware smoke", lambda r: r.update(backend="hardware")),
            ("invented wire count", lambda r: r.update(physical_wire_bytes=8)),
            ("overlapping requests", lambda r: r.update(outstanding_exchanges=2)),
            ("invented timing scope", lambda r: r.update(timing_scope="physical-link latency only")),
            ("changed channel partition", lambda r: r["channel_mapping"]["h2d"][0].update(row_end=2)),
        ]
        for name, change in changes:
            with self.subTest(mutation=name):
                record = copy.deepcopy(original)
                change(record)
                driver.write(output/"capture.json", record)
                with self.assertRaises(ValueError):
                    driver.audit(output)
        driver.write(output/"capture.json", original)
        self.assertEqual(driver.audit(output)["status"], "PASS")

    def test_audit_rejects_wrong_raw_witness_even_with_updated_file_hash(self):
        output, record, _ = self.capture()
        witness = record["witnesses"]["0"]
        path = output/witness["path"]
        values = np.load(path, allow_pickle=False)
        values[0] ^= np.uint32(1)
        np.save(path, values, allow_pickle=False)
        witness["sha256"] = driver.sha(path)
        driver.write(output/"capture.json", record)
        with self.assertRaisesRegex(ValueError, "raw witness equality"):
            driver.audit(output)

    def test_audit_rejects_rebound_source_and_changed_generated_metadata(self):
        output, record, _ = self.capture()
        source = output/"source/pe.csl"
        original = source.read_bytes()
        source.write_bytes(original + b"\n// modified frozen source\n")
        record["source_sha256"]["pe.csl"] = driver.sha(source)
        driver.write(output/"capture.json", record)
        with self.assertRaisesRegex(ValueError, "source identity"):
            driver.audit(output)
        source.write_bytes(original)
        record["source_sha256"]["pe.csl"] = driver.sha(source)
        driver.write(output/"capture.json", record)
        metadata = output/"metadata/out.json"
        metadata.write_text(metadata.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "metadata identity"):
            driver.audit(output)

    def test_failed_reply_is_preserved_and_cwd_restored(self):
        previous = Path.cwd()
        with self.assertRaisesRegex(ValueError, "echo reply differs"):
            self.capture(corrupt_reply=True)
        output = self.root/"capture-1"
        record = driver.read(output/"capture.json")
        self.assertEqual(Path.cwd(), previous)
        self.assertEqual(record["status"], "FAILED")
        self.assertEqual(len(record["rows"]), 1)
        self.assertIs(record["rows"][0]["equal"], False)
        self.assertTrue((output/record["witnesses"]["0"]["path"]).is_file())
        self.assertIn("stop", record["setup_seconds"])
        with self.assertRaises(ValueError):
            driver.audit(output)

    def test_partial_runtime_and_failed_stop_are_not_successful_captures(self):
        previous = Path.cwd()
        for fault, error in (({"fail_h2d": 2}, "H2D failure"), ({"fail_stop": True}, "stop failure")):
            with self.subTest(fault=fault):
                with self.assertRaisesRegex(RuntimeError, error):
                    self.capture(**fault)
                output = self.root/f"capture-{self.serial}"
                record = driver.read(output/"capture.json")
                self.assertEqual(Path.cwd(), previous)
                self.assertEqual(record["status"], "FAILED")
                self.assertEqual(len(record["rows"]), 1 if "fail_h2d" in fault else 4)
                self.assertIn("error" if "fail_h2d" in fault else "stop_error", record)
                with self.assertRaises(ValueError):
                    driver.audit(output)

    def test_hardware_rejects_smoke_before_starting_a_runtime(self):
        build = self.build("zero-p1")
        api, instances = fake_runtime()
        output = self.root/"hardware-smoke"
        with self.assertRaisesRegex(ValueError, "execution backend"):
            driver.capture(driver.HERE, build, output, "zero-p1", 0,
                           backend="hardware", cmaddr="unused:0", smoke=True, runtime_api=api)
        self.assertEqual(instances, [])
        self.assertFalse(output.exists())

    def test_sdk_workdir_cannot_contain_the_capture(self):
        build = self.build("zero-p1")
        api, instances = fake_runtime()
        with self.assertRaises(ValueError):
            driver.capture(driver.HERE, build, self.root/"nested-capture", "zero-p1", 0,
                           sdk_workdir=self.root, smoke=True, runtime_api=api)
        self.assertEqual(instances, [])


if __name__ == "__main__":
    unittest.main()
