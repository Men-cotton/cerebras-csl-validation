"""Offline synthetic evidence: no SDK, PMU, compiler or device execution."""
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_appliance as adapter
import clock_common as common
import clock_plan
import clock_report as report
import test_clock_common as fixtures
import test_clock_extensions as extensions


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build = self.root / "builds" / "fixture"
        self.build.mkdir(parents=True)
        self.plan = dict(profile="fixture", backend="local-sim", host_timeout=300,
                         builds=[dict(id="fixture", params=fixtures.PARAMS)], runs=[],
                         independent_unit="fresh runtime", evidence_scope="synthetic test only")
        self.manifest = dict(schema=1, action="compile", compile_id=str(uuid.uuid4()),
                             backend="local-sim", params=fixtures.PARAMS,
                             **common.source_identity(report.ROOT),
                             launcher_sha256=common.digest(report.ROOT / "clock_local.py"),
                             compiler_args=["synthetic-only"])
        (self.build / "build").mkdir()
        (self.build / "build" / "fixture.bin").write_bytes(b"synthetic build identity")
        write(self.build / "compile-request.json", self.manifest)
        self.manifest["artifacts_sha256"] = {"fixture.bin": common.digest(self.build / "build" / "fixture.bin")}
        write(self.build / "clock-build.json", self.manifest)
        write(self.build / "compile-status.json", dict(status="complete", manifest_sha256=common.digest(self.build / "clock-build.json")))

    def entry(self, name, arm="rectangle", block=1, position=1):
        entry = dict(id=name, cohort="pair", block=block, position=position, arm=arm,
                     build="fixture", config=common.validate_config(dict(arm=arm)))
        self.plan["runs"].append(entry)
        return entry

    def make_capture(self, entry, run_id=None):
        directory = self.root / "runs" / entry["id"]
        directory.mkdir(parents=True)
        (directory / "clock-build.json").write_bytes((self.build / "clock-build.json").read_bytes())
        request = dict(config=entry["config"], run_id=run_id or str(uuid.uuid4()),
                       manifest_sha256=common.digest(self.build / "clock-build.json"))
        write(directory / "clock-request.json", request)
        meta, calls, raw, _, _ = fixtures.envelope(entry["config"])
        if entry["config"].get("markers"):
            raw, steps = extensions.marked_raw(entry["config"])
            meta["steps"] = steps
        if entry["config"].get("markers") or entry["config"].get("host_pmu"):
            calls = extensions.extended_calls(entry["config"])
        meta.update(backend=self.manifest["backend"], run_id=request["run_id"],
                    manifest_sha256=request["manifest_sha256"],
                    source_sha256=self.manifest["source_sha256"], host_sha256=self.manifest["host_sha256"])
        capture = directory / "capture"
        capture.mkdir()
        (capture / "raw.npz").write_bytes(fixtures.npz_bytes(raw))
        write(capture / "calls.json", calls)
        meta.update(raw_sha256=common.digest(capture / "raw.npz"), calls_sha256=common.digest(capture / "calls.json"))
        write(capture / "capture.json", meta)
        if self.plan["backend"] == "local-sim":
            shutil.copytree(self.build / "build", directory / "build")
            write(directory / "launch.json", dict(command=["synthetic-only"],
                  run_launcher_sha256=self.manifest["launcher_sha256"], **request))
            write(directory / "local-status.json", dict(status="pass", exit_code=0))
        else:
            target = directory / "capture.tar.gz"
            with tarfile.open(target, "w:gz") as archive:
                for path in capture.iterdir():
                    data = path.read_bytes()
                    member = tarfile.TarInfo("capture/" + path.name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
            (directory / "host.log").write_text("synthetic log\n")
            (directory / "host-exit.txt").write_text("0\n")
            _, validation = adapter.validate_capture(target, self.manifest, request)
            downloaded = {}
            for remote, name in (("capture", "capture.tar.gz"), ("host.log", "host.log"), ("host-exit.txt", "host-exit.txt")):
                path = directory / name
                downloaded[remote] = dict(path=name, bytes=path.stat().st_size, sha256=common.digest(path))
            write(directory / "request.json", dict(action="run", backend="hardware",
                  command=adapter.host_command("hardware", 300), artifact_sha256=self.manifest["artifact_sha256"], **request))
            write(directory / "collection.json", dict(status="complete", worker_status="pass", host_exit=0,
                  run_id=request["run_id"], manifest_sha256=request["manifest_sha256"],
                  downloaded=downloaded, errors={}, capture_validation=validation, accounting={"status": "complete"}))
        return directory

    def use_hardware(self):
        self.plan["backend"] = "hardware"
        self.manifest.pop("artifacts_sha256")
        self.manifest.update(backend="hardware", launcher_sha256=common.digest(report.ROOT / "clock_appliance.py"))
        write(self.build / "request.json", self.manifest)
        (self.build / "artifact.tar.gz").write_bytes(b"synthetic artifact identity")
        self.manifest.update(artifact_path="artifact.tar.gz", artifact_sha256=common.digest(self.build / "artifact.tar.gz"))
        write(self.build / "clock-build.json", self.manifest)
        write(self.build / "compile-status.json", dict(status="complete", compile_id=self.manifest["compile_id"],
              backend="hardware", manifest_sha256=common.digest(self.build / "clock-build.json"), accounting={"status": "complete"}))

    def admitted(self, entry, run_id=None):
        directory = self.make_capture(entry, run_id)
        return report.analyze_run(directory, self.build, entry, self.plan)

    def test_raw_admission_and_compact_members(self):
        entry = self.entry("r001")
        result = self.admitted(entry)
        self.assertEqual(result["status"], "pass")
        self.assertEqual([s["step"] for s in result["steps"]], [0, 1, 2])
        self.assertEqual(set(result["observed_sha256"]), {"raw.npz", "capture.json", "calls.json"})
        self.assertEqual(result["calls"][0]["pmu"], {"status": "not_requested"})

    def add_runtime_auxiliaries(self, directory):
        for name in report.LOCAL_RUNTIME_AUXILIARIES:
            path = directory / "build" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic runtime auxiliary " + name).encode())

    def test_local_runtime_generated_auxiliaries_preserve_compiled_pins(self):
        entry = self.entry("r001")
        directory = self.make_capture(entry)
        self.add_runtime_auxiliaries(directory)
        analysis = report.analyze_run(directory, self.build, entry, self.plan)
        self.assertEqual(set(analysis["runtime_auxiliary_sha256"]), report.LOCAL_RUNTIME_AUXILIARIES)
        self.assertEqual(report.read_compact_run(directory, self.build, entry, self.plan,
                         omitted_paths=[]), analysis)
        self.assertEqual(report.analyze_run(directory, self.build, entry, self.plan), analysis)

    def test_local_runtime_auxiliaries_do_not_allow_modified_missing_or_unknown_files(self):
        for index, mutation in enumerate(("modified", "missing", "unknown"), 1):
            with self.subTest(mutation=mutation):
                entry = self.entry("r%03d" % index)
                directory = self.make_capture(entry)
                self.add_runtime_auxiliaries(directory)
                report.analyze_run(directory, self.build, entry, self.plan)
                original = directory / "build/fixture.bin"
                if mutation == "modified":
                    original.write_text("changed original")
                elif mutation == "missing":
                    original.unlink()
                else:
                    (directory / "build/generated/unknown.elf").write_text("unknown extra")
                expected = "unexpected local runtime artifact" if mutation == "unknown" else "original compiled artifact hashes"
                with self.assertRaisesRegex(ValueError, expected):
                    report.analyze_run(directory, self.build, entry, self.plan)
                with self.assertRaisesRegex(ValueError, expected):
                    report.read_compact_run(directory, self.build, entry, self.plan, omitted_paths=[])

    def test_readmission_is_idempotent_and_rejects_retained_tamper(self):
        entry = self.entry("r001")
        first = self.admitted(entry)
        directory = self.root / "runs/r001"
        before = (directory / "analysis.json").read_bytes()
        self.assertEqual(report.analyze_run(directory, self.build, entry, self.plan), first)
        self.assertEqual((directory / "analysis.json").read_bytes(), before)
        path = directory / "observed/raw.npz"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "oversized file|bytes differ"):
            report.analyze_run(directory, self.build, entry, self.plan)

    def test_local_manifest_bytes_and_plan_configuration_are_bound(self):
        entry = self.entry("r001")
        directory = self.make_capture(entry)
        (directory / "clock-build.json").write_text((directory / "clock-build.json").read_text() + " ")
        with self.assertRaisesRegex(ValueError, "manifest bytes"):
            report.analyze_run(directory, self.build, entry, self.plan)
        (directory / "clock-build.json").write_bytes((self.build / "clock-build.json").read_bytes())
        altered = dict(entry, config=common.validate_config(dict(arm="strips")))
        with self.assertRaisesRegex(ValueError, "configuration"):
            report.analyze_run(directory, self.build, altered, self.plan)

    def test_download_hash_is_checked_before_raw_admission(self):
        self.use_hardware()
        entry = self.entry("r001")
        directory = self.make_capture(entry)
        with (directory / "capture.tar.gz").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "download hash/size"):
            report.analyze_run(directory, self.build, entry, self.plan)

    def test_hardware_capture_admission(self):
        self.use_hardware()
        entry = self.entry("r001")
        self.assertEqual(self.admitted(entry)["backend"], "hardware")

    def test_compact_hardware_replays_selected_raw_with_explicit_omission(self):
        self.use_hardware()
        entry = self.entry("r001")
        expected = self.admitted(entry)
        directory = self.root / "runs/r001"
        (directory / "capture.tar.gz").unlink()
        with self.assertRaisesRegex(ValueError, "undeclared missing original"):
            report.read_compact_run(directory, self.build, entry, self.plan, omitted_paths=[])
        actual = report.read_compact_run(directory, self.build, entry, self.plan,
                                        omitted_paths=["runs/r001/capture.tar.gz"])
        self.assertEqual(actual, expected)
        # Remaining downloads are still checked, even though the original archive is omitted.
        (directory / "host.log").write_text("changed log")
        with self.assertRaisesRegex(ValueError, "remaining download hash"):
            report.read_compact_run(directory, self.build, entry, self.plan,
                                    omitted_paths=["runs/r001/capture.tar.gz"])

    def test_compact_local_build_copy_omission_and_cached_original_validation(self):
        entry = self.entry("r001")
        expected = self.admitted(entry)
        directory = self.root / "runs/r001"
        shutil.rmtree(directory / "build")
        with self.assertRaisesRegex(ValueError, "undeclared missing local build"):
            report.read_compact_run(directory, self.build, entry, self.plan, omitted_paths=[])
        cache = {}
        actual = report.read_compact_run(directory, self.build, entry, self.plan,
                    omitted_paths=["runs/r001/build/"], validated_builds=cache)
        self.assertEqual(actual, expected)
        self.assertEqual(len(cache), 1)
        self.assertEqual(report.read_compact_run(directory, self.build, entry, self.plan,
                    omitted_paths=["runs/r001/build/"], validated_builds=cache), expected)
        # A new replay cache must validate original compiled artifacts again.
        (self.build / "build/fixture.bin").write_text("changed build")
        with self.assertRaisesRegex(ValueError, "original compiled artifact hashes"):
            report.read_compact_run(directory, self.build, entry, self.plan,
                                    omitted_paths=["runs/r001/build/"], validated_builds={})

    def test_compact_collection_cannot_change_the_submitted_configuration(self):
        self.use_hardware()
        entry = self.entry("r001")
        self.admitted(entry)
        directory = self.root / "runs/r001"
        request = common.read_json(directory / "request.json")
        request["config"]["arm"] = "strips"
        write(directory / "request.json", request)
        with self.assertRaisesRegex(ValueError, "original launch request"):
            report.read_compact_run(directory, self.build, entry, self.plan, omitted_paths=[])

    def test_compact_uses_frozen_sources_instead_of_the_invoking_checkout(self):
        entry = self.entry("r001")
        expected = self.admitted(entry)
        frozen = self.root / "driver"
        for name in ("clock_probe.py", "clock_common.py", "clock_pmu.py", "clock_local.py",
                     "clock_appliance.py", "clock_source/layout.csl", "clock_source/pe.csl"):
            target = frozen / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((report.ROOT / name).read_bytes())
        with mock.patch.object(report, "ROOT", self.root / "unrelated-checkout"):
            self.assertEqual(report.read_compact_run(self.root / "runs/r001", self.build, entry,
                             self.plan, omitted_paths=[]), expected)

    def test_compact_rejects_symlinked_observed_directory(self):
        entry = self.entry("r001")
        self.admitted(entry)
        directory = self.root / "runs/r001"
        (directory / "observed").rename(directory / "saved-observed")
        (directory / "observed").symlink_to(directory / "saved-observed", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "without symlink traversal"):
            report.read_compact_run(directory, self.build, entry, self.plan, omitted_paths=[])

    def test_compact_rejects_noncanonical_runtime_uuid(self):
        entry = self.entry("r001")
        self.admitted(entry)
        directory = self.root / "runs/r001"
        request = common.read_json(directory / "clock-request.json")
        request["run_id"] = "{" + request["run_id"] + "}"
        write(directory / "clock-request.json", request)
        with self.assertRaisesRegex(ValueError, "canonical runtime UUID"):
            report.read_compact_run(directory, self.build, entry, self.plan, omitted_paths=[])

    def test_numeric_failure_is_retained_without_performance_promotion(self):
        entry = self.entry("r001")
        directory = self.make_capture(entry)
        capture = directory / "capture"
        raw = common.read_raw((capture / "raw.npz").read_bytes())
        raw["seen_1"][0, 0, 0] ^= 1
        (capture / "raw.npz").write_bytes(fixtures.npz_bytes(raw))
        meta = common.read_json(capture / "capture.json")
        meta.update(status="observed_numeric_fail", raw_sha256=common.digest(capture / "raw.npz"))
        write(capture / "capture.json", meta)
        write(directory / "local-status.json", dict(status="observed_numeric_fail", exit_code=1))
        analysis = report.analyze_run(directory, self.build, entry, self.plan)
        self.assertEqual(analysis["status"], "numeric_fail")
        self.assertEqual(analysis["steps"], [])
        result = report.write_report(self.root, self.plan,
                    dict(status="completed_with_failures", runs={"r001": {"status": "numeric_fail"}}))
        self.assertEqual(result["counts"]["numeric_fail"], 1)
        self.assertEqual(result["passed_fresh_runtimes"], 0)
        self.assertEqual(result["pe_intervals"], 0)

    def test_markers_thread_observations_and_unavailable_pmu_are_preserved(self):
        entry = self.entry("r001", "tiles")
        entry["config"].update(markers=True, host_pmu=True, pmu_controls=3)
        directory = self.make_capture(entry)
        capture = directory / "capture"
        calls = common.read_json(capture / "calls.json")
        calls[0]["pmu"] = dict(status="unavailable", scope="thread-self-user", error=dict(stage="open", errno=1, message="denied"))
        write(capture / "calls.json", calls)
        meta = common.read_json(capture / "capture.json")
        meta["calls_sha256"] = common.digest(capture / "calls.json")
        write(capture / "capture.json", meta)
        report.analyze_run(directory, self.build, entry, self.plan)
        result = report.write_report(self.root, self.plan, dict(status="completed", runs={"r001": {"status": "pass"}}))
        self.assertEqual(result["marker_observations"], 48)
        self.assertEqual(result["pmu_unavailable"], 1)
        with (self.root / "host.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["pmu_cycles"], "")
        self.assertEqual(rows[0]["thread_time_ns"], "100")
        self.assertEqual(rows[1]["pmu_cycles"], "900")
        self.assertEqual(rows[1]["pmu_scope"], "thread-self-user")

    def test_partial_summary_keeps_failed_and_unrun_entries(self):
        first, second = self.entry("r001"), self.entry("r002", "strips", position=2)
        third = self.entry("r003", block=2)
        self.admitted(first)
        state = dict(status="running", runs={first["id"]: {"status": "pass"},
                     second["id"]: {"status": "unknown", "remote_stop_unconfirmed": True}})
        summary = report.write_report(self.root, self.plan, state)
        self.assertEqual(summary["counts"]["unknown"], 1)
        self.assertEqual(summary["counts"]["pending"], 1)
        self.assertEqual(summary["passed_fresh_runtimes"], 1)
        self.assertEqual(summary["nested_generations"], 3)
        self.assertEqual(summary["pe_intervals"], 12)
        contrasts = common.read_json(self.root / "contrasts.json")
        self.assertEqual(contrasts["pair_records"][0]["status"], "unavailable")
        self.assertEqual(contrasts["pair_records"][0]["generations"], [])
        with (self.root / "pe.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 12)
        self.assertEqual(third["id"], "r003")

    def test_paired_generations_are_not_three_independent_repeats(self):
        first, second = self.entry("r001"), self.entry("r002", "strips", position=2)
        self.admitted(first)
        self.admitted(second)
        state = dict(status="completed", runs={e["id"]: {"status": "pass"} for e in (first, second)})
        summary = report.write_report(self.root, self.plan, state)
        contrasts = common.read_json(self.root / "contrasts.json")
        self.assertEqual(summary["passed_fresh_runtimes"], 2)
        self.assertEqual(summary["nested_generations"], 6)
        self.assertEqual(contrasts["paired_records"], 1)
        self.assertEqual(len(contrasts["pair_records"][0]["generations"]), 3)
        self.assertEqual(contrasts["pair_records"][0]["generations"][0]["right_minus_left_cycles"], 0)

    def test_duplicate_runtime_uuid_invalidates_summary(self):
        first, second = self.entry("r001"), self.entry("r002", "strips", position=2)
        runtime_id = str(uuid.uuid4())
        self.admitted(first, runtime_id)
        self.admitted(second, runtime_id)
        state = dict(status="completed", runs={e["id"]: {"status": "pass"} for e in (first, second)})
        summary = report.write_report(self.root, self.plan, state)
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["analysis_errors"], 1)
        self.assertEqual(summary["passed_fresh_runtimes"], 1)
        self.assertIn("duplicate", summary["runs"][1]["analysis_error"])

    def test_tampered_derived_cycles_do_not_enter_report(self):
        entry = self.entry("r001")
        self.admitted(entry)
        path = self.root / "runs/r001/analysis.json"
        analysis = common.read_json(path)
        analysis["steps"][0]["max_span_cycles"] += 1000
        write(path, analysis)
        summary = report.write_report(self.root, self.plan, dict(status="completed", runs={"r001": {"status": "pass"}}))
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["pe_intervals"], 0)
        self.assertEqual(summary["analysis_errors"], 1)

    def test_all_six_primary_orders_keep_three_shared_runtime_pairs(self):
        plan = clock_plan.make_plan()
        entries = [e for e in plan["runs"] if e["cohort"] == "granularity"]
        plan["runs"] = entries
        analyses = {e["id"]: dict(status="pass", steps=[dict(step=s, max_span_cycles=100+s) for s in range(3)]) for e in entries}
        result = report._contrasts(plan, analyses)
        self.assertEqual(len(entries), 18)
        self.assertEqual(result["paired_records"], 18)
        for block in range(1, 7):
            rows = [r for r in result["pair_records"] if r["block"] == block]
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({tuple(r["runtime_order"]) for r in rows}), 1)
            self.assertEqual(len(rows[0]["runtime_order"]), 3)


if __name__ == "__main__":
    unittest.main()
