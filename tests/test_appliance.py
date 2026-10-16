"""SDK-free regression checks for artifact selection and result collection."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import appliance
import inspect_result


class ApplianceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ("source", "diagnostics"):
            (self.root / directory).mkdir()
            for name in ("layout.csl", "pe.csl"):
                (self.root / directory / name).write_text(directory + name)
        for host in ("run.py", "diagnostic.py"):
            (self.root / host).write_text("# " + host)
        self.compile_calls = []

    def compile_record(self, variant="baseline", backend="simulator"):
        calls = self.compile_calls

        class Compiler:
            def __init__(self, **options):
                calls.append(options)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def compile(self, source, layout, flags, destination):
                calls.append((source, layout, flags))
                artifact = Path(destination) / "artifact.tar.gz"
                artifact.write_bytes(b"synthetic artifact")
                return str(artifact)

        args = SimpleNamespace(variant=variant, backend=backend, namespace="test-space",
                               out=self.root / ("build-" + variant + "-" + backend), fabric_offsets="4,1")
        self.assertEqual(appliance.compile_program(args, self.root, Compiler), 0)
        return args.out / "artifact.json"

    def capture(self, variant="baseline", backend="simulator", status="pass"):
        calls = [dict(index=0, api="stop", phase=phase, stage="teardown", detail={})
                 for phase in ("enter", "exit")]
        calls = json.dumps(calls).encode()
        raw = b"synthetic raw data; hash test only"
        host = "run.py" if variant == "baseline" else "diagnostic.py"
        meta = dict(config=dict(offset=3, length=8, mode=0), backend=backend, variant=variant,
                    status=status, host_sha256=appliance.digest(self.root / host),
                    baseline_host_sha256=appliance.digest(self.root / "run.py"),
                    panels=[dict(panel=index, errors={}) for index in range(3)],
                    **{"calls.json_sha256": hashlib.sha256(calls).hexdigest(),
                       "raw_device.npz_sha256": hashlib.sha256(raw).hexdigest()})
        if status == "observed_numeric_fail":
            meta["panels"] = [dict(panel=0, errors={"received_rows": {"count": 1}})]
        elif status == "observed_payload_difference":
            meta["panels"] = [dict(panel=0, errors={"payload_stream": {"matching_representations": []}})]
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as output:
            for name, data in (("capture.json", json.dumps(meta).encode()), ("calls.json", calls),
                               ("raw_device.npz", raw)):
                info = tarfile.TarInfo("capture/" + name)
                info.size = len(data)
                output.addfile(info, io.BytesIO(data))
        return archive.getvalue()

    def run_with_launcher(self, record, variant="baseline", backend="simulator", missing=None,
                          status="pass", run_error=False):
        output = self.root / "result"
        payloads = {"host.log": b"host result\n", "host-exit.txt": b"0\n" if status == "pass" else b"1\n",
                    "capture": self.capture(variant, backend, status)}
        seen = dict(staged=[])

        class Launcher:
            def __init__(self, artifact, **options):
                seen.update(artifact=artifact, options=options)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def stage(self, name):
                seen["staged"].append(Path(name).name)

            def run(self, command):
                seen["command"] = command
                if run_error:
                    raise RuntimeError("transport interrupted")
                return None

            def download_artifact(self, remote, path):
                if remote != missing:
                    Path(path).write_bytes(payloads[remote])
                return None  # SDK 2.10.0 wrapper intentionally returns no path.

        args = SimpleNamespace(artifact_record=record, variant=variant, backend=backend,
                               out=output, case="3:8:0", host_timeout=300)
        return appliance.run_program(args, self.root, Launcher), seen, output

    def test_compile_pins_sources_hosts_and_variant(self):
        record = self.compile_record("events", "hardware")
        saved, artifact = appliance.load_record(record, "hardware", "events", self.root)
        self.assertEqual(set(saved["host_sha256"]), {"run.py", "diagnostic.py"})
        self.assertIn("--params=OBSERVE:1", saved["compiler_args"])
        self.assertIn("--fabric-dims=762,1172", saved["compiler_args"])
        self.assertTrue(artifact.is_file())

    def test_wrong_backend_and_variant_rejected(self):
        record = self.compile_record()
        for backend, variant in (("hardware", "baseline"), ("simulator", "events")):
            with self.assertRaisesRegex(ValueError, "differs"):
                appliance.load_record(record, backend, variant, self.root)

    def test_changed_host_or_artifact_rejected(self):
        record = self.compile_record()
        host = self.root / "run.py"
        original = host.read_bytes()
        host.write_text("changed")
        with self.assertRaisesRegex(ValueError, "compiled host_sha256"):
            appliance.load_record(record, "simulator", "baseline", self.root)
        host.write_bytes(original)
        (record.parent / "artifact.tar.gz").write_bytes(b"changed artifact")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            appliance.load_record(record, "simulator", "baseline", self.root)

    def test_none_download_return_succeeds_and_sim_has_no_cmaddr(self):
        code, seen, output = self.run_with_launcher(self.compile_record())
        self.assertEqual(code, 0)
        self.assertNotIn("%CMADDR%", seen["command"])
        self.assertEqual(seen["staged"], ["run.py"])
        collection = json.loads((output / "collection.json").read_text())
        self.assertTrue(all(isinstance(path, str) for path in collection["downloaded"].values()))

    def test_hardware_stages_both_hosts_and_preserves_numeric_failure(self):
        code, seen, _ = self.run_with_launcher(self.compile_record("payload", "hardware"),
                                              "payload", "hardware", status="observed_numeric_fail")
        self.assertEqual(code, 1)
        self.assertIn("%CMADDR%", seen["command"])
        self.assertEqual(set(seen["staged"]), {"run.py", "diagnostic.py"})
        self.assertIn("--variant payload", seen["command"])

    def test_missing_download_is_collection_failure(self):
        with self.assertRaisesRegex(ValueError, "collection_failed"):
            self.run_with_launcher(self.compile_record(), missing="host-exit.txt")
        result = json.loads((self.root / "result/collection.json").read_text())
        self.assertEqual(set(result["errors"]), {"host-exit.txt"})

    def test_payload_difference_preserves_distinct_status(self):
        code, _, output = self.run_with_launcher(self.compile_record("payload"), "payload",
                                                 status="observed_payload_difference")
        self.assertEqual(code, 1)
        meta = appliance.capture_metadata(output / "capture")
        self.assertEqual(meta["status"], "observed_payload_difference")

    def test_transport_error_keeps_collected_files(self):
        with self.assertRaisesRegex(RuntimeError, "launcher failed"):
            self.run_with_launcher(self.compile_record(), run_error=True)
        self.assertTrue((self.root / "result/capture").is_file())
        self.assertTrue((self.root / "result/launcher-error.txt").is_file())

    def test_existing_output_is_not_reused(self):
        record = self.compile_record()
        self.run_with_launcher(record)
        with self.assertRaises(FileExistsError):
            self.run_with_launcher(record)

    def test_case_bounds_and_worker_timeout_quoting(self):
        self.assertEqual(appliance.parse_case("3:0:0"), "3:0:0")
        self.assertEqual(appliance.parse_case("3:1:0"), "3:1:0")
        for value in ("3:17:0", "31:2:0", "3:8:2", "3:8:0; echo BAD"):
            with self.assertRaises(Exception):
                appliance.parse_case(value)
        command = appliance.host_command("hardware", "events", "3:8:0", 300)
        self.assertIn("timeout --signal=INT --kill-after=15s 300s", command)

    def test_inspector_rejects_unclosed_api(self):
        with self.assertRaisesRegex(ValueError, "unclosed"):
            inspect_result.validate_calls([dict(index=0, api="stop", phase="enter")])

    def test_inspector_rejects_empty_pass_panels(self):
        calls = [dict(index=0, api="stop", phase=phase, stage="teardown", detail={})
                 for phase in ("enter", "exit")]
        with self.assertRaisesRegex(ValueError, "panel results"):
            inspect_result.validate_result(dict(status="pass", panels=[]), calls)

    def test_payload_difference_rejects_wrong_variant_and_structural_errors(self):
        calls = [dict(index=0, api="stop", phase=phase, stage="teardown", detail={})
                 for phase in ("enter", "exit")]
        for variant, errors in (("events", {"payload_stream": {}}),
                                ("payload", {"payload_stream": {}, "payload_marker": {}})):
            meta = dict(status="observed_payload_difference", variant=variant,
                        panels=[dict(panel=0, errors=errors)])
            with self.assertRaisesRegex(ValueError, "payload difference requires"):
                inspect_result.validate_result(meta, calls)


if __name__ == "__main__":
    unittest.main()
