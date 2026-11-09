"""Portable selective-case rejection tests; no repository model imports at export."""
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    from .test_appliance import HERE, PACKAGE, common, runner, fake_build, fake_sdk
except ImportError:
    from test_appliance import HERE, PACKAGE, common, runner, fake_build, fake_sdk


class SelectivePackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="e229-portable-")
        cls.root = Path(cls.temporary.name)
        if (HERE/"cases").exists():
            cases = [case for case in sorted((HERE/"cases").iterdir())
                     if common.read_json(case/"case.json")["params"].get("UNICAST") == 1]
            cls.case = min(cases, key=lambda case: common.read_json(case/"case.json")["nodes"])
        else:
            driver = importlib.import_module(PACKAGE+".run")
            cls.case = cls.root/"tiny"
            with mock.patch.object(driver, "RUNS", cls.root):
                driver.prepare(cls.case, columns=2, rows=2, tile=16,
                               transport="selective", tiny_demand=True)
        cls.manifest = common.load(cls.case)
        cls.build = cls.root/"build"
        fake_build(cls.build, cls.manifest)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_capture_and_exact_transport_counter_rejection(self):
        output = self.root/"capture"
        sdk, handles = fake_sdk(self.case)
        runner.main(self.case, self.build, output, "simulator", sdk=sdk)
        result = common.audit(self.case, output)
        self.assertEqual(result["numerical_status"], "PASS")
        self.assertEqual(handles[0].stops, 1)
        raw = common.load_npz(output/"raw.npz")
        raw["transport_stats"][0, 0] += np.uint32(1)
        np.savez_compressed(output/"raw.npz", **raw)
        execution = common.read_json(output/"execution.json")
        execution["raw_sha256"] = common.sha(output/"raw.npz")
        common.write_json(output/"execution.json", execution)
        with self.assertRaisesRegex(ValueError, "transport_stats"):
            common.audit(self.case, output)

    def test_frozen_directory_corruption_rejected(self):
        arrays = common.load_npz(self.case/"inputs.npz")
        nodes = common.load_npz(self.case/"node_ids.npz")["nodes"]
        original = common.native_inputs(self.case, self.manifest)
        for name in ("send_ptr", "send_slots", "recv_lengths"):
            bad = {key: value.copy() for key, value in arrays.items()}
            bad[name][0, 0] += np.uint16(1)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "director"):
                common.decode_uploads(self.manifest["params"], bad, nodes, original)


if __name__ == "__main__":
    unittest.main()
