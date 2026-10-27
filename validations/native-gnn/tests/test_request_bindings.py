"""Real frozen-case registration and dependency checks; no SDK or submissions."""
import copy
import json
from pathlib import Path
import unittest
from unittest import mock

import appliance as app
import saved_audit as common

ROOT = Path(__file__).resolve().parents[1]
REQUESTS = ROOT.parents[1] / "requests/native-gnn"
NEW = {"tank-layernorm-composition-20261027-r002": (17, 5, 6),
       "tank-indexed-cache-scale-20261027-r002": (55, 13, 21)}


class RequestBindingTests(unittest.TestCase):
    def test_new_requests_bind_all_cases_hosts_and_prior_compile_outputs(self):
        for ident, (count, builds_expected, runs_expected) in NEW.items():
            request = common.read_json(REQUESTS / (ident + ".json"))
            self.assertEqual(request["id"], ident)
            self.assertEqual(len(request["steps"]), count)
            compiled, outputs, run_count, audit_count, previous = {}, set(), 0, 0, None
            for step in request["steps"]:
                argv = step["args"]
                if step["kind"] == "appliance":
                    args = app.parser().parse_args(argv)
                    destination = str(args.out)
                    self.assertNotIn(destination, outputs)
                    outputs.add(destination)
                    if args.action == "compile":
                        pinned = app.identity(ROOT, args.case)
                        self.assertEqual(pinned["case_name"], args.case)
                        compiled[str(args.out / "native-build.json")] = args.case
                    else:
                        self.assertEqual(compiled[str(args.artifact_record)], args.case)
                        run_count += 1
                    previous = args
                else:
                    self.assertEqual(step["script"], "validations/native-gnn/saved_audit.py")
                    self.assertIsNotNone(previous)
                    self.assertEqual(previous.action, "run")
                    self.assertEqual(argv[0], str(previous.out / "capture-data"))
                    self.assertNotIn(argv[2], outputs)
                    outputs.add(argv[2])
                    audit_count += 1
                    previous = None
            self.assertEqual(len(compiled), builds_expected)
            self.assertEqual(run_count, runs_expected)
            self.assertEqual(audit_count, runs_expected)

    def test_indexed_replacement_preserves_original_step_contracts(self):
        old = common.read_json(REQUESTS / "tank-indexed-cache-scale-20261027-r001.json")
        new = common.read_json(REQUESTS / "tank-indexed-cache-scale-20261027-r002.json")
        self.assertEqual(old["steps"], new["steps"])
        self.assertNotEqual(old["id"], new["id"])

    def test_layernorm_continuation_omits_completed_full_runs(self):
        old = common.read_json(REQUESTS / "tank-layernorm-composition-20261027-r001.json")
        new = common.read_json(REQUESTS / "tank-layernorm-composition-20261027-r002.json")
        expected = old["steps"][:7] + old["steps"][21:]
        self.assertEqual(new["steps"], expected)

    def test_missing_package_registration_rejects_before_sdk(self):
        case = "e207-mixed-c8-r2-d1-indexed"
        package = common.read_json(ROOT / "package.json")
        self.assertIn(case, package["cases"])
        changed = copy.deepcopy(package)
        del changed["cases"][case]
        original = common.read_json
        def missing(path, *args, **kwargs):
            return changed if Path(path) == ROOT / "package.json" else original(path, *args, **kwargs)
        with mock.patch.object(common, "read_json", side_effect=missing), \
             mock.patch.object(app, "sdk_classes", side_effect=AssertionError("SDK forbidden")), \
             self.assertRaisesRegex(ValueError, "package case identity"):
            app.identity(ROOT, case)
