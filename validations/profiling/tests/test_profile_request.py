"""Check the published request against the actual entrypoint parsers without SDK calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import profile_appliance as appliance
import profile_compare as comparison


class RequestTests(unittest.TestCase):
    def test_manifest_arguments_and_capture_paths_match_entrypoints(self):
        root = Path(__file__).resolve().parents[3]
        request = json.loads((root / "requests/profiling/tank29-n4-timestamps-20261023.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            outputs, flags = [], []
            for step in request["steps"]:
                argv = [word.replace("{output}", temporary) for word in step["args"]]
                if step["kind"] == "appliance":
                    self.assertEqual(step["script"], "validations/profiling/profile_appliance.py")
                    args = appliance.parser().parse_args(argv)
                    self.assertEqual(args.backend, "hardware")
                    if args.action == "compile":
                        build = args.out / "profile-build.json"
                    else:
                        self.assertEqual(args.artifact_record, build)
                        outputs.append(args.out / "capture-data")
                        flags.append(args.flag)
                else:
                    self.assertEqual(step["script"], "validations/profiling/profile_compare.py")
                    with patch.object(comparison, "compare", return_value={"status": "pass"}) as compare:
                        self.assertEqual(comparison.main(argv), 0)
                        compare.assert_called_once_with(outputs)
            self.assertEqual(flags, [0, 1, 1, 0])
