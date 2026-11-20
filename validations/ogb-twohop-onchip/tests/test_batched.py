import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

BATCHED = Path(__file__).resolve().parents[1] / "batched"
sys.path.insert(0, str(BATCHED))
spec = importlib.util.spec_from_file_location(
    "batched_controls", BATCHED / "appliance.py"
)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class BatchedControls(unittest.TestCase):
    def test_registered_pairs(self):
        for name in sorted(app.CASES):
            case = BATCHED / "cases" / name
            path = case / "release.json"
            with self.subTest(case=case.name):
                _, refs = app.checked_case(case.name, app.sha(path))
                self.assertEqual({a.shape[0] for a in refs["logits"].values()}, {32})
                with self.assertRaisesRegex(ValueError, "frozen batch release"):
                    app.checked_case(case.name, "0" * 64)

    def test_changed_selected_roots(self):
        for release in sorted((BATCHED / "cases").glob("*/release.json")):
            case = release.parent
            with tempfile.TemporaryDirectory() as tmp:
                bad = Path(tmp)
                for name in ("plan.json", "one", "window.npz"):
                    (bad / name).symlink_to(
                        case / name, target_is_directory=(case / name).is_dir()
                    )
                value = app.read(case / "batch.json")
                value["original_targets"] = value["original_targets"][::-1]
                (bad / "batch.json").write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "selected batch"):
                    app.verify_window(bad)
