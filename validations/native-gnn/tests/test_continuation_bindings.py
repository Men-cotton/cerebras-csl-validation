"""Frozen E208/E209 request admission; no SDK or remote actions.

Install beside the native-gnn adapter tests. Inspect the actual published-tree
package registrations rather than a separate generator's in-memory registry.
"""
import copy
import json
from pathlib import Path
import sys
import unittest

FAMILY = Path(__file__).resolve().parents[1]
REPOSITORY = FAMILY.parents[1]
if str(FAMILY) not in sys.path:
    sys.path.insert(0, str(FAMILY))
import appliance

SPECS = {
    "tank-adjacent-owner-phases-20261028-r001": (58, 14, 22, 22, 0),
    "tank-adjacent-owner-phases-20261028-r002": (32, 8, 12, 12, 0),
    "tank28-resident-root-statistics-20261028-r001": (24, 4, 8, 8, 4),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def option(step, key):
    args = step["args"]
    require(args.count(key) == 1 and args.index(key)+1 < len(args), "one value for " + key)
    return args[args.index(key)+1]


def inspect_request(request, expected):
    steps = request["steps"]
    require(request["schema"] == 1 and len(steps) == expected[0], "request schema/step count")
    require(len({step["name"] for step in steps}) == len(steps), "unique step names")
    outputs, compiles, runs, audits, comparisons = set(), {}, {}, set(), []
    cases = set()
    for index, step in enumerate(steps):
        require(type(step["timeout_seconds"]) is int and 0 < step["timeout_seconds"] <= 1200,
                "bounded foreground step")
        output = option(step, "--out")
        require(output.startswith("{output}/") and output not in outputs, "unique output destination")
        outputs.add(output)
        script = step["script"]
        if script == "validations/native-gnn/appliance.py":
            require(step["kind"] == "appliance" and option(step, "--backend") == "hardware", "appliance backend")
            case = option(step, "--case")
            cases.add(case)
            if step["args"][0] == "compile":
                require(case not in compiles and option(step, "--compile-workers") == "1", "unique serial compile")
                compiles[case] = output + "/native-build.json"
            else:
                require(step["args"][0] == "run" and case in compiles, "run requires prior own compile")
                require(option(step, "--artifact-record") == compiles[case], "run uses own compiled case")
                require(option(step, "--host-timeout") == "300", "bounded worker")
                capture = output + "/capture-data"
                require(capture not in runs, "fresh runtime capture")
                runs[capture] = case
                require(index+1 < len(steps) and steps[index+1]["script"] == "validations/native-gnn/saved_audit.py"
                        and steps[index+1]["args"][0] == capture, "immediate audit of this capture")
        elif script == "validations/native-gnn/saved_audit.py":
            require(step["kind"] == "host", "host numerical audit")
            capture = step["args"][0]
            require(capture in runs and capture not in audits and index > 0
                    and steps[index-1]["script"] == "validations/native-gnn/appliance.py"
                    and steps[index-1]["args"][0] == "run"
                    and option(steps[index-1], "--out") + "/capture-data" == capture,
                    "audit binds immediately preceding runtime")
            audits.add(capture)
        else:
            require(step["kind"] == "host" and script in
                    ("validations/native-gnn/compare_rollouts.py", "validations/native-gnn/compare_statistics.py"),
                    "registered comparison entrypoint")
            pair = step["args"][:2]
            require(len(pair) == 2 and pair[0] != pair[1] and all(capture in audits for capture in pair),
                    "comparison consumes two already audited captures")
            comparisons.append((script, tuple(pair)))
    require((len(steps), len(compiles), len(runs), len(audits), len(comparisons)) == expected,
            "compile/runtime/audit/comparison census")
    require(set(runs) == audits and set(compiles) == cases == set(runs.values()), "all runtimes audited and cases compiled")
    return cases, comparisons


class ContinuationBindingTests(unittest.TestCase):
    def load_request(self, name):
        path = REPOSITORY / "requests/native-gnn" / (name + ".json")
        request = json.loads(path.read_text())
        self.assertEqual(request["id"], name)
        return request

    def test_real_requests_have_prior_compile_immediate_audit_and_unique_outputs(self):
        for name, counts in SPECS.items():
            with self.subTest(request=name):
                cases, comparisons = inspect_request(self.load_request(name), counts)
                self.assertEqual(len(cases), counts[1])
                if counts[-1]:
                    self.assertEqual(sum(path.endswith("compare_rollouts.py") for path, _ in comparisons), 1)
                    self.assertEqual(sum(path.endswith("compare_statistics.py") for path, _ in comparisons), 3)

    def test_actual_adapter_admits_every_referenced_case_registration(self):
        package = json.loads((FAMILY / "package.json").read_text())
        references = set()
        for name, counts in SPECS.items():
            references.update(inspect_request(self.load_request(name), counts)[0])
        self.assertEqual(len(references), 18)
        for name in sorted(references):
            with self.subTest(case=name):
                admitted = appliance.identity(FAMILY, name)
                self.assertEqual(admitted["case_name"], name)
                self.assertEqual(admitted["case_sha256"], package["cases"][name])

    def test_cross_case_artifact_reference_rejected(self):
        name = next(iter(SPECS))
        request = copy.deepcopy(self.load_request(name))
        run = next(step for step in request["steps"] if step["args"][0] == "run")
        run["args"][run["args"].index("--artifact-record")+1] = "{output}/compile-another-case/native-build.json"
        with self.assertRaisesRegex(ValueError, "own compiled case"):
            inspect_request(request, SPECS[name])

    def test_delayed_audit_rejected(self):
        name = next(iter(SPECS))
        request = copy.deepcopy(self.load_request(name))
        steps = request["steps"]
        index = next(i for i, step in enumerate(steps) if step["args"][0] == "run")
        steps[index+1], steps[index+2] = steps[index+2], steps[index+1]
        with self.assertRaisesRegex(ValueError, "immediate audit"):
            inspect_request(request, SPECS[name])

    def test_comparison_cannot_reference_future_or_absent_capture(self):
        name = "tank28-resident-root-statistics-20261028-r001"
        request = copy.deepcopy(self.load_request(name))
        step = next(step for step in request["steps"] if step["script"].endswith("compare_statistics.py"))
        step["args"][1] = "{output}/future-runtime/capture-data"
        with self.assertRaisesRegex(ValueError, "already audited"):
            inspect_request(request, SPECS[name])

    def test_output_reuse_rejected(self):
        name = next(iter(SPECS))
        request = copy.deepcopy(self.load_request(name))
        first, second = request["steps"][:2]
        second["args"][second["args"].index("--out")+1] = option(first, "--out")
        with self.assertRaisesRegex(ValueError, "unique output"):
            inspect_request(request, SPECS[name])


if __name__ == "__main__":
    unittest.main()
