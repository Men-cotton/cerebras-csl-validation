"""Schedule contracts only; these tests submit no jobs and import no SDK."""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_plan


class PlanTests(unittest.TestCase):
    def test_standard_inventory_and_fresh_runtime_units(self):
        plan = clock_plan.make_plan()
        self.assertEqual(plan["schema"], 1)
        self.assertEqual(len(plan["builds"]), 4)
        self.assertEqual(len(plan["runs"]), 80)
        self.assertEqual(plan["expected_fresh_runtimes"], 80)
        self.assertEqual(plan["generations_per_runtime"], 3)
        self.assertEqual([r["id"] for r in plan["runs"]], [f"r{i:03d}" for i in range(1, 81)])
        self.assertEqual(len({r["id"] for r in plan["runs"]}), 80)
        counts = Counter(r["cohort"] for r in plan["runs"])
        self.assertEqual(counts, {
            "granularity": 18, "payload": 8, "geometry": 8, "piece-order": 8,
            "submission-wait": 8, "markers": 8, "empty-markers": 4,
            "marked-pause": 4, "host-pmu": 8, "internal-work": 4,
            "reverse-marked-diagnostic": 2,
        })
        self.assertEqual({b["id"] for b in plan["builds"]}, {r["build"] for r in plan["runs"]})
        self.assertTrue(all(r["config"]["steps"] == 3 for r in plan["runs"]))

    def test_six_permutations_and_position_balance(self):
        records = [r for r in clock_plan.make_plan()["runs"] if r["cohort"] == "granularity"]
        blocks = defaultdict(list)
        for record in records:
            blocks[record["block"]].append(record)
        orders = {tuple(r["arm"] for r in group) for group in blocks.values()}
        self.assertEqual(orders, {
            ("rectangle", "strips", "tiles"), ("rectangle", "tiles", "strips"),
            ("strips", "rectangle", "tiles"), ("strips", "tiles", "rectangle"),
            ("tiles", "rectangle", "strips"), ("tiles", "strips", "rectangle"),
        })
        for position in (1, 2, 3):
            self.assertEqual(Counter(r["arm"] for r in records if r["position"] == position),
                             dict(rectangle=2, strips=2, tiles=2))
        self.assertEqual({r["timed_h2d_bytes_per_generation"] for r in records}, {1024})
        self.assertEqual({r["arm"]: r["h2d_calls_per_generation"] for r in records},
                         dict(rectangle=1, strips=2, tiles=4))

    def test_pair_order_and_matched_transfer_work(self):
        plan = clock_plan.make_plan()
        for cohort in plan["cohorts"]:
            if cohort["order"] not in ("AB,BA,BA,AB", "AB,BA"):
                continue
            records = [r for r in plan["runs"] if r["cohort"] == cohort["id"]]
            a, b = records[0]["arm"], records[1]["arm"]
            self.assertNotEqual(a, b)
            expected = [a, b, b, a, b, a, a, b][:len(records)]
            self.assertEqual([r["arm"] for r in records], expected, cohort["id"])
            self.assertEqual([r["position"] for r in records], [1, 2] * cohort["blocks"])
            self.assertEqual([r["block"] for r in records],
                             [i for i in range(1, cohort["blocks"] + 1) for _ in range(2)])
            self.assertEqual(len({r["timed_h2d_bytes_per_generation"] for r in records}), 1,
                             cohort["id"])
        geometry = [r for r in plan["runs"] if r["cohort"] == "geometry"]
        self.assertEqual({r["h2d_calls_per_generation"] for r in geometry}, {8})
        self.assertEqual({r["timed_h2d_bytes_per_generation"] for r in geometry}, {2048})

    def test_observer_controls_and_deliberate_pause(self):
        plan = clock_plan.make_plan()
        groups = defaultdict(list)
        for run in plan["runs"]:
            groups[run["cohort"]].append(run["config"])
        for cohort, field in (("piece-order", "piece_order"), ("submission-wait", "transfer"),
                              ("markers", "markers"), ("empty-markers", "markers"),
                              ("marked-pause", "pause_seconds"), ("internal-work", "work")):
            a, b = copy.deepcopy(groups[cohort][:2])
            self.assertNotEqual(a.pop(field), b.pop(field), cohort)
            self.assertEqual(a, b, cohort)
        self.assertTrue(all(c["kind"] == "empty" for c in groups["empty-markers"]))
        self.assertTrue(all(c["markers"] and c["pause_at"] == "between_h2d"
                            for c in groups["marked-pause"]))
        self.assertEqual({c["pause_seconds"] for c in groups["marked-pause"]}, {0.0, 0.01})
        self.assertEqual({(c["host_pmu"], c["pmu_controls"]) for c in groups["host-pmu"]},
                         {(False, 0), (True, 3)})

    def test_smoke_has_all_required_paths(self):
        plan = clock_plan.make_plan(profile="smoke")
        self.assertEqual(len(plan["builds"]), 1)
        self.assertEqual(len(plan["runs"]), 4)
        internal, rectangle, marked, pmu = [r["config"] for r in plan["runs"]]
        self.assertEqual((internal["kind"], internal["start"]), ("internal", "blocking"))
        self.assertEqual((rectangle["kind"], rectangle["arm"]), ("transfer", "rectangle"))
        self.assertEqual((marked["arm"], marked["piece_order"], marked["markers"]),
                         ("tiles", "reverse", True))
        self.assertEqual((pmu["arm"], pmu["host_pmu"], pmu["pmu_controls"]),
                         ("rectangle", True, 3))

    def test_deterministic_serializable_and_no_shared_mutable_results(self):
        original = clock_plan.make_plan()
        repeated = clock_plan.make_plan()
        self.assertEqual(json.dumps(original, sort_keys=True), json.dumps(repeated, sort_keys=True))
        original["runs"][0]["config"]["arm"] = "changed"
        original["builds"][0]["params"]["W"] = 999
        self.assertEqual(clock_plan.make_plan(), repeated)
        self.assertEqual(repeated["runs"][1]["config"]["arm"], "strips")

    def test_options_and_evidence_based_estimates(self):
        plan = clock_plan.make_plan(backend="simulator", namespace="existing-namespace",
                                    fabric_dims="012,009", fabric_offsets="04,01", host_timeout=123)
        self.assertEqual((plan["backend"], plan["namespace"], plan["host_timeout"]),
                         ("simulator", "existing-namespace", 123))
        self.assertEqual((plan["fabric_dims"], plan["fabric_offsets"]), ("12,9", "4,1"))
        estimates = plan["estimates"]
        self.assertAlmostEqual(estimates["hardware_wall_seconds_low"], 5794.356)
        self.assertAlmostEqual(estimates["hardware_wall_seconds_high"], 8197.796)
        self.assertIn("Not a timeout or a simulator estimate", estimates["scope"])
        self.assertGreater(estimates["retained_bytes_approx"], 800000000)

    def test_rejects_invalid_admission_options(self):
        for arguments in (
            {"profile": "all"}, {"backend": "unsupported"}, {"namespace": ""},
            {"namespace": "name\n"}, {"namespace": 3},
            {"host_timeout": True}, {"host_timeout": 0}, {"host_timeout": 1801},
            {"fabric_dims": "0,10"}, {"fabric_dims": "8,4"}, {"fabric_dims": "x,10"},
            {"fabric_dims": "10,10,10"}, {"fabric_dims": [10, 10]},
            {"fabric_offsets": "-1,0"}, {"fabric_offsets": "0, 1"},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                clock_plan.make_plan(**arguments)


if __name__ == "__main__":
    unittest.main()
