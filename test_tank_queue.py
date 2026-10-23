"""Cross-campaign exclusion tests; no SDK client or tmux is launched."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import tank_appliance as a
import tank_campaign as c


class QueueTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.old = self.root / "mesh-followup-test"
        self.old.mkdir()
        cohorts = []
        for name in ("h4-s2", "h8-s2"):
            child = self.old / "cohorts" / name
            (child / "driver").mkdir(parents=True)
            (child / "driver/source.py").write_text("# frozen child")
            a.write_json(child / "plan.json", dict(builds=[dict(id="b")], runs=[dict(id="r")],
                source_sha256={"source.py": a.digest(child / "driver/source.py")}))
            a.write_json(child / "state.json", dict(status="complete", plan_sha256=a.digest(child / "plan.json"),
                builds={"b": dict(status="pass")}, runs={"r": dict(status="pass")}))
            cohorts.append(dict(id=name, path="cohorts/"+name, plan_sha256=a.digest(child / "plan.json")))
        (self.old / "driver").mkdir()
        (self.old / "driver/source.py").write_text("# frozen parent")
        a.write_json(self.old / "plan.json", dict(kind="mesh-followup", cohorts=cohorts,
            source_sha256={"source.py": a.digest(self.old / "driver/source.py")}))
        a.write_json(self.old / "state.json", dict(status="complete", plan_sha256=a.digest(self.old / "plan.json"),
            cohorts={x["id"]: dict(status="complete") for x in cohorts}))
        self.entry = dict(path=str(self.old), plan_sha256=a.digest(self.old / "plan.json"))

    def set_status(self, status):
        state = a.read_json(self.old / "state.json")
        state["status"] = status
        a.write_json(self.old / "state.json", state)

    def test_detects_only_unfinished_followups_unless_explicit(self):
        self.assertEqual(c.predecessors(self.root, []), [])
        self.set_status("running")
        self.assertEqual(c.predecessors(self.root, []), [self.entry])
        self.set_status("complete")
        self.assertEqual(c.predecessors(self.root, [self.old]), [self.entry])

    def test_active_predecessor_waits_and_plan_changes_reject(self):
        with c.output_lock(self.old):
            self.assertFalse(c.predecessor_ready(self.entry))
        self.assertTrue(c.predecessor_ready(self.entry))
        (self.old / "plan.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "plan changed"):
            c.predecessor_ready(self.entry)

    def test_holds_parent_child_and_shared_locks_through_campaign(self):
        plan = dict(operator_root=str(self.root), after=[self.entry])
        with c.hardware_queue(self.root, plan):
            for path in [self.old, self.old / "cohorts/h4-s2", self.old / "cohorts/h8-s2"]:
                with self.assertRaisesRegex(ValueError, "another controller"):
                    with c.output_lock(path):
                        self.fail("predecessor lock released too soon")
            with self.assertRaisesRegex(ValueError, "another controller"):
                with c.output_lock(self.root, ".tank-hardware.lock"):
                    self.fail("new campaigns overlap")
        with ExitStack() as stack:
            for path in [self.old, self.old / "cohorts/h4-s2", self.old / "cohorts/h8-s2"]:
                stack.enter_context(c.output_lock(path))

    def test_wait_then_finish_without_submitting_a_job(self):
        self.set_status("running")
        with mock.patch.object(c.time, "sleep", side_effect=lambda _: self.set_status("complete")) as pause:
            with c.hardware_queue(self.root, dict(operator_root=str(self.root), after=[self.entry])):
                self.assertEqual(pause.call_count, 1)

    def test_uncertain_predecessor_and_changed_source_reject(self):
        child = self.old / "cohorts/h8-s2"
        state = a.read_json(child / "state.json")
        state["runs"]["r"]["remote_stop_unconfirmed"] = True
        a.write_json(child / "state.json", state)
        with self.assertRaisesRegex(ValueError, "unresolved jobs"):
            c.predecessor_ready(self.entry)
        state["runs"]["r"].pop("remote_stop_unconfirmed")
        a.write_json(child / "state.json", state)
        (self.old / "driver/source.py").write_text("# changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            c.predecessor_ready(self.entry)


if __name__ == "__main__":
    unittest.main()
