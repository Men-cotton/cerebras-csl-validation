"""Persistent queue failures and frozen-controller resume; no SDK or exec calls."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import tank_appliance as appliance
import tank_campaign as campaign


class ExecHandoff(BaseException):
    """os.execv never returns; this sentinel models the process handoff."""


class QueueExtraTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "campaign"
        (self.root / "driver").mkdir(parents=True)
        self.current = self.base / "current"
        self.current.mkdir()
        # Deliberately opaque old schema: the current load/verify implementation
        # must never inspect it before dispatching resume to its frozen source.
        pins = {}
        for name in ("tank_campaign.py", "tank_audit.py"):
            path = self.root / "driver" / name
            path.write_text("# Frozen previous controller: " + name + "\n")
            pins[name] = appliance.digest(path)
        self.plan = dict(source_sha256=pins, previous_schema="opaque-to-current")
        self.state = dict(status="pending", plan_sha256="fixture-pin",
                          builds={"done": dict(status="pass"), "later": dict(status="pending")},
                          runs={"r001": dict(status="pending")})
        appliance.write_json(self.root / "plan.json", self.plan)
        appliance.write_json(self.root / "state.json", self.state)

    def foreground_failure(self, error):
        @contextmanager
        def unavailable_queue(root, plan):
            self.assertEqual(root, self.root)
            self.assertEqual(plan, self.plan)
            raise error
            yield  # This remains a context manager whose entry raises.

        previous_signal = object()
        with mock.patch.object(campaign, "load", return_value=(self.plan, self.state)), \
             mock.patch.object(campaign, "hardware_queue", side_effect=unavailable_queue), \
             mock.patch.object(campaign, "run_campaign") as run, \
             mock.patch.object(campaign, "bundle", return_value=self.root.with_suffix(".zip")) as bundle, \
             mock.patch.object(campaign.signal, "signal", return_value=previous_signal) as signals, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = campaign.foreground(self.root)
        run.assert_not_called()
        bundle.assert_called_once_with(self.root)
        self.assertEqual(signals.call_count, 2)
        self.assertIs(signals.call_args.args[1], previous_signal)
        # The failed queue entry must release ownership and preserve every
        # prior attempt. A resumed controller may then consider pending work.
        with campaign.output_lock(self.root):
            pass
        state = appliance.read_json(self.root / "state.json")
        launcher = appliance.read_json(self.root / "launcher.json")
        self.assertEqual(state["builds"], self.state["builds"])
        self.assertEqual(state["runs"], self.state["runs"])
        self.assertIn("finished_at", state)
        self.assertIn("finished_at", launcher)
        self.assertTrue((self.root / "driver.log").is_file())
        return code, state, launcher

    def test_queue_entry_error_persists_blocked_state_without_starting_jobs(self):
        code, state, launcher = self.foreground_failure(ValueError("predecessor has unresolved jobs"))
        self.assertEqual(code, 2)
        self.assertEqual(state["status"], "blocked")
        self.assertEqual(state["exit_code"], 2)
        self.assertEqual(state["error"], "predecessor has unresolved jobs")
        self.assertEqual(launcher["status"], "completed_with_failures")
        self.assertEqual(launcher["exit_code"], 2)

    def test_queue_keyboard_interrupt_persists_interrupted_state_and_exit_130(self):
        code, state, launcher = self.foreground_failure(KeyboardInterrupt())
        self.assertEqual(code, 130)
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(state["exit_code"], 130)
        self.assertIn("waiting for hardware queue", state["error"])
        self.assertEqual(launcher["status"], "interrupted")
        self.assertEqual(launcher["exit_code"], 130)

    def test_resume_executes_frozen_controller_before_current_schema_load(self):
        with mock.patch.object(campaign, "ROOT", self.current), \
             mock.patch.object(campaign, "prepare") as prepare, \
             mock.patch.object(campaign, "load") as load, \
             mock.patch.object(campaign.tank_audit, "verify_package") as verify, \
             mock.patch.object(campaign.os, "execv", side_effect=ExecHandoff()) as execute:
            with self.assertRaises(ExecHandoff):
                campaign.main(["--resume", str(self.root), "--foreground"])
        prepare.assert_not_called()
        load.assert_not_called()
        verify.assert_not_called()
        execute.assert_called_once_with(sys.executable,
            [sys.executable, "-B", str(self.root / "driver/tank_campaign.py"), "--resume", str(self.root), "--foreground"])

    def test_changed_frozen_source_blocks_resume_before_handoff_or_current_load(self):
        (self.root / "driver/tank_audit.py").write_text("# Changed after the frozen plan\n")
        errors = io.StringIO()
        with mock.patch.object(campaign, "ROOT", self.current), \
             mock.patch.object(campaign, "prepare") as prepare, \
             mock.patch.object(campaign, "load") as load, \
             mock.patch.object(campaign.tank_audit, "verify_package") as verify, \
             mock.patch.object(campaign.os, "execv") as execute, redirect_stderr(errors):
            code = campaign.main(["--resume", str(self.root)])
        self.assertEqual(code, 2)
        self.assertIn("frozen resume source changed", errors.getvalue())
        prepare.assert_not_called()
        load.assert_not_called()
        verify.assert_not_called()
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
