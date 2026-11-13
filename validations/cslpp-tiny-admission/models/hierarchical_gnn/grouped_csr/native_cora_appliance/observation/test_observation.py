"""Reject stale completion, invalid clocks and numerically wrong repeat results."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import run


class ProfileTests(unittest.TestCase):
    def raw(self, profile=1):
        return dict(clocks=np.tile(np.array([65530, 65535, 2, 994, 0, 3], np.uint32), (4, 1)),
                    stage_cycles=np.full((4, 14), profile, np.uint32),
                    stage_calls=np.full((4, 14), profile, np.uint32), stage_invalid=np.zeros((4, 1), np.uint32))

    def test_timestamp_word_carry_and_same_pe_sum(self):
        observed = run.decode_profile(self.raw(), 1)
        self.assertEqual(observed['local_interval_cycles']['per_pe'], [1000]*4)
        self.assertEqual(len(observed['stages']), 14)

    def test_reject_overflow_and_impossible_composition(self):
        for mutation in ('stage_invalid', 'stage_cycles', 'clocks'):
            raw = self.raw()
            raw[mutation][0, 0] = 10000 if mutation == 'stage_cycles' else 65536
            with self.assertRaises(Exception):
                run.decode_profile(raw, 1)

    def test_disabled_profile_is_empty(self):
        self.assertEqual(run.decode_profile(self.raw(0), 0)['stages'], {})
        with self.assertRaises(Exception):
            run.decode_profile(self.raw(), 0)

    def test_completion_requires_every_pe_and_fresh_epoch(self):
        value = np.full((4, 1), 2, np.uint32)
        run.completion_check(value, 2, 4)
        value[3] = 1
        with self.assertRaises(Exception):
            run.completion_check(value, 2, 4)


class RuntimeTests(unittest.TestCase):
    def exercise(self, stale=False, wrong=False):
        parameters = dict(C=2, R=2, L=1, K=2, T=16)
        expected = dict(pre=np.zeros((4, 16), np.float64), logits=np.tile(np.arange(7, dtype=np.float64), (4, 1)),
                        panel_sum=np.zeros((4, 32), np.float64), stats=np.tile(np.array([0x600D]+[0]*7, np.uint32), (4, 1)))
        outputs = {name: [value.shape[1], 'uint32' if name == 'stats' else 'float32'] for name, value in expected.items()}
        outputs.update(clocks=[6, 'uint16'], **run.EXTRA)
        manifest = dict(params=parameters, inputs=dict(x_val=[1, 'float32']), outputs=outputs)
        original = {name: np.ones((1,), np.float32) for name in ('x', 'src', 'dst', 'coeff', 'w0', 'w1')}
        original.update(full=False, valid=None)
        context = dict(nodes=np.arange(4).reshape(4, 1), logits=expected['logits'], original=original)
        arrays = dict(x_val=np.ones((4, 1), np.float32))
        instances = []

        class SDK:
            def __init__(self, *_args, **_kwargs):
                self.epoch, self.stops, self.events = 0, 0, []
                instances.append(self)
            def get_id(self, name): return name
            def load(self): pass
            def run(self): pass
            def stop(self): self.stops += 1
            def memcpy_h2d(self, name, *_args, **_kwargs): self.events.append(('upload', name))
            def launch(self, *_args, **_kwargs):
                self.epoch += 1
                self.events.append(('launch', self.epoch))
            def memcpy_d2h(self, destination, name, *_args, **_kwargs):
                if name == 'completion':
                    destination[:] = min(self.epoch, 1) if stale else self.epoch
                elif name in expected:
                    destination[:] = expected[name].ravel()
                    if wrong and self.epoch == 2 and name == 'logits': destination[0] += 1
                elif name == 'x_val': destination[:] = 1
                elif name == 'clocks': destination[:] = np.tile([10, 0, 0, 1010, 0, 0], 4)
                else: destination[:] = 0

        class DataType:
            MEMCPY_16BIT, MEMCPY_32BIT = 0, 1
        class Order:
            ROW_MAJOR = 0

        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            for name, result in (('check_abi', None), ('compiled_inventory', {}), ('admit_compiled_inventory', None),
                                 ('load', manifest), ('independent_expected', (expected, context)),
                                 ('load_npz', arrays), ('sha', 'a'*64), ('named_bytes', 1)):
                stack.enter_context(patch.object(run.audit, name, return_value=result))
            stack.enter_context(patch.object(run, 'manifest_for', return_value=manifest))
            stack.enter_context(patch.object(run, 'sources', return_value={}))
            output = Path(temp)/'capture'
            if stale or wrong:
                with self.assertRaises(Exception):
                    run.capture(temp, temp, output, 0, 1, 2, sdk=(SDK, DataType, Order))
                record = run.audit.read_json(output/'execution.json', limit=run.audit.INVENTORY_JSON_LIMIT)
                self.assertEqual(record['status'], 'FAILED')
            else:
                record = run.capture(temp, temp, output, 0, 1, 2, sdk=(SDK, DataType, Order))
                self.assertEqual(len(record['iterations']), 3)
                self.assertEqual([it['warmup'] for it in record['iterations']], [True, False, False])
                self.assertTrue(all(it['numerics']['status'] == 'PASS' for it in record['iterations']))
            self.assertEqual(instances[0].stops, 1)
            self.assertEqual(instances[0].events[0], ('upload', 'x_val'))
            self.assertTrue(all(event[0] == 'launch' for event in instances[0].events[1:]))

    def test_same_runtime_same_input_three_epochs(self): self.exercise()
    def test_stale_second_epoch_fails_and_stops(self): self.exercise(stale=True)
    def test_wrong_second_logits_fails_and_stops(self): self.exercise(wrong=True)


class SimulationFailureTests(unittest.TestCase):
    def test_final_build_identity_failure_cannot_retain_pass(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            case, output = root/'case', root/'out'
            (case/'observation-0/build').mkdir(parents=True)
            record = dict(inventory={})
            def dedicated(path, fresh=False):
                path = Path(path)
                if fresh: path.mkdir(parents=True)
                return path
            stack.enter_context(patch.object(run, 'dedicated', side_effect=dedicated))
            stack.enter_context(patch.object(run, 'compiled', side_effect=[record, ValueError('changed ELF')]))
            stack.enter_context(patch.object(run.audit, 'compiled_inventory', return_value={}))
            stack.enter_context(patch.object(run.audit, 'sha', return_value='a'*64))
            stack.enter_context(patch.object(run.audit, 'read_json', return_value=dict(status='PASS')))
            stack.enter_context(patch.object(run.subprocess, 'run', return_value=type('Process', (), {'returncode': 0})()))
            with self.assertRaisesRegex(ValueError, 'changed ELF'):
                run.simulate(case, output, 0)
            import json
            self.assertEqual(json.loads((output/'result.json').read_text())['status'], 'FAILED')

    def test_raw_archive_entry_budget_before_sdk_import(self):
        with self.assertRaisesRegex(ValueError, 'archive inventory'):
            run.capture('.', '.', '.', 0, 10, 30)


if __name__ == '__main__':
    unittest.main()
