"""Exercise completed scheduling and independent capture admission without the SDK."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import driver


class MemcpyDataType:
    MEMCPY_32BIT = 'u32'


class MemcpyOrder:
    ROW_MAJOR = 'row-major'


def fake_runtime(clock, *, corrupt_reply=False, corrupt_readback=False, permute_reply=False,
                 fail_h2d=None, fail_stop=False):
    instances = []

    class Runtime:
        def __init__(self, build, **kwargs):
            params = json.loads((Path(build)/'out.json').read_text())['params']
            self.width, self.height = int(params['WIDTH']), int(params['HEIGHT'])
            self.pes, self.ack = self.width*self.height, int(params['REPLY_ACK']) == 1
            self.pending = self.stored = self.config = None
            self.events, self.h2d_count = [], 0
            self.loaded = self.started = self.stopped = False
            self.cwd, self.options = Path.cwd(), kwargs
            instances.append(self)

        def load(self):
            self.loaded = True
            self.events.append(('load',))

        def run(self):
            assert self.loaded
            self.started = True
            self.events.append(('run',))

        def get_id(self, name):
            return name

        def launch(self, name, words, *, nonblock):
            assert not clock['active'] and name == 'configure' and nonblock and self.pending is None
            self.config = int(words)
            self.events.append(('configure', self.config))

        def input(self, mode, symbol, source, x, y, width, height, count, **kwargs):
            self.h2d_count += 1
            if self.h2d_count == fail_h2d:
                raise RuntimeError('injected H2D failure')
            assert clock['active'] and self.started and self.pending is None
            assert symbol == 0 and (x, y, width, height) == (0, 0, self.width, self.height)
            assert count == self.config+1 and source.dtype == np.uint32 and source.ndim == 1
            assert kwargs == dict(streaming=True, nonblock=True,
                                  data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR)
            if mode == 'row-bcast':
                full = np.repeat(source.reshape(height, 1, count), width, axis=1)
            elif mode == 'col-bcast':
                full = np.repeat(source.reshape(1, width, count), height, axis=0)
            else:
                full = source.reshape(height, width, count).copy()
            full = full.reshape(self.pes, count)
            # Simulate PE control stamping directly from the actual submitted data.
            for pe in range(self.pes):
                full[pe, 0] ^= np.uint32((pe*0x85EBCA6B) & 0xffffffff)
            self.stored = full.reshape(-1)
            self.pending = full[:, 0].copy() if self.ack else self.stored.copy()
            self.events.append((mode, width, height, count, source.size))

        def memcpy_h2d(self, *args, **kwargs):
            self.input('unicast', *args, **kwargs)

        def memcpy_h2d_rowbcast(self, *args, **kwargs):
            self.input('row-bcast', *args, **kwargs)

        def memcpy_h2d_colbcast(self, *args, **kwargs):
            self.input('col-bcast', *args, **kwargs)

        def memcpy_d2h(self, destination, symbol, x, y, width, height, count, **kwargs):
            assert (x, y, width, height) == (0, 0, self.width, self.height)
            assert kwargs.get('nonblock') is False
            if kwargs.get('streaming'):
                assert clock['active'] and symbol == 1 and self.pending is not None
                assert count == (1 if self.ack else self.config+1)
                np.copyto(destination, self.pending)
                if permute_reply:
                    destination[:] = destination[::-1]
                if corrupt_reply:
                    destination[-1] ^= np.uint32(1)
                self.pending = None
                self.events.append(('reply', count))
            elif symbol == 'frame_data':
                assert not clock['active'] and self.ack and self.pending is None
                assert count == self.config+1 and self.stored is not None
                np.copyto(destination, self.stored)
                if corrupt_readback:
                    destination[-1] ^= np.uint32(1)
                self.events.append(('readback', count))
            else:
                assert not clock['active'] and count == 1 and symbol in ('config_words', 'status')
                destination.fill(self.config if symbol == 'config_words' else 0)
                self.events.append(('configuration-read', symbol))

        def stop(self):
            self.stopped = True
            self.events.append(('stop',))
            if fail_stop:
                raise RuntimeError('injected stop failure')

    return (Runtime, MemcpyDataType, MemcpyOrder), instances


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.serial = 0

    def build(self, case='smoke-echo'):
        self.serial += 1
        folder = self.root/f'build-{self.serial}'
        (folder/'bin').mkdir(parents=True)
        plan = driver.oracle.plan(case, smoke=case.startswith('smoke-'))
        driver.write(folder/'out.json', {'params': driver.oracle.compiler_params(plan)})
        (folder/'bin/out_0_0.elf').write_bytes(b'not executed; mock program identity')
        (folder/'bin/out_rpc.json').write_text('{}')
        self.channel_metadata(folder, plan)
        return folder, plan

    def channel_metadata(self, folder, plan, heights=None):
        if heights is None:
            q, r = divmod(plan['height'], plan['channels'])
            heights = [q+(i < r) for i in range(plan['channels'])]
        for side, count, prefix, color in (
                ('west', 'num_demux', 'height_demux_', 'FABRIC_H2D_DATA_ID_'),
                ('east', 'num_mux', 'height_mux_', 'FABRIC_D2H_DATA_ID_')):
            params = {'height': plan['height'], count: len(heights), 'width_'+side+'_buf': plan[side]}
            if side == 'west':
                params['num_inputs'] = 1
            for index, height in enumerate(heights):
                params[prefix+str(index)] = height
                params[color+str(index)] = 5+index
            (folder/side).mkdir(exist_ok=True)
            driver.write(folder/side/'out.json', {'params': params})

    def capture(self, case='smoke-echo', order=0, **faults):
        build, _ = self.build(case)
        output = self.root/f'capture-{self.serial}'
        clock = dict(active=False, value=1000)
        api, instances = fake_runtime(clock, **faults)
        def tick():
            clock['active'] = not clock['active']
            clock['value'] += 100
            if instances:
                instances[-1].events.append(('timer', clock['active']))
            return clock['value']
        frames, digest = driver.oracle.frames, driver.oracle.digest
        def outside_frames(*args):
            assert not clock['active'], 'generation inside measured interval'
            return frames(*args)
        def outside_digest(*args):
            assert not clock['active'], 'hashing inside measured interval'
            return digest(*args)
        with mock.patch.object(driver.time, 'perf_counter_ns', side_effect=tick), \
             mock.patch.object(driver.oracle, 'frames', side_effect=outside_frames), \
             mock.patch.object(driver.oracle, 'digest', side_effect=outside_digest):
            record = driver.capture(driver.HERE, build, output, case, order, smoke=True, runtime_api=api)
        return output, record, instances[-1]

    def test_echo_completed_rounds_and_generation_stamping(self):
        previous = Path.cwd()
        output, record, runtime = self.capture(order=2)
        result = driver.audit(output)
        self.assertEqual(Path.cwd(), previous)
        self.assertTrue(runtime.stopped)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(record['rows'][1]['generation']-record['rows'][0]['generation'], 16)
        self.assertEqual(len(record['rows']), len(record['plan']['sizes'])*4)
        self.assertTrue(all(row['elapsed_ns'] == 100 for row in record['rows']))
        self.assertTrue(all(row['samples'] == 3 for row in result['summary']))
        self.assertEqual(runtime.cwd, output.parent/(output.name+'-sdk-work'))
        self.assertEqual(sum(event[0] == 'configure' for event in runtime.events), len(record['plan']['sizes']))
        self.assertFalse(record['full_readback_outside_timing'])

    def test_all_broadcast_and_replicated_controls_deliver_same_full_frames(self):
        captures = {}
        for axis in ('row', 'col'):
            for mode in ('bcast', 'unicast'):
                case = f'smoke-{axis}-{mode}'
                output, record, runtime = self.capture(case)
                result = driver.audit(output)
                self.assertEqual(result['status'], 'PASS')
                self.assertTrue(record['full_readback_outside_timing'])
                self.assertEqual(sum(event[0] == 'readback' for event in runtime.events), len(record['rows']))
                self.assertTrue(all(event[1] == 1 for event in runtime.events if event[0] == 'reply'))
                self.assertTrue(all(chunk['readback_sha256'] for row in record['rows'] for chunk in row['chunks']))
                captures[(axis, mode)] = record
            a, b = captures[(axis, 'bcast')], captures[(axis, 'unicast')]
            self.assertEqual([r['chunks'][0]['readback_sha256'] for r in a['rows']],
                             [r['chunks'][0]['readback_sha256'] for r in b['rows']])
            self.assertEqual([r['chunks'][0]['received_sha256'] for r in a['rows']],
                             [r['chunks'][0]['received_sha256'] for r in b['rows']])
            self.assertNotEqual(a['rows'][0]['chunks'][0]['sent_sha256'], b['rows'][0]['chunks'][0]['sent_sha256'])

    def test_split_round_timer_encloses_every_sequential_chunk(self):
        output, record, runtime = self.capture('smoke-split16')
        result = driver.audit(output)
        self.assertEqual(result['exchanges'], result['logical_rounds']*16)
        self.assertTrue(all(len(row['chunks']) == 16 for row in record['rows']))
        self.assertEqual(result['summary'][0]['control_d2h_bytes'], 4*4*16)
        events = [event[0] for event in runtime.events if event[0] in ('timer', 'unicast', 'reply')]
        self.assertEqual(events, (['timer']+['unicast', 'reply']*16+['timer'])*len(record['rows']))
        self.assertEqual([c['generation'] for c in record['rows'][0]['chunks']], list(range(1, 17)))
        self.assertEqual(record['rows'][1]['generation'], 17)
        self.assertEqual(len(record['witnesses']['0']), 16)

    def test_ack_does_not_admit_corrupted_full_readback(self):
        with self.assertRaisesRegex(ValueError, 'full readback differs'):
            self.capture('smoke-row-bcast', corrupt_readback=True)
        record = driver.read(self.root/'capture-1/capture.json')
        self.assertEqual(record['status'], 'FAILED')
        self.assertFalse(record['rows'][0]['equal'])
        witness = record['witnesses']['0'][0]
        self.assertTrue((self.root/'capture-1'/witness['readback']['path']).is_file())

    def test_zero_payload_detects_duplicate_or_permuted_pe_ack(self):
        with self.assertRaisesRegex(ValueError, 'received reply'):
            self.capture('smoke-col-bcast', permute_reply=True)

    def test_partial_runtime_and_stop_failures_remain_failed(self):
        previous = Path.cwd()
        for fault, message in (({'fail_h2d': 2}, 'H2D failure'), ({'fail_stop': True}, 'stop failure')):
            with self.subTest(fault=fault):
                with self.assertRaisesRegex(RuntimeError, message):
                    self.capture(**fault)
                output = self.root/f'capture-{self.serial}'
                record = driver.read(output/'capture.json')
                self.assertEqual(record['status'], 'FAILED')
                self.assertEqual(Path.cwd(), previous)
                with self.assertRaises(ValueError):
                    driver.audit(output)

    def test_actual_buffers_and_unequal_channel_partitions_are_checked(self):
        case = next(key for key, value in driver.oracle.CASES.items()
                    if value['height'] == 92 and value['channels'] == 16)
        build, plan = self.build(case)
        mapping = driver.check_program(build, plan)
        self.assertEqual(mapping['h2d'][-1]['row_end'], 92)
        self.assertEqual(len(mapping['d2h']), 16)
        for side, field, value in (('west', 'width_west_buf', 1), ('east', 'width_east_buf', 1),
                                   ('west', 'num_inputs', 0), ('east', 'num_mux', 15),
                                   ('west', 'height_demux_0', 0), ('east', 'height_mux_15', 10)):
            with self.subTest(side=side, field=field):
                self.channel_metadata(build, plan)
                path = build/side/'out.json'
                data = driver.read(path)
                data['params'][field] = value
                driver.write(path, data)
                with self.assertRaises(ValueError):
                    driver.check_program(build, plan)

    def test_wrong_compile_reply_mode_fails_before_runtime(self):
        build, _ = self.build('smoke-row-bcast')
        params = driver.read(build/'out.json')
        params['params']['REPLY_ACK'] = 0
        driver.write(build/'out.json', params)
        api, instances = fake_runtime(dict(active=False))
        with self.assertRaisesRegex(ValueError, 'compiled parameter REPLY_ACK'):
            driver.capture(driver.HERE, build, self.root/'wrong-compile', 'smoke-row-bcast', 0,
                           smoke=True, runtime_api=api)
        self.assertEqual(instances, [])

    def test_audit_rejects_schedule_scope_generation_and_digest_mutations(self):
        output, original, _ = self.capture('smoke-split16')
        changes = [
            lambda r: r['rows'][0].update(elapsed_ns=0),
            lambda r: r['rows'][0].update(elapsed_ns=True),
            lambda r: r['rows'][0].update(elapsed_ns=600*10**9),
            lambda r: r['rows'][1].update(generation=2),
            lambda r: r['rows'][0]['chunks'].pop(),
            lambda r: r['rows'][0]['chunks'][0].update(index=1),
            lambda r: r['rows'][0]['chunks'][0].update(received_sha256='0'*64),
            lambda r: r['rows'][0]['chunks'][0].update(sent_sha256='0'*64),
            lambda r: r['rows'][0]['chunks'][0].update(readback_sha256='0'*64),
            lambda r: r['rows'][0].update(phase='measured'),
            lambda r: r['rows'].pop(),
            lambda r: r['plan'].update(chunks=8),
            lambda r: r.update(backend='hardware'),
            lambda r: r.update(outstanding_exchanges=16),
            lambda r: r.update(physical_wire_bytes=100),
            lambda r: r.update(full_readback_outside_timing=True),
            lambda r: r.update(timing_scope='only one of the chunks'),
        ]
        for change in changes:
            with self.subTest(change=change):
                record = copy.deepcopy(original)
                change(record)
                driver.write(output/'capture.json', record)
                with self.assertRaises(ValueError):
                    driver.audit(output)
        driver.write(output/'capture.json', original)
        self.assertEqual(driver.audit(output)['status'], 'PASS')

    def test_raw_readback_rejects_rehashed_corruption(self):
        output, record, _ = self.capture('smoke-row-bcast')
        witness = record['witnesses']['0'][0]['readback']
        path = output/witness['path']
        data = np.load(path, allow_pickle=False)
        data[0] ^= np.uint32(1)
        np.save(path, data, allow_pickle=False)
        witness['sha256'] = driver.sha(path)
        driver.write(output/'capture.json', record)
        with self.assertRaisesRegex(ValueError, 'raw witness equality'):
            driver.audit(output)

    def test_source_and_metadata_cannot_be_rebound(self):
        output, record, _ = self.capture()
        source = output/'source/pe.csl'
        saved = source.read_bytes()
        source.write_bytes(saved+b'\n// modified source\n')
        record['source_sha256']['pe.csl'] = driver.sha(source)
        driver.write(output/'capture.json', record)
        with self.assertRaisesRegex(ValueError, 'source identity'):
            driver.audit(output)
        source.write_bytes(saved)
        record['source_sha256']['pe.csl'] = driver.sha(source)
        driver.write(output/'capture.json', record)
        path = output/'metadata/out.json'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'metadata identity'):
            driver.audit(output)

    def test_hardware_smoke_and_nested_scratch_are_rejected(self):
        build, _ = self.build()
        api, instances = fake_runtime(dict(active=False))
        with self.assertRaises(ValueError):
            driver.capture(driver.HERE, build, self.root/'hardware-smoke', 'smoke-echo', 0,
                           backend='hardware', cmaddr='unused:0', smoke=True, runtime_api=api)
        with self.assertRaises(ValueError):
            driver.capture(driver.HERE, build, self.root/'nested-capture', 'smoke-echo', 0,
                           sdk_workdir=self.root, smoke=True, runtime_api=api)
        self.assertEqual(instances, [])


if __name__ == '__main__':
    unittest.main()
