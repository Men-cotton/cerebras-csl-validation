"""PMU ABI and failure tests with synthetic reads; never open real counters."""
import errno
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_pmu as pmu


def record(enabled, running, cycles=0, instructions=0, reverse=False):
    pairs = [cycles, 101, instructions, 202]
    if reverse:
        pairs = pairs[2:] + pairs[:2]
    return struct.pack('=7Q', 2, enabled, running, *pairs)


class FakeBackend:
    def __init__(self, records=(), fail_open=None, fail_control=None):
        self.records = iter(records)
        self.calls, self.closed = [], []
        self.fail_open, self.fail_control = fail_open, fail_control

    def open(self, config, leader):
        self.calls.append(('open', config, leader))
        if config == self.fail_open:
            raise PermissionError(errno.EACCES, 'PMU denied')
        return 10 + config

    def event_id(self, fd):
        return {10: 101, 11: 202}[fd]

    def control(self, fd, operation):
        self.calls.append(('control', fd, operation))
        if operation == self.fail_control:
            raise OSError(errno.EIO, 'control failed')

    def read(self, fd):
        self.calls.append(('read', fd))
        return next(self.records)

    def close(self, fd):
        self.closed.append(fd)


class CounterTests(unittest.TestCase):
    def test_native_syscall_abi_without_executing_syscall(self):
        for machine, number in (('x86_64', 298), ('aarch64', 241)):
            libc = mock.Mock()
            libc.syscall.return_value = 77
            with mock.patch.object(pmu.sys, 'platform', 'linux'), \
                    mock.patch.object(pmu.sys, 'byteorder', 'little'), \
                    mock.patch.object(pmu.platform, 'machine', return_value=machine), \
                    mock.patch.object(pmu.ctypes, 'sizeof', return_value=8), \
                    mock.patch.object(pmu.ctypes, 'CDLL', return_value=libc), \
                    mock.patch.dict(sys.modules, {'fcntl': mock.Mock()}):
                backend = pmu._LinuxBackend()
                self.assertEqual(backend.open(1, 22), 77)
            args = libc.syscall.call_args.args
            self.assertEqual([args[i].value for i in (0, 2, 3, 4, 5)],
                             [number, 0, -1, 22, 8])
            self.assertEqual(args[1]._obj.raw[:64], pmu._attr(1))
            libc.syscall.return_value = -1
            with mock.patch.object(pmu.ctypes, 'get_errno', return_value=errno.EPERM):
                with self.assertRaises(OSError) as failure:
                    backend.open(0, -1)
            self.assertEqual(failure.exception.errno, errno.EPERM)

    def test_attr_abi_excludes_kernel_hv_without_inheritance(self):
        for config in (0, 1):
            data = pmu._attr(config)
            self.assertEqual(len(data), 64)
            self.assertEqual(struct.unpack_from('=IIQ', data), (0, 64, config))
            self.assertEqual(struct.unpack_from('=Q', data, 32)[0], 15)
            self.assertEqual(struct.unpack_from('=Q', data, 40)[0], 0x61)
            self.assertEqual(data[48:], bytes(16))

    def test_repeated_intervals_ids_and_no_scaling(self):
        backend = FakeBackend([record(20, 10), record(120, 60, 900, 400, True),
                               record(120, 60), record(150, 80, 42, 17)])
        with pmu.CounterGroup(backend) as group:
            self.assertTrue(group.available)
            self.assertIsNone(group.begin())
            self.assertEqual(group.end(), dict(status='ok', scope='thread-self-user',
                cycles=900, instructions=400, time_enabled_ns=100,
                time_running_ns=50, scaled=False))
            group.begin()
            second = group.end()
            self.assertEqual((second['time_enabled_ns'], second['time_running_ns']), (30, 20))
            self.assertEqual(second['cycles'], 42)
        self.assertEqual(backend.closed, [11, 10])
        self.assertEqual(backend.calls[:2], [('open', 0, -1), ('open', 1, 10)])
        controls = [c[2] for c in backend.calls if c[0] == 'control']
        self.assertEqual(controls, [0x2403, 0x2400, 0x2401] * 2)
        group.close()
        self.assertEqual(backend.closed, [11, 10])

    def test_partial_open_denial_closes_leader_and_has_no_counts(self):
        backend = FakeBackend(fail_open=1)
        group = pmu.CounterGroup(backend)
        self.assertFalse(group.available)
        self.assertEqual(backend.closed, [10])
        for result in (group.begin(), group.end()):
            self.assertEqual(result['status'], 'unavailable')
            self.assertEqual(result['error']['errno'], errno.EACCES)
            self.assertFalse({'cycles', 'instructions', 'time_running_ns'} & set(result))

    def test_platform_unavailability_does_not_open_anything(self):
        with mock.patch.object(pmu, '_LinuxBackend', side_effect=OSError(errno.ENOSYS, 'ABI')):
            group = pmu.CounterGroup()
        self.assertEqual(group.end()['error']['errno'], errno.ENOSYS)

    def test_ioctl_failure_retires_group(self):
        backend = FakeBackend([record(0, 0)], fail_control=pmu.ENABLE)
        group = pmu.CounterGroup(backend)
        self.assertEqual(group.begin()['error']['stage'], 'begin')
        self.assertEqual(backend.closed, [11, 10])
        self.assertEqual(group.end()['status'], 'unavailable')

    def test_disable_or_read_failure_closes_both_fds(self):
        for stage in ('disable', 'read'):
            backend = FakeBackend([record(0, 0)])
            group = pmu.CounterGroup(backend)
            group.begin()
            if stage == 'disable':
                backend.fail_control = pmu.DISABLE
            else:
                backend.read = mock.Mock(side_effect=OSError(errno.EIO, 'read failed'))
            result = group.end()
            self.assertEqual(result['error']['stage'], 'end')
            self.assertEqual(result['error']['errno'], errno.EIO)
            self.assertNotIn('cycles', result)
            self.assertEqual(backend.closed, [11, 10])

    def test_id_failure_closes_newly_opened_fd(self):
        backend = FakeBackend()
        backend.event_id = mock.Mock(side_effect=OSError(errno.EIO, 'ID failed'))
        group = pmu.CounterGroup(backend)
        self.assertFalse(group.available)
        self.assertEqual(backend.closed, [10])

    def test_reset_must_clear_counts(self):
        group = pmu.CounterGroup(FakeBackend([record(0, 0, 1, 0)]))
        self.assertEqual(group.begin()['status'], 'unavailable')

    def test_malformed_and_unscheduled_reads_are_unavailable(self):
        invalid = [b'', record(10, 5)[:-1],
                   struct.pack('=7Q', 3, 10, 5, 8, 101, 9, 202),
                   struct.pack('=7Q', 2, 10, 5, 8, 101, 9, 101),
                   struct.pack('=7Q', 2, 10, 5, 8, 101, 9, 999),
                   record(10, 0), record(10, 11)]
        for data in invalid:
            with self.subTest(data=data):
                backend = FakeBackend([record(0, 0), data])
                group = pmu.CounterGroup(backend)
                group.begin()
                result = group.end()
                self.assertEqual(result['status'], 'unavailable')
                self.assertNotIn('cycles', result)
                self.assertEqual(backend.closed, [11, 10])

    def test_time_baseline_cannot_regress(self):
        group = pmu.CounterGroup(FakeBackend([record(100, 80), record(90, 85, 1, 2)]))
        group.begin()
        self.assertEqual(group.end()['status'], 'unavailable')

    def test_lifecycle_and_thread_ownership(self):
        backend = FakeBackend([record(0, 0), record(10, 10, 1, 2)])
        group = pmu.CounterGroup(backend)
        with self.assertRaises(RuntimeError):
            group.end()
        group.begin()
        with self.assertRaises(RuntimeError):
            group.begin()
        with mock.patch.object(pmu.threading, 'get_native_id', return_value=-1):
            with self.assertRaises(RuntimeError):
                group.end()
        group.end()
        group.close()
        with self.assertRaises(RuntimeError):
            group.begin()

    def test_context_exception_still_closes_active_group(self):
        backend = FakeBackend([record(0, 0)])
        with self.assertRaisesRegex(ValueError, 'caller failed'):
            with pmu.CounterGroup(backend) as group:
                group.begin()
                raise ValueError('caller failed')
        self.assertEqual(backend.closed, [11, 10])


if __name__ == '__main__':
    unittest.main()
