"""Optional Linux current-thread user PMU counts; importing opens no counters.

No scaling or frequency conversion. The enabled interval includes Python/ctypes
instrumentation around the caller's operation. Other threads, kernel execution,
and off-CPU waiting are outside this scope. Use an empty-region control.

ABI: https://man7.org/linux/man-pages/man2/perf_event_open.2.html and Linux
include/uapi/linux/perf_event.h (PERF_ATTR_SIZE_VER0, group read with IDs).
"""
import array
import ctypes
import errno
import os
import platform
import struct
import sys
import threading


ENABLE, DISABLE, RESET, EVENT_ID = 0x2400, 0x2401, 0x2403, 0x80082407
GROUP = 1
READ_FORMAT = 1 | 2 | 4 | 8  # enabled, running, ID, group
ATTR_FLAGS = 1 | (1 << 5) | (1 << 6)  # disabled, exclude_kernel, exclude_hv
SCOPE = 'thread-self-user'


def _attr(config):
    # Native byte order, fixed u32/u64 sizes: original 64-byte perf_event_attr.
    return struct.pack('=IIQQQQQIIQ', 0, 64, config, 0, 0,
                       READ_FORMAT, ATTR_FLAGS, 0, 0, 0)


class _LinuxBackend:
    def __init__(self):
        machine = platform.machine().lower()
        numbers = {'x86_64': 298, 'amd64': 298, 'aarch64': 241, 'arm64': 241}
        if (sys.platform != 'linux' or machine not in numbers or
                ctypes.sizeof(ctypes.c_void_p) != 8 or sys.byteorder != 'little'):
            raise OSError(errno.ENOSYS, 'supported ABI: Linux x86_64/aarch64 LP64 little-endian')
        import fcntl
        self.ioctl = fcntl.ioctl
        self.number = numbers[machine]
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.syscall.restype = ctypes.c_long

    def open(self, config, leader):
        attr = ctypes.create_string_buffer(_attr(config))
        # pid=0: calling thread; cpu=-1: any CPU; FD_CLOEXEC=8; no inheritance.
        fd = self.libc.syscall(ctypes.c_long(self.number), ctypes.byref(attr),
                               ctypes.c_int(0), ctypes.c_int(-1),
                               ctypes.c_int(leader), ctypes.c_ulong(8))
        if fd < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        return int(fd)

    def event_id(self, fd):
        value = array.array('Q', [0])
        self.ioctl(fd, EVENT_ID, value, True)
        return value[0]

    def control(self, fd, operation):
        self.ioctl(fd, operation, GROUP)

    @staticmethod
    def read(fd):
        return os.read(fd, 56)  # nr, enabled, running, two (value, ID) pairs

    close = staticmethod(os.close)


class CounterGroup:
    """Reuse begin()/end() on the creating thread; close() releases both FDs.

    begin returns None when started. end returns an 'ok' record with raw
    counts and interval enabled/running ns. OS/ABI/read failures return an
    'unavailable' record, without count fields, and permanently retire the group.
    Lifecycle misuse raises RuntimeError. backend is injectable for offline tests.
    """
    def __init__(self, backend=None):
        self._owner = (os.getpid(), threading.get_native_id())
        self._fds, self._ids = [], []
        self._active = self._closed = False
        self.error = None
        self.available = False
        self._backend = backend
        try:
            if self._backend is None:
                self._backend = _LinuxBackend()
            for config in (0, 1):
                fd = self._backend.open(config, self._fds[0] if self._fds else -1)
                self._fds.append(fd)
                self._ids.append(self._backend.event_id(fd))
            if len(set(self._ids)) != 2:
                raise ValueError('duplicate PMU event IDs')
            self.available = True
        except (OSError, ValueError) as exc:
            self._fail('open', exc)

    def _check(self):
        if self._closed:
            raise RuntimeError('PMU group is closed')
        if self._owner != (os.getpid(), threading.get_native_id()):
            raise RuntimeError('PMU group belongs to a different process/thread')

    def _release(self):
        for fd in reversed(self._fds):
            try:
                self._backend.close(fd)
            except OSError as exc:
                if self.error is None:
                    self.error = dict(stage='close', type=type(exc).__name__,
                                      errno=exc.errno, message=str(exc))
        self._fds.clear()

    def _record_error(self):
        return dict(status='unavailable', scope=SCOPE, error=dict(self.error))

    def _fail(self, stage, exc):
        self.error = dict(stage=stage, type=type(exc).__name__,
                          errno=getattr(exc, 'errno', None), message=str(exc))
        self.available = self._active = False
        self._release()
        return self._record_error()

    def _read(self):
        data = self._backend.read(self._fds[0])
        if len(data) != 56:
            raise ValueError('PMU group read must contain exactly 56 bytes')
        nr, enabled, running, v0, id0, v1, id1 = struct.unpack('=7Q', data)
        if nr != 2 or id0 == id1 or {id0, id1} != set(self._ids):
            raise ValueError('PMU group count/ID mismatch')
        if running > enabled:
            raise ValueError('PMU running time exceeds enabled time')
        values = {id0: v0, id1: v1}
        return enabled, running, [values[i] for i in self._ids]

    def begin(self):
        self._check()
        if not self.available:
            return self._record_error()
        if self._active:
            raise RuntimeError('PMU group is already measuring')
        try:
            self._backend.control(self._fds[0], RESET)
            # RESET clears counts but does not reset enabled/running time.
            enabled, running, values = self._read()
            if values != [0, 0]:
                raise ValueError('PMU group counts did not reset')
            self._baseline = enabled, running
            self._backend.control(self._fds[0], ENABLE)
            self._active = True
        except (OSError, ValueError) as exc:
            return self._fail('begin', exc)

    def end(self):
        self._check()
        if not self.available:
            return self._record_error()
        if not self._active:
            raise RuntimeError('PMU group has no active measurement')
        try:
            self._backend.control(self._fds[0], DISABLE)
            self._active = False
            enabled, running, values = self._read()
            enabled -= self._baseline[0]
            running -= self._baseline[1]
            if not 0 < running <= enabled:
                raise ValueError('PMU interval did not run or has invalid timing')
            return dict(status='ok', scope=SCOPE, cycles=values[0],
                        instructions=values[1], time_enabled_ns=enabled,
                        time_running_ns=running, scaled=False)
        except (OSError, ValueError) as exc:
            return self._fail('end', exc)

    def close(self):
        if not self._closed:
            self._release()
            self.available = self._active = False
            self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()
