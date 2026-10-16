#!/usr/bin/env python3
"""Standalone two-PE native16 report reproducer: one condition per process.

Only NumPy and the selected Cerebras runtime are required. Successful conditions
run A/B/A. A completed numerical mismatch exits 1 after retaining raw carriers;
runtime/setup errors retain partial capture and propagate the exception.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np

VARS = {'source_rows': ('u16', 32), 'source_values': ('f32', 32),
        'local_copy': ('u16', 20), 'compact_rows': ('u16', 20),
        'received_rows': ('u16', 20), 'received_values': ('f32', 20),
        'status': ('u32', 8)}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 128*1024**2,
         'bounded regular hash input')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    need(Path(path).stat().st_size <= 2*1024**2, 'JSON size')
    return json.loads(Path(path).read_text())


def write(path, value):
    data = json.dumps(value, indent=2, sort_keys=True) + '\n'
    need(len(data.encode()) <= 2*1024**2, 'JSON size')
    Path(path).write_text(data)


def bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def runtime_options(backend, cmaddr):
    need(backend != 'hardware' or bool(cmaddr), 'hardware requires cmaddr')
    options = dict(cmaddr=cmaddr)
    if backend == 'simulator' and not cmaddr:
        options.update(simfab_numthreads=1, suppress_simfab_trace=True)
    return options


class IO:
    """Every transfer has a preserved HWL carrier and closed blocking API record.

    COL_MAJOR is the reversible flat mapping (length, column, row). Native u16
    values occupy low16 of uint32 host carriers; upper carrier bits are retained.
    """
    def __init__(self, runtime, out, case, symbols, bindings):
        self.runtime, self.out, self.case = runtime, Path(out), case
        self.symbols, self.bindings = symbols, bindings
        self.events, self.raw, self.stage, self.last_key = [], {}, "setup", None

    def call(self, api, fn, *args, detail=None, **kwargs):
        entry = dict(index=len(self.events)//2, api=api, phase="enter", seconds=time.monotonic(),
                     stage=self.stage, detail=detail or {})
        self.events.append(entry)
        write(self.out / "calls.json", self.events)
        value = fn(*args, **kwargs)
        self.events.append(dict(entry, phase="exit", seconds=time.monotonic()))
        write(self.out / "calls.json", self.events)
        return value

    def launch(self, name, *args):
        return self.call("launch:" + name, self.runtime.launch, name, *args,
                         detail=dict(arguments=[int(v) for v in args], nonblock=False), nonblock=False)

    def save_raw(self):
        np.savez(self.out / "raw_device.npz", **self.raw)
        need((self.out / "raw_device.npz").stat().st_size <= 32*1024**2, "raw capture bound")

    def transfer(self, symbol, array, direction, word_bits, x, y):
        array = np.asarray(array)
        need(array.ndim == 3 and word_bits in (16, 32), "HWL transfer")
        h, w, length = array.shape
        need(0 <= x < self.case.pe_cols and 0 <= y < self.case.pe_rows and
             x+w <= self.case.pe_cols and y+h <= self.case.pe_rows and min(h,w,length)>0, "transfer rectangle")
        if word_bits == 16:
            array = array.astype(np.uint32)
        need(array.dtype in (np.dtype('float32'), np.dtype('uint32')), "SDK carrier dtype")
        flat = np.ascontiguousarray(array.transpose(2, 1, 0)).reshape(-1)
        key = f"{len(self.events)//2:04d}_{direction}_{symbol}"
        detail = dict(symbol=symbol, raw_key=key, geometry=[x,y,w,h,length], word_bits=word_bits,
                      dtype=array.dtype.str, order="COL_MAJOR", streaming=False, nonblock=False)
        datatype = getattr(self.bindings[0], "MEMCPY_"+str(word_bits)+"BIT")
        kwargs = dict(streaming=False, nonblock=False, data_type=datatype, order=self.bindings[1].COL_MAJOR)
        if direction == "h2d":
            self.raw[key] = array.copy()
            self.save_raw()
            self.call("h2d:"+symbol, self.runtime.memcpy_h2d, self.symbols[symbol], flat, x,y,w,h,length, detail=detail, **kwargs)
        else:
            self.call("d2h:"+symbol, self.runtime.memcpy_d2h, flat, self.symbols[symbol], x,y,w,h,length, detail=detail, **kwargs)
            self.raw[key] = flat.reshape(length,w,h).transpose(2,1,0).copy()
            self.save_raw()
        self.last_key = key
        value = self.raw[key]
        return (value & np.uint32(65535)).astype(np.uint16) if word_bits == 16 else value.copy()

    def put(self, symbol, array, word_bits=32):
        return self.transfer(symbol, array, "h2d", word_bits, 0, 0)

    def get(self, symbol, length, dtype="f32", x=0, y=0, width=None, height=None):
        width = self.case.pe_cols if width is None else width
        height = self.case.pe_rows if height is None else height
        native = np.float32 if dtype == "f32" else np.uint32
        need(dtype in ("f32", "u16", "u32"), "read dtype")
        return self.transfer(symbol, np.zeros((height,width,length), native), "d2h", 16 if dtype == "u16" else 32, x,y)


def inputs(panel):
 rows=np.empty((1,2,32),np.uint16);values=np.empty((1,2,32),np.float32)
 for x in range(2):
  rows[0,x]=257+17*np.arange(32)+100*x+(1000 if panel==1 else 0)
  values[0,x]=((1024+3*np.arange(32)+64*x)/16)*(-0.5 if panel==1 else 1)
 return rows,values

def expected(panel,off,n,mode):
 rows,values=inputs(panel)
 arrays={'source_rows':rows,'source_values':values}
 for key in ('local_copy','compact_rows','received_rows'):
  a=np.full((1,2,20),0xd00d,np.uint16);a[:,:,[0,1,18,19]]=[0x1234,0x2345,0x6789,0x789a];arrays[key]=a
 a=np.full((1,2,20),-12345.5,np.float32);a[:,:,[0,1,18,19]]=[-10001,-10002,-10003,-10004];arrays['received_values']=a
 arrays['local_copy'][:,:,2:2+n]=rows[:,:,off:off+n]
 if mode:arrays['compact_rows'][0,0,2:2+n]=rows[0,0,off:off+n]
 arrays['received_rows'][0,1,2:2+n]=rows[0,0,off:off+n]
 arrays['received_values'][0,1,2:2+n]=values[0,0,off:off+n]
 cb=3 if n else 1
 arrays['status']=np.array([[[0,4,off,n,mode,0,cb,0],[0,4,off,n,mode,n,0,cb]]],np.uint32)
 return arrays


def run_case(args):
    build = args.name.resolve()
    out = args.out.absolute()
    need(out == out.resolve() and not out.exists(), 'fresh resolved capture directory')
    out.mkdir(parents=True)
    config = tuple(map(int, args.case.split(':')))
    need(len(config) == 3, 'offset:length:mode')
    off, n, mode = config
    need(0 <= off <= 31 and 0 <= n <= 16 and off+n <= 32 and mode in (0, 1), 'case bounds')
    options = runtime_options(args.backend, args.cmaddr)
    result = dict(status='incomplete', config=dict(offset=off, length=n, mode=mode),
                  panels=[], backend=args.backend, host_sha256=sha(__file__),
                  runtime_image=os.environ.get('SINGULARITY_CONTAINER') or os.environ.get('APPTAINER_CONTAINER'),
                  simfab_numthreads=options.get('simfab_numthreads'),
                  suppress_simfab_trace=options.get('suppress_simfab_trace'),
                  identity_scope='external compiler/runtime must be recorded by launcher operator')
    record = None
    if args.build_record:
        record = read(args.build_record)
        need(record['status'] == 'pass', 'compiled build')
        need(record['host_sha256'] == sha(__file__), 'compiled host source identity')
        need(args.backend == 'simulator' and not args.cmaddr, 'local build record is embedded simulator evidence')
        need(result['runtime_image'] == str(Path(record['sdk']['root'])/record['sdk']['sif_name']), 'selected SDK image')
        for name, pin in record['artifact_sha256'].items():
            path = Path(name)
            need(not path.is_absolute() and '..' not in path.parts, 'relative artifact path')
            need(sha(build/path) == pin, 'compiled artifact identity')
        (out/'build_record.json').write_bytes(args.build_record.read_bytes())
        result.update(source_sha256=record['source_sha256'], sdk=record['sdk'],
                      identity_scope='local build/source and selected SDK image')
    write(out/'capture.json', result)
    io = None
    old = Path.cwd()
    start = time.monotonic()
    try:
        os.chdir(out)
        from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
        runtime = SdkRuntime(str(build), **options)
        io = IO(runtime, out, SimpleNamespace(pe_rows=1, pe_cols=2), {}, (MemcpyDataType, MemcpyOrder))
        symbols = {key: runtime.get_id(key) for key in VARS}
        if record:
            need(symbols == record['symbol_ids'], 'runtime symbol identity')
        io.symbols = symbols
        io.call('load', runtime.load)
        io.call('run', runtime.run)
        for panel in range(3):
            io.stage = f'panel{panel}'
            rows, values = inputs(panel)
            io.put('source_rows', rows, word_bits=16)
            io.put('source_values', values)
            need(bits_equal(io.get('source_rows', 32, 'u16'), rows), 'source row input readback')
            need(bits_equal(io.get('source_values', 32), values), 'source value input readback')
            io.launch('configure', np.int16(off), np.int16(n), np.int16(mode))
            io.launch('run_probe')
            want, errors = expected(panel, off, n, mode), {}
            for key, (dtype, size) in VARS.items():
                got = io.get(key, size, dtype)
                view = np.uint16 if dtype == 'u16' else np.uint32
                bad = np.argwhere(got.view(view) != want[key].view(view))
                if len(bad):
                    errors[key] = dict(count=len(bad), indices=bad.tolist())
            result['panels'].append(dict(panel=panel, errors=errors))
            write(out/'capture.json', result)
            if errors:
                break
        io.stage = 'teardown'
        io.call('stop', runtime.stop)
        io = None
        result['status'] = 'observed_numeric_fail' if any(p['errors'] for p in result['panels']) else 'pass'
    except Exception as error:
        result.update(status='runtime_error', error=str(error))
        raise
    finally:
        if io is not None:
            io.stage = 'failed-teardown'
            try:
                io.call('stop', io.runtime.stop)
            except Exception as error:
                result['stop_error'] = str(error)
        os.chdir(old)
        result['wall_seconds'] = time.monotonic()-start
        for name in ('calls.json', 'raw_device.npz', 'build_record.json'):
            if (out/name).is_file():
                result[name+'_sha256'] = sha(out/name)
        write(out/'capture.json', result)
    print(json.dumps(dict(config=result['config'], status=result['status']), sort_keys=True), flush=True)
    return 0 if result['status'] == 'pass' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', required=True, help='offset:length:mode (mode 0 direct, 1 compact)')
    parser.add_argument('--backend', choices=('simulator', 'hardware'), default='simulator')
    parser.add_argument('--cmaddr')
    parser.add_argument('--build-record', type=Path)
    raise SystemExit(run_case(parser.parse_args()))


if __name__ == '__main__':
    main()

