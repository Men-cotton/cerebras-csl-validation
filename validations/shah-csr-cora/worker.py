#!/usr/bin/env python3
"""Appliance worker: bind the staged case/program, retain partial failure data."""
import argparse
from pathlib import Path
import traceback
import uuid

from archive import check_npz, digest, program_tree, read, require, unpack, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--name', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cmaddr')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    status = dict(status='incomplete')
    try:
        request = read(args.record)
        build, run_id, mode = request['compile_record'], request['run_id'], request['mode']
        require(str(uuid.UUID(run_id)) == run_id and mode in ('all', 'gcn', 'ah16', 'ah7'), 'run identity/mode')
        status.update(run_id=run_id, compile_record=build, mode=mode)
        for name, pin in {**build['host_sha256'], **build['source_sha256']}.items():
            require(Path(name).name == name and digest(name, 2 << 20) == pin, 'staged execution source mismatch')
        item = build['case']
        bundle = Path(item['bundle']).name
        require(digest(bundle) == item['sha256'], 'staged case bundle mismatch')
        case = unpack(bundle, Path('worker-case').absolute())
        require(digest(case / 'case.json') == item['case_sha256'], 'staged case manifest mismatch')
        for name in read(case / 'case.json')['files']:
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'case payload path')
            if name.endswith('.npz'):
                check_npz(case / name)
        actual = program_tree(args.name)
        require(actual == build['program_inventory'], 'unpacked program differs from compiler archive')
        status['program_inventory'] = actual
        write(case / 'compile.json', dict(status='PASS', case_sha256=item['case_sha256'], inventory=actual['elf']))
        import driver
        driver.capture(case, args.name, args.out / 'runtime', cmaddr=args.cmaddr, mode=mode, run_id=run_id)
        status['status'] = 'PASS'
        return 0
    except BaseException as error:
        status.update(status='FAIL', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        write(args.out / 'status.json', status)


if __name__ == '__main__':
    raise SystemExit(main())
