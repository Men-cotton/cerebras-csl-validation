"""Local SDK compile/run driver. Place external resource supervision around it."""
import argparse
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import uuid
import clock_common as common

ROOT = Path(__file__).resolve().parent


def inventory(root):
    result, entries, size, started = {}, 0, 0, time.monotonic()
    def fail(error):
        raise error
    for directory, dirs, files in os.walk(root,onerror=fail):
        for name in dirs + files:
            path = Path(directory)/name
            entries += 1
            common.require(entries <= 4096 and time.monotonic()-started <= 5, 'bounded artifact inventory')
            common.require(not path.is_symlink(), 'no artifact symlinks')
            if path.is_file():
                size += path.stat().st_size
                common.require(size <= 128 << 20, 'artifact size bound')
                result[str(path.relative_to(root))] = common.digest(path)
            else:
                common.require(path.is_dir(), 'regular artifact files')
    common.require(result, 'nonempty artifact inventory')
    return result


def compile_case(args):
    params = common.validate_params(dict(C=args.cols,R=args.rows,W=args.words))
    out = Path(args.out).absolute()
    out.mkdir(parents=True,exist_ok=False)
    cmd = [args.compiler,str(ROOT/'clock_source/layout.csl'),'--arch=wse3',
           f"--fabric-dims={params['C']+7},{params['R']+2}",'--fabric-offsets=4,1',
           f"--params=C:{params['C']},R:{params['R']},W:{params['W']}",'--memcpy','--channels=1',
           '--width-west-buf=0','--width-east-buf=0','--max-parallelism=1','-o='+str(out/'build')]
    manifest = dict(schema=1,action='compile',backend='local-sim',compile_id=str(uuid.uuid4()),
                    params=params,compiler_args=cmd,**common.source_identity(ROOT),
                    launcher_sha256=common.digest(ROOT/'clock_local.py'))
    common.write_json(out/'compile-request.json',manifest)
    status = dict(status='incomplete')
    try:
        result = subprocess.run(cmd,timeout=180,check=False)
        common.require(result.returncode == 0,'compiler exit '+str(result.returncode))
        common.require(common.source_identity(ROOT) == {k:manifest[k] for k in ('source_sha256','host_sha256')}, 'source changed during compile')
        manifest['artifacts_sha256'] = inventory(out/'build')
        common.write_json(out/'clock-build.json',manifest)
        status = dict(status='complete',manifest_sha256=common.digest(out/'clock-build.json'))
    finally:
        common.write_json(out/'compile-status.json',status)
    return 0


def run_case(args):
    manifest_path = Path(args.manifest).absolute()
    manifest = common.read_json(manifest_path)
    common.require(manifest['backend'] == 'local-sim','local backend')
    common.require(common.read_json(manifest_path.parent/'compile-request.json') ==
                   {k:v for k,v in manifest.items() if k != 'artifacts_sha256'}, 'original compile request')
    common.require(common.read_json(manifest_path.parent/'compile-status.json') ==
                   dict(status='complete',manifest_sha256=common.digest(manifest_path)), 'compile completion')
    common.require(inventory(manifest_path.parent/'build') == manifest['artifacts_sha256'], 'compiled artifact integrity')
    common.require(common.source_identity(ROOT) == {k:manifest[k] for k in ('source_sha256','host_sha256')}, 'source identity')
    provided = common.read_json(args.config_file)
    common.require(set(provided) == {'config'}, 'condition file key')
    config = common.validate_config(provided['config'])
    out = Path(args.out).absolute()
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(manifest_path.parent/'build',out/'build')
    common.require(inventory(out/'build') == manifest['artifacts_sha256'], 'fresh build copy')
    (out/'clock-build.json').write_bytes(manifest_path.read_bytes())
    request = dict(config=config,run_id=str(uuid.uuid4()),manifest_sha256=common.digest(manifest_path))
    common.write_json(out/'clock-request.json',request)
    cmd = [args.python,str(ROOT/'clock_probe.py'),'--name',str(out/'build'),
           '--out',str(out/'capture'),'--backend','local-sim','--manifest',str(out/'clock-build.json'),
           '--config-file',str(out/'clock-request.json')]
    common.write_json(out/'launch.json',dict(command=cmd,run_launcher_sha256=common.digest(ROOT/'clock_local.py'),**request))
    status = dict(status='incomplete')
    try:
        result = subprocess.run(cmd,timeout=180,check=False)
        status.update(exit_code=result.returncode)
        common.require(result.returncode in (0,1,2),'worker exit')
        from clock_appliance import validate_capture
        meta, _ = validate_capture(out/'capture',manifest,request)
        common.require(meta['status'] == {0:'pass',1:'observed_numeric_fail',2:'incomplete'}[result.returncode], 'worker exit/status')
        status.update(status=meta['status'])
        return result.returncode
    finally:
        common.write_json(out/'local-status.json',status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    comp = sub.add_parser('compile')
    comp.add_argument('--compiler',default='cslc')
    for name in ('cols','rows','words'):
        comp.add_argument('--'+name,type=int,default=16 if name == 'words' else 2)
    comp.add_argument('--out',required=True)
    run = sub.add_parser('run')
    run.add_argument('--python',default='cs_python')
    run.add_argument('--manifest',required=True)
    run.add_argument('--config-file',required=True)
    run.add_argument('--out',required=True)
    args = parser.parse_args(argv)
    try:
        return compile_case(args) if args.action == 'compile' else run_case(args)
    except Exception as error:
        print('error: '+str(error),file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
