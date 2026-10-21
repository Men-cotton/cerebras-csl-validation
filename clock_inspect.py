"""Offline raw/API revalidation; never submits a job or trusts cached cycle summaries."""
import argparse
import json
import sys
from pathlib import Path
import clock_common as common
from clock_appliance import validate_capture


def inspect(manifest_path, request_path, capture_path):
    manifest, request = common.read_json(manifest_path), common.read_json(request_path)
    common.require(request['manifest_sha256'] == common.digest(manifest_path), 'original manifest bytes')
    meta, derived = validate_capture(capture_path,manifest,request)
    return dict(run_id=request['run_id'],backend=manifest['backend'],params=manifest['params'],
                config=request['config'],**derived)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--request',required=True)
    parser.add_argument('--capture',required=True)
    args = parser.parse_args(argv)
    try:
        result = inspect(args.manifest,args.request,args.capture)
        print(json.dumps(result,indent=2,sort_keys=True))
        return {'pass':0,'observed_numeric_fail':1,'incomplete':2}[result['status']]
    except Exception as error:
        print('error: '+str(error),file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
