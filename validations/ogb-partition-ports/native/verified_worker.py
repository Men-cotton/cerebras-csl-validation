"""Validate the physical compiler firmware and staged host program before running the native rectangle."""

import argparse
from pathlib import Path
from inventory import actual, read, require, sha
from worker_rectangle import execute

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cmaddr',required=True)
    a=p.parse_args()
    root=Path.cwd();build=read(root/'build.json')
    require(build['role']=='hardware' and actual(root/'out')==build['inventory'],'staged physical firmware')
    for name,pin in build['worker_sha256'].items():
        require(sha(root/name)==pin,'staged native host program '+name)
    execute(root,root/'fixture.npz',root/'capture',a.cmaddr,root/'out')
