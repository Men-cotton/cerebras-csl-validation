"""Run the automatic and fixed-coordinate controls consecutively on the same allocation."""
import argparse
from pathlib import Path
import subprocess
import sys

from probe import save


def run(source, artifact, build, out, cmaddr):
    out.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, worker, args in [
        ('automatic', 'probe.py', [source, out/'automatic']),
        ('fixed', 'fixed_worker.py', [artifact, build, out/'fixed', '--current-fabric', out/'automatic/fabric-connections.json']),
    ]:
        command = [sys.executable, '-B', source/worker, *args]
        if cmaddr:
            command.extend(['--cmaddr', cmaddr])
        with (out/(name+'.log')).open('w') as log:
            try:
                rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=480, check=False).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        results[name] = rc
        save(out/'driver.json', results)
    if any(results.values()):
        raise RuntimeError('one or both endpoint controls failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('source', 'artifact', 'build', 'out'):
        p.add_argument(arg, type=Path)
    p.add_argument('--cmaddr')
    a = p.parse_args()
    run(a.source.resolve(), a.artifact.resolve(), a.build.resolve(), a.out.resolve(), a.cmaddr)
