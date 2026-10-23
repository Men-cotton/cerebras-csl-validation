"""Offline checks discovered from per-family metadata; never appliance commands."""
import argparse
import subprocess
import sys
from . import catalog


SUITE_RUNNER = """import sys, unittest
suite = unittest.defaultTestLoader.discover(sys.argv[1])
if suite.countTestCases() == 0:
    print('No tests discovered: ' + sys.argv[1], file=sys.stderr)
    sys.exit(2)
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
"""


def suite_command(tests):
    # A fresh interpreter preserves each family's import namespace and cwd.
    return [sys.executable, "-B", "-c", SUITE_RUNNER, tests]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", help="only this validation family")
    args = parser.parse_args(argv)
    try:
        families = catalog.definitions()
        catalog.requests()
        if args.family:
            families = [(path, info) for path, info in families if info["id"] == args.family]
            catalog.require(families, "unknown family")
        suites = [] if args.family else [(catalog.ROOT, "tests"), (catalog.ROOT / "tools", "tests")]
        suites += [(path, test) for path, info in families for test in info["tests"]]
        for cwd, tests in suites:
            print("CHECK", cwd.relative_to(catalog.ROOT), tests, flush=True)
            result = subprocess.run(suite_command(tests), cwd=cwd)
            if result.returncode:
                return result.returncode
        return 0
    except (ValueError, OSError) as error:
        parser.exit(2, str(error) + "\n")
