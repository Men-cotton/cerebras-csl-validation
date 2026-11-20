"""Run the frozen eight-port GraphSAGE cases using the existing physical Direct Link driver."""
import argparse
import importlib.util
from pathlib import Path
import sys

FAMILY = Path(__file__).resolve().parent
BASE = FAMILY.parent / "ogb-parallel-ports"
sys.path.insert(0, str(BASE))
# The original driver pins its source files and 1/2/4-port maps for existing releases.
# An independent module instance leaves those release checks available in the same process.
spec = importlib.util.spec_from_file_location("graphsage_eight_port_core", BASE / "appliance.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
driver.PORTS = {8: "a86df9927b082418eeebdee80225858b45b29ec6f4e22e5960d456dcdf430dd9"}
driver.SCRIPTS += ("../ogb-eight-ports/physical.py",)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run","check"))
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--namespace", default="job-operator")
    args = parser.parse_args()
    case = args.case.resolve()
    if args.action == "check":
        driver.check(case, args.release_sha256)
    elif args.out:
        driver.run(case, args.out.resolve(), args.release_sha256, args.namespace)
    else:
        parser.error("run requires --out")
