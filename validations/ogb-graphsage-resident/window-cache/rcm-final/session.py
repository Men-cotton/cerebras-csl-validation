"""Run the six-frame worker against the verified system application and its memcpy helpers."""
import argparse
from pathlib import Path
import subprocess
import sys

from inventory import actual, read, require, save, sha


def verify(stage, artifact, pin):
    release = stage / "release.json"
    require(sha(release) == pin, "request release")
    pins = read(release)
    for name, expected in pins["runtime"].items():
        require(sha(stage / name) == expected, "execution source " + name)
    for name in ("build.json", "fixture.json"):
        require(sha(stage / name) == pins["files"]["physical/" + name], name)
    fixture = read(stage / "fixture.json")
    require(sha(stage / "fixture.npz") == fixture["files"]["fixture.npz"] == pins["files"]["fixture.npz"],
            "RCM window inputs")
    descriptor = read(stage / "build.json")
    hardware = actual(artifact)
    require(hardware == descriptor["inventory"] and descriptor["role"] == "hardware"
            and fixture["physical_size_sha256"] == sha(stage / "build.json"),
            "returned CS-3 firmware and hardware fixture")
    return dict(release_sha256=pin, build_sha256=sha(stage / "build.json"),
                fixture_sha256=sha(stage / "fixture.json"), inventory=hardware)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--release", required=True)
    parser.add_argument("--cmaddr", required=True)
    args = parser.parse_args()
    stage = Path(__file__).resolve().parent
    artifact = args.artifact.resolve()
    save(stage / "loaded.json", verify(stage, artifact, args.release))
    worker = stage / "runtime/host-device-transfer/ogb-traffic/hot-window/rcm-final/worker_final.py"
    subprocess.run([sys.executable, "-B", str(worker), str(stage), str(stage),
                    str(stage / "capture"), "--cmaddr", args.cmaddr, "--artifact", str(artifact)], check=True)
