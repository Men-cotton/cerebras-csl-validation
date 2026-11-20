from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1] / "window-cache"


class ResidentSourceTable(unittest.TestCase):
    def test_recreate_inputs_from_all_original_sdk_windows(self):
        releases = sorted((ROOT / "cases").glob("*/release.json"))
        self.assertGreater(len(releases), 0)
        import hashlib

        for release in releases:
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "physical.py"),
                    "check",
                    "--case",
                    release.parent.name,
                    "--release-sha256",
                    hashlib.sha256(release.read_bytes()).hexdigest(),
                ],
                check=True,
            )

    def test_staged_verifier_checks_a_changed_dma_helper(self):
        code = """from pathlib import Path
import sys,tempfile,shutil
sys.path.insert(0,sys.argv[1]);from physical import hosts,sha,read,save;from inventory import actual;from bootstrap import verified
root=Path(sys.argv[1]);case=sorted((root/'cases').iterdir())[0];sim=root/'base/sim';desc=read(sim/'build.json')
with tempfile.TemporaryDirectory() as td:
 work=Path(td);assets=hosts()
 for n,p in assets.items():shutil.copyfile(p,work/n)
 for n in ('fixture.json','fixture.npz'):shutil.copyfile(case/n,work/n)
 out=work/'out';shutil.copytree(sim/'out',out);inv=actual(out)
 desc.update(inventory=inv,host_sha256={n:sha(f) for n,f in assets.items()},fixture_sha256=sha(case/'fixture.json'));save(work/'build.json',desc)
 verified(work,out)
 target=next((out/'west/bin').glob('*.elf'));target.write_bytes(target.read_bytes()+b'X')
 try: verified(work,out)
 except ValueError: pass
 else: raise RuntimeError('modified DMA firmware accepted')
"""
        subprocess.run([sys.executable, "-B", "-c", code, str(ROOT)], check=True)

    def test_rejects_a_simulator_return_as_a_system_measurement(self):
        code = """from pathlib import Path
import sys
sys.path.insert(0,sys.argv[1]);from physical import observed,read
root=Path(sys.argv[1]);case=sorted((root/'cases').iterdir())[0]
try: observed(root/'base/physical',case,case/'sdk')
except Exception: pass
else: raise RuntimeError('simulator returned as a system frame')
"""
        subprocess.run(
            [sys.executable, "-B", "-c", code, str(ROOT)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
