"""Offline catalog, commit admission, and inherited-source integrity checks."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from csl_validation import catalog
import test_csl_watch as controller_fixtures


ROOT = Path(__file__).resolve().parents[1]


def descriptor(family="new-family"):
    return {"schema": 1, "id": family, "description": "Offline extension fixture",
            "entrypoints": ["adapter.py"], "tests": ["tests"]}


def request(ident="new-case", script="validations/new-family/adapter.py"):
    return {"schema": 1, "id": ident, "description": "Offline admission only",
            "steps": [{"name": "compile", "script": script, "args": [],
                       "timeout_seconds": 10, "kind": "appliance"}]}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def add_family(root):
    family = root / "validations/new-family"
    (family / "tests").mkdir(parents=True)
    (family / "adapter.py").write_text("raise RuntimeError('must never execute during discovery')\n")
    write_json(family / "validation.json", descriptor())
    return family


class ShippedCatalogTests(unittest.TestCase):
    def test_shipped_families_and_requests_are_discoverable(self):
        families = {value["id"]: value for _, value in catalog.definitions(ROOT)}
        expected = {"native16": "appliance.py", "clocks": "clock_appliance.py",
                    "meshgraphnet": "mesh_appliance.py", "primitives": "tank_appliance.py"}
        for family, entrypoint in expected.items():
            with self.subTest(family=family):
                self.assertIn(entrypoint, families[family]["entrypoints"])
        values = {value["id"]: (path, value) for path, value in catalog.requests(ROOT)}
        for ident, family in (("native16-baseline-3-7-0", "native16"),
                              ("topk-small-aba", "primitives")):
            with self.subTest(request=ident):
                path, value = values[ident]
                self.assertTrue(path.startswith("requests/" + family + "/"))
                self.assertEqual([step["kind"] for step in value["steps"]], ["appliance", "appliance"])
                for step in value["steps"]:
                    self.assertTrue(step["script"].startswith("validations/" + family + "/"))

    def test_relocated_sources_match_recorded_git_revision(self):
        record = json.loads((ROOT / "docs/architecture/relocation.json").read_text())
        self.assertEqual(record["schema"], 1)
        self.assertEqual(len(record["paths"]), 262)
        if shutil.which("git") is None:
            self.skipTest("Git is unavailable; relocation byte proof requires repository history")
        base = record["base_revision"]
        probe = subprocess.run(["git", "cat-file", "-e", base + "^{commit}"], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        if probe.returncode:
            self.skipTest("Recorded relocation revision is absent (source archive or shallow clone)")
        for old, new in record["paths"].items():
            with self.subTest(source=old, relocated=new):
                original = subprocess.run(["git", "show", base + ":" + old], cwd=ROOT,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          timeout=15, check=True).stdout
                # Prove the initial move, not permanent immutability of live code.
                # Before that move is committed, check the proposed working tree.
                additions = subprocess.run(
                    ["git", "log", "--no-renames", "--format=%H", "--diff-filter=A",
                     "--reverse", base + "..HEAD", "--", new], cwd=ROOT,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
                    check=True).stdout.decode().splitlines()
                relocated = (subprocess.run(
                    ["git", "show", additions[0] + ":" + new], cwd=ROOT,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
                    check=True).stdout if additions else (ROOT / new).read_bytes())
                self.assertEqual(relocated, original)


class CatalogExtensionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.family = add_family(self.root)
        self.path = Path("requests/new-family/sweep/small.json")
        self.value = request()
        write_json(self.root / self.path, self.value)

    def test_new_family_and_nested_request_need_no_central_list_edit(self):
        families = catalog.definitions(self.root)
        self.assertEqual([(path.name, value["id"]) for path, value in families],
                         [("new-family", "new-family")])
        self.assertEqual(catalog.requests(self.root), [(str(self.path), self.value)])

    def test_undeclared_script_rejected_even_when_it_exists(self):
        (self.family / "other.py").write_text("pass\n")
        self.value["steps"][0]["script"] = "validations/new-family/other.py"
        write_json(self.root / self.path, self.value)
        with self.assertRaisesRegex(ValueError, "allowlist"):
            catalog.request_file(self.root, self.path)

    def test_missing_descriptor_rejects_request(self):
        (self.family / "validation.json").unlink()
        with self.assertRaises((ValueError, OSError)):
            catalog.request_file(self.root, self.path)

    def test_missing_script_rejects_request_and_family(self):
        (self.family / "adapter.py").unlink()
        with self.assertRaisesRegex(ValueError, "missing request script"):
            catalog.request_file(self.root, self.path)
        with self.assertRaisesRegex(ValueError, "missing entrypoint"):
            catalog.definitions(self.root)

    def test_script_path_escape_and_noncanonical_paths_rejected(self):
        for script in ("validations/new-family/../../outside.py", "/tmp/outside.py",
                       "validations/new-family//adapter.py"):
            with self.subTest(script=script):
                value = copy.deepcopy(self.value)
                value["steps"][0]["script"] = script
                write_json(self.root / self.path, value)
                with self.assertRaises(ValueError):
                    catalog.request_file(self.root, self.path)

    def test_descriptor_path_escape_rejected(self):
        value = descriptor()
        value["entrypoints"] = ["../../outside.py"]
        write_json(self.family / "validation.json", value)
        with self.assertRaisesRegex(ValueError, "relative path"):
            catalog.definitions(self.root)
        with self.assertRaisesRegex(ValueError, "relative path"):
            catalog.request_file(self.root, self.path)

    def test_metadata_identity_must_match_family_directory(self):
        write_json(self.family / "validation.json", descriptor("different-family"))
        with self.assertRaisesRegex(ValueError, "identity"):
            catalog.definitions(self.root)

    def test_nested_campaign_cannot_be_declared_as_entrypoint(self):
        value = descriptor()
        value["entrypoints"] = ["nested_campaign.py"]
        write_json(self.family / "validation.json", value)
        with self.assertRaisesRegex(ValueError, "foreground action"):
            catalog.definitions(self.root)

    def test_duplicate_request_ids_rejected_across_nested_directories(self):
        write_json(self.root / "requests/another-group/deep/duplicate.json", self.value)
        with self.assertRaisesRegex(ValueError, "duplicate request ID"):
            catalog.requests(self.root)

    def test_symlinked_script_descriptor_and_request_rejected(self):
        for relative in ("validations/new-family/adapter.py",
                         "validations/new-family/validation.json", str(self.path)):
            with self.subTest(path=relative):
                path = self.root / relative
                original = path.read_bytes()
                target = self.root / "saved-file"
                target.write_bytes(original)
                path.unlink()
                path.symlink_to(target)
                try:
                    with self.assertRaisesRegex(ValueError, "symlink"):
                        catalog.request_file(self.root, self.path)
                finally:
                    path.unlink()
                    path.write_bytes(original)
                    target.unlink()

    def test_symlinked_family_directory_rejected(self):
        relocated = self.root / "family-target"
        self.family.rename(relocated)
        self.family.symlink_to(relocated, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            catalog.definitions(self.root)
        with self.assertRaisesRegex(ValueError, "symlink"):
            catalog.request_file(self.root, self.path)


class GitMetadataAdmissionTests(unittest.TestCase):
    def setUp(self):
        # Reuse only the isolated fixture, not its test methods. It uses local
        # file remotes and a fake scheduler; discovery never starts a child.
        self.fixture = controller_fixtures.GitControllerTests(methodName="runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def commit(self, message):
        fixture = self.fixture
        fixture.git("add", ".", cwd=fixture.source)
        fixture.git("commit", "-m", message, cwd=fixture.source)
        fixture.git("push", str(fixture.remote), "HEAD:refs/heads/main", cwd=fixture.source)
        return fixture.git("rev-parse", "HEAD", cwd=fixture.source).strip()

    def test_discovery_uses_descriptor_from_request_sha_not_latest_tip(self):
        fixture = self.fixture
        family = add_family(fixture.source)
        path = "requests/new-family/sweep/small.json"
        write_json(fixture.source / path, request())
        request_sha = self.commit("Request new family\n\nALCF-Request: new-case\nALCF-Manifest: " + path)
        later = descriptor()
        later["entrypoints"] = ["replacement.py"]
        write_json(family / "validation.json", later)
        (family / "replacement.py").write_text("raise RuntimeError('do not execute')\n")
        tip = self.commit("Change family metadata without requesting another run")
        fixture.watcher.discover()
        entry = fixture.watcher.ledger["requests"]["new-case"]
        self.assertEqual(entry["sha"], request_sha)
        self.assertEqual(entry["manifest_path"], path)
        self.assertEqual(entry["status"], "pending")
        self.assertEqual(fixture.watcher.ledger["cursor"], tip)
        self.assertNotIn("validations/new-family/adapter.py", fixture.config["allowed_scripts"])
        self.assertFalse((fixture.state / "requests/new-case/src").exists())

    def test_later_descriptor_cannot_retroactively_admit_earlier_request(self):
        fixture = self.fixture
        family = add_family(fixture.source)
        (family / "validation.json").unlink()
        path = "requests/new-family/sweep/small.json"
        write_json(fixture.source / path, request())
        self.commit("Request without descriptor\n\nALCF-Request: new-case\nALCF-Manifest: " + path)
        write_json(family / "validation.json", descriptor())
        self.commit("Add descriptor too late")
        with self.assertRaises((ValueError, RuntimeError, subprocess.CalledProcessError)):
            fixture.watcher.discover()
        self.assertEqual(fixture.watcher.ledger["cursor"], fixture.baseline)
        self.assertEqual(fixture.watcher.ledger["requests"], {})

    def test_git_symlink_descriptor_cannot_admit_request(self):
        fixture = self.fixture
        family = add_family(fixture.source)
        metadata = family / "validation.json"
        metadata.unlink()
        metadata.symlink_to(json.dumps(descriptor()))
        path = "requests/new-family/small.json"
        write_json(fixture.source / path, request())
        self.commit("Request with symlink descriptor\n\nALCF-Request: new-case\nALCF-Manifest: " + path)
        with self.assertRaisesRegex(ValueError, "regular Git file"):
            fixture.watcher.discover()
        self.assertEqual(fixture.watcher.ledger["requests"], {})


if __name__ == "__main__":
    unittest.main()
