"""Offline storage tests: temporary local Git only; no SDK or scheduler calls."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

from csl_validation import storage, watch
import test_csl_watch as fixtures


class StorageTests(unittest.TestCase):
    def setUp(self):
        # Composition avoids rediscovering the controller TestCase's inherited tests.
        self.fixture = fixtures.GitControllerTests(methodName="runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.watcher = self.fixture.watcher
        self.root = self.fixture.state
        self.cache = self.watcher.cache
        self.git = self.fixture.git
        self.watcher.fetch()  # Only the fixture's file-protocol temporary remote.

    def entry(self, ident, *, status="passed", packaged=True, sha=None):
        directory = self.root / "requests" / ident
        (directory / "output").mkdir(parents=True)
        (directory / "logs").mkdir()
        (directory / "logs/step.log").write_bytes(b"retained operator log\n")
        value = dict(id=ident, sha=sha or self.fixture.baseline, status=status,
                     packaged=packaged, steps=[], notified=False,
                     manifest=fixtures.request(ident), manifest_path="requests/example.json")
        watch.atomic_json(directory / "request.json", value["manifest"])
        watch.atomic_json(directory / "environment.json", {"python": "offline fixture"})
        if packaged:
            exports = self.root / "exports"
            exports.mkdir(exist_ok=True)
            archive = exports / (ident + ".zip")
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("receipt.json", json.dumps(value))
            value.update(bundle=str(archive), bundle_sha256=watch.digest(archive))
            archive.with_suffix(".zip.sha256").write_text(value["bundle_sha256"] + "\n")
        self.watcher.ledger["requests"][ident] = value
        self.watcher.ledger["order"].append(ident)
        self.watcher.record(value)
        return value

    def output(self, entry, name, data=b"equal retained payload\0" * 128, mode=0o600):
        path = self.root / "requests" / entry["id"] / "output" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(mode)
        return path

    def source(self, entry, *, clone=False, cache=None):
        path = self.root / "requests" / entry["id"] / "src"
        if clone:
            self.git("clone", "--no-hardlinks", "--no-checkout", str(self.cache), str(path))
            self.git("checkout", "--detach", entry["sha"], cwd=path)
        else:
            self.git("--git-dir", str(cache or self.cache), "worktree", "add", "--detach",
                     str(path), entry["sha"])
        return path

    @staticmethod
    def snapshot(paths):
        return {str(p): (p.read_bytes(), p.stat().st_ino, p.stat().st_mode) for p in paths}

    def evidence(self):
        # Git metadata and outputs have their own checks; these must never change.
        return [p for p in self.root.rglob("*") if p.is_file() and
                "source.git" not in p.parts and "src" not in p.parts and "output" not in p.parts]

    def prepare_without_sdk(self, entry):
        command = watch.command

        def offline(argv, *args, **kwargs):
            if argv[0] == "git":
                return command(argv, *args, **kwargs)
            if argv[0] == "OFFLINE-SDK":
                return json.dumps({"python": "offline", "executable": "offline", "packages": {}})
            if argv == ["OFFLINE-UV", "--version"]:
                return "uv offline fixture\n"
            self.fail("unexpected external command: " + repr(argv))

        with patch.object(watch, "command", side_effect=offline), \
             patch.object(watch, "sdk_argv", return_value=["OFFLINE-SDK"]), \
             patch.object(watch, "resolve_uv", return_value="OFFLINE-UV"), \
             patch.object(watch.shutil, "disk_usage", return_value=fixtures.SimpleNamespace(free=2 << 30)):
            self.watcher.prepare(entry)

    def test_prepare_shares_objects_but_keeps_detached_sha_and_files_independent(self):
        first_sha = self.fixture.submit(fixtures.request("first"))
        second_sha = self.fixture.submit(fixtures.request("second"))
        self.watcher.discover()
        paths = []
        for ident, sha in (("first", first_sha), ("second", second_sha)):
            entry = self.watcher.ledger["requests"][ident]
            self.prepare_without_sdk(entry)
            src = self.root / "requests" / ident / "src"
            paths.append(src)
            self.assertTrue((src / ".git").is_file())
            self.assertFalse((src / ".git").is_dir())
            self.assertEqual(self.git("rev-parse", "HEAD", cwd=src).strip(), sha)
            self.assertEqual(self.git("rev-parse", "--abbrev-ref", "HEAD", cwd=src).strip(), "HEAD")
            common = self.git("rev-parse", "--git-common-dir", cwd=src).strip()
            self.assertEqual((src / common).resolve(), self.cache)
            self.assertEqual(watch.read_json(src / "requests/example.json")["id"], ident)
            self.assertEqual(self.git("--git-dir", str(self.cache), "rev-parse",
                                      "refs/requests/" + ident).strip(), sha)
        self.assertNotEqual(first_sha, second_sha)
        original = (paths[1] / "fake_appliance.py").read_bytes()
        (paths[0] / "fake_appliance.py").write_bytes(b"private dirty source\n")
        self.assertEqual((paths[1] / "fake_appliance.py").read_bytes(), original)
        self.assertFalse(any((self.root / "requests").glob("*/output/invocations.txt")))

    def test_pin_is_idempotent_and_rejects_conflicting_revision(self):
        sha = self.fixture.submit()
        self.watcher.discover()
        entry = self.fixture.entry()
        ref = storage.pin_source(self.cache, entry, apply=False)
        self.assertEqual(self.git("--git-dir", str(self.cache), "for-each-ref", ref), "")
        self.assertEqual(storage.pin_source(self.cache, entry), ref)
        self.assertEqual(storage.pin_source(self.cache, entry), ref)
        for apply in (False, True):
            with self.subTest(apply=apply), self.assertRaisesRegex(ValueError, "pin conflict"):
                storage.pin_source(self.cache, dict(entry, sha=self.fixture.baseline), apply=apply)
        self.assertEqual(self.git("--git-dir", str(self.cache), "rev-parse", ref).strip(), sha)

    def test_clean_source_dry_run_and_apply_preserve_evidence_and_pin(self):
        entry = self.entry("clean")
        src = self.source(entry)
        self.output(entry, "capture.bin")
        before = self.snapshot(self.evidence())
        self.assertEqual(storage.clean_source(self.watcher, entry)["status"], "removable")
        self.assertTrue(src.exists())
        result = storage.clean_source(self.watcher, entry, apply=True)
        self.assertEqual((result["status"], result["kind"]), ("removed", "worktree"))
        self.assertFalse(src.exists())
        self.assertEqual(self.snapshot(self.evidence()), before)
        self.assertEqual(self.git("--git-dir", str(self.cache), "rev-parse", result["ref"]).strip(), entry["sha"])
        self.assertEqual(storage.clean_source(self.watcher, entry, apply=True)["status"], "absent")
        self.assertEqual(self.git("--git-dir", str(self.cache), "worktree", "list", "--porcelain").count(str(src)), 0)

    def test_legacy_clone_cleanup_preserves_clone_with_local_refs(self):
        entry = self.entry("legacy")
        src = self.source(entry, clone=True)
        self.assertTrue((src / ".git/objects").is_dir())
        self.git("branch", "operator-save", entry["sha"], cwd=src)
        result = storage.clean_source(self.watcher, entry, apply=True)
        self.assertEqual(result["status"], "retained")
        self.assertIn("local refs", result["reason"])
        self.assertTrue(src.exists())
        self.git("branch", "-D", "operator-save", cwd=src)
        result = storage.clean_source(self.watcher, entry, apply=True)
        self.assertEqual((result["status"], result["kind"]), ("removed", "clone"))
        self.assertFalse(src.exists())

    def test_sdk_sidecars_are_preserved_before_terminal_source_removal(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                entry = self.entry("sdk-legacy" if legacy else "sdk-worktree")
                entry["steps"] = [dict(jobs=["wsjob-%02d" % i for i in range(28)])]
                src = self.source(entry, clone=legacy)
                payloads = {"run_meta.json": b'{"last_job":"wsjob-27"}\n'}
                payloads.update({job + "-cluster-details.json": (job + "\n").encode()
                                 for job in entry["steps"][0]["jobs"]})
                for name, data in payloads.items():
                    (src / name).write_bytes(data)
                self.output(entry, "capture.bin")
                evidence = self.snapshot(self.evidence())
                output = self.snapshot(list((src.parent / "output").iterdir()))
                destination = src.parent / "sdk-metadata"
                dry = storage.clean_source(self.watcher, entry)
                self.assertEqual(dry["status"], "removable")
                self.assertEqual(len(dry["sdk_metadata"]), 29)
                self.assertFalse(destination.exists())
                self.assertEqual({p.name: p.read_bytes() for p in src.glob("*.json")}, payloads)
                result = storage.clean_source(self.watcher, entry, apply=True)
                self.assertEqual(result["status"], "removed")
                self.assertFalse(src.exists())
                self.assertEqual({p.name: p.read_bytes() for p in destination.iterdir()}, payloads)
                self.assertTrue(all(item["status"] == "preserved" for item in result["sdk_metadata"]))
                for item in result["sdk_metadata"]:
                    self.assertEqual(item["sha256"], watch.digest(destination / item["name"]))
                self.assertEqual(self.snapshot([Path(name) for name in evidence]), evidence)
                self.assertEqual(self.snapshot(list((src.parent / "output").iterdir())), output)
                self.assertEqual(storage.clean_source(self.watcher, entry, apply=True)["status"], "absent")

    def test_sdk_metadata_does_not_relax_other_source_guards(self):
        for kind in ("tracked", "ignored", "unknown-job", "nested", "symlink", "branch", "conflict", "dest-link"):
            with self.subTest(kind=kind):
                entry = self.entry(kind)
                src = self.source(entry, clone=True)
                meta = src / "run_meta.json"
                meta.write_bytes(b"original SDK metadata")
                destination = src.parent / "sdk-metadata"
                if kind == "tracked":
                    (src / "fake_appliance.py").write_bytes(b"local change")
                elif kind == "ignored":
                    (src / ".git/info/exclude").write_text("*.scratch\n")
                    (src / "user.scratch").write_bytes(b"keep me")
                elif kind == "unknown-job":
                    (src / "wsjob-other-cluster-details.json").write_bytes(b"unrecorded job")
                elif kind == "nested":
                    (src / "nested").mkdir()
                    (src / "nested/run_meta.json").write_bytes(b"nested file")
                elif kind == "symlink":
                    meta.unlink()
                    meta.symlink_to(self.fixture.source / "fake_appliance.py")
                elif kind == "branch":
                    self.git("branch", "operator-save", entry["sha"], cwd=src)
                elif kind == "conflict":
                    destination.mkdir()
                    (destination / meta.name).write_bytes(b"different earlier metadata")
                else:
                    outside = self.fixture.base / "outside-metadata"
                    outside.mkdir()
                    destination.symlink_to(outside, target_is_directory=True)
                original = meta.read_bytes()
                result = storage.clean_source(self.watcher, entry, apply=True)
                self.assertEqual(result["status"], "retained", result)
                self.assertTrue(src.exists())
                self.assertEqual(meta.read_bytes(), original)
                if kind not in ("conflict", "dest-link"):
                    self.assertFalse(destination.exists())

    def test_metadata_preflight_checks_every_destination_before_first_move(self):
        entry = self.entry("late-conflict")
        entry["steps"] = [dict(jobs=["wsjob-last"])]
        src = self.source(entry)
        (src / "run_meta.json").write_bytes(b"first")
        last = "wsjob-last-cluster-details.json"
        (src / last).write_bytes(b"last")
        destination = src.parent / "sdk-metadata"
        destination.mkdir()
        (destination / last).write_bytes(b"previous contents")
        result = storage.clean_source(self.watcher, entry, apply=True)
        self.assertEqual(result["status"], "retained")
        self.assertFalse((destination / "run_meta.json").exists())
        self.assertEqual((src / "run_meta.json").read_bytes(), b"first")
        self.assertEqual((src / last).read_bytes(), b"last")
        self.assertEqual((destination / last).read_bytes(), b"previous contents")

    def test_interrupted_metadata_preservation_resumes_without_overwriting(self):
        entry = self.entry("partial-metadata")
        entry["steps"] = [dict(jobs=["wsjob-last"])]
        src = self.source(entry)
        metadata = {"run_meta.json": b"first", "wsjob-last-cluster-details.json": b"last"}
        for name, data in metadata.items():
            (src / name).write_bytes(data)
        destination = src.parent / "sdk-metadata"
        evidence = self.snapshot(self.evidence())
        native_link = os.link
        calls = []

        def interrupted(source, target, **kwargs):
            native_link(source, target, **kwargs)
            calls.append(Path(target).name)
            if len(calls) == 2:
                raise OSError("injected interruption after publishing destination")

        with patch.object(storage.os, "link", side_effect=interrupted):
            report = storage.compact(self.watcher, apply=True)
        self.assertTrue(report["errors"])
        result = report["sources"][0]
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["sdk_metadata"][0]["status"], "preserved")
        self.assertFalse((src / "run_meta.json").exists())
        self.assertTrue((src / "wsjob-last-cluster-details.json").exists())
        self.assertEqual({p.name: p.read_bytes() for p in destination.iterdir()}, metadata)
        # An independent but identical existing destination is also admissible.
        target = destination / "wsjob-last-cluster-details.json"
        target.unlink()
        target.write_bytes(metadata[target.name])
        result = storage.clean_source(self.watcher, entry, apply=True)
        self.assertEqual(result["status"], "removed", result)
        self.assertEqual({p.name: p.read_bytes() for p in destination.iterdir()}, metadata)
        after = {name: value for name, value in self.snapshot(self.evidence()).items()
                 if "/sdk-metadata/" not in name}
        self.assertEqual(after, evidence)

    def test_live_sdk_metadata_is_not_moved(self):
        for status in ("pending", "running", "blocked"):
            entry = self.entry("live-" + status, status=status)
            src = self.source(entry)
            (src / "run_meta.json").write_bytes(b"active SDK consumer")
            result = storage.clean_source(self.watcher, entry, apply=True)
            self.assertEqual(result["status"], "ineligible")
            self.assertEqual((src / "run_meta.json").read_bytes(), b"active SDK consumer")
            self.assertFalse((src.parent / "sdk-metadata").exists())

    def test_changed_untracked_and_ignored_source_are_retained(self):
        (self.cache / "info/exclude").write_text("*.scratch\n")
        for kind in ("changed", "untracked", "ignored"):
            with self.subTest(kind=kind):
                entry = self.entry(kind)
                src = self.source(entry)
                name = {"changed": "fake_appliance.py", "untracked": "capture.bin",
                        "ignored": "capture.scratch"}[kind]
                path = src / name
                path.write_bytes(b"operator data must survive\n")
                before = self.snapshot([path])
                result = storage.clean_source(self.watcher, entry, apply=True)
                self.assertEqual(result["status"], "retained")
                self.assertIn("changed, untracked or ignored", result["reason"])
                self.assertEqual(self.snapshot([path]), before)

    def test_source_symlink_foreign_store_and_wrong_revision_are_retained(self):
        outside = self.fixture.base / "outside"
        outside.mkdir()
        marker = outside / "retain.txt"
        marker.write_bytes(b"outside source")
        entry = self.entry("symlink")
        src = self.root / "requests/symlink/src"
        src.symlink_to(outside, target_is_directory=True)
        self.assertEqual(storage.clean_source(self.watcher, entry, apply=True)["status"], "retained")
        self.assertTrue(src.is_symlink())
        self.assertEqual(marker.read_bytes(), b"outside source")
        entry = self.entry("foreign")
        src = self.source(entry, cache=self.fixture.remote)
        result = storage.clean_source(self.watcher, entry, apply=True)
        self.assertEqual(result["status"], "retained")
        self.assertIn("foreign shared", result["reason"])
        self.assertTrue(src.exists())
        entry = self.entry("wrong-revision")
        src = self.source(entry)
        sha = self.fixture.submit()
        self.watcher.fetch()
        entry["sha"] = sha
        result = storage.clean_source(self.watcher, entry, apply=True)
        self.assertEqual(result["status"], "retained")
        self.assertIn("revision changed", result["reason"])
        self.assertTrue(src.exists())

    def test_missing_source_is_safe_but_missing_commit_is_not_pinned(self):
        entry = self.entry("missing")
        self.assertEqual(storage.clean_source(self.watcher, entry, apply=True)["status"], "absent")
        self.assertFalse((self.root / "requests/missing/src").exists())
        entry = self.entry("unknown", sha="f" * 40)
        self.assertEqual(storage.clean_source(self.watcher, entry, apply=True)["status"], "retained")
        self.assertEqual(self.git("--git-dir", str(self.cache), "for-each-ref", "refs/requests/unknown"), "")

    def test_tick_cleans_only_packaged_terminal_source_and_keeps_ignored_data(self):
        clean = self.entry("terminal")
        clean_src = self.source(clean)
        dirty = self.entry("ignored")
        dirty_src = self.source(dirty)
        (self.cache / "info/exclude").write_text("*.scratch\n")
        (dirty_src / "capture.scratch").write_bytes(b"retained")
        unfinished = self.entry("unpackaged", packaged=False)
        unfinished_src = self.source(unfinished)
        # Busy fake scheduler prevents discovery; no request is executed.
        self.fixture.scheduler.busy = True
        with patch.object(self.watcher, "package") as package, \
             patch.object(self.watcher, "start_step", side_effect=AssertionError("must not run")):
            self.watcher.tick()
        package.assert_called_once_with(unfinished)
        self.assertFalse(clean_src.exists())
        self.assertEqual(clean["source_cleanup"]["status"], "removed")
        self.assertEqual(dirty["source_cleanup"]["status"], "retained")
        self.assertEqual((dirty_src / "capture.scratch").read_bytes(), b"retained")
        self.assertTrue(unfinished_src.exists())
        self.assertNotIn("source_cleanup", unfinished)
        with patch.object(storage, "clean_source", side_effect=AssertionError("do not retry cleanup")), \
             patch.object(self.watcher, "package"):
            self.watcher.tick()

    def test_compact_exact_bytes_keeps_paths_modes_and_evidence_and_is_idempotent(self):
        first, second = self.entry("first"), self.entry("second", status="failed")
        a = self.output(first, "nested/capture.bin")
        b = self.output(second, "renamed.bin", a.read_bytes())
        unequal = self.output(second, "different.bin", b"x" * len(a.read_bytes()))
        mode = self.output(second, "executable.bin", a.read_bytes(), mode=0o700)
        empty = self.output(second, "empty.bin", b"")
        paths = [a, b, unequal, mode, empty]
        before = self.snapshot(paths)
        evidence = self.snapshot(self.evidence())
        dry = storage.compact(self.watcher)
        self.assertEqual(dry["errors"], [])
        self.assertEqual((dry["duplicate_files"], dry["duplicate_bytes"], dry["linked_files"]), (1, a.stat().st_size, 0))
        self.assertEqual(self.snapshot(paths), before)
        self.assertEqual(self.snapshot(self.evidence()), evidence)
        applied = storage.compact(self.watcher, apply=True)
        self.assertEqual(applied["errors"], [])
        self.assertEqual((applied["duplicate_files"], applied["linked_files"]), (1, 1))
        self.assertTrue(a.samefile(b))
        self.assertFalse(a.samefile(unequal))
        self.assertFalse(a.samefile(mode))
        self.assertEqual({str(p): (p.read_bytes(), p.stat().st_mode) for p in paths},
                         {name: (value[0], value[2]) for name, value in before.items()})
        self.assertEqual(self.snapshot(self.evidence()), evidence)
        repeated = storage.compact(self.watcher, apply=True)
        self.assertEqual(repeated["errors"], [])
        self.assertEqual((repeated["duplicate_files"], repeated["linked_files"]), (0, 0))

    def test_compact_ignores_active_blocked_and_unpackaged_requests(self):
        done = self.entry("done")
        canonical = self.output(done, "result.bin")
        protected = []
        for status, packaged in (("pending", True), ("running", True), ("blocked", True),
                                 ("passed", False), ("failed", False), ("interrupted", False)):
            entry = self.entry(status + str(packaged).lower(), status=status, packaged=packaged)
            protected.append(self.output(entry, "result.bin", canonical.read_bytes()))
            src = self.root / "requests" / entry["id"] / "src"
            src.mkdir()
            (src / "marker").write_bytes(b"never inspect or delete active source")
        before = self.snapshot(protected)
        result = storage.compact(self.watcher, apply=True)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["files"], 1)
        self.assertEqual(result["linked_files"], 0)
        self.assertEqual(self.snapshot(protected), before)
        self.assertTrue(all(item["status"] == "ineligible" for item in result["sources"][1:]))
        self.assertEqual(len(list((self.root / "requests").glob("*/src/marker"))), 6)

    def test_request_filter_does_not_link_to_another_request(self):
        first, second = self.entry("first"), self.entry("second", status="interrupted")
        a = self.output(first, "a.bin")
        b = self.output(second, "b.bin", a.read_bytes())
        result = storage.compact(self.watcher, apply=True, request="second")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["files"], 1)
        self.assertFalse(a.samefile(b))
        self.assertEqual([item["id"] for item in result["sources"]], ["second"])

    def test_preexisting_alias_to_active_source_is_not_adopted(self):
        done = self.entry("done")
        a = self.output(done, "a.bin")
        b = self.output(done, "b.bin", a.read_bytes())
        active = self.entry("active", status="running", packaged=False)
        source = self.root / "requests" / active["id"] / "src"
        source.mkdir()
        alias = source / "runtime-data.bin"
        os.link(a, alias)
        before = self.snapshot([a, b, alias])
        result = storage.compact(self.watcher, apply=True)
        self.assertEqual(result["errors"], [])
        self.assertGreaterEqual(result["external_links_skipped"], 1)
        self.assertEqual(result["linked_files"], 0)
        self.assertEqual(self.snapshot([a, b, alias]), before)
        self.assertFalse(b.samefile(alias))

    def test_existing_links_fully_inside_selected_outputs_remain_admissible(self):
        entry = self.entry("internal")
        a = self.output(entry, "a.bin")
        b = self.output(entry, "b.bin", a.read_bytes())
        alias = a.parent / "alias.bin"
        os.link(a, alias)
        result = storage.compact(self.watcher, apply=True)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["external_links_skipped"], 0)
        self.assertEqual(result["linked_files"], 1)
        self.assertTrue(a.samefile(b) and a.samefile(alias))

    def test_output_symlink_and_missing_inventory_fail_closed(self):
        entry = self.entry("linked")
        a = self.output(entry, "a.bin")
        b = self.output(entry, "b.bin", a.read_bytes())
        (a.parent / "escape").symlink_to(self.fixture.source, target_is_directory=True)
        before = self.snapshot([a, b])
        result = storage.compact(self.watcher, apply=True)
        self.assertTrue(result["errors"])
        self.assertEqual(result["linked_files"], 0)
        self.assertEqual(self.snapshot([a, b]), before)
        entry = self.entry("missing-output")
        (self.root / "requests/missing-output/output").rmdir()
        result = storage.compact(self.watcher, apply=True, request=entry["id"])
        self.assertTrue(result["errors"])
        self.assertEqual(result["files"], 0)

    def test_link_boundary_source_mutation_or_symlink_never_replaces_target(self):
        native_link = os.link
        for kind in ("mutation", "symlink"):
            with self.subTest(kind=kind):
                entry = self.entry(kind)
                source = self.output(entry, "a.bin")
                target = self.output(entry, "b.bin", source.read_bytes())
                target_before = self.snapshot([target])
                old = source.stat()
                replacement = b"!" * old.st_size
                outside = self.fixture.base / (kind + "-outside")
                outside.write_bytes(replacement)

                def interfere(src, dst, **kwargs):
                    self.assertEqual(Path(src), source)
                    self.assertIs(kwargs.get("follow_symlinks"), False)
                    if kind == "mutation":
                        source.write_bytes(replacement)
                        # Equal size and restored mtime defeat a metadata-only check.
                        os.utime(source, ns=(old.st_atime_ns, old.st_mtime_ns))
                    else:
                        source.unlink()
                        source.symlink_to(outside)
                    native_link(src, dst, **kwargs)

                with patch.object(storage.os, "link", side_effect=interfere) as link:
                    result = storage.compact(self.watcher, apply=True, request=entry["id"])
                link.assert_called_once()
                self.assertTrue(result["errors"])
                self.assertEqual(result["linked_files"], 0)
                self.assertEqual(self.snapshot([target]), target_before)
                self.assertEqual(outside.read_bytes(), replacement)
                self.assertFalse(list(target.parent.parent.glob(".compact-*")))

    def test_hash_collision_is_rejected_by_fresh_byte_comparison(self):
        entry = self.entry("collision")
        a = self.output(entry, "a.bin", b"left")
        b = self.output(entry, "b.bin", b"rght")
        before = self.snapshot([a, b])
        with patch.object(storage, "file_hash", return_value=b"same-hash"):
            result = storage.compact(self.watcher, apply=True)
        self.assertTrue(result["errors"])
        self.assertIn("unequal contents", result["errors"][0]["error"])
        self.assertEqual(self.snapshot([a, b]), before)

    def test_canonical_mode_change_after_hash_does_not_change_target_access(self):
        entry = self.entry("chmod-race")
        a, b = self.output(entry, "a.bin"), self.output(entry, "b.bin")
        before = self.snapshot([b])
        original_hash = storage.file_hash

        def hash_then_change_canonical(path, deadline):
            if path == b:
                a.chmod(0o700)
            return original_hash(path, deadline)

        with patch.object(storage, "file_hash", side_effect=hash_then_change_canonical):
            result = storage.compact(self.watcher, apply=True)
        self.assertTrue(result["errors"])
        self.assertIn("canonical output metadata changed", result["errors"][0]["error"])
        self.assertEqual(result["linked_files"], 0)
        self.assertEqual(self.snapshot([b]), before)
        self.assertFalse(a.samefile(b))

    def test_partial_clone_deletion_reports_incomplete_without_demoting_result(self):
        entry = self.entry("partial-delete")
        src = self.source(entry, clone=True)
        self.output(entry, "capture.bin")
        before = self.snapshot(self.evidence())
        original = copy.deepcopy(entry)

        def interrupted_delete(argv, **kwargs):
            self.assertEqual(Path(argv[-1]), src)
            self.assertGreater(kwargs["seconds"], 0)
            self.assertLessEqual(kwargs["seconds"], 120)
            (src / "fake_appliance.py").unlink()
            raise ValueError("injected removal timeout")

        with patch.object(storage, "command", side_effect=interrupted_delete) as delete:
            result = storage.compact(self.watcher, apply=True)
        delete.assert_called_once()
        record = result["sources"][0]
        self.assertEqual(record["status"], "incomplete")
        self.assertEqual(record["path"], str(src))
        self.assertEqual(record["ref"], "refs/requests/partial-delete")
        self.assertIn("injected removal timeout", record["reason"])
        self.assertTrue(result["errors"])
        self.assertEqual(entry, original)
        self.assertEqual(self.snapshot(self.evidence()), before)
        self.assertTrue(src.exists())
        self.assertFalse((src / "fake_appliance.py").exists())
        self.assertEqual(self.git("--git-dir", str(self.cache), "rev-parse", record["ref"]).strip(), entry["sha"])

    def test_time_budget_and_unknown_request_fail_before_mutating(self):
        entry = self.entry("bounded")
        a, b = self.output(entry, "a.bin"), self.output(entry, "b.bin")
        before = self.snapshot([a, b, self.root / "ledger.json"])
        for kwargs in ({"seconds": 0}, {"request": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                storage.compact(self.watcher, apply=True, **kwargs)
        with patch.object(storage.time, "monotonic", side_effect=[0, 2]):
            result = storage.compact(self.watcher, apply=True, seconds=1)
        self.assertIn("time budget exhausted", result["errors"][0]["error"])
        self.assertEqual(result["linked_files"], 0)
        self.assertEqual(self.snapshot([a, b, self.root / "ledger.json"]), before)

    def test_compact_cli_does_not_discover_schedule_or_run_sdk(self):
        entry = self.entry("complete")
        a, b = self.output(entry, "a.bin"), self.output(entry, "b.bin")
        self.entry("blocked", status="blocked", packaged=False)
        config = dict(self.fixture.config, idle_seconds=60, poll_seconds=1)
        path = self.fixture.base / "config.json"
        watch.atomic_json(path, config)
        before = self.snapshot([a, b, *self.evidence()])
        output = io.StringIO()
        with patch.object(watch, "Scheduler") as scheduler, \
             patch.object(watch.Watcher, "tick", side_effect=AssertionError("must not tick")), \
             patch.object(watch.Watcher, "discover", side_effect=AssertionError("must not fetch")), \
             patch.object(watch.Watcher, "prepare", side_effect=AssertionError("must not prepare")), \
             patch.object(watch, "sdk_argv", side_effect=AssertionError("must not use SDK")), \
             patch.object(watch, "resolve_uv", return_value="OFFLINE-UV"), \
             patch.object(watch.os, "umask"), redirect_stdout(output):
            status = watch.main(["compact", "--config", str(path), "--seconds", "30"])
            self.assertEqual(scheduler.return_value.mock_calls, [])
        self.assertEqual(status, 0)
        report = json.loads(output.getvalue())
        self.assertFalse(report["apply"])
        self.assertEqual(report["duplicate_files"], 1)
        self.assertEqual(self.snapshot([a, b, *self.evidence()]), before)


if __name__ == "__main__":
    unittest.main()
