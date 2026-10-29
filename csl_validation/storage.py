"""Completed-request storage maintenance; never discovers or executes work."""
import hashlib
from collections import Counter
import os
from pathlib import Path
import sys
import stat
import tempfile
import time

from .watch import NAME, SHA, command, contained, files_under, git, require

FINISHED = {"passed", "failed", "interrupted"}


def completed(entry):
    return entry["status"] in FINISHED and entry.get("packaged") is True


def remaining(deadline):
    if deadline is None:
        return 120
    check_time(deadline)
    return min(120, deadline - time.monotonic())


def source_git(args, cwd=None, deadline=None):
    return git(args, cwd, seconds=remaining(deadline))


def pin_source(cache, entry, apply=True, deadline=None):
    require(NAME.fullmatch(entry["id"]) and SHA.fullmatch(entry["sha"]), "invalid source identity")
    require(cache.is_dir() and cache.resolve() == cache, "invalid shared source store")
    ref = "refs/requests/" + entry["id"]
    prefix = ["--git-dir", str(cache)]
    source_git([*prefix, "cat-file", "-e", entry["sha"] + "^{commit}"], deadline=deadline)
    refs = source_git([*prefix, "for-each-ref", "--format=%(refname) %(objectname)", ref], deadline=deadline).splitlines()
    current = next((line.split()[1] for line in refs if line.split()[0] == ref), None)
    require(current in (None, entry["sha"]), "request source pin conflict")
    if apply:
        source_git([*prefix, "update-ref", ref, entry["sha"], current or "0" * 40], deadline=deadline)
    return ref


def source_tree(cache, src, sha, deadline=None):
    require(src.is_dir() and src.resolve() == src, "source path is not a regular directory")
    require(Path(source_git(["rev-parse", "--show-toplevel"], src, deadline).strip()) == src, "foreign source worktree")
    require(source_git(["rev-parse", "HEAD"], src, deadline).strip() == sha, "source revision changed")
    git_dir = Path(source_git(["rev-parse", "--absolute-git-dir"], src, deadline).strip())
    common = Path(source_git(["rev-parse", "--git-common-dir"], src, deadline).strip())
    common = (src / common).resolve()
    require(not (src / ".git").is_symlink(), "linked Git metadata")
    if (src / ".git").is_file():
        require(common == cache and git_dir.parent == cache / "worktrees", "foreign shared source store")
        return "worktree"
    require((src / ".git").is_dir() and not (src / ".git").is_symlink() and
            common == src / ".git" and git_dir == common, "foreign source clone")
    return "clone"


def clean_source(watcher, entry, apply=False, deadline=None):
    """Drop only reproducible, clean source; never delete generated files."""
    deadline = deadline or time.monotonic() + 120
    result = {"id": entry["id"], "status": "ineligible"}
    removing = False
    if not completed(entry):
        return result
    try:
        src = contained(watcher.root, "requests/" + entry["id"] + "/src")
        ref = pin_source(watcher.cache, entry, apply=False, deadline=deadline)
        if not src.exists():
            if apply:
                pin_source(watcher.cache, entry, deadline=deadline)
            return dict(result, status="absent", ref=ref)
        kind = source_tree(watcher.cache, src, entry["sha"], deadline)
        require(source_git(["status", "--porcelain", "--untracked-files=all", "--ignored"], src, deadline) == "",
                "source contains changed, untracked or ignored files")
        if kind == "clone":
            # Only the watcher's inherited remote ref is disposable. Preserve
            # locally created branches/tags/stashes even if their objects exist.
            refs = source_git(["for-each-ref", "--format=%(refname) %(objectname)"], src, deadline)
            for line in refs.splitlines():
                name, sha = line.split()
                require(name == "refs/remotes/origin/watched", "source has additional local refs")
                source_git(["--git-dir", str(watcher.cache), "merge-base", "--is-ancestor",
                            sha, "refs/heads/watched"], deadline=deadline)
        if apply:
            pin_source(watcher.cache, entry, deadline=deadline)
            result.update(ref=ref, path=str(src))
            removing = True
            if kind == "worktree":
                source_git(["--git-dir", str(watcher.cache), "worktree", "remove", str(src)], deadline=deadline)
            else:
                command([sys.executable, "-B", "-c", "import shutil,sys; shutil.rmtree(sys.argv[1])", str(src)],
                        seconds=remaining(deadline))
        return dict(result, status="removed" if apply else "removable", kind=kind, ref=ref)
    except (ValueError, OSError) as error:
        return dict(result, status="incomplete" if removing else "retained", reason=str(error)[:4096])


def stamp(path):
    info = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(info.st_mode), "output is no longer a regular file")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            info.st_mode, info.st_uid, info.st_gid)


def check_time(deadline):
    require(time.monotonic() < deadline, "compaction time budget exhausted; completed changes are retained")


def file_hash(path, deadline):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            check_time(deadline)
            value.update(block)
    return value.digest()


def equal_bytes(left, right, deadline):
    # A fresh comparison, independent of cached filecmp/stat results.
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            check_time(deadline)
            block = a.read(1 << 20)
            if block != b.read(1 << 20):
                return False
            if not block:
                return True


def replace_with_link(source, target, scratch, expected, expected_source, sha, deadline):
    """Keep the original path intact until an atomic replacement is ready."""
    fd, name = tempfile.mkstemp(prefix=".compact-", dir=scratch)
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.unlink()
        require(stamp(source) == expected_source, "source changed before linking")
        os.link(source, temporary, follow_symlinks=False)
        linked = stamp(temporary)
        # Linking changes ctime/nlink; verify the selected inode and its bytes again.
        require(linked[:4] == expected_source[:4] and linked[5:] == expected_source[5:],
                "source changed during linking")
        require(file_hash(temporary, deadline) == sha and stamp(temporary) == linked,
                "source changed after comparison")
        require(stamp(target) == expected, "output changed during compaction")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def compact(watcher, apply=False, request=None, seconds=300):
    """Share bytes in terminal outputs; preserve names, contents and ZIP exports.

    Completed output is immutable. To edit a retained capture, make an independent
    copy first. Runtime source trees are never linked to output files.
    """
    require(seconds > 0, "positive compaction time budget required")
    require(request is None or request in watcher.ledger["requests"], "unknown request")
    deadline = time.monotonic() + seconds
    entries = [watcher.ledger["requests"][ident] for ident in watcher.ledger["order"]
               if request is None or ident == request]
    report = dict(apply=apply, sources=[], files=0, duplicate_bytes=0,
                  duplicate_files=0, linked_files=0, released_allocated_bytes=0, external_links_skipped=0, errors=[])
    groups, inode_paths = {}, Counter()
    try:
        for entry in entries:
            check_time(deadline)
            result = clean_source(watcher, entry, apply, deadline)
            report["sources"].append(result)
            if result["status"] == "incomplete":
                report["errors"].append({"id": entry["id"], "error": result["reason"]})
            if not completed(entry):
                continue
            output = contained(watcher.root, "requests/" + entry["id"] + "/output")
            try:
                paths = files_under(output, None)
            except (ValueError, OSError) as error:
                report["errors"].append({"id": entry["id"], "error": str(error)[:4096]})
                continue
            for path in paths:
                check_time(deadline)
                before = stamp(path)
                report["files"] += 1
                if not before[2]:
                    continue
                # Same filesystem, size, mode and owner: preserve access semantics.
                key = (before[0], before[2], *before[5:])
                inode_paths[before[:2]] += 1
                groups.setdefault(key, []).append((path, before, output.parent, path.stat().st_nlink))
        # Hash only equal-size candidates; unique files need no payload read.
        for key, candidates in groups.items():
            if len(candidates) < 2:
                continue
            hashes = {}
            for path, before, scratch, original_links in candidates:
                check_time(deadline)
                if original_links != inode_paths[before[:2]]:
                    report["external_links_skipped"] += 1
                    continue
                current = stamp(path)
                # Earlier links in this group can change a shared inode's ctime.
                require(current[:4] == before[:4] and current[5:] == before[5:], "output changed during compaction")
                before = current
                sha = file_hash(path, deadline)
                require(stamp(path) == before, "output changed while hashing")
                canonical = hashes.get(sha)
                if canonical is None:
                    hashes[sha] = path
                    continue
                other = stamp(canonical)
                require((other[0], other[2], *other[5:]) == key, "canonical output metadata changed")
                if before[:2] == other[:2]:
                    continue
                require(equal_bytes(canonical, path, deadline), "equal hash but unequal contents")
                require(stamp(canonical) == other and stamp(path) == before, "output changed during comparison")
                report["duplicate_files"] += 1
                report["duplicate_bytes"] += before[2]
                if apply:
                    info = path.stat()
                    # A preexisting external hardlink can retain the old allocation.
                    released = info.st_blocks * 512 if info.st_nlink == 1 else 0
                    replace_with_link(canonical, path, scratch, before, other, sha, deadline)
                    report["linked_files"] += 1
                    report["released_allocated_bytes"] += released
    except (ValueError, OSError) as error:
        report["errors"].append({"error": str(error)[:4096]})
    return report
