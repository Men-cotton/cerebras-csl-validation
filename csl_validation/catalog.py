"""Discover requests from JSON rather than maintaining a second job inventory."""
import argparse
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
REQUEST_SCAN_ENTRIES = 16384
REQUEST_SCAN_SECONDS = 10


def require(test, message):
    if not test:
        raise ValueError(message)


def relative(value, suffix=None):
    require(isinstance(value, str) and value and not Path(value).is_absolute() and
            ".." not in Path(value).parts and not value.startswith("-"), "invalid relative path")
    require(str(Path(value)) == value, "noncanonical relative path")
    if suffix:
        require(value.endswith(suffix), "wrong path extension")
    return value


def definition(value, family):
    require(isinstance(value, dict) and value.get("schema") == 1 and value.get("id") == family,
            "validation metadata identity")
    require(IDENTIFIER.fullmatch(family), "validation family name")
    require(isinstance(value.get("description"), str) and 1 <= len(value["description"]) <= 2000,
            "validation description")
    entries, tests = value.get("entrypoints"), value.get("tests")
    require(isinstance(entries, list) and 1 <= len(entries) <= 64 and len(set(entries)) == len(entries), "validation entrypoints")
    require(isinstance(tests, list) and 1 <= len(tests) <= 8, "validation tests")
    for name in entries:
        relative(name, ".py")
        require(not name.endswith(("_campaign.py", "_followup.py")), "entrypoint must be one foreground action")
    for name in tests:
        relative(name)
    return value


def declared_scripts(request, read_definition):
    """Read only the metadata needed by this request, at the same Git revision."""
    require(isinstance(request, dict) and isinstance(request.get("steps"), list), "request steps")
    families, allowed = {}, set()
    for step in request["steps"]:
        require(isinstance(step, dict) and isinstance(step.get("script"), str), "request script")
        path = step["script"]
        if not path.startswith("validations/"):
            continue  # Compatibility allowlist handles requests from older commits.
        relative(path, ".py")
        parts = Path(path).parts
        require(len(parts) >= 3 and IDENTIFIER.fullmatch(parts[1]), "validation script location")
        family = parts[1]
        if family not in families:
            families[family] = definition(read_definition("validations/" + family + "/validation.json"), family)
        allowed.update("validations/" + family + "/" + name for name in families[family]["entrypoints"])
    return allowed


def definitions(root=ROOT):
    from .watch import contained, read_json
    found = []
    for path in sorted((root / "validations").glob("*/validation.json")):
        contained(root, str(path.relative_to(root)))
        value = definition(read_json(path), path.parent.name)
        for name in value["entrypoints"]:
            require(contained(path.parent, name).is_file(), "missing entrypoint: " + name)
        for name in value["tests"]:
            require(contained(path.parent, name).is_dir(), "missing test directory: " + name)
        found.append((path.parent, value))
    return found


def request_file(root, path):
    from .watch import contained, manifest, read_json
    path = contained(root, str(path))
    require(path.is_relative_to(root / "requests") and path.suffix == ".json", "request must be under requests/")
    value = read_json(path)
    allowed = declared_scripts(value, lambda p: read_json(contained(root, p)))
    manifest(value, allowed)
    for step in value["steps"]:
        require(contained(root, step["script"]).is_file(), "missing request script")
    return value


def request_paths(root):
    """Scan completely or fail; glob silently omits linked/unreadable directories."""
    from .watch import contained
    pending = [contained(root, "requests")]
    found, entries = [], 0
    deadline = time.monotonic() + REQUEST_SCAN_SECONDS
    while pending:
        require(time.monotonic() <= deadline, "request scan time limit")
        with os.scandir(pending.pop()) as directory:
            for entry in directory:
                entries += 1
                require(entries <= REQUEST_SCAN_ENTRIES, "request scan entry limit")
                require(time.monotonic() <= deadline, "request scan time limit")
                require(not entry.is_symlink(), "symlink in request inventory: " + entry.path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                else:
                    require(entry.is_file(follow_symlinks=False), "nonregular request inventory entry")
                    if entry.name.endswith(".json"):
                        require(len(found) < 4096, "request inventory limit")
                        found.append(Path(entry.path))
        require(time.monotonic() <= deadline, "request scan time limit")
    return sorted(found)


def requests(root=ROOT):
    root = Path(root).resolve()
    found, identifiers = [], set()
    for path in request_paths(root):
        value = request_file(root, path.relative_to(root))
        require(value["id"] not in identifiers, "duplicate request ID: " + value["id"])
        identifiers.add(value["id"])
        found.append((str(path.relative_to(root)), value))
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--query", default="")
    listing.add_argument("--json", action="store_true")
    show = sub.add_parser("show")
    show.add_argument("path")
    sub.add_parser("check")
    args = parser.parse_args(argv)
    try:
        if args.action == "show":
            print(json.dumps(request_file(ROOT, args.path), indent=2))
            return 0
        families = definitions()
        values = requests()
        if args.action == "check":
            print(f"PASS: {len(families)} validation families, {len(values)} request manifests; no jobs submitted")
        else:
            matches = [dict(path=path, id=value["id"], description=value.get("description", ""), steps=len(value["steps"]))
                       for path, value in values if args.query.lower() in (path + json.dumps(value)).lower()]
            if args.json:
                print(json.dumps(matches, indent=2))
            else:
                for value in matches:
                    print(value["id"] + "\t" + value["path"] + "\t" + value["description"])
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, str(error) + "\n")
