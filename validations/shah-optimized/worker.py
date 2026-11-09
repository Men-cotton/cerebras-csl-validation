#!/usr/bin/env python3
"""Appliance-side frozen-host admission and failure capture."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback
import uuid

from compiled import tree_inventory


def digest(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 2 << 20:
        raise ValueError(f"invalid staged source {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--name", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cmaddr")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    status = dict(status="incomplete")
    try:
        if args.record.stat().st_size > 2 << 20:
            raise ValueError("oversize case record")
        request = json.loads(args.record.read_text())
        record = request["compile_record"]
        output_mode = request["output_mode"]
        run_id = request["run_id"]
        if str(uuid.UUID(run_id)) != run_id:
            raise ValueError("invalid run identity")
        status["run_id"] = run_id
        status["compile_record"] = record
        status["output_mode"] = output_mode
        for table in ("source_sha256", "host_sha256"):
            for name, pin in record[table].items():
                if Path(name).name != name or digest(name) != pin:
                    raise ValueError(f"staged file mismatch: {name}")
        inventory = tree_inventory(args.name)
        if inventory != record["compiled_inventory"]:
            raise ValueError("unpacked program differs from compiled archive")
        (args.out / "compiled_inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
        layout = inventory["io_layout"]
        (args.out / "io_layout.json").write_text(json.dumps(layout, indent=2, sort_keys=True) + "\n")
        import driver
        # run_case executes the exact numerical/streaming path also used locally.
        local_record = dict(record)
        local_record["source_sha256"] = {**record["source_sha256"],
            **{name: record["host_sha256"][name] for name in ("driver.py", "oracle.py")}}
        driver.run_case(local_record, args.name, args.out, cmaddr=args.cmaddr, output_mode=output_mode)
        status["status"] = "PASS"
        return 0
    except BaseException as error:
        status.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        (args.out / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
