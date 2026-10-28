#!/usr/bin/env python3
"""Freeze the E210 input-free pilot without materializing an expected array."""
import argparse
from pathlib import Path
import re
import subprocess

if __package__:
    from . import chunks, halo, stream_contract as contract
else:
    import chunks
    import halo
    import stream_contract as contract


def export(source_root, destination, source_revision):
    contract.require(isinstance(source_revision, str) and re.fullmatch(r"[0-9a-f]{40}", source_revision),
                     "pass the full committed source revision")
    root = Path(source_root).resolve()
    base = root/"kernels/stencil/independent-halo"
    vendor = root/"sdk-2.10.2/csl-extras-202604101435-6-d2f7d96e/examples/benchmarks/benchmark-libs/stencil_3d_7pts"
    sources = {"layout.csl": base/"runtime_coords/layout.csl", "pe.csl": base/"runtime_coords/pe.csl",
               "vendor_layout.csl": vendor/"layout.csl", "vendor_pe.csl": vendor/"wse3/pe.csl"}
    # The revision is a verified source pin, rather than a label supplied by
    # the caller. Vendor sources are immutable imports and retain byte hashes.
    tracked = [base/"runtime_coords/layout.csl", base/"runtime_coords/pe.csl", base/"oracle.py"]
    tracked += [root/"tools/csl_capture"/n for n in contract.APP_FILES]
    for path in tracked:
        relative = path.relative_to(root).as_posix()
        committed = subprocess.run(["git", "-C", str(root), "show", source_revision+":"+relative],
                                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10).stdout
        contract.require(len(committed) <= contract.JSON_LIMIT and committed == halo.bounded(path),
                         "source bytes differ from committed revision: "+relative)
    contract.require(all(halo.bounded(root/"tools/csl_capture"/n) == halo.bounded(contract.ROOT/n)
                         for n in contract.APP_FILES), "exporter installation differs from source revision")
    # Files are copied byte-for-byte, including immutable vendor code.
    payloads = {"source/"+name: halo.bounded(path) for name, path in sources.items()}
    payloads["oracle.py"] = halo.bounded(base/"oracle.py")
    operations = []
    for index, pair in enumerate(contract.PILOT_SEQUENCE):
        operations.append(dict(op="launch", symbol="apply_epochs", args=pair))
        for symbol in halo.SYMBOLS:
            operation = dict(op="d2h", symbol=symbol, array=f"s{index}_{symbol}", rect=[0, 0, 3, 3])
            if symbol == "clocks":
                operation["clock"] = True
            operations.append(operation)
    case = dict(schema=2, kind="halo", grid=[3, 3], params=dict(width=3, height=3),
                symbols=halo.SYMBOLS, functions=halo.FUNCTIONS, operations=operations, validations=dict(restore=[]),
                census=dict(epochs_per_launch=64, max_n=32, block_size=16, sequence=contract.PILOT_SEQUENCE),
                capture=dict(chunk_bytes=128, total_bytes=7668, max_chunks=66),
                oracle=dict(name=contract.ORACLE, source_sha256=halo.file_hash(contract.ROOT/"halo.py"),
                            reference_sha256=chunks.digest(payloads["oracle.py"])),
                source_files={name: chunks.digest(payloads["source/"+name]) for name in sources})
    contract.pilot(case)
    payloads["case.json"] = chunks.encoded(case)
    destination = contract.new_directory(destination)
    folder = destination/"cases"/contract.CASE_NAME
    (folder/"source").mkdir(parents=True)
    for name, value in payloads.items():
        (folder/name).write_bytes(value)
    manifest = dict(schema=2, source_revision=source_revision,
                    cases={contract.CASE_NAME: dict(case_sha256=chunks.digest(payloads["case.json"]), params=case["params"])},
                    files={f"cases/{contract.CASE_NAME}/{name}": chunks.digest(value) for name, value in payloads.items()})
    contract.write_json(destination/"manifest.json", manifest)
    contract.package_identity(destination, contract.CASE_NAME)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.source_root, args.out, args.source_revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
