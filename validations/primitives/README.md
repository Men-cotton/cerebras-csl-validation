# Primitive validations

This family contains the Top-K and halo validation adapters, frozen cases,
host oracles, and capture audits. Existing `tank_*` filenames retain the
original experiment's identity and source hashes. A small-case pass establishes
that case's behavior; scaling requires separate measurements.
[`TANK_CAMPAIGN.md`](TANK_CAMPAIGN.md) preserves the original campaign record.

Run host tests from this directory after the repository's uv setup:

```sh
uv run --project ../.. --locked --group dev python -B -m unittest discover -s . -v
```

Inspect the approved appliance entrypoint with the site's SDK Python:

```sh
uv run --no-project --no-sync --offline --python /SDK/bin/python -- /SDK/bin/python tank_appliance.py --help
```

Replace `/SDK/bin/python` with the actual absolute path. Keep `tank_cases/`
intact beside the adapter. Its export and provenance checks depend on exact
source bytes, including `export_tank_cases.py`; changes require a deliberate
package regeneration. Automated requests live under
[`requests/primitives/`](../../requests/primitives/) and select the entrypoint
in [`validation.json`](validation.json). See the
[repository workflow](../../README.md) for submission and collection.
