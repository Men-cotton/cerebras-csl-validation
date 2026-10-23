# Native16 transfers

This family contains the two-PE native16 transfer reproducer, diagnostic
variants, and capture checks. A passing control establishes only its selected
offset, length, and mode; it does not resolve the known length-8 boundary.
[`USAGE.md`](USAGE.md) preserves the original reproducer instructions.

Run host tests from this directory after the repository's uv setup:

```sh
uv run --project ../.. --locked --group dev python -B -m unittest discover -s tests -v
```

For appliance commands, select the site's SDK Python explicitly through uv.
Replace `/SDK/bin/python` with its actual absolute path:

```sh
uv run --no-project --no-sync --offline --python /SDK/bin/python -- /SDK/bin/python appliance.py --help
```

Keep `source/`, `diagnostics/`, `run.py`, and `diagnostic.py` beside the adapter;
the adapter records their source hashes. Automated requests use the approved
entrypoint in [`validation.json`](validation.json) and live under
[`requests/native16/`](../../requests/native16/). See the
[repository workflow](../../README.md) for submission and result collection.
