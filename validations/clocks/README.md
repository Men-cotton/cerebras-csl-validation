# Clock measurements

This family owns the clock probes, source package, plans, result inspectors,
and reports. Interpret PE-local timing and host elapsed time separately.
[`CLOCKS.md`](CLOCKS.md) and [`CLOCK_CAMPAIGN.md`](CLOCK_CAMPAIGN.md) retain
the original measurement definitions and campaign commands.

Run host tests from this directory after the repository's uv setup:

```sh
uv run --project ../.. --locked --group dev python -B -m unittest discover -s tests -v
```

Inspect the approved appliance entrypoint with the site's SDK Python:

```sh
uv run --no-project --no-sync --offline --python /SDK/bin/python -- /SDK/bin/python clock_appliance.py --help
```

Replace `/SDK/bin/python` with the actual absolute path. Keep `clock_source/`
and the clock helpers beside the adapter to preserve its source records.
Create individual compile/run requests under `requests/clocks/` when needed;
campaign and retry helpers are not approved appliance entrypoints because
they can submit more than one job. See [`validation.json`](validation.json)
and the [repository workflow](../../README.md).
