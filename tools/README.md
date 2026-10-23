# Operator utilities

`collect_jobs.py` extracts appliance job IDs from existing client logs. Its
default mode only reads local files and emits JSON. With `--collect`, it queries
job metadata and exports logs through `csctl` into a new output directory.

From this directory, inspect usage and run its host tests through uv:

```sh
uv run --project .. --locked python collect_jobs.py --help
uv run --project .. --locked --group dev python -B -m unittest discover -s tests -v
```

These utilities are shared operator tools. Keep validation-specific sources,
checks, and fixtures in the corresponding `validations/<family>/` directory.
