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


`publish_request.py` prepares the GitHub Git Data objects for a single new watcher request. It reads the exact full commit message and verifies the two discovery trailers. With `publish` it uploads the objects, checks the remote commit and request blob, and advances main without forcing a ref update. Run from a checkout containing the intended commit. A GitHub CLI login with access to the selected upstream is required for publication.

```sh
uv run --project .. --locked python publish_request.py plan .. HEAD --out /tmp/watcher-request-plan
uv run --project .. --locked python publish_request.py publish .. HEAD
```
