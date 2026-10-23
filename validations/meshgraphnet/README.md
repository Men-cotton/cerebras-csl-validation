# MeshGraphNet validations

This family owns the MeshGraphNet appliance adapter, frozen cases, follow-up
packages, and capture audits. `mesh_local_evidence/` records local simulation;
it is not hardware evidence. [`MESHGRAPHNET.md`](MESHGRAPHNET.md) and
[`MESH_FOLLOWUP.md`](MESH_FOLLOWUP.md) preserve the original experiment details.

Run host tests from this directory after the repository's uv setup:

```sh
uv run --project ../.. --locked --group dev python -B -m unittest discover -s . -v
```

Inspect the approved appliance entrypoint with the site's SDK Python:

```sh
uv run --no-project --no-sync --offline --python /SDK/bin/python -- /SDK/bin/python mesh_appliance.py --help
```

Replace `/SDK/bin/python` with the actual absolute path. Keep package contents
and their relative paths intact so their hashes, host oracles, and captures
remain comparable. Run standalone audits from this directory or supply an
explicit package path; the audit's default `mesh_cases` path is relative to the
working directory. Individual execution requests belong in
`requests/meshgraphnet/`; see [`validation.json`](validation.json) and the
[repository workflow](../../README.md).
