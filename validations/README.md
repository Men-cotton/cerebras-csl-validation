# Validation families

Each immediate subdirectory owns one validation family: its appliance adapter,
CSL sources or frozen packages, result checks, tests, and detailed documentation.
Its `validation.json` declares the family ID, approved appliance entrypoints,
and test directories. Keep family details here so the root README remains a
guide to the development and operator workflows.

To add a family, create a descriptive directory and a schema-1 descriptor with
`id`, `description`, `entrypoints`, and `tests`. Paths in that descriptor are
relative to its directory. Add a bounded compile/run adapter and host checks;
put execution requests in `requests/<family>/`. Registering a family or adding
a request file does not itself submit a hardware job.

Frozen packages retain their internal paths and bytes. Their source hashes and
provenance records describe those exact files. Update a package through its
export process when changing the experiment, and distinguish host checks,
local simulation, and hardware evidence in its documentation.

The initial relocation is recorded in
[`relocation.json`](../docs/architecture/relocation.json). Historical guides
inside each family retain their original commands; the family README and
[repository workflow](../README.md) describe the current uv entrypoints.
