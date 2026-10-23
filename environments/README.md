# SDK environment profiles

Profiles here describe external user-node environments. They are separate from
the controller's root `pyproject.toml` and `uv.lock`. The watcher selects an
already prepared environment through `--sdk-env`; request clones never sync it.

The [ALCF CSL profile](alcf-csl/README.md) contains the installation inputs and
hashed transitive dependency lock. Start with the
[SDK setup workflow](../docs/workflows/sdk-environment.md) to create a new environment.
Keep profile changes reviewable and create a new environment when changing a
profile used by active jobs.
