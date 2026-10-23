# Cerebras CSL hardware validation

Submit reproducible CSL validation requests, preserve their source revision and
results, and collect verified bundles for local analysis. The operator checks new
Git requests while idle and runs appliance actions serially. Server-side Git push
is optional; the laptop can collect status and results over authenticated SSH.

Use uv and Python 3.11 or newer. The controller has no SDK dependency; an existing
site-compatible SDK environment is selected separately for hardware actions.

```bash
unset UV_PROJECT_ENVIRONMENT
uv run --locked -m csl_validation jobs list
uv run --locked -m csl_validation jobs check
```

Keep the controller in its own `.venv`; do not redirect `UV_PROJECT_ENVIRONMENT`
to the shared SDK environment.

| Task | Start here |
| --- | --- |
| Set up or recover the server operator | [Operator workflow](docs/workflows/operator.md) |
| Retrieve results from a laptop | [Collection workflow](docs/workflows/collection.md) |
| Create a request | [Request contract](requests/README.md) |
| Add or change a validation family | [Validation ownership](validations/README.md) |
| Run checks or develop the tooling | [Development workflow](docs/workflows/development.md) |
| Understand directory boundaries | [Layout and design sources](docs/architecture/layout.md) |
| Interpret inherited sources and evidence | [Provenance](PROVENANCE.md) |

Requests are discovered from `requests/**/*.json`; the README is not a second job
inventory. Use `jobs list --query TEXT` to search and `jobs show PATH` to inspect a
request. Merely adding a manifest does not execute it: the request commit must
carry the documented `ALCF-Request` and `ALCF-Manifest` trailers.

Read [AGENTS.md](AGENTS.md) before changing code. Runtime state and result bundles
belong outside this checkout. Existing frozen request clones remain tied to their
original revisions when this repository evolves.
