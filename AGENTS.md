# Working in this repository

Read README.md, then the relevant area's README and any local guidance. Root
sessions must explicitly read local guidance; do not load every family's sources
for an unrelated tooling change.

- Use uv: `uv run --locked -m csl_validation ...`. Run offline host checks with
  `uv run --locked --group dev -m csl_validation check`; `--family NAME` narrows
  them. These checks must not submit appliance jobs or start remote sessions.
- `csl_validation/` owns orchestration. `validations/<family>/` owns its numerical
  contract, entrypoints, fixtures and tests. Register entrypoints and test roots
  in that family's `validation.json`. Do not add a handwritten job list to README.
- `requests/<family>/` owns small invocation manifests. Metadata is read at the
  triggering commit SHA. Maintenance commits must not carry execution trailers.
- Keep the controller uv project separate from the existing site SDK environment.
  Clear `UV_PROJECT_ENVIRONMENT` before project commands; use this project's `.venv`.
  Do not sync, upgrade or copy SDK environments from individual request clones.
- Preserve frozen package bytes, source hashes and historical evidence unless
  the task explicitly changes their contract. A move is not new hardware evidence.
- Keep source, controller state and downloaded results separate. Never infer
  remote completion from a client timeout or retry an ambiguous submission.
- Complete relevant host tests and review changed behavior before publishing.
  Report mocked checks, local simulation and hardware measurements separately.
- Write durable policy here, operating procedures in docs/workflows/, and design
  rationale in docs/architecture/. Add task-specific detail only where it belongs.
