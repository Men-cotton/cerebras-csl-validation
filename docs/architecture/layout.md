# Repository ownership and growth

The repository has two growth rates. Operator commands and recovery rules change
occasionally; validation families, cases and requested runs accumulate. The root
entry page therefore routes tasks to stable interfaces. It does not enumerate
every job. Current inventory is derived from the metadata and request files that
already own those facts.

## Ownership

```text
README.md                         purpose, uv entry commands, stable navigation
AGENTS.md                         durable working rules and verification contract
pyproject.toml / uv.lock           controller and host-development environment
csl_validation/                   shared watcher, collector, discovery and CLI
tests/                            shared-controller and discovery tests
validations/<family>/
  validation.json                 entrypoints, descriptions and test locations
  README.md / local guides        interface and result interpretation
  implementation and tests        family-owned validation behavior
  fixed cases and provenance      inputs/oracles bound to their original sources
requests/<family>/*.json           individual invocation descriptions
docs/workflows/                   operation, collection and development procedures
docs/architecture/               design decisions and compatibility boundaries
```

Operator configuration, ledgers, source clones and generated results live in an
external state directory. The SDK virtual environment is external too. Neither
belongs in the source inventory or inside every request clone.

A new invocation adds a request file under its owner. A new method adds a family
directory with the corresponding metadata, code, tests and local explanation.
The discovery command finds both from disk. A root README edit is needed only
when a stable interface or ownership boundary changes.

`validation.json` owns entrypoint metadata; the request owns the concrete args,
timeout and output plan. Local guides explain interpretation rather than copying
those fields into a second catalog. This keeps the executable inventory from
drifting away from prose lists.

## uv and the SDK boundary

The shared controller uses the committed uv project and lockfile. Host-development
dependencies belong to its development group. Site-specific appliance packages
remain in the operator-selected SDK environment. The watcher invokes that
environment through uv without discovering/syncing the request clone's project
or resolving packages online. This prevents each new source SHA from creating or
upgrading an SDK environment merely because a job was requested.

The outer controller command clears `UV_PROJECT_ENVIRONMENT` so its normal uv
sync cannot target a shared environment. uv documents that reusing an absolute
project environment path across projects can overwrite that environment.
[Project environment path](https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path)

Source pinning and environment recording remain separate. A clone at a fixed
SHA identifies the code and request, while the captured environment metadata
identifies the interpreter and installed packages used for that run. The operator
must keep a shared SDK environment unchanged while requests are active.

## Migration boundary

The former root-level family scripts, guides, tests and fixed packages move
together beneath their owners. Relative imports and package-internal paths must
be checked as part of the move. Preserve the bytes and source identities of
frozen case packages; adapt surrounding imports or entrypoint routing where
needed instead of changing evidence to fit the new directory name.

The root `csl_watch.py` and `csl_fetch.py` remain compatibility entrypoints. New
instructions use the uv module CLI. Existing operator configuration and ledgers
are retained, including the original SDK interpreter selection. Requests pinned
to older commits retain those commits' source paths. Stop the controller at an
idle boundary before updating its own checkout; running source clones and
external results remain separate from that update.

The migration does not add new hardware evidence. Compatibility requires the
old numerical/identity tests plus focused checks for metadata discovery,
relocated imports, request admission and uv command construction.

## Official workflow guidance applied here

OpenAI's best-practices guidance recommends a short, accurate `AGENTS.md` that
states layout, run/test commands, constraints and completion criteria, with
task-specific details linked separately. This repository applies that principle
through a small root contract and owner-local documentation.
[Best practices](https://learn.chatgpt.com/docs/learn/best-practices)

Codex discovers instruction files along the path from project root to the
launch working directory. It does not automatically load every nested owner's
guidance into a root-launched task. Root instructions therefore tell contributors
to read the target area's files explicitly.
[AGENTS.md discovery](https://learn.chatgpt.com/docs/agent-configuration/agents-md)

The development-workflow cookbook separates durable instructions, plans and
observed verification, and describes its additional harness files as optional
conventions. Here, code and metadata own behavior, workflow documents own
procedures, and retained receipts own observed execution. The directory names
above are this project's design choices; OpenAI does not prescribe them. We do
not add a parallel set of GOALS, PLANS or harness records merely to copy the
cookbook's example tree.
[Iterating Development Workflows with Codex](https://developers.openai.com/cookbook/examples/codex/iterating-development-workflows-with-codex)
