# Develop and request a validation

Start with the [layout contract](../architecture/layout.md), then read the owning
validation directory and its applicable guidance. The root README explains how
to enter the repository; the metadata and request files describe its growing
inventory. Do not add another manually maintained job list to the README.

## Discover current work

```bash
unset UV_PROJECT_ENVIRONMENT
uv run --locked -m csl_validation jobs list
uv run --locked -m csl_validation jobs list --query topk
uv run --locked -m csl_validation jobs list --json
uv run --locked -m csl_validation jobs show requests/FAMILY/REQUEST.json
uv run --locked -m csl_validation jobs check
```

Discovery reads local metadata and requests; it does not submit jobs. Adding a
request or validation must make it discoverable without editing a central
narrative table. Keep request descriptions in their own JSON and validation
interfaces in the owning family's `validation.json`.

## Add code under its owner

Reusable validation code, its tests, fixtures and interpretation belong under
`validations/<family>/`. Its `validation.json` identifies entrypoints and test
locations. Keep cross-family orchestration in the controller package, not copied
into each validation. A different shape, input or repetition of an existing
method normally needs another request, not another implementation directory.

Keep fixed input/oracle packages and their provenance together. Changes to a
package's internal layout or bytes need corresponding identity checks; moving
the enclosing family directory does not authorize rewriting historical hashes,
commands, source revisions or result claims.

Each automated step runs a single foreground action. Appliance steps must expose
the SDK's job IDs so the watcher can confirm remote completion before admitting
the next action. Host steps must not submit remote work. Do not detach, daemonize,
launch tmux or wrap a legacy multiple-job campaign inside a request step.

## Describe one requested execution

Place a JSON manifest under `requests/<family>/`. It contains a unique request
ID, a description, and ordered steps naming the owning entrypoint, arguments,
kind (`appliance` or `host`) and timeout. The watcher substitutes `{output}` with
an absolute request-specific directory. Use the previous compile step's exact
artifact record for a run step.

To submit that request, put these trailers in the commit message:

```text
ALCF-Request: halo-small-20261023-01
ALCF-Manifest: requests/primitives/halo-small-20261023-01.json
```

Push the commit to the configured source branch. The watcher uses its exact SHA;
ordinary commits without trailers do not execute anything. Use a new request ID
for each intended execution, including a retry after resolving an uncertain
attempt. Merely storing a reusable example manifest does not submit it.

The watched branch is permission to execute its declared validation scripts
under the operator account. Review its code and restrict write access to the
intended authors. The result branch, when enabled, is separate from this source
branch.

## Verify a change

Use the committed uv lockfile for host development:

```bash
uv run --locked -m csl_validation jobs check
uv run --locked --group dev -m csl_validation check
```

The host suite covers controller behavior, collection, metadata and the relocated
validation tests. The development group supplies host dependencies such as
NumPy; it does not install the appliance SDK. Run focused tests when changing a
single owner, then the checks needed to establish compatibility across the changed
interfaces. Keep tests that exercise observable behavior, including malformed
state, interrupted execution and exact source/fixture identities.

Record the checks that actually ran and retain skips or unresolved failures.
Host tests, local simulator execution, actual appliance execution and numerical
correctness are distinct evidence. A file move or a new repository entry does
not promote an old result to a newer validation stage.

For controller changes, test request discovery, SHA pinning, interrupted
submission, terminal-state gating, ledger recovery and notification retry. For
collector changes, test authenticated-connection failure, identity collisions,
partial downloads and checksum verification. Real ALCF authentication and transfer
behavior still require deployment evidence.

## Layout and uv migration verification

On 2026-10-23, `uv run --locked --offline --group dev -m csl_validation check`
ran 344 host tests: 343 passed and one inherited SDK-source comparison skipped
because the standalone repository has no original SDK tree. This includes six
uv helper/runtime tests, with actual uv subprocesses checking fake SDK interpreter
selection, inherited lock descriptors and timeout process-group cleanup. No ALCF
connection, appliance submission or numerical hardware result was produced.

The relocation check compares 262 inherited files with their recorded pre-move
revision. After the move is committed, it verifies the initial addition revision
at each destination so later implementation changes remain possible.
