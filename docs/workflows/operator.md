# Operate the validation watcher

The watcher reads a trusted Git branch, freezes each requested revision in a
separate detached worktree, and runs individual appliance actions serially. Its configuration,
ledger, captures and result archives live outside the controller checkout.
Read [result collection](collection.md) for the laptop side and
[development](development.md) for adding validation code and requests.

## Prepare the user node

On ALCF, enter the Cerebras user node through `cerebras.alcf.anl.gov`. Reconnect to
the same user node for later operator sessions. Keep the existing, site-compatible
SDK virtual environment, or create an independent one with the
[SDK environment workflow](sdk-environment.md). The watcher never upgrades it.

```bash
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export https_proxy="$HTTPS_PROXY"
git clone https://github.com/Men-cotton/cerebras-csl-validation.git
cd cerebras-csl-validation
unset UV_PROJECT_ENVIRONMENT
uv --version
uv run --locked -m csl_validation --help
timeout 20s csctl --version
timeout 20s csctl get jobs --my-jobs -o json
```

The controller's uv project contains no Cerebras SDK dependency. `uv run --locked`
uses its committed project metadata and lockfile; the host test group is separate.
The SDK actions use the existing environment selected below, through uv with
project discovery, synchronization and online resolution disabled. A new
request clone does not create a new SDK virtual environment.

Clear `UV_PROJECT_ENVIRONMENT` before the outer controller command so uv cannot
sync this project into a shared SDK environment. The tmux launch below also
clears any override inherited from an existing tmux server. The project uses its
own `.venv`; `--sdk-env` selects the independent SDK environment.

Existing campaigns must finish before the watcher starts new appliance work.
The admission check covers this user's jobs in the configured namespace,
`job-operator` by default; it does not discover other namespaces. Queued and
compiling jobs count as busy. A failed query, malformed response, unknown phase
or missing individual job never establishes that remote work has ended.

## Initialize the operator

Replace the SDK environment path with the separately prepared environment from
the [SDK setup workflow](sdk-environment.md), or another site-compatible SDK
environment. Do not point it at the new controller project's
`.venv`. Keep the configuration and state directory outside both that project
and the request clones.

```bash
uv run --locked -m csl_validation watch init \
  --config "$HOME/csl-operator.json" \
  --repo https://github.com/Men-cotton/cerebras-csl-validation.git \
  --branch main \
  --state-dir "$HOME/csl-validation-state" \
  --sdk-env /ABSOLUTE/PATH/TO/EXISTING/SDK/ENV \
  --baseline "$(git rev-parse HEAD)"
```

Initialization does not submit a job. The explicit baseline excludes all older
commits; later request commits become eligible. Use `--uv` if the child actions
must use a particular uv executable. The older `--python` option remains a
compatibility route for an existing SDK interpreter, but uv still launches the
action. Existing configuration and ledger files remain usable; do not reinitialize
them just to update the controller.

The default requires only Git read access on the user node. It writes status and
results to shared home for the laptop collector. Server-side Git push and mail
are not required. Optional `--publish-status` publishes small result records on
the separate `alcf-status` branch and requires a working Git write credential.
Credentials do not belong in a URL, commit, or operator configuration.

## Start and inspect the watcher

```bash
uv run --locked -m csl_validation watch status --config "$HOME/csl-operator.json"
uv run --locked -m csl_validation watch once --config "$HOME/csl-operator.json"
```

`status` only reads the saved state. `once` can execute an eligible action; it is
not a dry run. To run continuously in tmux, launch from the repository root:

```bash
printf -v CSL_WATCH_COMMAND '%q ' \
  env -u UV_PROJECT_ENVIRONMENT "$(command -v uv)" --directory "$PWD" run --locked \
  -m csl_validation watch run --config "$HOME/csl-operator.json"
tmux new-session -d -s csl-validation "$CSL_WATCH_COMMAND"
tmux attach-session -t csl-validation
```

Detach with `Ctrl-b`, then `d`. This uses the tmux 2.7 interface reported on the
operator node. The default intervals are 3,600 seconds between idle Git checks
and 60 seconds between remote-state polls. The watcher checks for updates again
after a request finishes. tmux survives an SSH disconnection, not a node reboot.
Restart on the same user node with the existing configuration and ledger.

The request's source SHA, selected interpreter and installed NumPy/SDK versions
are retained in its result bundle. This records the environment but does not
freeze a mutable virtual environment. Keep the SDK environment unchanged while
requests run. Stop the watcher at an idle boundary before updating its own
checkout; then restart it with the same state.

## Recover an ambiguous attempt

If a client times out or the watcher is interrupted around submission, inspect
its saved job IDs, local processes and remote state. The watcher will not replay
an uncertain submission, even when a later active-job list is empty.

```bash
uv run --locked -m csl_validation watch status --config "$HOME/csl-operator.json"
timeout 20s csctl get job JOB_ID -o json
```

After establishing that the local processes and remote jobs have ended, close
the attempt with an explicit explanation:

```bash
uv run --locked -m csl_validation watch resolve \
  --config "$HOME/csl-operator.json" \
  --request REQUEST_ID \
  --reason 'Checked local and remote termination after client interruption'
```

Resolution checks the recorded processes/jobs, marks the request `interrupted`,
and packages its evidence. It neither cancels a remote job nor retries the
calculation. A retry is a new request with a new ID.

## Retain and clean results

Results are stored below `STATE_DIR/requests/REQUEST_ID/output`; finalized
bundles are `STATE_DIR/exports/REQUEST_ID.zip` and its `.sha256` file. The atomic
`STATE_DIR/status.json` identifies source revisions and finalized bundle
checksums/sizes. Failed notification delivery does not resubmit a calculation.

`output_bytes` (default 2 GiB) is an advisory threshold for accumulated request
output, not an execution limit. Job-ID discovery bounds directory traversal and
log bytes independently of numerical payload size. The watcher checks at least
1 GiB free before each new step. Logs retain their separate bounds and completed
ZIP bundles retain the collector's 4 GiB ceiling. Accumulated source and result
size does not gate new requests: preparation no longer scans the entire state
directory. Older configurations may retain `state_bytes`; it is ignored.
The free-space check still applies before preparation and each step. Keep the
ledger and receipts when cleaning artifacts after verified collection and analysis.

New work stops at 16 MiB of ledger metadata or 1,024 request records. The 32 MiB
ledger read/write bound reserves room for active-request recovery. If admission
stops at that bound, preserve the state and reconcile pending requests before
selecting a baseline for a new state directory.

## Compact completed storage

New requests share the Git objects in `STATE_DIR/source.git`. Each request has
an independent detached working tree and a permanent `refs/requests/ID` source
pin. Once execution and packaging finish, the watcher removes a clean `src/`.
Before removing a terminal source, it preserves the SDK's root-level
`run_meta.json` and `wsjob-…-cluster-details.json` files under the request's
`sdk-metadata/` directory. Job-detail names must match this request's recorded
job IDs. Original filenames and bytes are retained; the cleanup report includes
sizes and SHA-256 values. This operational metadata remains local and is not
added retroactively to a finalized ZIP. It is separate from numerical captures.

Any other changed, untracked or ignored file, additional local ref, nested
sidecar, symlink or conflicting destination keeps the source in place. All
candidates are checked before the first relocation. An existing destination with
identical bytes is reused when resuming interrupted preservation; another value
is never overwritten. These rules apply to legacy independent clones too.
The pinned source can be restored from `source.git` at its recorded SHA.
This does not change frozen validation code or the source revision of a request.

For existing results, stop the watcher at an idle boundary, update the controller
with `git pull --ff-only`, and use the same configuration and ledger. Do not run
`watch init` again. Compaction holds the operator lock and never queries the
scheduler, discovers requests or starts a validation:

```bash
uv --directory "$HOME/cerebras-csl-validation" run --locked -m csl_validation \
  watch compact --config "$HOME/csl-operator.json"
uv --directory "$HOME/cerebras-csl-validation" run --locked -m csl_validation \
  watch compact --config "$HOME/csl-operator.json" --apply
```

The first command reports candidates. `--apply` removes only clean, packaged,
terminal sources and hardlinks byte-identical files in completed `output/`
trees. It keeps every output pathname, ZIP export, checksum, receipt and log;
active, pending, blocked and unpackaged requests are excluded. File size,
permissions, owner, filesystem, SHA-256 and a fresh byte comparison must match.
An output tree containing a symlink or an incomplete inventory is skipped.
Inodes with existing hardlinks outside the selected completed outputs are also
skipped; they could still have a writer outside this maintenance operation.
Completed outputs are immutable after sharing: make a separate copy before
editing them. Runtime source files are never linked to output files.

The report distinguishes logical duplicate bytes from allocated blocks released
by replacement. Already shared inodes are skipped. `du` measures the remaining
allocated storage; JSON `output_bytes` and ZIP members retain their original
logical sizes. ZIP files are neither rewritten nor deduplicated with raw trees.
A compaction timeout leaves completed output replacements valid and can be retried.
Interrupted metadata preservation or source deletion is reported as `incomplete`
and returns a nonzero compact exit status. Already preserved metadata stays in
`sdk-metadata/`; retrying compact finishes remaining relocations. The retained
source pin can restore a partially removed tree. No execution record is replayed.
Use `--request ID` to work on one request or `--seconds N` to adjust the default
300-second maintenance budget. These are bounds for this maintenance command,
not admission limits on accumulated results. Cleanup errors do not block the
next request. Dirty sources are retained for inspection rather than force-cleaned.
Previously recorded cleanup failures are not automatically retried every tick;
use `compact --apply` after updating the controller at an idle boundary.

## Packaging failures

Execution `status` remains `passed`, `failed`, or `interrupted` when export
packaging fails. Public status records `packaged: false`, `packaging_status:
failed`, and `packaging_error`. Independent requests may proceed; the watcher
does not rerun completed steps or repeatedly rebuild a failed archive. After
correcting the packaging problem, retry packaging only:

```bash
uv run --locked -m csl_validation watch package \
  --config "$HOME/csl-operator.json" --request REQUEST_ID
```

The existing 4 GiB collector ceiling still applies. An oversized bundle needs
an independently designed sharded collection path; raising `output_bytes`
does not raise that transfer limit. Older `blocked` attempts remain explicit
recovery cases; updating the controller does not silently mark them successful
or modify their frozen request clones. `resolve` closes them as `interrupted`.
Native-GNN adapters newly frozen after the retention update remove only the
verified redundant capture transport archive; complete capture-data and failed
transfers remain available.

## Discover without execution and skip an unstarted request

Stop the continuous watcher before these commands; they acquire the same
operator lock. `discover` fetches the watched branch and records new pending
requests even when an older request is blocked. It never calls the scheduler,
prepares a source clone or executes a validation step. This differs from `once`,
which can execute work.

```bash
uv run --locked -m csl_validation watch discover --config "$HOME/csl-operator.json"
uv run --locked -m csl_validation watch status --config "$HOME/csl-operator.json"
uv run --locked -m csl_validation watch skip \
  --config "$HOME/csl-operator.json" --request REQUEST_ID \
  --reason 'Unstarted request superseded by corrected REQUEST_ID'
```

`skip` accepts only `pending` requests whose step list is exactly empty. It
preserves the original request ID, SHA and manifest, and records `interrupted`,
`execution_status: NOT_RUN`, the reason, and `packaging_status: not_applicable`.
The authoritative ledger is saved before auxiliary receipts, so interruption
while writing a receipt cannot make the skipped request runnable again. There
is no fabricated SDK environment, execution output, ZIP bundle or Git result
notification. Public status and the local receipt retain the skip evidence.
Even an explicit package retry does not package a never-started request.

Started or blocked attempts still require `resolve` and its process/job
termination checks. A skipped ID cannot be reused; submit a fresh request ID.
See [the October recovery sequence](recovery-20261027.md) for the LayerNorm
output-limit interruption and indexed-case registration replacement.
