# Operate the validation watcher

The watcher reads a trusted Git branch, freezes each requested revision in a
separate clone, and runs individual appliance actions serially. Its configuration,
ledger, captures and result archives live outside the controller checkout.
Read [result collection](collection.md) for the laptop side and
[development](development.md) for adding validation code and requests.

## Prepare the user node

On ALCF, enter the Cerebras user node through `cerebras.alcf.anl.gov`. Reconnect to
the same user node for later operator sessions. Keep the existing, site-compatible
SDK virtual environment: this repository does not choose or upgrade its SDK.

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

Replace the SDK environment path with an existing environment that already runs
the site's appliance jobs. Do not point it at the new controller project's
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

Request output is capped at 2 GiB, with separately bounded logs and a 4 GiB
archive ceiling. Before cloning, the watcher checks a 32 GiB total state budget,
100,000 entries and at least 1 GiB free space. It does not automatically delete
completed clones or results. Keep the ledger and receipts when cleaning
artifacts after verified collection and analysis.

New work stops at 16 MiB of ledger metadata or 1,024 request records. The 32 MiB
ledger read/write bound reserves room for active-request recovery. If admission
stops at that bound, preserve the state and reconcile pending requests before
selecting a baseline for a new state directory.
