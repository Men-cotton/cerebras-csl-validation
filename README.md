# Cerebras CSL hardware validation

Run reproducible CSL checks on a Cerebras appliance, retain complete numerical
captures, and bring results back to a local analysis environment. This repository
includes Native16 transfer diagnostics, command-clock probes, MeshGraphNet
validation, Top-K capacity cases, and halo communication cases. It also provides
a Git-driven operator for submitting new validation requests serially.

The initial source and case packages come from `cerebras-native16-repro` at
`6467a527e07417fb5cd2c7f742eb92cbce266951`. Existing scripts and frozen case packages
retain their paths and contents; see [provenance](PROVENANCE.md). The old repository
and campaigns already running from it can remain in place.

## Choose a validation

| Question | Guide | Entry points |
| --- | --- | --- |
| Does Native16 transfer preserve adjacent data? | [Transfer diagnostics](USAGE.md) | `appliance.py`, `inspect_result.py` |
| What do host submissions and device command clocks measure? | [Clock probes](CLOCKS.md), [clock campaign](CLOCK_CAMPAIGN.md) | `clock_appliance.py`, `clock_report.py` |
| Does resident MeshGraphNet preserve full numerical results? | [MeshGraphNet](MESHGRAPHNET.md), [two-block follow-up](MESH_FOLLOWUP.md) | `mesh_appliance.py`, `mesh_audit.py` |
| How do Top-K capacity and halo work scale? | [Top-K and halo](TANK_CAMPAIGN.md) | `tank_appliance.py`, `tank_audit.py` |
| Which remote jobs belong to saved client logs? | [Job collection](USAGE.md#existing-job-information) | `collect_jobs.py` |

The inherited campaign commands start their own private tmux sessions. They remain
available for manual use. Automated requests use individual foreground steps so
the operator can check remote state between submissions. Do not place a nested
campaign controller in a watcher request.

## Prepare the user node

On ALCF, connect to `cerebras.alcf.anl.gov`, then to the same Cerebras user node for
each operator session. Use the existing Python environment containing NumPy and
the site-compatible `cerebras_appliance` and `cerebras_sdk` packages. The operator
does not install or upgrade the SDK. Keep the shared SDK environment outside the
per-request clones.

Use Python 3.11 or newer for the watcher. Its subprocess supervision uses process
groups without starting detached sessions. The supplied server reports Python 3.11.11.

```bash
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export https_proxy="$HTTPS_PROXY"
git clone https://github.com/Men-cotton/cerebras-csl-validation.git
cd cerebras-csl-validation
```

Confirm the selected environment and job visibility:

```bash
python3 --version
tmux -V
timeout 20s csctl --version
timeout 20s csctl get jobs --my-jobs -o json
```

Existing jobs from the old repository must finish before the watcher starts new
appliance work. The gate covers your jobs in the configured namespace (`job-operator`
by default); it does not discover other namespaces. A queued or compiling job also occupies this gate. A failed query
or an unrecognized state does not establish that the account is idle.

## Initialize and start the operator

Keep its configuration and persistent state outside this checkout. Select the
absolute path to the Python interpreter in the existing SDK environment. Replace
the example interpreter path below. The initial baseline deliberately skips all
existing commits; only later request commits are eligible.

```bash
python3 csl_watch.py init \
  --config "$HOME/csl-operator.json" \
  --repo https://github.com/Men-cotton/cerebras-csl-validation.git \
  --branch main \
  --state-dir "$HOME/csl-validation-state" \
  --python /ABSOLUTE/PATH/TO/SDK/ENV/bin/python \
  --baseline "$(git rev-parse HEAD)"
```

The default needs only Git read access on the user node. It writes status and
results to shared home, and the laptop collector reads them over an authenticated
SSH connection. Server-side Git push and mail are not required.

Optional: add `--publish-status` to `init` to publish completion records on the
separate `alcf-status` branch. This requires Git push credentials on the user node;
clone/read access alone does not prove write access. If this transport is needed,
an HTTPS fine-grained token limited to this repository with Contents read/write
matches the existing proxy route. Store it through an appropriate credential
helper, not in a URL, command argument, commit, or operator configuration.
[GitHub token documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)

For one foreground inspection/execution cycle:

```bash
python3 csl_watch.py once --config "$HOME/csl-operator.json"
python3 csl_watch.py status --config "$HOME/csl-operator.json"
```

`once` can execute an eligible request; it is not a dry-run command. To keep the
operator running, use absolute paths in a named tmux session:

```bash
tmux new-session -d -s csl-validation \
  '/ABSOLUTE/PATH/TO/SDK/ENV/bin/python /ABSOLUTE/PATH/TO/cerebras-csl-validation/csl_watch.py run --config /ABSOLUTE/PATH/TO/csl-operator.json'
tmux attach-session -t csl-validation
```

The command works with the tmux 2.7 interface reported on the operator node.
Detach with `Ctrl-b`, then `d`. The default intervals are 3,600 seconds for an idle
Git update check and 60 seconds for job polling; the default namespace is
`job-operator`. The operator checks for new requests again after completing a
request. tmux preserves the session across SSH disconnection, but does not restart
it after a node reboot.

The selected interpreter and installed NumPy/SDK package versions are recorded in
each result bundle. This records the environment; it does not freeze a mutable
virtual environment. Keep that environment unchanged while requests run. Do not
update the controller checkout while its process is running; stop it at an idle
boundary, update, then restart with the same configuration and ledger.

Output is bounded to 2 GiB per request, with separately bounded logs and a 4 GiB
archive ceiling shared with the collector. Before cloning, the watcher checks a
32 GiB total state budget, 100,000 entries and at least 1 GiB free space. It does
not delete completed source clones or results automatically. After collection and
analysis, retain the ledger and receipts when cleaning completed artifacts.
New work stops at 16 MiB of ledger metadata or 1,024 request records; the 32 MiB
read/write bound reserves space for the active request's recovery records. If that
admission limit is reached, preserve the old state and reconcile pending requests
before choosing an explicit baseline for a new controller state directory.

## Submit a new request

Add a request manifest under `requests/` and include these trailers in the commit
message that introduces it:

```text
ALCF-Request: halo-small-20261023-01
ALCF-Manifest: requests/halo-small-20261023-01.json
```

Use a new request ID for each intended execution. The manifest lists foreground
steps, with `kind` set to `appliance` or `host`, the script to run, and its arguments.
Inspect the [request format and examples](requests/README.md) before
adding a new case. Appliance steps must expose the job IDs produced by the SDK;
host steps perform local analysis and must not submit remote work. A request must
not detach, daemonize, launch tmux, or start a background campaign.

Push the request commit to the configured branch. The operator freezes a separate
clone at that commit SHA and records the request outside the clone. It does not
pull into a running request's source tree. Ordinary commits without request
trailers do not submit jobs. A status notification cannot trigger work because
`alcf-status` is separate from the watched source branch.

Treat the watched branch as permission to execute its request scripts under the
operator account. Review request code and keep access limited to intended authors.

## Recover and collect results

The persistent ledger distinguishes submitted work from completed results. If a
client times out or the watcher restarts around submission, inspect the saved job
IDs and remote state before resolving the request. Do not start a fresh copy as a
retry while remote work may still be active.

```bash
python3 csl_watch.py status --config "$HOME/csl-operator.json"
timeout 20s csctl get job JOB_ID -o json
```

After establishing the disposition of an ambiguous request, record the operator's
reason using `resolve`. It checks that recorded local processes and remote jobs
have ended, closes the request as `interrupted`, and packages its evidence; it
does not retry the calculation or cancel remote jobs.

```bash
python3 csl_watch.py resolve --config "$HOME/csl-operator.json" \
  --request REQUEST_ID --reason 'Checked job termination after client interruption'
```

Request output is under `STATE_DIR/requests/REQUEST_ID/output`. Final bundles are
`STATE_DIR/exports/REQUEST_ID.zip` and `REQUEST_ID.zip.sha256`. The always-written
`STATE_DIR/status.json` identifies requests, source SHAs, results, and finalized
bundle checksums/sizes. Failed notification delivery does not cause a calculation
to be resubmitted. ZIP creation and transfer do not by themselves establish
numerical correctness; use the relevant audit above.

## Collect from the laptop without server-side Git push

The AI Testbed shares home across login and user nodes, so the laptop can retrieve
the small status file through `cerebras.alcf.anl.gov`. It does not need to SSH into
the internal user node or run the SDK remotely for this operation.
[ALCF shared-home and transfer documentation](https://docs.alcf.anl.gov/ai-testbed/data-management/data-management-overview/)

On the laptop, create a private directory for SSH control sockets:

```bash
mkdir -p "$HOME/.ssh/csl-control"
chmod 700 "$HOME/.ssh/csl-control"
```

Add an alias to the laptop's `~/.ssh/config`, replacing the account name:

```sshconfig
Host alcf-csl-results
    HostName cerebras.alcf.anl.gov
    User ALCF_USER
    ControlMaster auto
    ControlPath ~/.ssh/csl-control/%C
    ControlPersist 8h
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Authenticate once interactively, completing MFA, and start the collector from a
local checkout of this repository:

```bash
ssh -MNf alcf-csl-results
ssh -O check alcf-csl-results
python3 csl_fetch.py --host alcf-csl-results \
  --state-dir /home/ALCF_USER/csl-validation-state \
  --out ./csl-results --watch --interval 60
```

Omit `--watch` for one collection cycle. The collector requires an existing SSH
master, sets `BatchMode=yes`, and stops on a connection/transfer error. It does not
retry MFA. After laptop sleep or connection loss, authenticate again and restart
the collector. `ControlPersist 8h` is an idle connection timeout; it does not
guarantee eight hours of connectivity or bypass site session limits.
[OpenSSH connection-sharing documentation](https://man.openbsd.org/ssh_config)

The collector accepts a status snapshot up to 4 MiB and bundles up to 4 GiB. Each
bundle downloads to a private `.partial` path, is checked against the declared size
and SHA-256, and only then becomes `REQUEST_ID.zip`. Existing verified bundles are
skipped; a conflicting file or symlink stops collection. It saves local status and
checksum files, prints request states, and never extracts ZIP contents. A transfer
has a 30-minute limit; use a separate transfer path for larger/slower results.

The laptop agent can watch local `csl-results/status.json` and analyze newly
verified ZIPs. With optional Git publication enabled, it can also fetch
`alcf-status`; this is an alternative notification transport.

Manual download remains available:

```bash
scp ALCF_USER@cerebras.alcf.anl.gov:/home/ALCF_USER/csl-validation-state/exports/REQUEST_ID.zip .
scp ALCF_USER@cerebras.alcf.anl.gov:/home/ALCF_USER/csl-validation-state/exports/REQUEST_ID.zip.sha256 .
sha256sum -c REQUEST_ID.zip.sha256
unzip -t REQUEST_ID.zip
```

On macOS, use `shasum -a 256 -c REQUEST_ID.zip.sha256`. A fresh laptop-to-ALCF
authentication requires the site's MFA flow; notification receipt does not
guarantee unattended SCP reconnection. Use Globus for transfers large enough to burden login nodes.
Retain the server copy until local verification and analysis have succeeded.

## Validation boundaries

Inherited guides describe their own prior host, local-SIM, and hardware evidence.
Moving files into this repository does not add hardware validation. The watcher
can be tested with a local Git remote and simulated csctl responses without
submitting appliance jobs. Real operator-node authentication, notifications and
new hardware results require deployment evidence.

Existing mocked host suites live both under `tests/` and in root `test_*.py` files;
run both when changing shared helpers. Preserve source/case hashes when reviewing
legacy results. The source paths and original revisions in frozen manifests are
provenance and should not be renamed to the new repository name.

Offline validation on 2026-10-23 passed 32 watcher and 10 collector tests,
including local Git request discovery, interruption recovery, terminal-state
gating and notification retry. The inherited suites passed 177 Native16/clock
tests and 89 root tests. One inherited comparison against original SDK trees was
skipped because those trees are absent from this standalone checkout. The 262
inherited files other than the extended `.gitignore` match the migration source
byte for byte. No ALCF connection or new hardware submission was made by these checks.

With NumPy available, run both test locations from the repository root:

```bash
python3 -B -m unittest discover -s tests -v
python3 -B -m unittest discover -s . -v
```
