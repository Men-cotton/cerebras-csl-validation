# Collect results on the laptop

The default path needs no Git write credential on Argonne. The watcher saves
an atomic status file and finalized ZIPs under shared home; the laptop reads
them through the public login host. ALCF documents home sharing across the AI
Testbed and supports SCP/rsync/Globus transfers.
[ALCF storage and transfer guide](https://docs.alcf.anl.gov/ai-testbed/data-management/data-management-overview/)

## Authenticate one SSH connection

Create a private directory for control sockets on the laptop:

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

Authenticate interactively and complete MFA:

```bash
ssh -MNf alcf-csl-results
ssh -O check alcf-csl-results
```

Connection sharing lets later SSH/SCP operations reuse the authenticated
transport. `ControlPersist 8h` is an idle timeout, not a guarantee that the
connection lasts eight hours. Laptop sleep, network loss and site session limits
can end it. [OpenSSH configuration](https://man.openbsd.org/ssh_config)

## Run the collector

From a local checkout of this repository:

```bash
unset UV_PROJECT_ENVIRONMENT
uv run --locked -m csl_validation fetch \
  --host alcf-csl-results \
  --state-dir /home/ALCF_USER/csl-validation-state \
  --out "$HOME/csl-results" --watch --interval 60
```

Omit `--watch` for one collection cycle. The collector first checks that an SSH
master exists, uses `BatchMode=yes`, and stops on connection or transfer error.
It does not loop through MFA attempts. Authenticate again and restart the
collector after the connection is lost.

Status snapshots are limited to 4 MiB and bundles to 4 GiB. Each download goes
to a private `.partial` path. Only a file with the declared size and SHA-256
becomes `REQUEST_ID.zip`. Existing verified ZIPs are skipped; conflicting files
and symlinks stop collection. A transfer has a 30-minute bound.

The collector publishes local `$HOME/csl-results/status.json` only after the referenced
bundles have been verified locally. A laptop agent can inspect that file and the
downloaded ZIPs to begin analysis. The collector never extracts an archive or
turns transport integrity into a claim of numerical correctness. Use the owning
validation's audit and interpretation guide.

## Manual transfer and optional Git notification

Manual download remains available:

```bash
scp ALCF_USER@cerebras.alcf.anl.gov:/home/ALCF_USER/csl-validation-state/exports/REQUEST_ID.zip .
scp ALCF_USER@cerebras.alcf.anl.gov:/home/ALCF_USER/csl-validation-state/exports/REQUEST_ID.zip.sha256 .
sha256sum -c REQUEST_ID.zip.sha256
unzip -t REQUEST_ID.zip
```

On macOS, use `shasum -a 256 -c REQUEST_ID.zip.sha256`. Retain the server copy
until local verification and analysis succeed. For larger or slower transfers,
use the site's appropriate transfer service; ALCF recommends Globus for large
data transfers. [SFTP/SCP guidance](https://docs.alcf.anl.gov/data-management/data-transfer/sftp-scp/)

If `--publish-status` was enabled during watcher initialization, the separate
`alcf-status` branch carries small completion records. This is an optional
notification transport. A repository-scoped HTTPS credential can use the
existing proxy route, but configuring one is unnecessary for collection above.
Keep credentials out of repository URLs and files.
[GitHub token guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
