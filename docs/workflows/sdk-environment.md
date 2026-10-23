# Create an independent CSL SDK environment

Create a fresh environment for this repository. It does not reuse or copy the
old `cerebras-native16-repro` environment. The controller keeps its own `.venv`,
and the SDK environment lives outside the checkout so source updates cannot
replace it.

## Source contract and uv translation

Checked on 2026-10-23: [ALCF's CSL appliance setup](https://docs.alcf.anl.gov/ai-testbed/cerebras/csl/#setup)
uses Python 3.11 and installs `cerebras_appliance==2.10.0` plus
`cerebras_sdk==2.10.0`. The vendor's
[appliance guide](https://sdk.cerebras.ai/appliance-mode) agrees on package pins
but still shows Python 3.8 in its environment example. Use the site-specific
3.11 instruction and the published cp311 wheel tags for this profile.

| Python/pip operation in the installation procedure | uv operation used here |
| --- | --- |
| `python3.11 -m venv ENV` | `uv venv --python 3.11 ENV` |
| Activate the environment | Select `ENV/bin/python` explicitly in each command |
| Upgrade pip and install the SDK packages | `uv pip sync --python ENV/bin/python --require-hashes LOCK` |
| Check installed dependencies | `uv pip check --python ENV/bin/python` |
| Run an import check with that Python | `uv run --no-project --offline --python ENV/bin/python -- ENV/bin/python ...` |

uv provides the installer; bootstrapping pip is not a prerequisite. The vendor
dependency graph itself includes pip, so it is present in the locked SDK
environment. The [requirements profile](../../environments/alcf-csl/README.md)
pins transitive dependencies before installation. The controller's NumPy 2
development constraint is not applied to this SDK's NumPy 1.25.0 environment.

This follows uv's current official documentation for
[explicit environment selection](https://docs.astral.sh/uv/pip/environments/),
[locking and syncing](https://docs.astral.sh/uv/pip/compile/), and
[dependency inspection](https://docs.astral.sh/uv/pip/inspection/).
The wrapper rejects any existing destination, including a dangling symlink:
[`uv venv` can replace an existing environment](https://docs.astral.sh/uv/reference/cli/#uv-venv).

## Create and verify on the ALCF user node

From the updated repository root:

```bash
unset UV_PROJECT_ENVIRONMENT
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export https_proxy="$HTTPS_PROXY"
CSL_SDK_ENV="$HOME/.local/share/csl-validation/sdk-2.10.0"

uv run --locked -m csl_validation sdk-env create \
  --path "$CSL_SDK_ENV" --python 3.11

uv run --locked -m csl_validation sdk-env check --path "$CSL_SDK_ENV"
```

The command selects an installed Python 3.11 and refuses automatic interpreter
downloads. ALCF documents `/usr/bin/python3.11`; pass that absolute path with
`--python /usr/bin/python3.11` if several installations exist. If no 3.11 is
available, explicitly install it with `uv python install 3.11`, then repeat
creation at a new unused path if the first attempt already created a directory.
See [uv Python installation](https://docs.astral.sh/uv/guides/install-python/).

`create` reserves a new directory, creates the venv, installs the hashed lock,
checks dependency consistency, then imports NumPy, `cerebras.appliance.logger`,
`SdkCompiler` and `SdkLauncher`. It records all installed versions, interpreter,
uv version and lock SHA-256 in `ENV/csl-sdk-setup.json`. Only successful completion
writes `status: ready`. A failed environment remains for diagnosis and is never
silently reused or deleted. Use a new path for another attempt.

No SDK client is constructed and no job is submitted. Import checks establish
installation and the API names used by our adapters; they do not establish
cluster authentication or client/server compatibility. `csctl` remains the site's
existing command. Remote compiler/runtime software and worker-side Python are
not installed by this command. `SdkLauncher` runs the staged worker on the
appliance; the local import check therefore does not require worker-only
`sdkruntimepybind`.
[Cerebras appliance API](https://sdk.cerebras.ai/api-docs/appliance-api)

## Connect the watcher

For a new watcher, pass the new environment directly:

```bash
uv run --locked -m csl_validation watch init \
  --config "$HOME/csl-operator.json" \
  --repo https://github.com/Men-cotton/cerebras-csl-validation.git \
  --branch main --state-dir "$HOME/csl-validation-state" \
  --sdk-env "$CSL_SDK_ENV" --baseline "$(git rev-parse HEAD)"
```

Initialization submits nothing. If a watcher is already initialized, stop it at
an idle boundary and update only its config's `python` field to the new absolute
`ENV/bin/python` path; retain its ledger, baseline and state. Do not reinitialize
to switch environments. Continue with the [operator workflow](operator.md).

Subsequent SDK actions use uv with project discovery, synchronization and
downloads disabled. The environment is created once and kept unchanged while
jobs are active. For another SDK release, prepare and verify another environment.

## Verification recorded for this profile

On 2026-10-23, the commands were checked against the current official uv pages
linked above and executed with uv 0.7.13. A new Linux x86_64 environment with
CPython 3.11.13 installed all 19 locked packages, passed `uv pip check`, and
imported the four APIs listed above. The independent `sdk-env check` command
also passed. No existing SDK environment was used or modified.

The 12 setup regressions passed in the controller environment. The full host
suite then ran under the newly installed SDK interpreter and NumPy 1.25.0:
356 tests, 355 passed, one inherited comparison skipped because the original SDK
source trees are absent. This covers host inspector compatibility as well as
installation. It does not establish ALCF connectivity or device execution.
