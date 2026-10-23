# ALCF CSL appliance client profile

Target: CPython 3.11, Linux x86_64. Direct dependencies are in `requirements.in`;
`requirements.txt` fixes their transitive versions and distribution hashes.
NumPy 1.25.0 matches the appliance client's dependency and supports this repo's
local inspectors. It is independent of the controller development group's NumPy.

ALCF documents `cerebras_appliance==2.10.0` and `cerebras_sdk==2.10.0` on Python
3.11. Both published wheels use `cp311-none-manylinux2014_x86_64`. These are
client package versions; a `csctl` server version such as 3.2.1 does not select
the Python package version.
[ALCF setup](https://docs.alcf.anl.gov/ai-testbed/cerebras/csl/#setup),
[SDK wheel](https://pypi.org/project/cerebras-sdk/2.10.0/#files),
[appliance wheel](https://pypi.org/project/cerebras-appliance/2.10.0/#files).

## Maintain the lock

From the repository root, after reviewing changes to `requirements.in`:

```bash
uv pip compile --no-config --no-python-downloads \
  --python-version 3.11 --python-platform x86_64-manylinux2014 \
  --only-binary :all: --generate-hashes --no-header \
  --output-file environments/alcf-csl/requirements.txt \
  environments/alcf-csl/requirements.in
```

Resolve on the intended platform, review the changed versions/hashes, then test
creation and imports in a new environment. The setup command validates this
profile's direct pins; change that contract and its tests deliberately when
updating the SDK release. Do not update an environment serving active jobs.
The root host suite mocks setup operations; package downloads and actual SDK
imports are separate validation steps.

This uses uv's documented [requirements locking and synchronization](https://docs.astral.sh/uv/pip/compile/).
Requirements locks are supported by uv; a nested project is unnecessary for an
environment that the watcher deliberately uses without project discovery.
