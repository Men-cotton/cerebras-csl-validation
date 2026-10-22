# CylinderFlow selective halo on the appliance

This branch adds a resident MeshGraphNet experiment to the existing command-clock
repo. The target is the public CylinderFlow validation mesh: 1,896 vertices and
10,908 directed edges. It retains every edge, mesh feature, weight and evolving
velocity on device across all processor blocks and physical steps in a launch.
The bundled models have deterministic untrained weights, H4 and one or two
processor blocks; they do not reproduce the original H128/S15 physical model.

## Run the hardware campaign

Use the SDK Python environment already used for `clock_campaign.py` on the
operator node. From this repo on branch `tank28-selective-halo`, run:

```bash
uv run python mesh_campaign.py
```

The command starts a private tmux session and prints OUTPUT, ATTACH, LOG and
STATUS commands. It compiles five cases and executes eight serial runtime jobs:

| Case | Graph / PE grid | Blocks / steps | Transport |
| --- | --- | --- | --- |
| small-colored | adversarial N8/E9 / 2×2 | S2 / chunks2,2 | parallel selective, Q2 |
| crossing-colored | N64/E24 shared-link fixture / 4×2 | S2 / chunks2,2 | parallel selective, Q2 |
| full-broadcast | full CylinderFlow / 8×8 | S1 / two steps in one launch | broadcast |
| full-selective | same full graph and arithmetic | same | source-turn selective, 64 rounds |
| full-colored | same full graph and arithmetic | same | parallel selective, 9 rounds |

Each full case runs twice with the second transport order reversed. One compiled
artifact is reused for the two fresh runtime sessions. A failed job is recorded
and the campaign continues; a failed build skips only its dependent runtimes.
The final `mesh-campaign-*.zip` includes the frozen sources, package, artifacts,
requests, raw captures, logs and validation reports. `summary.json` includes
all passing full-run comparisons: six arrays must agree bit for bit at every
checkpoint, in addition to each run passing the independent numerical oracle.

Hardware defaults match the existing ALCF path: WSE3, fabric762×1172, offsets4,1,
channels1 and compiler parallelism1. Override the management namespace when your
site requires it:

```bash
uv run python mesh_campaign.py --namespace YOUR_NAMESPACE
```

`--host-timeout 300` bounds each worker and `--client-timeout 1200` bounds each
local client by default. A client timeout does not prove the remote job stopped.
The elapsed campaign time is measured, not predicted from simulated PE cycles.
Previous small command-clock jobs needed roughly50–60 seconds per compile and
70–190 seconds per runtime including collection; this GNN's hardware turnaround
has not yet been measured. Its full source-turn local SIM needed46 minutes.

## Inspect or resume

To prepare the exact plan without submitting jobs:

```bash
uv run python mesh_campaign.py --plan-only --out ./mesh-plan
```

To submit the pending entries of that plan, or resume another output directory:

```bash
uv run python mesh_campaign.py --resume ./mesh-plan
```

A completed entry is revalidated. An attempted entry without terminal evidence
is marked unknown and is not automatically submitted again. An active controller
holds an OS lock; do not delete its lock file to bypass it. After checking the
site scheduler for any uncertain remote job, retry selected cases in a **fresh**
output directory, for example:

```bash
uv run python mesh_campaign.py --cases full-colored --full-repeats 1
```

Use `--foreground` when running under your own job supervisor. `--backend
appliance-sim` explicitly requests the appliance simulator and is not hardware
validation. The campaign never connects to a server by SSH.

## Measurement and validation

- `state.json` records outer compile/runtime client wall intervals and job IDs.
- `compile-status.json` measures compiler-client work through artifact retrieval.
- `collection.json` measures launcher setup, staging, execution, download and
  validation together. It is not download time alone.
- `execution.json` records nested worker/API wall times and the exact resident
  upload/launch/readback sequence. It records the SDK identity available on that
  worker; the local SDK2.10.2 pin does not assert the appliance SDK version.
- `audit.json` checks velocity, node/edge latents, geometry cache, aggregate,
  sender cache, boundaries, padding, immutable inputs, lifecycle and communication
  counters. It reports each PE's local48-bit timestamp interval, not a globally
  synchronized makespan. First-chunk cycles include geometry initialization.

No host call occurs inside a processor block or between physical steps in the
full-mesh two-step request. Readbacks before and after the launch deliberately
validate residency and affect worker wall time. PE intervals exclude those
readbacks. Queue wait, setup and download are not separately instrumented.

`mesh_cases/manifest.json` binds every prepared input/CSL source and frozen
Python oracle to gnn-csl source commit231ff3c8. The worker additionally validates
the appliance's compiled params and `run(steps:u16)` RPC ABI. The controller
binds the source package to the compiled artifact and run request; this is not
an independent worker-side ELF hash or an appliance SRAM measurement.

## Local verification status

The source has41 passing host tests covering worker calls, capture validation,
ABI and identity rejection, failed compile/launch/collection, failure continuation,
resume, locks and matched-run comparison. The new worker also passed a genuine
local SDK2.10.2 run of the existing2×2 compiled kernel (24.536 seconds worker,
26.865 seconds supervised invocation). `mesh_local_evidence/` contains that
LOCAL SIM capture and its offline audit fixture. Its artifact field identifies
the local build inventory, not a hardware archive. Hardware compile and runtime
remain unrun.

```bash
uv run python -B -m unittest -q test_mesh_worker test_mesh_audit test_mesh_appliance
```

The package exporter is `models/meshgraphnet/export_hardware.py` in gnn-csl.
Export always creates a new package; it rejects source changes relative to
prepared cases and never rewrites an existing experiment's identity.
