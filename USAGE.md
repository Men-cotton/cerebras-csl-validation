# Running the transfer diagnostics

Use the existing Python environment with NumPy and the site-provided
`cerebras_appliance` / `cerebras_sdk` client. These scripts use the ordinary
`SdkCompiler` and `SdkLauncher` path. They do not require a debug session,
reserved device, support request, or `rg`.

The original `source/` and `run.py` are unchanged. Results, downloaded logs,
compiled artifacts and the virtual environment are ignored by Git.

## Existing job information

From the repository root on the user node:

```bash
git pull --ff-only
uv run python collect_jobs.py 'compile-*.log' 'launch-hw-*.log'
```

The default command only reads local logs. To fetch JSON and logs for those
jobs with the existing account:

```bash
csctl get --help
csctl log-export --help
uv run python collect_jobs.py 'compile-*.log' 'launch-hw-*.log' \
  --collect --namespace job-operator --out job-diagnostics
```

Use the namespace already used by your jobs. Each job gets a subdirectory
with its command outputs. `jobs.json` keeps the log-to-job mapping and
`status.systems` when present. Compile and launch jobs remain distinct.
A missing field or unavailable log remains missing; the scripts do not
infer compiler/runtime versions from client versions or connection IPs.
No result is uploaded to GitHub.

## Compile a diagnostic variant

| Variant | Receiver and observations |
| --- | --- |
| `baseline` | Original source and host |
| `control` | Diagnostic sibling with additional arrays, no event recording |
| `events` | Original native receiver plus callback queue flags and timestamps |
| `payload` | u32 data task records complete payload words through an end marker |

Start with `control`, then `events`, then `payload`. A separate `control`
checks whether the changed program layout already changes the outcome.
For each variant choose a new artifact directory. Example:

```bash
set -o pipefail
timeout --signal=INT --kill-after=30s 15m \
  uv run python appliance.py compile hardware --variant control \
  --out artifacts-control-hardware 2>&1 | tee compile-control-hardware.log
```

Use `simulator` instead of `hardware` for an appliance SIM run. The default
fabric dimensions are 9x3 for SIM and 762x1172 for hardware, with application
offsets 4,1. These hardware dimensions are for WSE-3 at ALCF. If a namespace
is required, pass `--namespace NAME` to compile; run reuses that value.
The compiler validates optional `--fabric-offsets X,Y`. Changing offsets
also produces a different artifact and does not pin a physical CS-X.

## Run and inspect one condition

```bash
set -o pipefail
timeout --signal=INT --kill-after=30s 15m \
  uv run python appliance.py run hardware --variant control \
  --artifact-record artifacts-control-hardware/artifact.json \
  --case 3:7:0 --out results-control-hardware-3-7-0 \
  2>&1 | tee launch-hw-control-3-7-0.log
uv run python inspect_result.py results-control-hardware-3-7-0/capture
```

Run one condition at a time. Cases are `offset:length:mode`; mode 0 sends
rows directly and mode 1 first copies rows to an aligned buffer. Use
`3:7:0`, `3:8:0`, `2:9:0`, `3:8:1` for the boundary and controls. Also use
`3:0:0` and `3:1:0` to check the recorder's empty and single-element paths.
For the next variant replace `control` in both commands and output paths.

Host exit 0 means all A/B/A panels matched. Exit 1 means a completed numerical
mismatch (`observed_numeric_fail`) or a payload-template difference
(`observed_payload_difference`); inspect `capture.json`, panel errors and raw words. Exit 2 from the
launcher means a setup, execution or collection error. Retain that directory
and check job termination before continuing. The outer timeout does not
prove that a cluster job has ended; use `csctl get jobs -a` to check your jobs.

The launcher requires the matching backend, variant and source/host hashes
from `artifact.json`. Legacy records created by separately saved scripts do
not contain these fields; keep those artifacts/results and use a fresh build
for this launcher. Output directories must be new, so reruns cannot replace
previous captures. A directory download is saved as a tarball named `capture`.

## Interpret the additional records

`events` saves per-PE event IDs, phase, 48-bit timestamp and IQ/OQ flags.
Event IDs 0/1/2/3/4 mean start/header/rows/values/finish. Queue flags store
empty bits in the low byte and full bits in the high byte; queue 2 is the
transfer queue. Reads are sequential, not an atomic snapshot. Compare time
differences within one PE only. Host wall time is not a kernel timing result.

`payload` replaces the native receiver and appends `0xfefedcba` on the same
sender queue. It records up to 64 u32 words including the marker, with count,
overflow, guard and completion checks. It compares the entire stream against
scalar16 and packed16 representations; ambiguous zero/one-element cases
retain both matches. A completed recording alone is not numerical success.
It observes data-task payloads, not physical-wire traces or control bits.
A template difference preserves the words without establishing that an
unmatched packing representation is itself invalid.

If instrumentation changes an outcome, retain both results. A different
receiver, memory placement or backpressure may hide an error. These captures
do not by themselves identify a compiler, router or hardware defect.

## Host checks

```bash
uv run python -m unittest discover -s tests
```

Tests use local mocks and independent payload fixtures; they submit no jobs.
API references: [appliance](https://sdk.cerebras.ai/api-docs/appliance-api),
[data tasks](https://sdk.cerebras.ai/csl/language/task-ids),
[queue and time libraries](https://sdk.cerebras.ai/csl/language/libraries),
[job monitoring](https://training-docs.cerebras.ai/rel-2.10.0/cluster-monitoring/cerebras-job-scheduling-and-monitoring/cli-for-job-monitoring-csctl).

## Host submission and device clocks

The separate [command-clock probe](CLOCKS.md) provides start/end witnesses,
full-word A/B/A validation, independent start/H2D modes, pause placement,
rectangle/row/tile transfers, device-contained work and optional host-thread
PMU counters. It has local SIM and appliance/hardware drivers; this addition
does not change the native16 tests above. See the guide for representative
comparisons, source-bound captures and the current validation limits.
