# Complete Cora inference observation

This observer holds the compact grouped-CSR graph, sparse X and both weights in
one runtime and executes the same complete two-layer model repeatedly. It reads
a monotonically increasing completion epoch from **every application PE** after
each launch. A launch API return alone does not establish completion. The host
completed interval ends after that blocking epoch read; it includes launch and
completion transport. No disk writes or numerical audit occur inside it.

The original native cases and their source hashes remain intact. `PROFILE=0`
is a new control with completion epochs; it is distinct from the frozen E226
binary. `PROFILE=1` adds seven stage accumulators for each layer. These are
same-PE intervals for projection, row reduction wait, row broadcast wait,
packing, column delivery wait, aggregation and local control. Collective waits
include communication, scheduling and synchronization, which this observer
cannot separate. Timestamp/accounting overhead is charged to the following
interval. Comparison of the two profiles measures the observer's perturbation.

Each local interval and counter must fit its checked range. The stage decoder
rejects malformed 48-bit timestamp carriers, interval or accumulator overflow,
missing stage calls and sums exceeding that PE's whole interval. Stages can be
stacked for one PE; maxima across PEs must not be added. These unsynchronized
PE clocks do not establish a global makespan or measured link congestion.

For each invocation the host independently reconstructs FP64 `pre`, logits and
final projection panel from the original input, checks counters and all outputs,
reports maximum absolute, floored elementwise relative and relative L2 errors,
and verifies repeated output bits. Full Cora additionally requires all 2,708
predictions to match and reports the existing custom validation accuracy. It
checks all uploaded data before the first and after the last invocation.
Diagnostic output reads and checks between invocations remain outside timing;
this is a resident repeated cohort, not a diagnostic-free throughput loop.

## Run locally

Run each command through `tools/run_with_budget.py` from the repository root,
with its recorded estimate, timeout and dedicated artifact budget. First run
`python3 -B -m unittest models.hierarchical_gnn.grouped_csr.native_cora_appliance.observation.test_observation`.
The smallest legal fixture is 2×2 and includes duplicate/signed/zero edge
occurrences, an empty owner, partial/dense/empty compact demands and O7 tails.

```sh
python3 -B models/hierarchical_gnn/grouped_csr/native_cora_appliance/observation/run.py prepare --case research/runs/E217/observation/tiny --kind tiny --columns 2 --rows 2 --tile 16
python3 -B models/hierarchical_gnn/grouped_csr/native_cora_appliance/observation/run.py compile --case research/runs/E217/observation/tiny --profile 1
python3 -B models/hierarchical_gnn/grouped_csr/native_cora_appliance/observation/run.py simulate --case research/runs/E217/observation/tiny --profile 1 --out research/runs/E217/observation/tiny-profile --warmup 1 --repeats 3
```

Compile and simulate `--profile 0` into a different output directory to obtain
the matched control. Existing admitted compact native cases can be passed
straight to `compile`; no preparation or weight regeneration is required.
Compiled static endpoints are retained in the result. Dynamic stack peak remains
unmeasured. The added named arrays occupy 132 bytes/PE before compiler alignment
and code; only the actual compiler inventory decides SRAM admission.

## Hardware adapter

The local `compile` and `simulate` subcommands are simulator rungs. The
`capture()` Python entrypoint accepts an explicit hardware backend and the
endpoint supplied inside an existing registered validation worker; it does not submit jobs,
connect by SSH or implement a controller. That workflow must compile these
`layout.csl`/`pe.csl` files with the original case parameters plus `PROFILE:0`
or `PROFILE:1`, and bind the actual generated artifact and source identity.
`capture` verifies the complete exported ABI and static SRAM admission itself.
The worker stages a private build directory and invokes it with absolute case,
build and output paths. The local CLI accepts simulator capture only. A hardware artifact has to be compiled by that existing
workflow before the hardware capture can run.

`result.json` retains backend, exact input hashes, parameters, compile identity,
per-invocation numerical checks, per-PE cycles and stage totals, host SDK setup,
upload and diagnostic-read durations. `runtime/raw.npz` owns raw observations;
`execution.json` remains available on failures. Simulator process wall time,
host completed windows, device local cycles and code-derived traffic are
separate measurements. No hardware execution or H100 comparison is implied by
successful host checks or simulator results.

Official source patterns read for this implementation are both immutable SDK
versions of `benchmarks/gemv-collectives_2d/pe.csl` and
`benchmarks/gemm-collectives_2d/pe.csl`. They establish collective callback
ordering, explicit queue/DSR ownership and command-stream release on each PE.
The timestamp carrier follows the existing native GCN and transfer-granularity
cycle probe. No official example is modified.

The portable exporter prepares a separate registered family and one independent
request per shape and profile. It copies only frozen cases and required Python/
CSL source, then binds them to the compiled artifact and returned actual ELF
inventory. It neither registers nor publishes nor submits the requests:

```sh
python3 -B models/hierarchical_gnn/grouped_csr/native_cora_appliance/observation/export.py --case tiny=research/runs/E217/observation/tiny --out research/runs/E217/observation/export --request-id cora-observe-20261112 --warmup 1 --repeats 5
```

The resulting `validation.json`, `hardware_entry.py`, source tree, case tree and
`package.tar.gz` belong under `validations/native-cora-observation/` in the
existing validation repository. Its generated `request-*.json` use that
repository's compile/run/audit entrypoint contract. Registration, checks and
publication follow that repository's workflow. Each `run` uses `SdkLauncher`;
the worker receives `%CMADDR%` from that launcher, never a direct connection
initiated here. `audit` reconstructs the independent FP64 reference, replays all
raw epochs and verifies source/input/build/request bindings and host windows.
Archive transport copies are removed only after that replay succeeds; failed
worker downloads remain available. `test_hardware` uses stub clients and tests
archive rejection, source staging, artifact confinement and partial-failure
retention without contacting a device.

The hardware compile record also inventories every generated kernel ELF and
application/helper `out.json`/`out_rpc.json` directly from the returned compiler
archive, and verifies its embedded CSL against the frozen sources. Before any
launch, the worker requires its actual unpacked program to match that complete
inventory. Offline replay joins the same inventory to the runtime's observed
ELFs. A valid ABI or SRAM limit alone cannot substitute for this identity check.
Request steps use the controller's exact `name`, `script`, `args`, `kind` schema;
the bounded worker timeout belongs to the adapter, rather than an extra request
step field.

`export.py --runtimes 2` gives each full case/profile two independent runtimes
sharing one compile artifact. The tiny pilot retains one runtime per profile.
Each runtime has its own capture and audit, with `--warmup 1 --repeats 5` by
default. Compare runtime cohorts separately; five calls in one runtime are not
five independent device trials, and two runtimes do not establish a confidence
interval. The worker's default 600-second command timeout does not configure
the appliance scheduler's compile or queue time limit.

Offline audit rechecks raw outputs against its own FP64 reference and retains
fresh per-epoch numerical summaries bound to `execution.json` by SHA256.
Outcome, accuracy and prediction counts must match exactly; maximum absolute
and relative L2 diagnostics permit small cross-host floating-point roundoff.
The elementwise relative maximum may amplify reference differences near zero,
so only its finiteness, sign and absolute-error/floor bound are compared.
Reports use the fresh audit values via `--wse-audit`, not this remote diagnostic.
