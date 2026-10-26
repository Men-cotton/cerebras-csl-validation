# Native H128 GNN appliance validation

This family exports native-width GNN cases for one compiler action followed by
one resident runtime action. Cases retain their original manifest and frozen
files, a portable array/launch contract, independently computed NumPy outputs,
exact counters and padding masks. The package needs NumPy and the site's SDK
to execute; its local numerical auditor needs NumPy only. No training, gradient,
optimizer or CPU GNN evaluation runs inside the appliance worker.

The worker admits completed SDK calls and complete hash-bound captures. The
collector reports `execution_status=PASS` and `numerical_status=NOT_RUN` for
such a capture, including a complete capture containing NaNs or wrong values.
The local auditor then checks every frozen numerical output, immutable input
and bank, counter, padding element and PE-local clock interval. Numerical
failure leaves the original raw arrays and worker records intact.

The adapter stages all graph states and weight banks once, reads the uploaded
values back, executes the frozen launch schedule, and collects all outputs
plus final inputs. Schema1 keeps the single zero-argument `run()` contract.
Schema2 calls `run(steps: u32)` for each declared chunk without another upload.
Each case states which inputs are immutable;
any mutable input must also be a numerically checked output. This transport
contract supports the banked processor block and the general graph-group
native mesh adapter. A processor-only case does not establish encoded input
generation, physical rollout, model training or fluid prediction accuracy.

`export_appliance.py` converts a prepared native-block case through its checked
NumPy oracle. Mesh and complete-model cases declare
`portable_oracle_module=models.meshgraphnet.native_mesh.driver`,
`models.meshgraphnet.native_full.driver`, or
`models.meshgraphnet.native_rollout.driver`. Only these module names are admitted.
Their `portable_expectations(case)` returns packed expectations, exact-field
names, zero masks, default tolerances, immutable inputs and scope. Optional
`tolerances` entries set `rtol`/`atol` for individual approximate outputs; the
physical velocity contract uses3e-5/1e-6, while normalized arrays use3e-4/3e-4.
Export never submits work.

Schema2 freezes one to16 launches, each exactly
`{"entrypoint": "run", "steps": N}`. Positive chunk lengths must total at most
the compiled `MAX_STEPS`, which is bounded to256. The originating model's
`residency.chunks` must equal this schedule. Compiled ABI checking requires
the u32 argument and a four-word u32 `lifecycle` output. Initial lifecycle and
stats must be zero; the model oracle supplies each checkpoint's exact counters
and lifecycle values. The model also owns arithmetic-counter and generation
bounds for its selected shape.

Every output is read after every launch. Captures, expectations and masks use
`launch0_FIELD`, `launch1_FIELD`, and so on; ABI declarations and numerical
tolerances remain keyed by device field. Input readbacks occur once before the
first launch and once after the last. The exact recorded API sequence and
launch argument inventory reject missing checkpoints, extra uploads and changed
chunk lengths. Execution admission still does not compare numerical values.
The numerical audit checks every checkpoint, including nonzero fixed-boundary
bits, padding, immutable banks and PE-local clocks. Its total cycle statistic
sums chunk intervals on each same PE and excludes host checkpoint gaps; each
chunk's interval remains separately reported.

New schema2 expectation and capture archives use lossless ZIP compression.
Prepare source inputs, expectations and masks in compressed form before export:
the exporter preserves every originating byte and cannot remove an uncompressed
source file merely because another encoding is smaller. The128MiB case,
256MiB capture and256-entry limits remain unchanged. Projected capture admission
counts all expanded raw arrays for every checkpoint, including initial and final
input readbacks, so compression does not bypass that limit. Individual NPZ
expanded extents remain bounded too.

Changing the chunk schedule changes the frozen case identity. The existing
artifact record therefore cannot be reused across differently scheduled cases;
compile each case and compare generated kernel hashes before claiming identical
ELFs. Repeating one unchanged case in fresh sessions can reuse its exact build
record without another compile.

`compare_rollouts.py FIRST_CAPTURE SECOND_CAPTURE --out FRESH_REPORT` first
runs the strict auditor on both captures. It then requires matching backends,
parameters, uploaded and original arrays, source files and model provenance,
with different chunk schedules ending at the same physical step. All ten final
numerical arrays must match bit for bit. Initialization, completed steps and
geometry count must agree, while each launch count must match its own schedule.
The report retains each PE's sum of chunk intervals and both generated ELF hash
inventories. Different executable hashes are reported explicitly and do not
turn numerical agreement into a failure. This host-only comparator is included
and registered by export; it submits no jobs and changes no captures.

An optional `exactmasks` dictionary supplies uint32 zero/one masks with each
selected float output's shape. The exporter freezes these arrays and their
selected-element counts. The local auditor requires bit equality to the
finite frozen expectation at every selected element, including nonzero fixed
boundary velocities. A zero selected-element count is explicit when a case
has no such boundary elements; zero-padding checks remain separate. The model oracle owns which boundary
elements must be selected, and the adapter verifies the frozen mask's complete
field inventory, shape, type, values, counts and file identity.

When original expectation or mask NPZ files contain exactly the contract's
arrays, export keeps those original file bytes and maps them directly into
the portable case. Matching requires identical keys, shapes, dtypes and array
bits; it does not require the same ZIP encoding. Differing original files are
preserved under a collision-free `origin-files/` path. This avoids storing a
second copy of large equal expectations without changing the128MiB case or
256MiB capture admission bounds. Both bounds include the mask files.

The generated `validation.json` and `request-template.json` can be integrated
into `cerebras-csl-validation` after host checks and review. Each case produces
three foreground steps in order: appliance compile, appliance run, then a
separate `kind=host` invocation of `saved_audit.py`. The descriptor registers
these entrypoints and the rollout comparator. The controller advances to the next case only after the
audit exits with numerical PASS. The run's execution-admission record remains
separate, and a failed audit preserves the complete capture for diagnosis.
For a paired rollout request, append a `kind=host` step invoking
`validations/native-gnn/compare_rollouts.py` with both runs' `capture-data`
directories after both numerical audits. The exporter registers this entrypoint
but does not infer which case pairs should be compared.

```text
python3 -B -m models.meshgraphnet.native_appliance.export_appliance --case NAME=/ABS/CASE --output /ABS/FRESH --request-id UNIQUE-ID
python3 appliance.py compile --case NAME --backend hardware --namespace job-operator --out /ABS/BUILD
python3 appliance.py run --case NAME --backend hardware --artifact-record /ABS/BUILD/native-build.json --host-timeout 300 --out /ABS/RUN
python3 saved_audit.py /ABS/RUN/capture-data --out /ABS/NEW-numerical-report.json
python3 compare_rollouts.py /ABS/WHOLE/capture-data /ABS/SPLIT/capture-data --out /ABS/NEW-comparison.json
```

The first command runs from the source repository using its budget runner; the
other commands are standalone validation-family entrypoints. In the validation
repository, use its locked uv environment and configured SDK interpreter.
Compile and run steps have a 1200-second client limit; each host audit has a
120-second limit and checks saved arrays without recomputing the GNN oracle.
The controller runs host steps through its configured SDK Python with uv's
`--no-project --no-sync --offline` options, so that interpreter must provide
NumPy. The run worker has a separate 300-second limit and 15-second termination
grace. A client timeout does not establish remote termination or authorize
automatic retry.

Compiler parallelism is explicit: appliance `compile`, the exporter and the
local `native_full/driver.py compile` accept `--compile-workers 1|4|8`, default1.
The appliance requests1000 millicores per compiler worker and sets the matching
`--max-parallelism`. Compile records bind both values and the complete compiler
arguments; execution rechecks that binding. Runtime jobs still request1000
millicores, and local simulation still uses one simulator thread. E200 permits
a bounded1-versus4 compiler study; using8 on hardware requires the root agent's
forecast review before submission. A larger worker count alone is not evidence
of compile speedup.

Compile records bind package, host source, frozen case, compiler arguments,
namespace and artifact bytes. Run records consume that exact artifact record.
The worker records compiled ABI metadata and an inventory of generated kernel
ELF hashes and allocated static SRAM ranges before launch. It never inspects
compiler or runtime binary internals. The compact inventory is an observation
made by the pinned worker; retained captures do not contain all generated ELF
bytes. Static allocation does not measure dynamic stack or runtime high-water.

Case admission is limited to128MiB. Captures are limited to256MiB and256 regular
archive entries; links, traversal, duplicate members and malformed NPY extents
are rejected. Preparation also checks the frozen-case bytes plus the complete
raw array extent and32MiB of metadata allowance against the capture limit.
ELF inventory has separate30-second/20,000-entry/1GiB total limits, with32MiB
per generated ELF. Only `build/inventory.json` uses the explicit16MiB JSON
limit; ordinary records remain limited to4MiB. Collection and numerical reports
retain the inventory's hash, count and resource summary, while its complete
per-file metadata stays in that captured file. The32MiB metadata allowance and
the128/256MiB case/capture bounds are unchanged. The local full-model driver's
private ELF helper also permits1GiB total, retaining that helper's stricter
entry and scan-time limits. Its task-local section reader permits32MiB per ELF
and retains the same44KiB static endpoint guard.
Unreadable artifact subtrees fail closed. Before constructing the SDK runtime,
the worker applies the same 44 KiB static SRAM endpoint limit as the local
native drivers; an oversized inventory remains in the failed capture.
Download limits are checked after transfer, rather than imposing a remote disk
quota. SDK runtime working files stay outside the exported `capture/` folder.
Partial raw captures and logs are downloaded whenever available after failure.

The local auditor exits0 for numerical PASS,1 for numerical FAIL after complete
execution, and2 for incomplete/corrupt capture. `test_appliance.py` uses mock
compiler/launcher objects and a fake SDK memory implementation. These tests
establish adapter behavior only; they are not simulation or hardware results.
No hardware result is implied by exporting a package or publishing a request.
