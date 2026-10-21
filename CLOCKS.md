# Host command and device clocks

For the fixed hardware campaign with tmux, continued execution after failures,
and one-command collection, use [CLOCK_CAMPAIGN.md](CLOCK_CAMPAIGN.md).

This additive probe investigates host command submission without changing the
native16 reproducer. It runs the same CSL source and full-word oracle on local
SDK SIM, appliance SIM, and hardware. Only local SDK 2.10.2 SIM has been run by
the author. Appliance and hardware commands below are prepared for an operator;
mock client tests establish controller behavior, not appliance compatibility.

Each fresh runtime runs A/B/A inputs. Those three generations share a runtime
and are not three independent repetitions. Each PE snapshots every input word,
exports start/end generations and guards, and publishes three raw 48-bit clocks:

| Quantity | Observation | Limit |
|---|---|---|
| `span_cycles` | start to end on the same PE | Includes host arrival gaps when the interval spans commands. |
| `work_cycles` | start of device copy loop to its end | Includes loop/clock overhead; final full snapshot checks the data, not each repeated copy separately. |
| `pre_work_cycles` | span minus work | Includes dispatch, instrumentation and any witness transfer; not pure RPC latency. |
| host wall / process CPU ns | each native API call | Neither is a CPU cycle counter; process CPU can include SDK threads. |

PE clocks are not assumed synchronized. Do not subtract clocks from different
PEs or convert them using an assumed MHz. The maximum *local duration* is not a
synchronized global makespan. A blocking API return and `task_wait` are measured
boundaries; the probe does not assume either proves execution of a device marker.

`start=witness` performs a **blocking tic plus a blocking D2H** of its start
record. Its generation and raw start clock must match the final record. It is
an instrumented control with additional traffic. `after_end_witness` reads the
end record, sleeps, and reads that same record again; all 16 words must match.
`after_end_return` only follows the end API return. The `internal` entrypoint
places both clocks and the work inside one device invocation.

## Local SDK SIM

Use an SDK environment that provides `cslc`, `cs_python` and numpy. The drivers
cap the compile helper at one thread and disable SIM traces. Place them under
your resource supervisor for process-tree time, memory and disk limits. The
local driver has a 180-second child timeout, and copies each pristine build
into a fresh run directory because the runtime creates auxiliary ELF files.

```bash
python clock_local.py compile --cols 2 --rows 2 --words 64 --out build-clock
printf '%s\n' '{"config":{"kind":"empty","start":"witness","pause_at":"after_start","pause_seconds":0.25}}' > condition.json
python clock_local.py run --manifest build-clock/clock-build.json \
  --config-file condition.json --out run-clock
python clock_inspect.py --manifest run-clock/clock-build.json \
  --request run-clock/clock-request.json --capture run-clock/capture
```

`--compiler /path/to/cslc` and `--python /path/to/cs_python` select local wrappers.
Run each contrast in fresh output directories. Keep the same compiled artifact
within a pair and record the actual compiler/runtime version and container
identity; they may differ between the operator's SIM and appliance installations.

## Appliance SIM and hardware

Use the site's SDK client environment and existing configuration. Compilation
and execution are separate actions; no command below is run by importing a
module. The host worker executes on the appliance worker, so its deliberate
pause and host timing concern that worker, not the laptop launching the job.

```bash
python clock_appliance.py compile simulator --cols 2 --rows 2 --words 64 \
  --out build-clock-appliance
python clock_appliance.py run simulator \
  --artifact-record build-clock-appliance/clock-build.json \
  --config-file condition.json --out run-clock-appliance

python clock_appliance.py compile hardware --cols 2 --rows 2 --words 64 \
  --out build-clock-hardware
python clock_appliance.py run hardware \
  --artifact-record build-clock-hardware/clock-build.json \
  --config-file condition.json --out run-clock-hardware
```

Hardware defaults explicitly target the ALCF WSE-3 fabric dimensions
`762,1172`. Override `--fabric-dims` and `--fabric-offsets` for the site's
configuration. `--namespace` on compile is retained by the launch manifest.
`--host-timeout` defaults to 300 seconds. The client requests one CPU and 8 GiB
for the worker; local artifact snapshots and bounded downloaded archives do not
enforce remote disk limits. Keep the site's job/system/runtime identity from
its logs alongside these records. Unknown versions or missing job IDs remain
unknown. There is no hardware frequency assumption.

`collection.json` records actual downloaded names (a directory may become
`capture.tar.gz`), log hashes, worker exit, and raw validation. Reinspect with:

```bash
python clock_inspect.py --manifest run-clock-hardware/clock-build.json \
  --request run-clock-hardware/clock-request.json \
  --capture run-clock-hardware/capture.tar.gz
```

Use the actual capture filename in `collection.json`. Archives are bounded
and read without extraction. For appliance runs, compile request/status, artifact, CSL source,
host worker and launcher hashes are joined before launch. Raw arrays, native
API arguments, complete call order, generation and clock summaries are checked
again offline. Status distinguishes `pass`, `observed_numeric_fail` and
`incomplete`; collecting a failed run does not make it a pass.

## Representative comparisons

Every condition file has one `config` key. Defaults are transfer,
blocking-tic-plus-witness, queued H2D, rectangle, no pause, work=1, three steps.
Start choices are `blocking`, `queued`, `wait`, `witness`; H2D choices are
`blocking`, `queued`, `wait`. Vary one independently of the other.

| Question | Conditions sharing a build |
|---|---|
| Does a host delay enter the PE interval? | empty, each start mode, after_start pause 0 versus 0.25 s |
| Was start executed before the pause? | witness; verify start generation and raw clock against final |
| Can an already observed end change? | after_end_witness; require bitwise identity within each generation |
| Does the control affect contained work? | internal, start=blocking, work=8, no pause versus before_start and after_end_witness |
| How does call granularity behave? | transfer rectangle / strips / tiles = 1 / R / C×R calls, identical bytes |
| Does mid-transfer arrival matter? | strips, between_h2d pause 0 versus 0.25 s |
| Do effects survive another work/shape? | 2×2 W16/64/256 and 4×2 W64; repeat matching conditions |

Other pause positions are `before_start`, `before_end`, `after_end_return`.
An explicit zero-second pause is a no-op, not `sleep(0)`. The upper bound is
1 second per generation. Packing, allocation, payload hashing and report
validation occur outside the timed interval; API bookkeeping remains inside
host-spanning intervals. Payloads remain alive until full blocking readback.

Select a contrast before its confirmation runs. Use four fresh pairs ordered
AB, BA, BA, AB; report every observation rather than selecting a favorable
warm window. Do not fit a per-byte/per-command latency model from a single
geometry. Preserve internal work, caller flags, witnesses, API overhead and
worker environment when comparing SIM with hardware. These small probes
identify measurement behavior; they do not establish a scaling frontier or
GNN throughput.

Apply the observation boundaries to other workloads as follows. This is a
measurement design for those tasks, not a claim that this probe ran them.

| Tank | Work interval and final completion witness |
|---|---|
| 01 resident layers | layer/update work; completed generation before resume |
| 03 irregular communication | receive/compute work; all required owners' outputs |
| 04 bounded CSR panels | panel compute; destination merge and release acknowledgement |
| 05 comparable completion | causal GO/result/all-ACK at one clock owner |
| 20 collectives | reduction work; root result and all participant acknowledgements |
| 24 message service | receiver service; sender join, receiver terminal and release |

Carry separate span/work/arrival observations to each actual workload, check
all outputs and preserve its own completion semantics. A successful memory-copy
probe does not establish a collective's or a receiver service's completion.

## Sources

- [SDK runtime API](https://sdk.cerebras.ai/api-docs/sdkruntime-api) defines native Tasks and submission calls.
- [Tensor streaming](https://sdk.cerebras.ai/tensor-streaming) describes nonblocking command aggregation; it does not specify all SIM idle behavior.
- [SDK release notes](https://sdk.cerebras.ai/sdk-release-notes/sdk-rel-notes-cumulative) document TSC reset on enable; this probe enables it once.
- [Appliance client API](https://sdk.cerebras.ai/api-docs/appliance-api) defines compiler, launcher, staging and artifact collection.
- Official SDK `tutorials/topic-02-libraries` supplies the same-PE clock pattern;
  `benchmarks/bandwidth-test` brackets host transfers with device timestamps.

Offline tests: `python -m unittest discover -s tests -p 'test_clock*.py'`.

## CPU submission counters

Add `"host_pmu":true` inside a condition's `config` to request Linux PMU counts
for each API invocation and one empty instrumentation region. This is opt-in:
PMU control calls add work between host commands and may perturb PE spans.
Compare it in separate runs with the default instrumentation.

`clock_pmu.py` opens a grouped `cycles:u` / `instructions:u` pair for the calling
thread (`pid=0,cpu=-1`, no inheritance). Other SDK threads, kernel execution and
off-CPU waits are outside its scope. It records raw counts and per-region
enabled/running nanoseconds, with no multiplexing correction. If running time
is smaller than enabled time, the counts cover only scheduled counter time.
Keep that fact; do not treat them as complete execution counts. CPU cycles can
still depend on CPU frequency and shared-resource contention. The measured
region includes the Python/ctypes instrumentation after enable and before
disable; the `pmu-empty` record is an observed control, not an automatic quantity
to subtract.

An unsupported platform, permission failure, unavailable event or invalid read
produces `status=unavailable` with its reason and **no count fields**. It does
not silently replace cycles with wall time, TSC ticks or zero. A numerically
passing capture can therefore have unavailable CPU counters. Linux x86-64 and
AArch64 LP64 little-endian ABIs are supported; no privileged system setting is
changed. PMU timing behavior on the operator's worker must be checked there.

The [Linux perf event ABI](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)
defines thread attachment, event groups and enabled/running times. RESET clears
counts but not those times; the implementation takes their per-region delta.
[Python CPU time](https://docs.python.org/3/library/time.html#time.process_time)
is a different observable. An invariant [CPU TSC](https://docs.kernel.org/virt/kvm/x86/timekeeping.html)
measures elapsed ticks even while the caller is off CPU, so reading it alone
does not isolate host instruction cost. Wrapping the entire simulator in
inheriting `perf stat` would also count simulator work and should not be called
submission-thread cycles.

A client interruption or a host-worker timeout does not establish remote job
termination. Preserve returned job IDs and confirm the site's job status before
retrying. The harness does not cancel an arbitrary remote job automatically.

## Local observations, 2026-10-21

SDK 2.10.2 local SIM passed 53 fresh runtimes / 159 A/B/A generations on
2×2 W16/64/256 and 4×2 W64. A separate, producer-free raw auditor checked
582 arrays / 119,616 u32 words, 684 PE intervals and all 1,597 SDK API records.
The repository's complete offline suite passed 93 tests (53 new, 40 existing).
These counts describe local evidence; no appliance or hardware job was run.

The four confirmation pairs used `kind=empty,start=witness,pause_at=after_start`.
A is pause0, B is pause0.25. Each entry is the three maximum **local** durations:

| Fresh pair/order | A span cycles | B span cycles |
|---|---|---|
| 1 AB | 2688 / 2691 / 2676 | 8315 / 7956 / 8988 |
| 2 BA | 2681 / 2686 / 2667 | 8174 / 7475 / 8626 |
| 3 BA | 2696 / 2692 / 2661 | 9182 / 9250 / 9280 |
| 4 AB | 2704 / 2709 / 2686 | 8798 / 8585 / 9227 |

Every corresponding PE interval increased; copy work stayed55 cycles. The
start witness establishes that the start marker existed before the pause, so
pre-start queue residence alone cannot explain this observation. The exact SIM
progress mechanism remains unresolved. The internal work=8 control stayed at
span300/work286 on W64 with pauses before start, after API return and after an
observed end. Re-reading an observed end was bitwise invariant. Witness reads
themselves add traffic and time; do not compare their spans as free probes.

Eight PMU-enabled runtimes returned 240 measured regions, all with equal
running/enabled times. For one queued rectangle's three H2D calls, the recorded
instructions were50,640–50,918 and cycles71,731–121,913; these include measurement
instrumentation and are not isolated SDK instructions. The0.25-second sleep
controls retained roughly250–255ms wall intervals while their thread PMU
enabled intervals were about0.185–0.248ms. This illustrates the distinct scopes;
no nominal CPU-frequency conversion or empty-region subtraction was applied.
PMU instrumentation is opt-in and its observations are retained separately
from the confirmation pairs.

Development history is separate from that final cohort: a controller tuple
unpack failed after a numerically correct worker; 3D D2H buffers produced SDK
warnings and were replaced by documented flat buffers; build reuse initially
rejected runtime-generated auxiliary ELF files and was repaired with fresh
copies. No timing threshold was chosen to discard a final-cohort observation.
