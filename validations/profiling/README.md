# Sparse PE timestamps: TANK29 feasibility

This first appliance request tests whether five receiver phase records can be
captured and compared with recording disabled. It uses two adjacent PEs and
N=4: eight useful 16-bit payload values carried in nine 32-bit DATA words,
including a terminal word. READY and ACK add one word each. There is no shape
sweep, per-element timestamp loop, raw performance-counter read or fabric trace.

One compiled artifact serves four serial fresh runtimes in order **0, 1, 1, 0**
(off/on/on/off). Each runtime executes three invocations with the original
A/B/A input sequence to check state reuse. These are nested observations;
there are two independent runtimes per arm. A final host step validates and
compares the four captures. This means one compile job and four run jobs.

## What the timestamps mean

The sender retains four original timestamps: DATA start, TX callback, tagged ACK
receipt, and joined completion. The primary comparison uses that sender's
start-to-joined-end interval. Receiver flag 1 adds exactly these five records:

1. Arm, before posting receive and sending READY.
2. RX completion callback entry, before validation work.
3. Validation and output-copy completion.
4. Before ACK issue; record stores precede the asynchronous send.
5. ACK TX callback, before unblocking the host command stream.

Each record holds generation, event ID and three zero-extended u16 TSC limbs in
u32 carriers. Readback follows blocking completion. The receiver phases include
observer instructions, validation and callback scheduling; they are not pure
compute/network counters. Never subtract clocks across PEs or add overlapping
intervals to construct a global makespan.

Flag 0 retains the same buffers, branches, event-call sites and receiver TSC
enable/disable. The difference measures the marginal effect of enabling records
in one ELF, not all instrumentation cost against an untouched kernel. The
comparison retains signed and relative changes without rejecting a slowdown.
If overhead is large, retain this small result before deciding on more probes.

The copied source also contains E195's optional flag-2 counter code. This
family's worker, adapter and oracle admission reject flag 2, and both admitted
arms require its exported samples to remain zero. Counter event semantics are
not part of this request.

## Evidence and outputs

Each runtime retains 60 input/output arrays and 80 closed native API calls.
Admission checks complete payloads, upper indices, poison tails, guards,
scratch mirrors, generation, callback/ACK/join counts, flag and bounded event
records. Missing records, overflow, invalid ordering, ambiguous receiver clock
wrap or wrong compile/source identity fail admission. The inherited sender
clock oracle rejects wrap rather than interpreting it.

Captures bind a unique run UUID to the exact build manifest and request. Raw
arrays, compiled parameter/RPC metadata, and the five frozen CSL/host files are
hashed and retained. `comparison.json` preserves each runtime and invocation,
same-PE phases, source work/traffic quantities and off/on differences.

Per invocation, source accounting gives 44 endpoint-injected bytes over the
adjacent links, including READY/ACK; 40 bytes belong to the sender's interval
because READY precedes it. Do not add TX and RX counts. Both arms export
548 bytes/PE (1,096 bytes per full H2D or D2H), including 168 bytes/PE beyond the
original transport ABI. Private stamps, code, descriptors and stack are separate.
These are logical operation/transfer counts, not measured retired instructions,
stall fractions or peak link bandwidth.

The client step bound is 1,200 seconds including queue/startup; the worker bound
is 300 seconds plus a 15-second termination grace. These are limits, not expected
kernel times. Ambiguous submission or a missing terminal job stops the watcher;
there is no automatic retry. Complete results enter the watcher's normal ZIP
and push-free laptop collection workflow.

## Develop and inspect

From the repository root:

```bash
uv run --locked --group dev -m csl_validation check --family profiling
uv run --locked -m csl_validation jobs list --query tank29
```

Each `profile_appliance.py compile` or `run` performs one appliance action.
`profile_compare.py` only reads four admitted capture directories and writes a
fresh comparison JSON. Neither the tests nor listing commands submit jobs.
The request selects `profile_appliance.py` for appliance steps and
`profile_compare.py` for the host step through `validation.json`.

## Source provenance and measurement limits

`source/layout.csl`, `source/pe.csl` and `oracle.py` are byte-for-byte copies of
the E195 phase-profile kernel and E178 matched-u32 oracle from research revision
`a5dcfe33800b438b5fb465d8d1bb95c672e90874`. The portable worker and appliance
adapter replace the original research-tree-specific local launcher. Their source
identities are recorded with every build/capture.

The official SDK 2.10.2 examples read for this port are
`topic-02-libraries/pe_program.csl` (TSC/export pattern) and
`topic-12-debug-library/sender.csl` (async callback boundary). These examples
remain unmodified. The inherited E195 local-SIM N4 result was 1,289 versus 1,446
sender cycles (+157, 12.18%); it is prior SDK 2.10.2 simulator evidence, not an
appliance prediction. New local checks and returned hardware captures must be
reported separately. This experiment does not yet instrument a GNN kernel or
separate arithmetic work from intentional receiver delay.
