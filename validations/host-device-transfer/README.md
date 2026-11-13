# Host-device transfer controls

E237 / H118 extends the E235 baseline with geometry, buffering, fixed-total
splitting and matched row/column broadcast. Existing E235 captures and frozen
requests retain their original source identity. This directory is frozen into
the separate `host-device-transfer` validation family.

| Campaign | Controls |
| --- | --- |
| Geometry | 64×64, 46×92, 32×128, 23×184, 16×256 and nonsquare transposes; channels 1 and 16 separately |
| Buffers | 64×64, channels 16; west/east columns (0,0), (1,0), (0,1), (1,1), (4,0), (0,4), (4,4), (16,0), (0,16), (16,16) |
| Splitting | 64×64, channels 16; one logical payload divided into 1, 2, 4, 8 or 16 sequential completed chunks |
| Broadcast | 64×64, channels 16; row broadcast versus replicated-row unicast, and column broadcast versus replicated-column unicast |

Geometry preserves exact ratios 1:1, 1:2, 1:4, 1:8 and 1:16. Rounding the
short dimension upward gives 4232 PEs for 1:2 and 1:8, including transposes;
the other shapes have 4096. Report the actual count, control bytes and padding.
Added I/O buffer columns are infrastructure PEs outside the compute rectangle.
Their allocation changes the total footprint even when compute PEs are fixed.

Each hardware case uses three fresh runtimes and 10 warmups plus 30 measured
logical rounds per size. Size order is ascending, descending or a deterministic
permutation. Total delivered payload is zero or a power of two through 64 MiB.
Broadcast sizes start at the replication factor (64 bytes here) so logical
payload can be exactly replicated. Cases are explicitly enumerated in oracle.py;
dedicated 2×2 smoke cases cannot be submitted through the hardware adapter.

An echo round spans the complete H2D/D2H sequence. Splitting times all chunks
in one interval; preparing frames, allocating arrays, checking replies and
hashing remain outside it. Concatenating each PE's unpadded chunk slices
reconstructs that PE's original tensor at every split count. Each chunk has a generation
word per PE, so control overhead increases with the split count. Only one chunk
is outstanding. This measures granularity, not concurrent request throughput.

Broadcast rounds time H2D delivery through one tagged u32 ACK from every PE.
The kernel sends ACK only after receiving the complete frame. The host then
reads every PE's entire frame outside the timed interval and verifies it before
another round. The matched unicast control delivers exactly the same replicated
data and uses the same ACK/readback protocol. Report delivered payload,
uploaded application payload, actual H2D carrier, ACK carrier and verification readback
as separate quantities. This is delivery-plus-ACK latency, not a physical
one-way latency. E235's full-payload echo RTT is a different timing scope.

Device replies tag each generation with the PE's runtime coordinates, avoiding
a different compiled kernel for every PE while detecting missing, duplicated
or misordered responses. The receive buffer is owned until asynchronous TX
finishes; changing frame length waits for that callback. The buffer remains
4097 u32 words per PE. Streaming uses 32-bit carriers; native16 API flags do
not implicitly pack two application values into a word.

Capture retains all measured and warmup times, per-chunk returned-data hashes,
full final replies/readbacks for each size, source hashes and generated program
metadata. The audit regenerates expected values and exact schedules independently
of SDK execution. The registered adapter binds source/package, compile archive,
actual runtime program, parent request and fresh run UUID. SDK scratch is kept
outside the capture. Package, artifact and capture bounds are 8/256/256 MiB;
scratch has a 512 MiB post-run accounting bound. The worker command is bounded
to 600 s after allocation, independently of scheduler queueing and job lifetime.

Local checks: `python3 -B -m unittest -v test_oracle test_driver` from this
directory through the repository budget runner. The E237 `run_local.py` records
separate budgeted compile and simulator rungs. Registered checks use
`uv run --locked --group dev -m csl_validation check --family host-device-transfer`.
Actual outcomes and publication identities belong to
[E237](https://github.com/Men-cotton/gnn-csl/blob/main/research/experiments/E237-host-device-transfer-controls.md).

Official sources read: old and SDK2.10.2 `benchmarks/bandwidth-test`,
`benchmarks/row-col-broadcast`, `tutorials/topic-03-streaming-wavelet-data`,
`tutorials/topic-09-fifos` and runtime-coordinate examples. Their patterns support
the implementation; old bandwidth/latency comments are not current measurements.
