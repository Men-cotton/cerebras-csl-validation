# Shah optimized streaming SpMM

This independent reconstruction follows Shah, Di and Becchi,
[arXiv:2604.27985v1](https://arxiv.org/abs/2604.27985v1), sections3.1.2–3.1.3.
The CSL execution files originate in gnn-csl revision `927a5d87` under
`kernels/sparse/shah-sellpack-spmm/optimized/`. The host driver/oracle
are exported from gnn-csl `94ca2a30`; T1527 adds controlled fixtures and the
broad campaign while preserving the CSL data path.
The local gnn-csl record `research/experiments/E231-shah-optimized-streaming.md`
contains simulator evidence and the exact source-to-paper distinctions. The
execution source is published in this family: [layout](source/layout.csl),
[PE program](source/pe.csl), [driver](driver.py), and [oracle](oracle.py).
The adapter is maintained here; SDK images and generated programs stay outside
the source checkout.

The initial request has15 serial steps: R8/F8/V16/Q8/B8 with1 and4 channels,
then R16/F16/V32/Q16/B16 with4 channels. Every build runs seeded random input
in buffered and per-chunk output modes, each followed by an offline audit.
R8 passed local numerical/counter validation; R16 is the next hardware rung.
These are single observations. Problem size and work per PE differ between
shapes, so they do not establish weak-scaling efficiency or paper throughput.
The initial request used300s worker and1200s client limits under its original
controller policy. Seven host oracle tests and six registered adapter tests
passed before publication. The returned initial cohort passed all six numerical/protocol audits (E231,
gnn-csl fe5a3165). Buffered windows were shorter in all single-sample comparisons;
channel superiority and weak scaling remain unestablished.

`compile` creates one artifact and pins the source, host and offline auditor.
`run` accepts that exact build record, stages the pinned files, and calls the
same `driver.run_case` function used for local simulation. All host commands
remain foreground and worker runtime is bounded independently of client waiting.
SDK INFO logs expose appliance job IDs. The adapter attempts host log, exit and
capture collection even after launcher failure; incomplete collection fails.

`audit` is a host-only step. It requires the exact `--artifact-record`, the run
directory as `--capture`, and a new JSON `--out`. It binds the returned archive
to the compile record, regenerates fixture identity, recomputes FP64 results
from captured original edge occurrences, checks all role counters and verifies
the input/output call order. It does not infer scheduler terminal status from
numerical success.

Use separate manifest compile, run and audit steps. Compile arguments include
`--R --F --V --Q --B --channels --workload --seed --degree`; run arguments include
`--artifact-record`, `--host-timeout` and `--output-mode buffered|per-chunk`.
The two output schedules use the same compiled accumulator topology. The
comparison changes host serialization and the number of output partitions used
by each call; it does not isolate those two effects or reproduce the original
paper's baseline layout.

The compiler archive reader locates the unique generated tree and hashes its
ELFs, `out.json` and `out_rpc.json` files under explicit entry, time and byte
bounds. The worker records the same inventory from the unpacked program before
execution. The offline auditor compares those bytes with the actual pinned
compiler archive, including the I/O metadata hashes and decoded partition sizes.
The observed E219 SDK 2.10.0 archive layout is `cs_<hash>/out/`; the reader admits
a unique wrapper prefix without assuming that hash value. A fresh UUID binds
each request, collection and capture, including runs that reuse one compile.

The worker captures partition heights from compiled `out/west/out.json` and
`out/east/out.json`. The audit computes which partitions intersect router and
accumulator rows. A channel flag alone does not prove multiple active input or
output channels; these mappings still do not establish physical bandwidth.

Before publication, the primary agent runs the focused adapter tests and the
validation repository's metadata checks. Publication requires a fresh request
ID, matching `ALCF-Request` and `ALCF-Manifest` commit trailers, and an ordinary
push to the watched branch. Requests compare mechanism and moderate shapes;
they do not reproduce the paper's throughput claim.

## Controlled parameter campaign

`campaign.py --out FRESH_DIRECTORY` prepares seven axis manifests and a host
work/padding plan; it submits no jobs. The complete batch has33 configurations,
33compiles,111fresh-runtime runs and111offline audits. Each arm has three
observations; selected anchors also repeat per-chunk output. Request axes are
independent failure domains, not step-count workarounds. The current controller
accepts their four-field steps without a local deadline.

The anchor is R8/F16/V32/Q16/B8/degree2. Axes cover channels1/2/4, balanced and
paired-skew inputs, R16/32/64, F8/32/64/128, fixedN128 Q/B factorizations,
degree1/4/16, original random controls, and F256,N=K=2048. Degree counts
occurrences per output row per original shard. Paired-skew reassigns odd shards
to the preceding even shard, preserving total work and H but changing the
matrix and possibly the output. Fixed-N Q/B preserves edges/H, while changing
accumulator topology and I/O coverage. R sweeps hold work per worker fixed but
grow logical degree; they are synthetic operator weak-work comparisons.

Report every sample and its median/range, actual active I/O groups, work/worker,
imbalance, useful/padding bytes, modeled communication, compiler/run-step wall
time and artifact size. The timer remains the host stream window after H
upload, including output allocation and Python overhead through final D2H.
No GCN, device-cycle or total-session speedup follows from that window. Three
repeats and the reconstructed accumulator differ from the paper's50-run study.

The allocated worker cap defaults to86400seconds. It bounds the foreground
worker only; it does not configure SDK compile/job_time_sec defaults. Each
request still submits one operation at a time and checks remote terminal state.
Do not automatically retry ambiguous submissions. Several hours of serial
service are expected from initial51–81s compile and81–121s run steps, with
frontier uncertainty and scheduler wait separate. No result is implied by
generation or publication.

Before publication,21host tests pass (oracle, adapter numerical corruption
checks and campaign contracts). R4/F4/V8/Q4/B4 paired-skew passes SDK2.10.2
compile and local SIM in both output schedules, with a separate capture replay.
The33-case host plan also passes. New hardware results remain unobserved.
