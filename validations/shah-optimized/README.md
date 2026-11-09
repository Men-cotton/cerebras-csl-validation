# Shah optimized streaming SpMM

This independent reconstruction follows Shah, Di and Becchi,
[arXiv:2604.27985v1](https://arxiv.org/abs/2604.27985v1), sections3.1.2–3.1.3.
The four frozen execution files come from gnn-csl revision
`927a5d87` under `kernels/sparse/shah-sellpack-spmm/optimized/`.
[E231](https://github.com/Men-cotton/gnn-csl/blob/927a5d87/research/experiments/E231-shah-optimized-streaming.md)
contains local simulator evidence and the exact source-to-paper distinctions.
The adapter is maintained here; SDK images and generated programs stay outside
the source checkout.

The initial request has15 serial steps: R8/F8/V16/Q8/B8 with1 and4 channels,
then R16/F16/V32/Q16/B16 with4 channels. Every build runs seeded random input
in buffered and per-chunk output modes, each followed by an offline audit.
R8 passed local numerical/counter validation; R16 is the next hardware rung.
These are single observations. Problem size and work per PE differ between
shapes, so they do not establish weak-scaling efficiency or paper throughput.
Worker limits are300s; client limits1200s follow the controller's documented
scheduler-wait policy. Seven host oracle tests and six registered adapter tests
passed before publication. Hardware completion remains unobserved.

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
