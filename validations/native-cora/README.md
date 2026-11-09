# Native-feature Cora inference on the appliance

This E217/H105 adapter executes the complete two-layer GCN
`Ahat ReLU(Ahat X W0) W1` with hidden width16 and7 output classes. The full case
uses all2,708 Cora nodes, all1,433 row-normalized input features, all5,278
undirected nonself pairs plus self loops, and the retained seed11 final200epoch
original-graph checkpoint from E087. Both learned projections, both graph
aggregations and ReLU execute in CSL. The weights were trained on the host;
this adapter performs inference.

The original E217 `pe.csl` and `layout.csl` copied E087's grouped forward source. The two
`comptime_struct` module-parameter annotations are removed to match the active
SDK2.10.2 official GEMV example. The current E224 extension adds matched full and
compact column panels without changing arithmetic or received-callback order.
The original E087/E217 files and receipts remain unchanged. The source patterns are
both immutable SDK versions of `benchmarks/gemv-collectives_2d/pe.csl`; the
current example uses untyped module parameters. Actual compile and execution
results belong to the E217 experiment card.


E229 adds `--transport selective` using a separate `selective/` CSL source.
Each source group sends T-lane vectors only to recipient rows. Four header
words identify epoch, phase/tile, source group and packet; a recipient ACK
confirms its cache, then a column RELEASE permits the original CSR arithmetic.
The input directory is immutable and independently reconstructed from the raw
graph. Dedicated guarded packet buffers and exact DATA/ACK/RELEASE counters
make protocol progress observable. The compact implementation remains the
matched baseline at the same owners and T16. Headers, ACKs, RELEASE payload,
directory bytes and actual ELF endpoints are charged separately from useful
arithmetic. Reduced modeled traffic is not measured speedup.

`repeat_probe.py` is a small local-only two-launch diagnostic. It changes X in
one loaded runtime and independently checks both complete models and epochs;
its receipt is deliberately separate from the one-launch portable family.
Partial-result aggregation remains a candidate evaluated by whole-model
quality and measured latency. Bit differences alone do not decide adoption.
The source research record is gnn-csl: research/experiments/E229-finite-column-delivery.md.

The minimum directed fixture has23 input features, signed duplicate edges,
an empty owner and a partial7-class output tile. Features17 and22 are nonzero;
preparation requires its logits to differ from a first16-feature fallback.
The native64-node calibration retains all1,433 features and the real checkpoint,
using the first64 corpus rows and their induced original graph with recomputed
degrees. Calibration subsets do not report validation accuracy.

Preparation reuses E087's packing and FP32 executor. Independently,
`saved_audit.py` reconstructs the raw Cora corpus and trained checkpoint,
decodes every sparse-X feature, weight shard, graph occurrence and owner slot,
and computes full FP64 activations and logits. The portable auditor imports no
repository model code. Source/data/checkpoint hashes, custom train/validation
split and original-graph identity remain explicit. Complete native predictions
must agree with the FP64 reference. Only the full graph reports accuracy on the
existing custom500-node validation set; this is not a Planetoid test score.

The audit checks all `pre`, `logits` and `panel_sum` values at
`atol=2e-5, rtol=2e-4`. It checks exact completion/FMA counters, structural zeros,
final panel replicas and every input before and after execution, including both
weight matrices; clocks must encode valid positive same-PE intervals. u16 uploads and downloads use
native uint32 carriers; nonzero upper halves are rejected before decoding.
Numerical PASS is separate from structural execution admission.

The dedicated `native-cora` family fixes H16/O7, one `f_run()` and the original
u16 pointer ABI. It does not widen the native H128 appliance schema. Its
transport, bounded file/archive/ELF helpers and failure-safe runtime journal
derive from the existing native appliance and E215 adapter. A runtime uses
explicit `load`, `run`, blocking copies/launch and one `stop`. The native handle
is retained before constructor-exit journal writes. Local simulation invokes
the SDK wrapper from the repository root, then executes against a verified
private build copy under `sdk-work`; SDK-generated ELFs cannot mutate the
canonical artifact shared by later runs.

Named arrays and scalar bytes, code reserve, useful FMAs/imbalance, sparse
counts and full-panel modeled traffic are recorded before compilation. An
18KiB code/SDK reserve is the initial forecast; actual generated ELF endpoints
must remain strictly below44KiB, with stack high-water unmeasured. Forecast
rejections remain prepared and cannot compile or export. Full geometry
preflights should include C16/R16 and C8/R64/R128 at common T4/T8 values before
choosing a hardware pair. Increasing rows reduces group scratch but does not
reduce the C8 W0 shard's11,520 bytes, and increases collective invocation count.
A larger PE rectangle does not by itself establish faster GCN inference.

Run each inner command below through the repository budget runner, with a
recorded estimate, one CPU thread and traces disabled. Root owns all executions.

```sh
python3 -B -m unittest -v models.hierarchical_gnn.grouped_csr.native_cora_appliance.test_appliance
python3 -B models/hierarchical_gnn/grouped_csr/native_cora_appliance/run.py prepare --case research/runs/E217/tiny --kind tiny --columns 2 --rows 2 --tile 4
python3 -B models/hierarchical_gnn/grouped_csr/native_cora_appliance/run.py compile --case research/runs/E217/tiny
python3 -B models/hierarchical_gnn/grouped_csr/native_cora_appliance/run.py simulate --case research/runs/E217/tiny --out research/runs/E217/tiny-run
python3 -B models/hierarchical_gnn/grouped_csr/native_cora_appliance/run.py prepare --case research/runs/E217/native64 --kind cora --nodes 64 --columns 8 --rows 2 --tile 8
python3 -B models/hierarchical_gnn/grouped_csr/native_cora_appliance/run.py prepare --case research/runs/E217/full-c8r128-t8 --kind cora --nodes 2708 --columns 8 --rows 128 --tile 8
python3 -B models/hierarchical_gnn/grouped_csr/native_cora_appliance/export_appliance.py --case tiny=research/runs/E217/tiny --case full-c8r128-t8=research/runs/E217/full-c8r128-t8 --output research/runs/E217/export --request-id tank01-native-cora-20261029-r001
```

The exporter supplies a new-family validation descriptor and an unsubmitted
request template. Each case compiles, captures, then undergoes independent
numerical audit before the next case. Register and publish through
`cerebras-csl-validation`; no direct SSH is used. The hardware worker timeout is
explicit and is not extrapolated from simulator process wall time. Cross-geometry
comparison uses each full FP64 tensor/argmax audit; reduction grouping may change
FP32 bits. Host windows and same-PE TSC durations remain separate, and neither
is synchronized device makespan. Compilation, local simulation, request
publication, hardware completion and numerical audit are separate evidence.

## Demand-compacted column panels

`demand.py` keeps the existing two-layer arithmetic and ownership, replacing each
column's full source panel with its remote demanded slots. Local sources keep
home-panel indexes; duplicate and zero-coefficient occurrences remain. Dense
pairs use the full panel without packing. The report includes packing copies,
unchanged callbacks and added directories, so payload savings are not presented
as speedups. [E223](../../../../research/experiments/E223-consumer-directed-reuse.md)
owns the host full-Cora results. T1445 adds the corresponding CSL transport;
compilation, local simulation and hardware results are recorded separately.

Run the seven `test_demand` unittest cases through the budget runner. The CLI
accepts `--case /ABS/existing-E217-case` (repeat for a second shape),
`--output /ABS/research/runs/E223/demand/report.json`, and optional `--execute`
for full FP32 baseline-bit and frozen-FP64 output comparisons. Without that flag
it only reports structural demand. The output path must be fresh.

New preparation accepts `--transport full|compact` and always adds `COMPACT`
and `U` parameters, immutable `sender_slots[K]` and `source_lengths[R]` inputs.
The matched full arm uploads the same directories but sends the entire panel.
Its metadata storage therefore exceeds the frozen E217 baseline by `2K+2R`
bytes/PE. Compact uses `U*T` RX storage, packs only partial-demand panels, and
uses the original panel directly for dense demand. An empty column demand
activates the received callback locally; source-row consumers still use their
full home panel. All rows of a column must have identical lengths.

The standalone auditor reconstructs exact directories and ordered occurrences
from the original graph and owner map. Legacy frozen E217 parameter sets and
schemas remain accepted. Newly prepared cases use the extended schema, and the
registered `native-cora` exporter/runtime handle its additional immutable inputs.
The historical `demand.py` exploration CLI accepts legacy full-panel cases only.

Run through the budget runner, one SDK command at a time:

```sh
python3 -B -m unittest models.hierarchical_gnn.grouped_csr.native_cora_appliance.test_compact
python3 -B research/experiments/data/E224/cora_prepare.py --output research/runs/E224/cora-tiny/cases --shapes tiny
python3 -B models/hierarchical_gnn/grouped_csr/native_cora_appliance/run.py compile --case research/runs/E224/cora-tiny/cases/tiny-compact
python3 -B models/hierarchical_gnn/grouped_csr/native_cora_appliance/run.py simulate --case research/runs/E224/cora-tiny/cases/tiny-compact --out research/runs/E224/cora-tiny/compact-sim
python3 -B research/experiments/data/E224/cora_prepare.py --output research/runs/E224/cora-full/cases --shapes r64 r128
```

The 2×2 demand fixture includes partial and dense panels, an empty broadcast,
an empty owner, local-only sources, duplicates, signed/zero coefficients and O7
tail lanes. Its dense pair keeps `U=K`: it checks branch behavior, not storage
savings. Full native R64/R128 cases retain E217's original graph and weights.
Package the chosen matched pairs with `export_appliance.py`; preparation and
package creation alone do not establish remote execution or performance.

E224/E225 now retain independent hardware audits at 256/512/1024 PE. Compact
panels reduce the matched PE intervals, while host timing does not establish
improvement. R32 trades half the R64 PE count for about 5.21% longer intervals
across separate campaigns. E226 holds compact C8/R64 ownership fixed and
compares T8/T16 in returned hardware captures. T16 preserves full-model outputs
and shortens the median maximum PE interval by23.48%, with larger panel storage
and modeled traffic. Host E2E improvement remains unestablished; see E226 for
the exact timing scope.
