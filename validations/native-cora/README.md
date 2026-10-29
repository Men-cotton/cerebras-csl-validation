# Native-feature Cora inference on the appliance

This E217/H105 adapter executes the complete two-layer GCN
`Ahat ReLU(Ahat X W0) W1` with hidden width16 and7 output classes. The full case
uses all2,708 Cora nodes, all1,433 row-normalized input features, all5,278
undirected nonself pairs plus self loops, and the retained seed11 final200epoch
original-graph checkpoint from E087. Both learned projections, both graph
aggregations and ReLU execute in CSL. The weights were trained on the host;
this adapter performs inference.

`pe.csl` and `layout.csl` copy E087's grouped forward source. The two
`comptime_struct` module-parameter annotations are removed to match the active
SDK2.10.2 official GEMV example. No arithmetic or callback sequence changes.
The original E087 files and receipts remain unchanged. The source patterns are
both immutable SDK versions of `benchmarks/gemv-collectives_2d/pe.csl`; the
current example uses untyped module parameters. Actual compile and execution
results belong to the E217 experiment card.

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
