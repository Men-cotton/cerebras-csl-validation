# Eight-step resident control appliance adapter

E215 carries E166's `mean`, signed `adversarial`, and `tiled6` fixtures through
an ordinary appliance launcher. It freezes E159's `control_split/{layout,pe}.csl`
bytes and reuses the E166 host model only during preparation. The remote package
contains arrays, FP64 expectations and source hashes; its auditor needs NumPy
and has no import dependency on the repository's model code.

One compiled artifact serves three fresh runtimes. Each uploads the same graph,
features, initial weights, supervision and learning rate. `control` selects
resident mode or staged mode; it has no fault injection. `f_run()` initializes
once. Staged and download modes call `f_resume(np.uint32(generation))` seven
times, after reading the completed-generation markers. The separate auditor
checks those values; the CSL entrypoint guards the requested generation.
Download mode also reads seven
intermediate two-slot histories. Resident mode reads the terminal marker eight
times; its markers do not observe intermediate execution. Every mode reads the
final outputs and all immutable inputs. `release` is read-only host evidence of
a device-internal broadcast token. No host operation writes it.

The compiler ABI must expose `f_run() -> void`, `f_resume(expected: u32) -> void`
and the fixed E166 pointer schemas. The runner uses
`cerebras.sdk.runtime.sdkruntimepybind.SdkRuntime`, explicit `load`, `run`, and
one `stop`, with blocking calls and copies. This differs from the SDK client's
context-managed automatic lifecycle. All u16 copies use uint32 native carriers;
the auditor rejects nonzero upper halves before interpreting timestamps.
The runtime handle is retained before constructor-exit journal persistence, so
a journal failure still attempts cleanup once.

The portable auditor checks the independent FP64 expectation for every retained
field, every download prefix, exact counters/markers, replica bits, old-weight
generations, overlapping history slots, structural zeros, final weights and
immutable inputs. Structural capture admission remains distinct from numerical
PASS. Each PE's eight active intervals, seven gaps and total span are separate
from host API durations. Host monotonic endpoints also retain construction to
stop and first launch to last marker windows; those windows include Python and
journal gaps. No cross-PE clock subtraction or speedup inference is made.

`appliance.py` and `appliance_worker.py` derive from
`models/meshgraphnet/native_appliance/`; `portable.py` extracts its bounded file,
archive and generated-ELF helpers. This dedicated family does not import or
change the native H128 loader. Actual generated ELF endpoints must remain
strictly below 44 KiB; stack and runtime high-water remain unmeasured.

Local simulation copies the verified canonical build into each runtime's
dedicated work directory before construction. The simulator may create extra
helper ELFs there; these never change the canonical build used by the other arms.
The wrapper starts from the repository root so its container can see the frozen
source, then the worker changes into that dedicated directory.

`compare_arms.py` first audits all three captures and checks one shared build
identity, then requires bit equality for all seven final non-clock/control
outputs. A one-ULP difference that individual tolerances admit still rejects.
Local driver regressions and comparator tests are in `test_compare.py`.

Read patterns: both immutable SDK versions of
`examples/tutorials/topic-11-collectives/run.py` use explicit ordinary runtime
lifecycle and multiple exported launches. Both versions of
`examples/benchmarks/bandwidth-test/run.appliance.py` show the SDK client
context-managed lifecycle; SDK2.10.2
`examples/benchmarks/7pt-stencil-spmv/run.appliance.py` uses typed launch scalars.
E159's exported `f_resume(expected: u32)` is the exact generation contract.
These sources establish API patterns; successful hardware execution is a
separate evidence rung.

From the repository root, wrap every check or SDK invocation with the budget
runner and record an estimate before starting it. These are the inner commands:

```sh
python3 -B -m unittest -v models.hierarchical_gnn.grouped_csr.checkpoint_training.control_appliance.test_appliance
python3 -B models/hierarchical_gnn/grouped_csr/checkpoint_training/control_appliance/run.py prepare --kind mean --case research/runs/E215/mean
python3 -B models/hierarchical_gnn/grouped_csr/checkpoint_training/control_appliance/run.py compile --case research/runs/E215/mean
python3 -B models/hierarchical_gnn/grouped_csr/checkpoint_training/control_appliance/run.py simulate --case research/runs/E215/mean --arm resident --out research/runs/E215/mean-resident
python3 -B models/hierarchical_gnn/grouped_csr/checkpoint_training/control_appliance/run.py simulate --case research/runs/E215/mean --arm staged --out research/runs/E215/mean-staged
python3 -B models/hierarchical_gnn/grouped_csr/checkpoint_training/control_appliance/run.py simulate --case research/runs/E215/mean --arm download --out research/runs/E215/mean-download
python3 -B models/hierarchical_gnn/grouped_csr/checkpoint_training/control_appliance/export_appliance.py --case research/runs/E215/mean --output research/runs/E215/export-mean --request-id tank01-resident-control-20261029-r001
```

Repeat the local sequence with `--kind adversarial` and fresh paths before
scaling to tiled6. Compile and simulator output roots are dedicated E215
descendants. The simulator is one-threaded with fabric traces disabled.

The export supplies `validation.json` for the new `resident-control` family and
an unsubmitted request template. It compiles once, runs the three arms, and
requires each numerical audit to pass before the next arm, then runs the
cross-arm comparator. Register and publish
through `cerebras-csl-validation`'s existing request workflow. The pilot is not a
balanced timing campaign. Fresh request and compile IDs bind collected captures;
no direct SSH or background controller is used. Preparation, local simulation,
request publication, hardware completion and numerical audit remain separate.
