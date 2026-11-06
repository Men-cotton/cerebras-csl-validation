# Bounded forward/adjoint appliance family

This dedicated `bounded-adjoint` family exports the existing E214/E223 CSL
without changing its arithmetic, directories or communication. One runtime
uploads the graph and Z, executes `Y = A Z` and `dZ = Aᵀ D`, uploads only the
second D, and executes the same zero-argument `run()` again. The family retains
native u16 directory pointers, uint32 traversal counters and six uint32 clock
words. It does not use the native-Cora or native-H128 numerical schema.

`export_appliance.py` validates an already prepared E214 case against the live
source/SDK before converting it. The exported case retains its original manifest,
all original arrays and unchanged CSL bytes. Workers and saved auditors require
only their frozen package; they do not import the model preparer or find local
SDK images. The exporter itself is a repository-side preparation tool.

The standalone auditor reconstructs the ordered directory from original edge
occurrences, including duplicates, zero coefficients, external IDs and padding.
It independently scatters FP64 forward/adjoint outputs, derives both operators'
scan/cursor counters and checks the frozen reference before trusting captures.
All 81 arrays are retained: initial counters, two sets of inputs before/after
execution, decoded u16 and their full uint32 carriers, owner results, panel
buffers, counters and clocks. Inputs must match bitwise in each epoch; Y must
repeat bitwise when D changes. Numerical bounds remain `atol=2e-5, rtol=2e-4`.

The original-edge oracle defines every final local partial panel and the final
source row's column/row reductions. Other non-root reduction scratch is checked
for finiteness and feature padding, as in E211. Retaining every array does not
give undefined scratch a new numerical meaning.

Each same-PE clock interval spans the first forward descriptor through the
last adjoint RELEASE/status stores. It excludes initialization, output clearing,
clock export and host transfers. Per-launch host windows span launch entry to
the completed stats read. The constructor-to-stop window includes diagnostic
readbacks and journaling. These scopes do not measure synchronized global
makespan or model training.

`source-origin.json` pins the native-Cora transport/archive/ELF/Journal sources
used here. The worker checks exact case, host source, request, artifact, generated
ELF inventory and pointer ABI bindings. Numerical audit remains a separate
registered step after structural capture admission. Native handles are owned
before constructor-exit journaling, and `stop()` is attempted once after failures.
Local SIM uses a verified private build copy so runtime metadata cannot modify
the canonical artifact reused by another run.

The original E214 capture ceiling remains 64 MiB of expanded arrays, with the
same conservative NPZ overhead allowance. Case and complete exported capture
bounds are 128 and 256 MiB. A complete capture forecast includes the case,
uncompressed arrays and 32 MiB for host/build metadata. The existing bounded
archive and ELF scans remain in force. Named arrays plus scalars and the explicit
code reserve govern preparation; generated static endpoints must be below
44 KiB. Dynamic stack high-water remains unmeasured.

Root runs these commands through `tools/run_with_budget.py`, with one CPU
thread and one global SDK slot. E225 records11host tests and the minimum new-runner compile/SIM passing;
all81arrays are retained and both output/gradient epochs have zero reference
error on that fixture. This is local evidence. Use fresh paths under `research/runs/`:

```text
python3 -B -m unittest discover -s models/hierarchical_gnn/grouped_csr/bounded_panels/adjoint_capacity/appliance -p 'test_*.py' -v
python3 -B models/hierarchical_gnn/grouped_csr/bounded_panels/adjoint_capacity/driver.py prepare --case research/runs/E225/min-cursor --mode cursor
python3 -B models/hierarchical_gnn/grouped_csr/bounded_panels/adjoint_capacity/appliance/export_appliance.py --case min-cursor=research/runs/E225/min-cursor --output research/runs/E225/package --request-id bounded-adjoint-minimum-r001
python3 -B models/hierarchical_gnn/grouped_csr/bounded_panels/adjoint_capacity/appliance/run.py compile --case research/runs/E225/package/cases/min-cursor
python3 -B models/hierarchical_gnn/grouped_csr/bounded_panels/adjoint_capacity/appliance/run.py simulate --case research/runs/E225/package/cases/min-cursor --out research/runs/E225/min-cursor-sim
```

The minimum remains the signed asymmetric 2×2/B2/T4/F3 case. E214's approximately
13.4 s compile and 13.5 s two-launch simulation support initial 25 s estimates,
90 s timeouts and 256/128 MiB growth limits; root records the final budget before
execution. Host mocks check ordering, D-only updates, native carriers, cleanup,
source/artifact binding and numerical rejection. The minimum SIM must exercise
this new runner and standalone audit before hardware publication.

The exporter emits a schema-1 descriptor and a simple compile/run/audit request
template, one case at a time and one compiler worker. The default worker timeout
is 600 s and remains explicit; it is not derived from simulator wall time.
Root registers `validations/bounded-adjoint/` in the sibling validation repository
and publishes reviewed requests. No direct SSH, retry loop or campaign controller
is included. The initial template is a numerical gate; balanced scan/cursor
performance requests and full-Cora cases are deferred until that gate passes.

Official precedents are the two immutable SDK versions of
`benchmarks/gemv-collectives_2d/pe.csl` (axis callbacks) and
`benchmarks/spmv-hypersparse/run.py` (uint32 host carriers for u16 copies), plus
the active SDK's `benchmarks/gemm-collectives_2d/pe.csl` (separate axis resources)
and `benchmarks/spmv-hypersparse/src/kernel.csl` (timestamp words). These patterns
support the interface choices; they do not establish this family's execution.
