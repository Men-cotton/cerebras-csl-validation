# Matched CSR Cora AH comparison

This `shah-csr-cora` family computes AH with the grouped-CSR path and the same
frozen Cora graph and FP32 projected inputs used by the resident Shah controls.
Each prepared case fixes an AH width of 16 or 7. The Cora corpus and
seed11/200epoch checkpoint are preserved so the saved-output audit can
reconstruct the inputs and independently evaluate AH from the original COO.
This family's execution scope is standalone AH; whole-GCN residency is
evaluated by the separate `shah-resident-cora` family.

Each `appliance.py compile --case LABEL --out PATH` submits one compile.
`run --artifact-record BUILD --mode ah16 --out PATH` submits one run for an
AH16 case; use `--mode ah7` for an AH7 case. The mode must match the compiled
case. `audit --artifact-record BUILD --capture RUN --out FILE` is host-only.
Request manifests use separate steps and the registered publication workflow.
Exporting or checking this family submits no request. The worker uses the
existing explicit lifetime limit of 86400 seconds.

Each run launches AH twice. The second launch uses H multiplied by -0.5 and
retains the graph on device. These changed inputs check repeated invocation
and input lifetime; they do not provide same-input latency repetitions. The
audit checks complete output tensors against an FP64 COO reference, input
readbacks, owner and feature padding, per-PE counters, and each launch's clocks.

The exporter maps `csr_driver.py` to `driver.py` and
`csr_compare/{layout,pe}.csl` to `{layout,pe}.csl`, alongside the shared
`oracle.py` and `proof.py`. Compilation flags come from that frozen driver's
CSR parameter contract. Prepared cases contain the exact source snapshot,
raw corpus/checkpoint provenance, original arrays, ownership, projected inputs,
and packed arrays. The worker and saved-output audit bind the case, compile,
generated ELF and I/O metadata, and fresh run UUID. Scheduler terminal status
remains a separate controller responsibility.

Saved and replayed audit reports must agree exactly in structure, identities,
tensor hashes, counters and clocks. Finite maximum-absolute-error diagnostics
allow a difference of at most `1e-12 + 1e-8 * max(abs(a), abs(b))` to accommodate
FP64 reference roundoff across NumPy environments. Both audits must independently
pass the tensor checks.

Static SRAM admission requires generated ELF sections to use strictly less
than 44 KiB per PE; runtime stack high-water is unmeasured. PE-local cycles and
host blocking-launch windows do not establish synchronized makespan or
application E2E. Compile, simulator, publication, and hardware outcomes belong
to the corresponding experiment evidence.

Partial runtime data and host failures are collected when available. If SDK
launcher construction fails, no worker collection may exist; the request and
available client log remain failure evidence.
