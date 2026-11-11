# Resident Shah-path Cora GCN

This family packages the E232 model implementation, prepared input cases and
an independent saved-output audit. Its router-local paired graph streams,
feature workers and southward partial reduction derive from E231. Resident
sparse-X projection, accumulator-row feature exchange, northward projected
feature return and repeated invocation control extend that standalone SpMM.
The preserved Cora corpus and seed11/200epoch weights define the model.

Each `appliance.py compile --case LABEL --out PATH` submits one compile.
`run --artifact-record BUILD --mode all --out PATH` submits one run of that
exact artifact. `audit --artifact-record BUILD --capture RUN --out FILE` is
host-only. Request manifests must use separate steps and the registered
publication workflow. No request is submitted by exporting or checking this
family. Worker lifetime uses the existing explicit maximum86400seconds.

The `all` schedule executes complete GCN twice (second input X multiplied by
-0.75), then fixed-input AH16 and AH7 twice each (second H multiplied by-0.5).
Changed inputs test reset and lifetime correctness; they are not same-input
latency repetitions. All graph, feature and weight data remain on device
between the two GCN layers. Host updates occur only between full invocations.
The standalone AH tests use identical predeclared FP32 projection matrices;
whole-model acceptance separately recomputes the original COO FP64 reference.

Frozen bundles contain source, raw corpus/checkpoint provenance, original
arrays and packed inputs. The worker checks the exact case and generated ELF/
I/O metadata inventory before calling the shared `driver.capture`. Each run
has a fresh UUID. Partial runtime data and host failures are collected when
available. The separate audit binds the compile/case/run identities and
replays complete tensors, immutable input readbacks, role counters, padding,
intermediate device transfer and each launch's PE clocks without loading SDK.
Scheduler terminal status remains a separate controller responsibility.

Saved and replayed audit reports must agree exactly in structure, identities,
tensor hashes, counters and clocks. Finite maximum-absolute-error diagnostics
allow a difference of at most `1e-12 + 1e-8 * max(abs(a), abs(b))` to accommodate
FP64 reference roundoff across NumPy environments. Both audits must independently
pass the tensor checks.

Code/SDK static SRAM admission is strictly below44KiB from generated ELF
sections; runtime stack high-water is unmeasured. PE-local cycles and host
blocking-launch windows are not synchronized makespan or application E2E.
Actual compile, simulator, publication and hardware outcomes belong to E232;
source presence does not establish any of those outcomes.

The exporter also accepts `--variant csr`. It installs the matched CSR driver
and CSL under the same execution-file names and registers `shah-csr-cora`.
Compilation flags are obtained from that frozen driver's parameter contract;
the adapter does not reinterpret the CSR geometry as a Shah rectangle. The
CSR variant supplies the same fixed AH inputs and numerical audit interface;
it is a standalone comparison, not a second resident whole-GCN implementation.

If SDK launcher construction fails, no worker collection may exist. The
request and available client log remain failure evidence; lack of a collection
is not interpreted as a successful or numerically audited run.
