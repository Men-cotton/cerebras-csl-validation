# Tiny CSL++ target-compiler admission

This family compiles the frozen E236 seven-node, 2x2 PE fixture only. It exposes
`hardware_entry.py compile --arm direct|frontend|async --out DIR` and an offline
`audit --arm ARM --build-record DIR/build.json --out FILE`. Each appliance step
submits exactly one compiler operation. There is no device run or full-Cora path.

Every generated layout/PE source is registered and matched to the returned
compiler archive. All application/helper static SRAM endpoints must remain
below 44 KiB. For each application coordinate, target code, function locations,
loaded memory and coroutine storage must match the pinned frame-measured E236
reference. Async has a measured 64 B root frame plus the resolver's 12 B argument area;
backend-generated call closure remains unverified, so its admission is HOLD. Different
code fails this deliberately narrow equivalence gate. Target metadata absence is
accepted only under exact machine-code equivalence; runtime peaks are unmeasured.

The package contains the original/generated sources, local reference ELF files,
generator provenance, frozen tiny input and offline mutation tests. A published
request is not proof of target compiler completion. Compile records preserve the
returned archive even when the subsequent stack gate fails. Source, target
compiler success, admission and numerical device execution are distinct claims.
