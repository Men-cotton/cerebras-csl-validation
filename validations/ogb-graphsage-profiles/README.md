# OGB GraphSAGE on CS-3

Each case freezes an exact unweighted neighbourhood, two trained projections, bias, original node IDs and all three input frames. The final-layer inputs are the full-graph hidden activations of a frozen two-layer model. See each case's provenance and reference for the CPU/GPU full-graph accuracy and graph and checkpoint hashes. Each request compiles CSL and executes both native CSL and Tungsten Direct Link on the same destination set.

Use `python3 -B appliance.py check --case CASE` to replay the physical input contract and SDK simulator outputs and clocks.
