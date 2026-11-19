# Fixed-ELF Cora final aggregation with cyclic source placement

Each geometry compiles one CSL artifact and runs CSL and Direct Link Tungsten once each. Each run processes 2,708 original destinations, 13,264 weighted edges and all seven trained logits over 11 windows. Source i is placed on shard i % R in slot i // R. The resident A, −A, A panels and original destination numbering are preserved. The contiguous case uses the exact revisions in [comparators.json](comparators.json).

Run `python3 -B appliance.py check` to replay four complete simulator captures and reconstruct every owner word from original COO edges, C99 fused multiplication/addition and the FP32 PE reduction order. Each physical worker returns the platform, input and graph checksums, raw owner words, and independent 48-bit setup and compute clocks. [Simulator placement and scaling](comparison.json) · [Source order changes](order.json) · [Host packing times](host-packing.json)

Run `python3 -B report.py hardware --help` to audit the two builds and four returned physical captures for this placement. After both placement reports exist, run `python3 -B physical_compare.py --help` with both reports and their build records. This comparison requires equal actual CSL ELF inventories or, if the ELF containers differ, identical loaded bytes for every CSL application and program segment.
