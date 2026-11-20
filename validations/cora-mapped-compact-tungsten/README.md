# Source-balanced all-FMAC Tungsten on 64 PEs

This single Direct Link job uses the physical ELF from the prior cyclic unrolled run and the source map from the prior balanced run. It processes all 2,708 original Cora nodes, 13,264 edges and seven trained classes in eleven windows with resident A, −A, A. The simulator measured maximum individual-PE setup plus one, two and three calculations summed over the windows at 26,580, 46,872 and 67,166 cycles. The prior best cyclic CSL measured 27,459, 53,659 and 79,859. Host packing, transfers and dispatch are separate. [Simulator replay](simulator.json) · [Earlier hardware request and matched identifiers](comparators.json)

Run `python3 -B returned.py OUTPUT/run-tungsten hardware.json --prior PRIOR_CROSSOVER_HARDWARE.json` to verify the full owner data, independently recomputed ordered original COO, physical target and ELF identity and recovered PE intervals, and to compare against all five earlier physical jobs.
