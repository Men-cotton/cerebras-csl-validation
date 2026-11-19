# Cora resident aggregation on 16 and 64 PEs

Each request compiles one CSL application and runs CSL and Direct Link Tungsten once each. Each run processes all 2,708 destinations, 13,264 normalized edges and seven trained logits in eleven windows. Both input banks stay on device while the host selects A, −A, A in each window.

Run `python3 -B appliance.py check` to verify both ELF geometries, input provenance, every simulator reply, numeric outputs, SRAM checksums and per-PE clocks. [Simulator results](simulator-comparison.json) · [Input provenance](provenance.json) · [Earlier windowed series](legacy-comparison.json)

After both physical requests return, run `python3 -B report.py hardware --help` for the four captures and two CSL build records. The report verifies actual platform, compiled ELF inventory, downloaded bytes and all 56,868 output words per application.
