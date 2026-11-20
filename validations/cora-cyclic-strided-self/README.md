# Strided self-edge SIMD on cyclic Cora sources

Each geometry uses one native compilation and one hardware run per language across all eleven destination windows. The native and Direct Link programs use separate feature-lane streams to handle source stride two and destination stride 2R. Every original Cora node and all seven trained outputs are checked against the original ordered COO FMA and north-to-south FP32 reduction. The resulting trained Cora logits recover 402 of the 500 validation labels. Each capture stores the original owner words and individual PE setup and computation clocks. Run the matching r4 or r16 appliance with the check action to replay its local simulator captures. Run return_all.py with the returned watcher step directories and the earlier cyclic hardware report to compare verified physical measurements. [Earlier physical request revisions](comparators.json)

| Active PEs | Language | Setup | First compute | Setup + one | Setup + two | Setup + three | Sum of phase maxima at three |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | csl | 12,074 | 99,979 | 111,977 | 211,880 | 311,783 | 312,011 |
| 16 | tungsten | 44,164 | 69,606 | 113,760 | 183,356 | 252,952 | 252,982 |
| 64 | csl | 5,964 | 26,846 | 32,640 | 59,352 | 86,084 | 86,502 |
| 64 | tungsten | 14,275 | 23,566 | 37,817 | 61,361 | 84,914 | 84,973 |

All entries are WSE-3 simulator cycles. Each inclusive value sums the longest individual PE's setup plus the indicated computations within each window. Host-side packet construction, transfers and inter-window dispatch use separate clocks.

[Same-source 16-to-64 PE scaling](simulator-strong-scaling.json)
