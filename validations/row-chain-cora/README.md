# Cora with a static row reduction

Each PE accumulates its feature shard. A fixed right-to-left FADD chain sums the projected row vectors. The leftmost PE multicasts the sum, then the grouped endpoint transport delivers each remote source vector to the destination PEs that need it.

The 16-PE calibration uses the induced first 256 original Cora nodes on a 4-by-4 rectangle with eight feature lanes. The node list, all 1,433 input features, trained H16/O7 checkpoint, graph occurrences and packet plan match the corresponding Tungsten case. Full cases use all 2,708 nodes and 13,264 occurrences. Every case stores its node count and source hashes.

The standard native-cora appliance runner retrieves the inputs before and after execution, all PE counters, clocks, first layer, padded projection panel and logits. The independent host audit rebuilds the graph and checkpoint calculation. The 44 KiB compiled static admission applies to all images. Run the offline plan checks with `python -B -m unittest discover -s tests`.
