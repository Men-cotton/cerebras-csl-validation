# Fixed H16 native Cora GCN on CS-3

The native kernel traverses each input feature row once to compute all 16 hidden channels. It reduces and broadcasts those channels in two groups of eight and writes each tile to all locally owned nodes with one multidimensional data move. The seven output channels use the original frozen checkpoint. Vertex 163 retains its owner and receives partial sums for an ordered suffix of its edges from a helper in the same row.

The case contains the original Cora data, checkpoint, node IDs, grouped endpoint packets and independent FP64 expectations. The appliance checks its entire compiled ELF inventory, transfers every input by standard SDK H2D, reads every PE before and after the launch, and records per-PE start and end clocks. The saved audit reconstructs the hidden state, final projection panel, logits, predictions, counters, packet traffic and partial-sum word counts from the original graph.
