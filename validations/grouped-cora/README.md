# Grouped endpoint Cora inference

This family measures complete inference with the native Cora 1,433→16→7 checkpoint and the original 2,708-node, 13,264-occurrence graph. It uses the native-cora compile, Worker, collection and numerical-audit flow. Each case freezes all packed tensors and unchanged corpus/checkpoint hashes.

Rows reduce and broadcast projected features. Each PE sends batches containing only the distinct remote source vectors required by each consumer in its column. Each message carries a destination slot and vector count. Startup configures the SDK message-passing bank. Separate microthreads drive the bounded outgoing packet buffer and incoming destination cache. The stored edge order is stable within each source group and receiver.

The 512-PE case uses C16/R32/T8 and the 1,024-PE case uses C8/R128/T16. Both change tile width and mesh shape. The numerical audit reconstructs sparse inputs, packet destinations and lengths, delivery counters, both GCN layers and the projection panel from the original graph and checkpoint. It checks every PE's clock interval and all validation predictions. Simulator image admission is checked before export. The appliance performs its own static ELF check after compile.

The request runs a separate compile and run for each case, followed by collection and an independent audit. Compare the maximum per-PE cycle interval and the host interval from launch to the blocking statistics read within their respective timing domains.

Run the host-only checks from the family directory with `python -B -m unittest discover -s tests`.
