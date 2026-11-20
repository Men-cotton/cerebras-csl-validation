# Two WEST inputs for complete two-layer trained GraphSAGE

One physical program receives a tagged first-layer input on each of two independently declared WEST ports. Eight first-layer owners per row retain the 16-owner reference assignment, original destination IDs, every input feature, weights and the per-destination reduction order. Raw intermediate activations flow across two fabric rows, meet at one final-layer PE and produce all logits. The four-output closures reuse the exact graph/weight fixtures verified by the one-input family. An additional one-owner-per-row calibration uses two original arxiv outputs.

The first of nine resident frames sends A concurrently and initializes both copies of the first-layer weights. Four warm pairs cover sequential and concurrent ingress for A with host echo, half-A with echo, half-A on chip, and A on chip. Both output ports participate in every frame. One acknowledges the lower row. The other performs the gate handshake and returns the final result and any requested complete intermediate. The collector records the clock from its data tag to completion and the enclosing clock from the earlier gate. Each frame retains the raw gate and lower-row acknowledgments, complete PE timing limbs, all logits and every original destination ID. Echo frames receive and resend all hidden FP32 words.

The standalone checker replays every returned word against the independent first-layer and full-model FP64 reference and the frozen one-input SDK measurements. Both target geometries have separate ELF files. All linked loads across the 8 × 4 active rectangle must agree before a release can run. The physical worker records the actual program and port-map digests, imported host files, initial packet, each of the two complete ingress buffers, every echo re-entry and the physical SDK target. The one-input times embedded in this request come from its frozen SDK reference. Comparing two physical port counts additionally requires the separately registered physical one-input return.

The pilot includes nine frames from both the original SDK worker and the physical worker running locally on the SDK simulator. The original research worker records the input and firmware identities. The source copy in `reference-source/` records the worker at export. The instrumented pilot and physical worker also record the exact imported files. [Return and numerical replay](report_parallel.py) · [Physical execution](appliance.py)

After both physical requests return and each passes its own worker, ELF and numerical audit, the two physical timing series can be joined with the same API. Run from this directory with the two returned capture directories. The joined record records hardware for both inputs and compares all recovered layer outputs and hidden-state words before listing the clocks.

```sh
one=../cases/arxiv-p16-e24
two=cases/arxiv-p16-e24
python3 -B report_parallel.py "$one/first.npz" "$one/last.npz" "$two" /path/to/two-port/capture "$one" /path/to/one-port/capture --output /tmp/physical-pair.json
```
