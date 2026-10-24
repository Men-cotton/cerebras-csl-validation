# Complete native H128 fluid inference campaign

The request `tank28-native-h128-s15-20261024-r001` advances the E200 resident
inference composition. Every runtime uploads its complete model once, executes
one zero-argument physical prediction step, and captures all declared outputs
and initial/final inputs. Encoders, untied processors, decoder, saved-statistic
denormalization and boundary masking execute in CSL.

| Case | Graph | Processor blocks | Feature lanes × owners | Transport |
| --- | --- | ---: | ---: | --- |
| minimum-s2 | 4 vertices,4 directed edge occurrences | 2 | 32 × 4 | streamed source panel |
| trained-c32-streamed | 1896 vertices,10908 directed edges | 15 | 32 × 128 | streamed source panel |
| trained-c64-streamed | same complete graph and partition | 15 | 64 × 128 | streamed source panel |
| trained-c64-colored | same complete graph and partition | 15 | 64 × 128 | selective delivery in11 endpoint-disjoint rounds |

The minimum case has duplicate remote edges, a local edge, two empty owners,
an isolated vertex and nonzero fixed boundaries. Its deterministic random
weights and nonidentity statistics exercise the complete numerical contract.
The three scale cases use the first official CylinderFlow validation record,
raw frame1, the same spatial16×8 owner partition and the same trained H128/S15
parameters. C32 versus C64 isolates feature-lane count at fixed ownership;
the two C64 cases isolate transport. Each owner holds at most15 vertices and92
incoming edge occurrences.

The checkpoint is the author-published `model.ckpt-200000` at
[revision85b450d16adb23daa92d6c6fe20550e0b41e11c0](https://github.com/ali-izhar/uq-for-meshsimulations-using-splitcp/tree/85b450d16adb23daa92d6c6fe20550e0b41e11c0/checkpoints/cylinder_ts).
Every trained case retains its `PUBLIC_MODEL_LICENSE` and
`public-model-import.json`, including original file hashes and saved normalizer
provenance. No fitting or training runs in this campaign. The CPU trajectory
accuracy study and the hardware's single-step arithmetic agreement are distinct
evaluations. A hardware PASS here establishes neither long device rollout nor
model convergence or cross-trajectory physical accuracy.

Each case has three foreground steps: compile, runtime capture and host numerical
audit. A nonzero audit stops the next scale case. The compile requests four CPU
workers, matching `--max-parallelism=4`; runtime remains one CPU worker. Client
steps have1200-second bounds, the runtime worker has300 seconds plus15 seconds
termination grace, and each host audit has120 seconds. Queue/service timing is
unmeasured. An ambiguous client timeout stops progress and does not authorize a
retry.

Approximate latent outputs use rtol=atol=3e-4. Physical velocity uses rtol=3e-5,
atol=1e-6; every fixed boundary component is bit-exact. Counters, padding and
immutable input/model banks are exact. Complete numerical failures remain in
the captured arrays and do not become execution-admission failures. Per-PE
clock intervals are separate from client/SDK wall times, which include transfer
and collection. Static ELF endpoints remain limited to44KiB; dynamic stack and
runtime high-water are unmeasured.

The C64 streamed source case projects263.75MiB including a complete raw capture
and metadata reserve. Its exported derivative uses lossless NPZ compression,
preserving all array bits, CSL bytes and the exact parent manifest. It projects
188.85MiB under the unchanged256MiB capture limit. The other three projected
sizes are38.62,206.10 and252.81MiB in table order. The128MiB frozen-case bound also
remains unchanged. Compression is an input-artifact encoding choice; it does
not remove any runtime readback or numerical check.

The original local compiler and host-test evidence is recorded in the owning
research repository's E199/E200 notes. All four portable cases passed export
admission, and this family ran40 mocked host tests in the validation
repository:39 passed and the local-only driver binding test was intentionally
skipped because that research driver is not exported. These observations precede the requested hardware execution.
