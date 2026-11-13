# Host-device ping-pong

This E235 / H116 kernel measures completed worker-side SDK echo RTT. A single
SdkLauncher runs a persistent Python SdkRuntime on the worker. Each exchange
contains one nonblocking streaming H2D and one blocking streaming D2H. There is
one outstanding exchange; configuration, allocation, validation, hashes and
logging are outside the timed interval. No launch is needed per exchange.

PE counts are fixed throughout each size sweep:1, 4, 16, 64, 256, 1024, 4096 in square
grids with side lengths 1, 2, 4, 8, 16, 32, 64. Every
case in the fixed-PE comparison uses one configured I/O channel and every PE is active, including for zero
payload. Each PE holds the complete frame before asynchronously sending it,
then rearms only after TX completion. Configuration waits for pending TX.

| Fixed PEs | Largest aggregate payload | Payload capacity per PE |
| --- | --- | --- |
|1|16 KiB|16 KiB|
|4|64 KiB|16 KiB|
|16|256 KiB|16 KiB|
|64|1 MiB|16 KiB|
|256|4 MiB|16 KiB|
|1024|16 MiB|16 KiB|
|4096|64 MiB|16 KiB|

The zero and powers hardware requests each contain seven compiles and three fresh runtime/audit
pairs per compiled shape. `zero-pP` runs 20 warmup and 200 measured exchanges.
`powers-pP` includes 0 then powers of two from 1 byte to the shape's maximum,
with 10 warmup and 30 measured exchanges per size. Runtime orders are ascending,
descending and a fixed permutation. The zero cases use the same framed protocol
as the corresponding size sweep. Fresh runtimes are independent observations;
exchanges within a runtime are nested measurements. A third request fixes 4096 PEs
and compares 1, 2, 4, 8, 16 configured I/O channels, each with a fresh compile and
three runtimes of the complete size sweep, including zero. All returned I/O
metadata are retained to verify actual channel placement and coverage.

Zero means 0 **application payload bytes**, not zero transport. A fresh u32
generation token tagged with the linear PE index requires 4 P bytes each direction.
The tag detects swapped or duplicated PE replies even in the zero-payload case. Small payloads are
padded to the uniform per-PE u32 extent; padding is separately reported. Host
carrier bytes exclude unknown SDK/network framing. The fixed-P series share
sizes through 16 KiB; larger sizes occur only in higher-capacity series.

The device performs a full-frame echo, not arithmetic. It receives the header
with a data task and the remaining payload with a vector fabric-to-memory move.
The complete per-PE buffer prevents the serialized H2D/D2H deadlock discussed
in official `topic-09-fifos`. Official `topic-03-streaming-wavelet-data` supplies
the scalar echo and asynchronous-H2D/blocking-D2H precedent. Old and active
SDK2.10.2 immutable source versions were read. The generation header is echoed
unchanged; payload patterns vary with generation, PE and element index.

Keep RTT as measured. RTT/2 is not a measured one-way latency, and subtracting
the zero baseline does not isolate wire time. `2*payload_bytes/RTT` describes
application echo throughput, not separate directional bandwidth. Device
buffering, WSE routing and host runtime overhead all contribute. Job scheduling,
loading and initial startup are outside steady RTT. Simulator wall time is
never hardware latency evidence.

Every reply is compared in full outside timing. Capture retains all warmup and
measurement RTTs, returned SHA256 per exchange, a full final reply per size,
and a failing reply if observed. The standalone audit regenerates every expected
frame/hash and checks raw witnesses, exact schedule, frozen source and compiler
metadata. The registered adapter joins package, compiler archive, actual ELF,
run UUID and parent request records. SDK scratch is separate from retained
capture; its 512 MiB bound is a post-run accounting bound, not a live quota.
The capture bound is 256 MiB: the largest runtime's full geometric witnesses
occupy about 128 MiB plus control and metadata. Compiler archives are bounded
at 256 MiB. Bounds are enforced; a larger artifact fails rather than silently
relaxing the request. Large-shape hardware execution remains unestablished.

Offline host checks use `python 3 -B -m unittest -v test_oracle test_driver` from
this directory under the gnn-csl budget runner. Local captures use
`driver.py capture --build PATH --case zero-p1 --out PATH --smoke`; `driver.py
audit --capture PATH` needs NumPy but no SDK. Production requests forbid smoke.

In the validation repository, run
`uv run --locked --group dev -m csl_validation check --family host-device-ping-pong`.
Each appliance manifest step performs one compile or one run. Publish with a
matching request ID and manifest trailer. Publication is not hardware execution.
