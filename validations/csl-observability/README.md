# CSL observation pilot

This frozen family belongs to gnn-csl E234 / TANK29. Queue N32 uses two PEs and
65 data words, with fresh runtimes for PROFILE1/delay0, PROFILE1/delay256 and
PROFILE0/delay256. Each runtime validates three complete epochs. Arena CAP64
uses one PE and four observer-off/on/on/off calls, checking every allocation,
capacity query, free/reuse, payload and completion epoch.

The queue records local empty/full masks, transmit callback, queue drain and
validated receiver ACK. These are endpoint observations, not physical-link
congestion fractions. The memory records occupied words inside an owned arena;
they do not measure total SRAM or compiler/SDK stack peaks. Local SIM proof is
recorded in E234; publication alone does not establish hardware execution.

Use the repository's compile/run/audit request workflow. Each appliance step
performs one SDK action. Frozen source, compiler archive, actual generated ELF,
run UUID and returned capture are bound before independent numerical replay.
The 600-second worker command limit does not configure scheduler queue lifetime.
No scheduler, SSH connection or endpoint discovery is added.

Run offline checks with `uv run --locked --group dev -m csl_validation check
--family csl-observability` (one command). Request publication follows the root
development workflow; adding files without execution trailers does not submit.
