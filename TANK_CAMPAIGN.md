# Top-K capacity and halo weak-scaling campaign

From the updated checkout on the operator server:

```bash
uv run python tank_campaign.py
```

The command creates a private tmux session, freezes the source and cases, and
submits clients for eight builds and sixteen hardware runtimes serially. Each case runs twice,
with reversed order in the second repetition. One failure does not cancel later
entries. The output directory, log and attach command are printed immediately;
the final ZIP contains source, inputs, expected arrays, artifacts, raw captures,
audits and failure records.

| Case | Tank | Work |
| --- | --- | --- |
| topk-small | 17 | 3×2 owners, including an empty and a short shard |
| topk24, topk32, topk64 | 17 | 24×24, 32×32, 64×64 owners; C16/D8/K4 fixed, 9,216 / 16,384 / 65,536 database rows |
| halo-small | 06 | 2×2; lengths/phases 17:0, 32:1, 32:0 in one runtime |
| halo16, halo32, halo64 | 06 | 16×16, 32×32, 64×64; 32 local cells and 64 independent epochs per launch |

Top-K keeps the database resident across queries A/B/A. It checks every local
distance, regional candidate, query and status word, resident input integrity
and exact restoration. It has no device timestamp; worker or client time must
not be described as query latency. The data is a deterministic capacity fixture,
not a learned GNN or a real-dataset accuracy evaluation.

Halo checks every y/Q/count and the six raw timestamp words. Its clock includes
input generation, zeroing, communication, stencil arithmetic, checksums and
callback scheduling over 64 independent epochs. Maximum PE-local intervals
support a fixed-work weak-scaling comparison; they are not synchronized global
makespan. This is a local communication control, not a time-evolving solver.
The wrapper retains compile-time coordinates. Its two vendor imports are
localized, and exported vendor copies adopt the SDK2.10.2 official Params type
and queue-DSD syntax. Export and audit require byte equality with the current
official source; both historical and current originals and hashes are saved.
The wrapper forwards stencil parameters explicitly using the current official
import pattern; callback, queues, DSR arguments and executable body are unchanged.

Before submitting jobs, the controller detects unfinished `mesh-followup-*`
directories in this checkout and waits for their verified successful terminal
states. Add `--after /absolute/path/to/mesh-followup-output` if its output lives
elsewhere. Predecessor parent/child locks are held throughout the new campaign;
a shared operator lock serializes new tank campaigns. Interrupted or ambiguous
predecessors stop submission with a reason in `state.json`; lock release alone
does not prove a remote job stopped. An external command that ignores these
locks is outside this serialization contract.

The default compiler/client timeout is 1,200 s. The coordinate-specialized
halo64 build alone allows 7,200 s, based on a prior local compiler extrapolation
of about 3,439 s. Hardware compiler and queue turnaround are unmeasured. Runtime
workers retain their separate 300 s bound. A client timeout cannot prove that a
remote job has stopped. Later clients still run after a failed attempt, as
requested; unfinished remote work can therefore overlap them after a timeout.
Attempted entries are never automatically retried.

To prepare without submitting, select cases, or resume the frozen campaign:

```bash
uv run python tank_campaign.py --plan-only --out ./tank-plan
uv run python tank_campaign.py --resume ./tank-plan
uv run python tank_campaign.py --cases topk-small,halo-small --repeats 1
```

Resume revalidates completed entries and submits only pending entries using the
saved implementation. Do not start a new campaign as a retry until the previous
remote jobs are known to have ended. `--namespace NAME` selects a site namespace;
`--backend appliance-sim` explicitly selects the appliance simulator.

Expected arrays can be regenerated offline from the included original oracles:

```bash
uv run python export_tank_cases.py --verify ./tank_cases
uv run python -B -m unittest -q test_tank_adapters test_tank_appliance test_tank_queue test_tank_queue_extras
```

These commands perform host checks only. Test results, local SDK checks and
hardware results are separate evidence. Full capture/ABI checks do not establish
dynamic SRAM high-water or independently attest which ELF ran on the worker.
