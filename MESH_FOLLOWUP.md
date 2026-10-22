# Two-block CylinderFlow continuation campaign

Run the next campaign after the currently submitted campaign has finished:

```bash
uv run python mesh_followup.py
```

The parent creates a private tmux session and runs two child campaigns serially.
Each child has its own frozen sources, package, plan, controller lock, captures,
validation reports and ZIP. Existing `mesh_campaign.py`, its worker and
`mesh_cases/` remain unchanged.

| Cohort | Full mesh | PE grid | Model | Resident launches | Selective packet capacity |
| --- | --- | --- | --- | --- | --- |
| h4-s2 | N1896 / E10908 | 8×8 | H4 / S2 | chunks2,2 | K6 |
| h8-s2 | N1896 / E10908 | 16×8 | H8 / S2 | chunks2,2 | K3 |

Each cohort contains `full-broadcast`, `full-selective` and `full-colored`.
Defaults compile all three once and run each twice, reversing transport order
in the second repetition: six builds and twelve runtime jobs in total.
The second two-step launch uses the evolving velocity and initialized geometry
already on device. Readback at the checkpoint lies between launches; processor
blocks and the two steps inside a launch have no host round trip. This is a
continuation measurement, not a repeat from identical state. No model weights
are trained by this campaign.

Comparisons require equality of six arrays at both checkpoints and occur only
inside a cohort. H4 and H8 differ in arithmetic and PE grid, so the parent
reports their summaries separately. Each child summary retains
`complete_transport_set`; a comparison among two passing transports does not
establish a complete three-way comparison when another build or run failed.

Prepare the complete frozen plan without submitting jobs:

```bash
uv run python mesh_followup.py --plan-only --out ./mesh-followup-plan
uv run python mesh_followup.py --resume ./mesh-followup-plan
```

Add `--namespace YOUR_NAMESPACE` when preparing a fresh plan if the site needs
it. `--foreground` uses your existing supervisor. `--backend appliance-sim`
requests the appliance simulator explicitly; the default is hardware.

Child failures are retained and the next cohort proceeds. Resume executes the
saved parent and children. It revalidates completed entries, submits pending
entries, and never automatically retries an attempted build or runtime without
terminal evidence. Such an entry remains unknown or incomplete. A client
timeout or a released local controller lock does not prove that a remote job
has stopped. Inspect the scheduler before creating a fresh retry campaign.
An active child controller prevents another parent from submitting work.

The parent ZIP contains its sources, plan and summary plus both child ZIPs.
If a child ZIP is unavailable or predates the current evidence, the current
expanded local evidence is included too. The parent does not merge hardware timing ratios across
cohorts. PE-local intervals, worker wall time and compiler/client turnaround
remain separate quantities. Neither queue delay nor remote-stop behavior has
been validated by the local mock tests.

The follow-up controller does not assert successful local or hardware
admission for any case. Hardware compile and runtime outcomes are determined
by the submitted jobs; a capacity failure remains a recorded failure at its
requested H, grid and transport. Independent local admission and numerical
checks are recorded in E190 in the gnn-csl repository.

The matched full-mesh H8/S2 local builds both compile and pass ABI checks.
Broadcast's maximum static ELF endpoint is44,784 B; colored delivery's is
47,408 B. The latter exceeds the local44 KiB admission guard by2,352 B.
This guard excludes dynamic stack/high-water storage. H8 colored is therefore
a capacity diagnostic in this hardware campaign, with runtime correctness and
memory headroom still unverified. The H4/S2 full cases are prepared; their new
builds and runtimes will be checked by the hardware campaign.

Source-only development check, after the existing four modules and both new
files have been placed together:

```bash
uv run python -B -m unittest -q test_mesh_followup
```

The tests use mocked clients and never start tmux or submit an SDK job.
