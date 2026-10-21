# One-command hardware transfer campaign

From this branch on the Argonne SDK client node, use the same Python environment
and site configuration as the previous hardware runs. `tmux` must be installed.
Start the complete campaign with:

```bash
uv run python clock_campaign.py
```

The command starts a detached session on a fresh private tmux server and prints
`OUTPUT`, `ATTACH`, `LOG`, and `STATUS` commands. The server inherits the current
SDK environment and uses the same Python interpreter. You can disconnect after
launch. The session closes when the campaign ends; its logs and results remain.
No remote execution occurs when these modules are imported.

The default has four compiles and 80 fresh runtimes submitted by sequential clients. E183's hardware
observations suggest about 97–137 minutes, plus queue and system variation.
Allow about 1 GB for original results and extra space for the ZIP. This is a
planning estimate, not a two-hour deadline. A worker has a 300-second timeout;
each local compile/run client has a separate 1,200-second wall limit including
queue wait. Override with `--host-timeout` or `--client-timeout` if needed.

If the prior runs required a namespace or a different fabric configuration,
append the same settings, for example `--namespace NAME`. Hardware defaults to
ALCF WSE-3 dimensions `762,1172` and offsets `4,1`. `--backend simulator` selects
appliance SIM. `--profile smoke` runs four plumbing checks instead of the full
comparison; it is not the default experiment. `--plan-only` saves the plan and
frozen source without launching tmux or SDK jobs.

## Failure, interruption, and resume

Each attempt is written to `state.json` before submission. A failed run is
retained and the next independent run starts. A failed compile skips only runs
that depend on it. Numerical failure, incomplete collection, uncertain execution,
and unattempted entries remain separate. The controller exits 2 when the campaign
has failures, while still writing a partial summary and ZIP.

SIGINT or SIGTERM stops the campaign and leaves unattempted entries pending.
The local client timeout does not confirm that the remote job stopped. Job IDs,
local logs and `remote_stop_unconfirmed` are retained for the site's job tools.
The next client action can therefore overlap a still-running remote job.
There is no automatic retry of a failed or uncertain attempt. To continue only
the entries that have never been attempted, launch:

```bash
uv run python clock_campaign.py --resume /absolute/path/to/clock-campaign-...
```

Resume revalidates completed originals and frozen source. It refuses a second
controller for the same output. An interrupted `started` entry with complete
valid artifacts can be recovered; otherwise it becomes `unknown` and is skipped.
Launching the default command again creates a new independent campaign. Do not
use that as an implicit retry of an uncertain remote job.

The implementation follows Modelzoo's `benchmark_launcher.py` private tmux
server, interpreter/environment inheritance, atomic state and locking, and its
`worker_campaign.py` / `non_gnn/campaign.py` per-attempt failure handling. It has
no Modelzoo dependency. Only the outer controller starts tmux; every SDK client
runs in the foreground of that controller. `--foreground` is available for an
existing supervisor or local validation.

## Fixed comparisons

Every run uses a fresh runtime and retains all three A/B/A generations and all
PEs. Generations within a runtime are nested observations. No warm-time cutoff,
favorable generation selection, fitted frequency, or timing-dependent scheduling
is used. The primary cohort balances all six orders of rectangle, strips and
tiles. Most other comparisons have four pairs ordered AB, BA, BA, AB.

| Comparison | Fresh runtimes | Question |
|---|---:|---|
| 2×2, W64 rectangle/strips/tiles | 18 | Does splitting the same 1,024 bytes into 1/2/4 calls change the result across balanced orders? |
| 2×2, W256 rectangle/strips | 8 | Does the difference survive four times the payload? |
| 2×4 versus 4×2, W64 tiles | 8 | Does a transposed shape change equal-byte, eight-call behavior? |
| Forward/reverse tile order | 8 | Does the PE spread follow submission position? |
| Queued/waited strips | 8 | Does waiting on each returned task change the result? |
| Tile markers off/on | 8 | How much does the additional observation perturb the interval? |
| Empty tile markers off/on | 4 | What happens with the same marker count and no timed H2D? |
| Marked tiles, zero/10 ms between-call pause | 4 | Where does a known host delay appear in the PE intervals? |
| Host PMU observer off/on | 8 | What changes when counters, thread diagnostics and repeated empty controls are enabled? |
| Internal work1/work8 | 4 | Does contained device work respond to more repetitions? |
| Reverse marked tiles | 2 | Retain two additional diagnostic observations of submission order. |

W is u32 words per PE. Each runtime also preloads and reads full arrays outside
the measured interval. Counts above concern runtime executions, not physical
systems or independent PEs. Smaller cohorts are diagnostic; missing/failed pairs
stay unavailable rather than being replaced with another run.

## Fine-grained observations

With markers enabled, a queued whole-rectangle function after each transfer
records one raw 48-bit clock per PE, generation, marker ordinal, and that PE's
first and last data words. The validator checks the expected prefix state of
those endpoint words after each piece, then validates **every** word in the final
data and copied snapshot. Unused marker slots retain checked sentinels. All
marker readback happens after the end observation.

`markers.csv` contains start→marker and previous-marker→marker intervals.
Subtract the last marker's start-relative interval from the matching PE's
`pre_work_cycles` in `pe.csv` to obtain last-marker→work on that same PE. These are command-progress observations, not DMA or
wire arrival timestamps. Marker RPCs can change batching. The marker-off control
uses the same new ELF, including marker-buffer clearing before the start clock.
It is not identical machine code to E183's older ELF.

The pause follows submission of a nonblocking marker. Its completion is not
witnessed before sleeping, so the delay can appear before the first marker or
between later markers depending on execution/aggregation. This distribution is
an observation to inspect, not a predeclared location for the delay. PE clocks
are never subtracted across PEs; the largest local duration is not a global
makespan.

The PMU cohort retains raw `cycles:u` and `instructions:u` for the submitting
thread, enabled/running time, and three empty regions before and after each
generation. Controls are outside the PE span and are not subtracted. Thread CPU
time, endpoint CPU IDs, context switches and faults accompany the observer-on
runs. Thread CPU time includes kernel execution; the PMU configuration excludes
it. Matching endpoint CPU IDs cannot exclude migration between samples. The
comparison measures the entire observer package, not just perf control calls.
Unavailable counters remain unavailable. A bounded host CPU/affinity and loaded
runtime-module identity record helps compare workers without inferring unknown
SDK versions.

## Results and offline replay

The output directory contains the frozen plan/source, build and job records,
per-attempt logs, `state.json`, and:

- `summary.md` / `summary.json`: status, counts, per-run outcomes and limitations.
- `pe.csv`: all PE intervals and generations.
- `markers.csv`: all marker intervals and endpoint observations.
- `host.csv`: raw API timing, PMU and optional thread diagnostics.
- `contrasts.json`: scheduled within-block pairs, separately for each generation.

`<output-directory>.zip` is created beside the output directory, including partial
results on failure. Original capture archives remain on the operator server.
The compact ZIP retains validated raw arrays, metadata and API calls; it records
exact archive omissions and hashes of the retained files in `bundle.json`.
An original archive's hash cannot be recomputed from its selected members.
Unvalidated captures are kept whole. Transfer this ZIP for analysis.

To revalidate an extracted ZIP without SDK access, use its frozen driver:

```bash
python /path/to/clock-campaign-.../driver/clock_campaign.py \
  --summarize /path/to/clock-campaign-...
```

This checks retained bytes, source/build/request bindings, raw numerical results
and derived clocks, then rebuilds the report and ZIP. Exit 2 also represents a
validly retained partial or failed campaign; inspect its summary. Offline replay
does not submit or retry jobs. The local tests and SIM checks described in the
branch history establish local behavior; new hardware timing conclusions require
the operator's captures from this campaign.

Local SDK 2.10.2 validation for this revision passed 163 host tests, the minimum
1×1 marker case, and an eight-run functional subset across all four builds.
The corrected subset produced 24 generations, 120 PE intervals and 144 marker
observations, then passed ZIP extraction and offline replay. Four earlier smoke
workers also passed; their original campaign remains incomplete because the old
reporter rejected SIM-generated auxiliary ELF files. That admission defect has
focused regression coverage. No new hardware campaign or real tmux session was
executed by the author in this validation.
