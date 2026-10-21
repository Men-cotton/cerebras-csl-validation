"""Fixed operator campaigns; importing or generating a plan submits no jobs.

One run entry means one fresh SDK runtime with three nested A/B/A generations.
The schedule never depends on measured timings, and retains the first generation.
"""
from itertools import permutations

import clock_common as common


BUILD_PARAMS = (
    ("c2-r2-w64", dict(C=2, R=2, W=64)),
    ("c2-r2-w256", dict(C=2, R=2, W=256)),
    ("c2-r4-w64", dict(C=2, R=4, W=64)),
    ("c4-r2-w64", dict(C=4, R=2, W=64)),
)
PAIR_ORDERS = ((0, 1), (1, 0), (1, 0), (0, 1))


def _pair(value, name, positive=False):
    common.require(isinstance(value, str), name + " must be X,Y")
    parts = value.split(",")
    common.require(len(parts) == 2 and all(p.isascii() and p.isdecimal() for p in parts),
                   name + " must be nonnegative integer X,Y")
    values = [int(p) for p in parts]
    common.require(not positive or min(values) > 0, name + " must be positive")
    return ",".join(str(v) for v in values)


def make_plan(profile="standard", backend="hardware", namespace=None,
              fabric_dims=None, fabric_offsets="4,1", host_timeout=300):
    """Return a JSON-serializable, deterministic plan with stable schedule IDs.

    Schedule IDs identify planned entries. The controller creates independent
    runtime UUIDs, pins this plan, and must not resubmit an uncertain remote job.
    """
    common.require(profile in ("standard", "smoke"), "campaign profile")
    common.require(backend in ("hardware", "simulator", "local-sim"), "campaign backend")
    common.require(namespace is None or isinstance(namespace, str) and
                   0 < len(namespace) <= 255 and all(ord(c) >= 32 for c in namespace),
                   "namespace must be a nonempty string or null")
    common.require(type(host_timeout) is int and 1 <= host_timeout <= 1800,
                   "host timeout must be 1..1800 seconds")
    offsets = _pair(fabric_offsets, "fabric offsets")
    dims = _pair(fabric_dims, "fabric dimensions", positive=True) if fabric_dims is not None else None
    selected = BUILD_PARAMS if profile == "standard" else BUILD_PARAMS[:1]
    builds = [dict(id=name, params=common.validate_params(params)) for name, params in selected]
    if dims is not None:
        width, height = map(int, dims.split(","))
        x, y = map(int, offsets.split(","))
        common.require(all(x + b["params"]["C"] <= width and
                           y + b["params"]["R"] <= height for b in builds),
                       "campaign application exceeds fabric")
    params_by_build = {b["id"]: b["params"] for b in builds}
    runs, cohorts = [], []
    base = builds[0]["id"]

    def condition(**changes):
        value = dict(piece_order="forward", markers=False, host_pmu=False, pmu_controls=0)
        value.update(changes)
        return common.validate_config(value)

    def add(cohort, block, position, arm, build, config):
        p = params_by_build[build]
        cfg = common.validate_config(config)
        transferring = cfg["kind"] == "transfer"
        calls = (1 if cfg["arm"] == "rectangle" else p["R"] if cfg["arm"] == "strips"
                 else p["C"] * p["R"]) if transferring else 0
        runs.append(dict(id=f"r{len(runs) + 1:03d}", cohort=cohort, block=block,
                         position=position, arm=arm, build=build, config=cfg,
                         timed_h2d_bytes_per_generation=p["C"] * p["R"] * p["W"] * 4 if transferring else 0,
                         h2d_calls_per_generation=calls))

    def pairs(name, hypothesis, left, right, count=4):
        cohorts.append(dict(id=name, hypothesis=hypothesis,
                            order="AB,BA,BA,AB" if count == 4 else "AB,BA",
                            blocks=count, fresh_runtimes=2 * count))
        for block, order in enumerate(PAIR_ORDERS[:count], 1):
            for position, index in enumerate(order, 1):
                arm, build, cfg = (left, right)[index]
                add(name, block, position, arm, build, cfg)

    if profile == "smoke":
        cohorts.append(dict(id="smoke", hypothesis="Verify operator compile, collection and instrumentation plumbing.",
                            order="fixed", blocks=1, fresh_runtimes=4))
        conditions = (
            ("internal", condition(kind="internal", start="blocking", work=8)),
            ("rectangle", condition(arm="rectangle")),
            ("reverse-marked-tiles", condition(arm="tiles", piece_order="reverse", markers=True)),
            ("rectangle-pmu", condition(arm="rectangle", host_pmu=True, pmu_controls=3)),
        )
        for position, (arm, cfg) in enumerate(conditions, 1):
            add("smoke", 1, position, arm, base, cfg)
    else:
        cohorts.append(dict(id="granularity", hypothesis="Compare identical bytes in one, two or four H2D calls with balanced runtime order.",
                            order="all six permutations of rectangle,strips,tiles", blocks=6, fresh_runtimes=18))
        for block, order in enumerate(permutations(("rectangle", "strips", "tiles")), 1):
            for position, arm in enumerate(order, 1):
                add("granularity", block, position, arm, base, condition(arm=arm))
        pairs("payload", "Repeat rectangle/strips comparison with four times as many words per PE.",
              ("rectangle", "c2-r2-w256", condition(arm="rectangle")),
              ("strips", "c2-r2-w256", condition(arm="strips")))
        pairs("geometry", "Compare transposed rectangles with equal PE count, bytes and eight tile calls.",
              ("c2-r4", "c2-r4-w64", condition(arm="tiles")),
              ("c4-r2", "c4-r2-w64", condition(arm="tiles")))
        pairs("piece-order", "Test whether row spread follows the order of tile submissions.",
              ("forward", base, condition(arm="tiles", piece_order="forward")),
              ("reverse", base, condition(arm="tiles", piece_order="reverse")))
        pairs("submission-wait", "Compare queued row transfers with waiting after each returned task.",
              ("queued", base, condition(arm="strips", transfer="queued")),
              ("wait", base, condition(arm="strips", transfer="wait")))
        pairs("markers", "Measure the perturbation from device marker invocations between tile transfers.",
              ("unmarked", base, condition(arm="tiles", markers=False)),
              ("marked", base, condition(arm="tiles", markers=True)))
        pairs("empty-markers", "Observe marker-only command overhead with no timed H2D payload.",
              ("unmarked", base, condition(kind="empty", arm="tiles", markers=False)),
              ("marked", base, condition(kind="empty", arm="tiles", markers=True)), count=2)
        pairs("marked-pause", "Locate a deliberate 10 ms host pause among same-PE marker intervals.",
              ("pause0", base, condition(arm="tiles", markers=True, pause_at="between_h2d", pause_seconds=0)),
              ("pause10ms", base, condition(arm="tiles", markers=True, pause_at="between_h2d", pause_seconds=0.01)), count=2)
        pairs("host-pmu", "Compare observer-off with PMU, thread diagnostics and repeated empty controls; retain raw counts without subtraction.",
              ("off", base, condition(arm="rectangle", host_pmu=False, pmu_controls=0)),
              ("on", base, condition(arm="rectangle", host_pmu=True, pmu_controls=3)))
        pairs("internal-work", "Vary device-contained copy repetitions independently of host-spanning intervals.",
              ("work1", base, condition(kind="internal", start="blocking", work=1)),
              ("work8", base, condition(kind="internal", start="blocking", work=8)), count=2)
        cohorts.append(dict(id="reverse-marked-diagnostic",
                            hypothesis="Retain two diagnostic repetitions of reverse tile order with markers; compare against earlier cohorts cautiously.",
                            order="same condition repeated", blocks=2, fresh_runtimes=2))
        for block in (1, 2):
            add("reverse-marked-diagnostic", block, 1, "reverse-marked", base,
                condition(arm="tiles", markers=True, piece_order="reverse"))

    nbuilds, nruns = len(builds), len(runs)
    estimates = dict(
        basis="E183 operator hardware: compile 40.849 s; fresh runtime 70.387..100.430 s; about 10.99 MB retained per run.",
        scope="Planning estimate for hardware; queue wait and other system conditions can increase elapsed time. Not a timeout or a simulator estimate.",
        hardware_wall_seconds_low=round(nbuilds * 40.849 + nruns * 70.387, 3),
        hardware_wall_seconds_high=round(nbuilds * 40.849 + nruns * 100.430, 3),
        hardware_compile_seconds_per_build=40.849,
        hardware_runtime_seconds_low=70.387,
        hardware_runtime_seconds_high=100.430,
        retained_bytes_approx=nbuilds * 51000 + nruns * 10990000,
        storage_scope="Approximate uncompressed campaign contents; allow additional space for the ZIP and version-dependent artifacts.",
    )
    return dict(schema=1, profile=profile, backend=backend, namespace=namespace,
                fabric_dims=dims, fabric_offsets=offsets, host_timeout=host_timeout,
                builds=builds, runs=runs, cohorts=cohorts, estimates=estimates,
                independent_unit="One run entry creates one fresh runtime; A/B/A are three nested generations.",
                generations_per_runtime=3, expected_fresh_runtimes=nruns,
                evidence_scope="Per-PE local durations and submitting-thread PMU observations; no synchronized global makespan or pure SDK cost.",
                selection="Fixed before measurements; retain all generations, failures and unavailable counters. No warm-window selection.")
