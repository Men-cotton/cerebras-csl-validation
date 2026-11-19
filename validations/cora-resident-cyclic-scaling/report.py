"""Reconstruct local or returned physical strong scaling from all original PE replies."""

import argparse
from pathlib import Path
import numpy as np
from appliance import ROOT, read, check_capture, application_elfs, verified
from compile_csl import flags
from contract import COLS, FEATURES, HEAD, LANES, ROWS, require, load_graph, save, sha, sizes, window


def compare(roots, role, builds=None):
    graph = load_graph(ROOT / "fixture.npz")
    release = read(ROOT / "release.json")
    data = {}
    capture_hashes = {}
    archives = {}
    launch_times = {}
    release_hash = sha(ROOT / "release.json")
    simulator = verified() if role == "hardware" else {}
    for r in (4, 16):
        compiled = None
        if role == "hardware":
            build = read(builds[r])
            compiled = build["application_elfs"]
            require(
                build["shards"] == r
                and build["flags"] == flags(r, True)
                and build["release_sha256"] == release_hash,
                "compiled physical geometry and release",
            )
            archive = builds[r].parent / build["artifact"]
            require(
                archive.parent.resolve() == builds[r].parent.resolve(),
                "compile archive under the result directory",
            )
            if archive.is_file():
                require(
                    sha(archive) == build["sha256"] and application_elfs(archive) == compiled,
                    "returned CSL archive",
                )
            archives[str(r)] = dict(
                archive_sha256=build["sha256"],
                returned_archive_rehashed=archive.is_file(),
                build_record_sha256=sha(builds[r]),
            )
        for lang in ("csl", "tungsten"):
            key = f"{lang}-r{r}"
            data[key] = check_capture(roots[key], lang, r, role, release, graph, compiled)
            capture_hashes[key] = dict(
                raw_sha256=sha(roots[key] / "raw.npz"), result_sha256=sha(roots[key] / "result.json")
            )
            if role == "hardware":
                record = read(roots[key] / "collection.json")
                require(record["launcher_run_elapsed_ns"] > 0, "launcher interval")
                launch_times[key] = record["launcher_run_elapsed_ns"]
                require(
                    record["backend"] == role
                    and record["release_sha256"] == release_hash
                    and record["language"] == lang
                    and record["shards"] == r,
                    "physical result collection",
                )
                for name in ("capture/raw.npz", "capture/result.json", "worker.log", "worker-exit.txt"):
                    require(
                        record["downloads"][name]["sha256"] == sha(roots[key] / Path(name).name),
                        "collected artifact",
                    )
                require(
                    (roots[key] / "worker-exit.txt").read_text().strip() == "0"
                    and data[key]["owner_sha256"] == simulator[key]["owner_sha256"],
                    "hardware completion and all owner bytes",
                )
        require(
            data[f"csl-r{r}"]["owner_sha256"] == data[f"tungsten-r{r}"]["owner_sha256"],
            "bitwise equality for all rows and panels",
        )
    scaled = {}
    for lang in ("csl", "tungsten"):
        a, b = (data[f"{lang}-r{r}"] for r in (4, 16))
        speed = np.array(a["compute_sum"]) / b["compute_sum"]
        inclusive = np.array(a["max_owner_cumulative_sum"]) / b["max_owner_cumulative_sum"]
        scaled[lang] = dict(
            max_pe_speedup=speed.tolist(),
            parallel_efficiency=(speed / 4).tolist(),
            inclusive_prefix_speedup=inclusive.tolist(),
        )
    rounding = []
    with np.load(roots["csl-r4"] / "raw.npz") as small, np.load(roots["csl-r16"] / "raw.npz") as large:
        for t in range(1, 4):
            count = 0
            error = 0.0
            for i in range(11):
                active = int(graph["bounds"][i + 1] - graph["bounds"][i])
                arr = []
                for n, z in ((4, small), (16, large)):
                    w = z[f"w{i}_t{t}"].reshape(COLS, -1)[:, HEAD * n :]
                    arr.append(
                        w.reshape(COLS, ROWS, LANES)
                        .transpose(1, 2, 0)
                        .reshape(ROWS, COLS * LANES)[:active, :FEATURES]
                    )
                count += int(np.count_nonzero(arr[0] != arr[1]))
                error = max(
                    error,
                    float(
                        np.max(
                            np.abs(arr[0].view(np.float32).astype(np.float64) - arr[1].view(np.float32))
                        )
                    ),
                )
            rounding.append(dict(unequal_words=count, max_abs=error))
    cost = {}
    for r in (4, 16):
        active = [int(window(graph, r, i)["counts"].max()) for i in range(11)]
        _, edge_cap = sizes(r)
        # SDK memcpy uploads both edge/weight arrays to all columns, zero-padding to the common capacity.
        cost[str(r)] = dict(
            maximum_shard_edges_by_window=active,
            physical_edges=13264,
            edge_capacity_per_pe=edge_cap,
            excess_capacity_slot_sum_per_pe_vs_tight_cyclic_windows=11 * edge_cap - sum(active),
            extra_csl_graph_transfer_words_vs_per_window_capacities_same_transport=2
            * r
            * 4
            * (11 * edge_cap - sum(active)),
            extra_reduction_link_words_per_panel=(11 * 256 - 2708) * 8 * (r - 1),
            extra_return_words_per_panel=(11 * 256 - 2708) * 8,
            timed_reduction_link_words_per_panel=11 * 256 * 8 * (r - 1),
        )
    return dict(
        backend=role,
        release_sha256=release_hash,
        capture_hashes=capture_hashes,
        physical_compile_archives=archives,
        physical_launcher_run_elapsed_ns=launch_times,
        series="cyclic source assignment; uniform 256 destinations; three resident bank selections A/-A/A",
        nodes=2708,
        edges=13264,
        active_features=7,
        active_fmas_per_panel=13264 * 7,
        measured_panels=3,
        metric="sum of window maxima of individual PE intervals; excludes host and inter-window dispatch",
        rounding_across_geometries=rounding,
        uniform_shape_cost=cost,
        strong_scaling=scaled,
        cases=data,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("role", choices=("simulator", "hardware"))
    p.add_argument("output", type=Path)
    for r in (4, 16):
        p.add_argument(f"--build-r{r}", type=Path)
        for lang in ("csl", "tungsten"):
            p.add_argument(f"--{lang}-r{r}", type=Path)
    a = p.parse_args()
    roots = {
        f"{l}-r{r}": getattr(a, f"{l}_r{r}") or ROOT / f"sim/{l}-r{r}"
        for r in (4, 16)
        for l in ("csl", "tungsten")
    }
    builds = {r: getattr(a, f"build_r{r}") for r in (4, 16)}
    if a.role == "hardware" and (
        not all(builds.values()) or any(getattr(a, k.replace("-", "_")) is None for k in roots)
    ):
        p.error("hardware requires both build records and all four capture roots")
    save(a.output, compare(roots, a.role, builds))
