"""Continue an interrupted fixed GraphSAGE schedule from its verified original SDK words."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np


def entries(path):
    return [json.loads(s) for s in path.read_bytes().splitlines()]


def verify(record):
    from graph import read, require, sha
    from pooled_resume import active_words

    cert = read(record / "continuation.json")
    old = read(record / "first-result.json")
    before = entries(record / "first-windows.jsonl")
    after = entries(record / "windows.jsonl")
    now = read(record / "result.json")
    require(
        old["stage"] == "running"
        and not old["stopped"]
        and now["stage"] == "PASS"
        and now["stopped"]
        and now["completed_frames"] == len(after) == old["planned_frames"],
        "interrupted first session and finished combined sequence",
    )
    require(
        old["completed_frames"] <= len(before) < old["completed_frames"] + 10
        and all(
            old[k] == now[k]
            for k in ("params", "host_sha256", "case_sha256", "build_sha256", "loaded_image", "platform")
        ),
        "same executable, model and device",
    )
    require(
        sha(record / "first-result.json") == cert["first_result_sha256"]
        and sha(record / "first-windows.jsonl") == cert["first_journal_sha256"]
        and len(before) == cert["completed_original_frames"]
        and sha(Path(__file__)) == cert["continuation_driver_sha256"],
        "immutable source run",
    )
    for source, measured in zip(before, after[: len(before)], strict=True):
        require(
            source.keys() == measured.keys()
            and all(source[k] == measured[k] for k in source if k not in ("max_abs", "local_max_abs"))
            and sha(record / measured["name"]) == measured["raw_sha256"],
            "immutable original packet",
        )
        for key in ("max_abs", "local_max_abs"):
            np.testing.assert_allclose(source[key], measured[key], rtol=1e-9, atol=3e-13)
    guard = before[cert["guard_tag"] - 1]
    cfg = old["params"]
    require(
        guard["mode"] == 0
        and sha(record / "guard.npz") == cert["guard_sha256"]
        and cert["newly_sent_frames"] == len(after) - len(before) + 1,
        "separate repeated guard",
    )
    with np.load(record / guard["name"]) as a, np.load(record / "guard.npz") as b:
        left, right = a["reply"], b["reply"]
        np.testing.assert_array_equal(left[:, :4], right[:, :4])
        rows = right[:, 2].astype(int)
        for x, y in zip(
            active_words(left, cfg, rows, 0, 0), active_words(right, cfg, rows, 0, 0), strict=True
        ):
            np.testing.assert_array_equal(x, y)
        require(
            cert["guard_projection_words"] == int(rows.sum()) * 2 * cfg["F"], "all repeated dense values"
        )
    return cert


def resume(a):
    sys.path.insert(0, str(a.host.resolve() / "resident"))
    import run
    from graph import read, require, save, sha
    from pooled_resume import active_words

    src, out = a.prefix.resolve(), a.output.resolve()
    source = read(src / "result.json")
    journal = entries(src / "windows.jsonl")
    count = len(journal)
    require(
        source["stage"] == "running"
        and not source["stopped"]
        and journal[-1]["trial"] == 3
        and journal[-1]["mode"] == 0
        and source["completed_frames"] <= count < source["completed_frames"] + 10,
        "incomplete restored-input prefix",
    )
    require(
        source["host_sha256"]
        == {
            q.name: sha(q)
            for q in (
                a.host / "resident/run.py",
                a.host / "resident/graph.py",
                a.host / "transfer/rectangles.py",
            )
        }
        and source["build_sha256"] == sha(a.build / "build.json")
        and source["case_sha256"] == sha(a.case / "case.json"),
        "original host, ELF and case",
    )
    guard = min((j for j in journal if j["trial"] == 1 and j["mode"] == 0), key=lambda j: j["rows"])
    count_new = source["planned_frames"] - count + 1
    real = run.Device
    cert = dict(
        continuation_driver_sha256=sha(Path(__file__)),
        first_result_sha256=sha(src / "result.json"),
        first_journal_sha256=sha(src / "windows.jsonl"),
        completed_original_frames=count,
        guard_tag=guard["tag"],
        newly_sent_frames=count_new,
        scope="two separate SDK loads of the same executable; original local clocks are retained",
    )

    class Continued(real):
        def exchange(self, batch, step, boot):
            tag = step["tag"]
            if tag == 1:
                self.boot = (count_new, boot[1])
                require(
                    self.image == source["loaded_image"] and self.platform == source["platform"],
                    "second SDK load matches original",
                )
                for name, target in (
                    ("result.json", "first-result.json"),
                    ("windows.jsonl", "first-windows.jsonl"),
                ):
                    shutil.copyfile(src / name, out / target)
            if tag > count:
                return super().exchange(batch, step, None)
            row = journal[tag - 1]
            require(
                row["tag"] == tag and sha(src / row["name"]) == row["raw_sha256"],
                "archived input reply identity",
            )
            with np.load(src / row["name"]) as z:
                previous = z["reply"]
            if tag == guard["tag"]:
                observed, sent, wall = super().exchange(batch, step, self.boot)
                np.testing.assert_array_equal(previous[:, :4], observed[:, :4])
                rows = previous[:, 2].astype(int)
                for x, y in zip(
                    active_words(previous, self.cfg, rows, 0, 0),
                    active_words(observed, self.cfg, rows, 0, 0),
                    strict=True,
                ):
                    np.testing.assert_array_equal(x, y)
                np.savez_compressed(out / "guard.npz", reply=observed)
                cert.update(
                    guard_projection_words=int(rows.sum()) * 2 * self.cfg["F"],
                    guard_sha256=sha(out / "guard.npz"),
                    guard_H2D_words=sent,
                    guard_D2H_words=observed.size,
                    guard_host_ns=wall,
                )
                save(out / "continuation.json", cert)
            return previous, row["input_words"], row["host_ns"]

    run.Device = Continued
    try:
        a.cmaddr = a.artifact = a.replay = None
        run.run(a)
    finally:
        run.Device = real
    verify(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    r = sub.add_parser("resume")
    for k in ("build", "case", "host", "prefix", "output"):
        r.add_argument(k, type=Path)
    v = sub.add_parser("verify")
    v.add_argument("record", type=Path)
    v.add_argument("host", type=Path)
    a = p.parse_args()
    if a.action == "resume":
        resume(a)
    else:
        sys.path.insert(0, str(a.host.resolve() / "resident"))
        verify(a.record.resolve())
