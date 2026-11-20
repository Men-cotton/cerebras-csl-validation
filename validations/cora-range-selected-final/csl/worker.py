"""Run one executable across all eleven Cora windows and retain the full device replies."""

import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from contract import (
    COLS,
    HEAD,
    LANES,
    ROWS,
    audit,
    banks,
    check_frame,
    check_original_plan,
    checksum,
    load_graph,
    require,
    save,
    sha,
    sizes,
    window,
    word_sha,
)


def run(a):
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime,
        SdkTarget,
        SimfabConfig,
        get_platform,
        MemcpyDataType,
        MemcpyOrder,
    )

    src = Path(__file__).resolve().parent
    out = a.output.resolve()
    artifact = a.artifact.resolve()
    manifest = a.manifest.resolve()
    release = json.loads(manifest.read_text())
    role = "hardware" if a.cmaddr else "simulator"
    r = a.shards
    require(sha(a.input) == release["input_sha256"], "frozen Cora bytes")
    z = load_graph(a.input)
    host = {n: sha(src / n) for n in ("worker.py", "contract.py")}
    require(host == release["host"], "worker identity")
    out.mkdir(parents=True, exist_ok=False)
    require(a.language=="csl" and r==16,"selected native 64-PE Cora program")
    state = dict(
        language=a.language,
        variant=release["method"],
        shards=r,
        role=role,
        release_sha256=sha(manifest),
        input_sha256=sha(a.input),
        host=host,
        windows=[],
        stopped=False,
    )

    def checkpoint(name):
        state["stage"] = name
        save(out / "result.json", state)

    checkpoint("platform")
    os.chdir(out)
    platform = get_platform(a.cmaddr, SimfabConfig(num_threads=1, suppress_trace=True), SdkTarget.WSE3)
    state["platform"] = {
        "system": bool(platform.is_system()),
        "simulation": bool(platform.is_simulation()),
    }
    require(
        state["platform"] == {"system": bool(a.cmaddr), "simulation": not bool(a.cmaddr)},
        "runtime target",
    )
    state["elf_sha256"] = {p.name: sha(p) for p in (artifact / "bin").glob("*.elf")}
    require(bool(state["elf_sha256"]), "CSL application inventory")
    compiled = json.loads((artifact / "out.json").read_text())["params"]
    cap, ec = sizes(r)
    require(
        {k: int(compiled[k]) for k in ("R", "F", "E")} == {"R": r, "F": cap * 2, "E": ec},
        "compiled source, bank and edge capacities",
    )
    if role == "simulator":
        require(
            state["elf_sha256"] == release["csl_simulator"][str(r)]["elf"], "CSL simulator inventory"
        )
    dev = SdkRuntime(
        str(artifact),
        cmaddr=a.cmaddr,
        **({} if a.cmaddr else dict(simfab_numthreads=1, suppress_simfab_trace=True)),
    )

    def io(name, ar, put=False, y=0):
        ar = np.ascontiguousarray(ar)
        h, w, n = ar.shape
        sym = dev.get_id(name)
        if put:
            dev.memcpy_h2d(
                sym,
                ar.ravel(),
                0,
                y,
                w,
                h,
                n,
                streaming=False,
                nonblock=False,
                order=MemcpyOrder.ROW_MAJOR,
                data_type=MemcpyDataType.MEMCPY_32BIT,
            )
        else:
            dev.memcpy_d2h(
                ar.ravel(),
                sym,
                0,
                y,
                w,
                h,
                n,
                streaming=False,
                nonblock=False,
                order=MemcpyOrder.ROW_MAJOR,
                data_type=MemcpyDataType.MEMCPY_32BIT,
            )
        return ar

    ab = banks(z, r)
    bank_wire = np.concatenate(ab, axis=2)
    state["bank_sha256"] = word_sha(bank_wire)
    bh = [
        np.array([[checksum(bank[y, x].view(np.uint32)) for x in range(COLS)] for y in range(r)])
        for bank in ab
    ]
    cap, ec = sizes(r)
    raw = {}
    checkpoint("load")
    dev.load()
    checkpoint("run")
    dev.run()
    try:
        io("banks", bank_wire, True)
        dev.launch("verify_banks", nonblock=False)
        for i in range(11):
            sl = window(z, r, i)
            entry = dict(index=i, packed_graph_sha256=word_sha(sl["wire"]), frames=[])
            state["windows"].append(entry)
            checkpoint(f"prepare-{i}")
            pack = np.zeros((r, 2 + 2 * ec), np.uint32)
            pack[:, 0] = i
            pack[:, 1] = sl.get("encoded_counts",sl["counts"])
            pack[:, 2 : 2 + ec] = sl["pairs"]
            pack[:, 2 + ec :] = sl["values"].view(np.uint32)
            io("graph", np.broadcast_to(pack[:, None, :], (r, COLS, pack.shape[1])), True)
            dev.launch("prepare", nonblock=False)
            actual_plan=io("plan",np.empty((r,COLS,3),np.uint32))
            check_original_plan(z,sl,i,r,actual_plan,release["method"])
            entry["device_plan_rows"]=actual_plan[:,0].tolist()
            entry["device_plan_all_pe"]=actual_plan.tolist()
            entry["device_plan_sha256"]=word_sha(actual_plan)
            setup_once=None
            frame_clocks=[]
            for t in (1, 2, 3):
                checkpoint(f"frame-{i}-{t}")
                control = np.array([t], np.uint32)
                begin = time.perf_counter_ns()
                dev.launch("compute", np.uint16(t), nonblock=False)
                meta = (
                    io("header", np.empty((r, COLS, HEAD), np.uint32))
                    .transpose(1, 0, 2)
                    .reshape(COLS, r * HEAD)
                )
                owners = io("owner", np.empty((1, COLS, ROWS * LANES), np.float32), y=r - 1)[0].view(
                    np.uint32
                )
                reply = np.concatenate([meta, owners], axis=1).ravel()
                dt = time.perf_counter_ns() - begin
                raw[f"w{i}_t{t}"] = reply.copy()
                entry["frames"].append(
                    dict(
                        trial=t,
                        control_sha256=word_sha(control),
                        reply_sha256=word_sha(reply),
                        host_interval_ns=dt,
                    )
                )
                logits,prep,cycles,error=check_frame(z, sl, i, t, r, reply, bh)
                if setup_once is None: setup_once=prep
                np.testing.assert_array_equal(prep,setup_once)
                frame_clocks.append(cycles)
                entry["frames"][-1].update(compute_max=int(cycles.max()),active_logit_words=logits.size,fp64_max_abs=error)
            entry["setup_max"]=int(setup_once.max())
            entry["max_owner_cumulative"]=(np.cumsum(frame_clocks,axis=0)+setup_once).max(axis=(1,2)).tolist()
            entry["sum_phase_max"]=(np.max(frame_clocks,axis=(1,2)).cumsum()+setup_once.max()).tolist()
            np.savez_compressed(out / "raw.npz", **raw)
            checkpoint(f"verified-window-{i}")
    finally:
        np.savez_compressed(out / "raw.npz", **raw)
        checkpoint("stop")
        dev.stop()
        state["stopped"] = True
    if role=="simulator":
        require(sha(src/"pointer_debug.py")==release["simulator_debug_sha256"],"simulator SRAM auditor")
        from pointer_debug import capture
        state["stopped_sram"]=capture(window(z,r,10),r,artifact,out,10)
    state["stage"] = "PASS"
    state["raw_sha256"] = sha(out / "raw.npz")
    state["summary"] = audit(z, state, raw)
    checkpoint("PASS")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", choices=("csl",), required=True)
    p.add_argument("--shards", type=int, choices=(16,), required=True)
    for name in ("input", "manifest", "artifact", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--cmaddr")
    run(p.parse_args())
