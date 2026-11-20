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
    original_plan,
    checksum,
    frame_packet,
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
        SdkCompileArtifacts,
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
    direct = a.language == "tungsten"
    state = dict(
        language=a.language,
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
    if direct:
        target = release["direct"][str(r)]["roles"][role]
        state["elf_sha256"] = sha(artifact / "program.elf")
        require(
            state["elf_sha256"] == target["elf_sha256"]
            and sha(artifact / "out_port_map.json") == release["port_sha256"],
            "Direct Link artifact",
        )
        dev = SdkRuntime(
            SdkCompileArtifacts(str(artifact)).add_port_mapping(str(artifact / "out_port_map.json")),
            platform,
            memcpy_required=False,
            msg_level="INFO",
        )
        state["coords"] = [
            list(dev.coord_logical_to_physical((x, y))) for y in range(r) for x in range(COLS)
        ]
        require(
            state["coords"] == [[x, y] for y in range(r) for x in range(COLS)],
            "unshifted Direct Link geometry",
        )
    else:
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
        if direct:
            dev.report_port_infos()
            rx, tx = dev.get_port_id("rx_port"), dev.get_port_id("tx_port")
            require(rx is not None and tx is not None and rx != tx, "Direct Link ports")
        else:
            io("banks", bank_wire, True)
            dev.launch("verify_banks", nonblock=False)
        for i in range(11):
            sl = window(z, r, i)
            entry = dict(index=i, packed_graph_sha256=word_sha(sl["wire"]), frames=[])
            expected_plan=original_plan(z,sl,i,r,release["mode"])
            entry["expected_plan"]=expected_plan.tolist()
            state["windows"].append(entry)
            checkpoint(f"prepare-{i}")
            if not direct:
                pack = np.zeros((r, 2 + 2 * ec), np.uint32)
                pack[:, 0] = i
                pack[:, 1] = sl["encoded_counts"]
                pack[:, 2 : 2 + ec] = sl["pairs"]
                pack[:, 2 + ec :] = sl["values"].view(np.uint32)
                io("graph", np.broadcast_to(pack[:, None, :], (r, COLS, pack.shape[1])), True)
                dev.launch("prepare", nonblock=False)
                pe_plan=io("plan",np.empty((r,COLS,3),np.uint32))
                np.testing.assert_array_equal(pe_plan,np.broadcast_to(expected_plan[:,None,:],pe_plan.shape))
                entry["device_plan_all_pe"]=pe_plan.tolist()
                entry["device_plan_sha256"]=word_sha(pe_plan)
            for t in (1, 2, 3):
                checkpoint(f"frame-{i}-{t}")
                control = frame_packet(bank_wire, sl, i, t) if direct else np.array([t], np.uint32)
                begin = time.perf_counter_ns()
                if direct:
                    guarded = np.full(COLS * (r * HEAD + ROWS * LANES) + 4, 0xBAD0C0DE, np.uint32)
                    reply = guarded[2:-2]
                    rec = dev.receive(tx, reply, reply.size, nonblock=True)
                    send = dev.send(rx, control, control.size, nonblock=True)
                    dev.task_wait(send)
                    dev.task_wait(rec)
                    require(np.all(guarded[[0, 1, -2, -1]] == 0xBAD0C0DE), "port guards")
                else:
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
                check_frame(z, sl, i, t, r, reply, bh)
            np.savez_compressed(out / "raw.npz", **raw)
            checkpoint(f"verified-window-{i}")
    finally:
        np.savez_compressed(out / "raw.npz", **raw)
        checkpoint("stop")
        dev.stop()
        state["stopped"] = True
    state["stage"] = "PASS"
    state["raw_sha256"] = sha(out / "raw.npz")
    state["summary"] = audit(z, state, raw)
    checkpoint("PASS")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", choices=("csl", "tungsten"), required=True)
    p.add_argument("--shards", type=int, choices=(4, 16), required=True)
    for name in ("input", "manifest", "artifact", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--cmaddr")
    run(p.parse_args())
