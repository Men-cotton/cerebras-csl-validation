"""Link two independent first-layer ingress rows to a single on-chip GraphSAGE output layer."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from two_rows import HERE, SCHEDULE, device, parts, prepare
from build_chain import module

sys.path.insert(0, str(HERE.parents[1] / "parallel-ports"))
from build_lanes import once, validate_ports


def build(first, last, reference, port, dest, readelf):
    baseline = json.loads((reference / "build.json").read_text())
    cfg = baseline["first"]
    data = prepare(first, last, cfg["P"], cfg["E"])
    if data[2] != cfg or data[7] != baseline["final"]:
        raise ValueError("unexpected original owner geometry")
    if baseline["first_sha256"] != device.sha(first) or baseline["last_sha256"] != device.sha(last):
        raise ValueError("unexpected original graph input")
    for name, expected in baseline["source_sha256"].items():
        if device.sha(reference / "src" / name) != expected:
            raise ValueError("original paired source changed")
    validate_ports(port, 2)
    rows = parts(data)
    w = rows[0][1]["P"]
    out = dest / "src"
    out.mkdir(parents=True, exist_ok=False)
    source = reference / "src"
    text = [
        "let down=color(23)",
        "let odd=color(21)",
        "let even=color(20)",
        "let cross=color(18)",
        "let in0=color(0)",
        "let out0=color(1)",
        "let in1=color(2)",
        "let out1=color(3)",
    ]
    tiles = []
    packet = 16 + cfg["R"] * (1 + cfg["Q"])
    collect = (source / "collect.w").read_text()
    (out / "collect.w").write_text(
        once(collect, f"frame in [0,{cfg['FRAMES']})", f"frame in [0,{len(SCHEDULE)})")
    )
    for lane, (_, node, _, blocks) in enumerate(rows):
        owner = (source / "owner.w").read_text()
        cap = max(1, max((len(ref) for _, owners in blocks for ref in owners), default=0))
        for name in ("selectors", "ignored"):
            owner, count = re.subn(rf"u32 {name}\[n=\d+\];", f"u32 {name}[n={cap}];", owner)
            if count != 1:
                raise ValueError("unknown selector bank")
        if lane:
            owner = once(owner, "if (X==0) { tx[] <- tag; tx[] <- (u32)echo; }", "")
            owner = once(owner, "tx[] <- (u32)X;", f"tx[] <- (u32)(X+{w});")
        else:
            gate = """u32 ticket=(u32)rx[]; u32 check=(u32)rx[];
        if (X==0) { tx[] <- ticket; tx[] <- check; }
        u32 tag=(u32)rx[];"""
            owner = once(owner, "u32 tag=(u32)rx[];", gate)
            if len(data[8]) % 2:
                owner = once(
                    owner,
                    "    frame in [0,FRAMES)",
                    "    u32 setup_pad=(u32)rx[];\n    frame in [0,FRAMES)",
                )
        (out / f"owner{lane}.w").write_text(owner)
        params = " ".join(
            f"{k}:{v}"
            for k, v in dict(node, N=data[7]["N"], SETUP=len(data[8]) if not lane else 0).items()
        )
        for x in range(w):
            o, g = f"o{lane}x{x}", f"g{lane}x{x}"
            tiles.extend((o, g))
            incoming = "even" if x % 2 == 0 else "odd"
            outgoing = "odd" if x % 2 == 0 else "even"
            text += [
                f"let {o}=tile({x} {lane * 2})",
                f"{o} : code('owner{lane} [X:{x} {params} rx:in{lane} tx:down])",
                f"{o} : paint(in{lane} [L>{'CR' if x + 1 < w else 'C'}])",
                f"{o} : paint(down [C>U])",
                f"let {g}=tile({x} {lane * 2 + 1})",
            ]
            if x:
                text += [
                    f"{g} : code('collect [OWN:{packet} TRAILING:{packet * (w - x - 1)} local_rx:down right_rx:{incoming} tx:{outgoing}])",
                    f"{g} : paint({outgoing} [C>L])",
                ]
            elif lane:
                text += [
                    f"{g} : code('bridge [P:{w} WORDS:{packet} FRAMES:{len(SCHEDULE)} local_rx:down right_rx:{incoming} fabric:cross done:out1])",
                    f"{g} : paint(cross [C>D])",
                    f"{g} : paint(out1 [C>L])",
                    f"{o} : paint(cross [U>D])",
                ]
            else:
                final = dict(data[7], FRAMES=len(SCHEDULE))
                params_final = " ".join(f"{k}:{v}" for k, v in final.items())
                text += [
                    f"{g} : code('final [{params_final} local_rx:down right_rx:{incoming} lower_rx:cross tx:out0])",
                    f"{g} : paint(cross [U>C])",
                    f"{g} : paint(out0 [C>L])",
                ]
            text.append(f"{g} : paint(down [D>C])")
            if x + 1 < w:
                text.append(f"{g} : paint({incoming} [R>C])")
    text.append("union(" + " ".join(tiles) + ")")
    (out / "program.paint").write_text("\n".join(text) + "\n")
    shutil.copyfile(HERE / "bridge.w", out / "bridge.w")
    final = (source / "final.w").read_text()
    final = once(final, "socket tx;", "socket lower_rx [[iq=2]];\nsocket tx;")
    final = once(
        final,
        "        u32 tag=(u32)local_rx[];",
        """        u32 ticket=(u32)local_rx[]; u32 check=(u32)local_rx[];
        stamp(0);
        tx[] <- ticket; tx[] <- check;
        u32 tag=(u32)local_rx[];""",
    )
    final = once(final, "        stamp(0);\n        i in [0,N)", "        i in [0,N)")
    final = once(final, "        u32 errors=0;", "        u32 errors=0; if (ticket!=tag) errors+=1;")
    final = once(
        final,
        "            else { i in [0,16+R*(1+H)) packet[i] <- (u32)right_rx[]; }",
        f"            else if (who<{w}) {{ i in [0,16+R*(1+H)) packet[i] <- (u32)right_rx[]; }}\n"
        "            else { i in [0,16+R*(1+H)) packet[i] <- (u32)lower_rx[]; }",
    )
    final = once(final, "u16 clocks[n=12];", "u16 clocks[n=12];\nu16 data_start[n=3];")
    final = once(
        final,
        "        i in [0,N) seen[i] <- 0;",
        """        data_start[0]=tsc[0]; data_start[1]=tsc[1]; data_start[2]=tsc[2];
        i in [0,N) seen[i] <- 0;""",
    )
    trailer = "        if ((17+P*16+Z*(1+Q))%2!=0) tx[] <- tag;"
    final = once(
        final, trailer, trailer + "\n        tx[] <- tag; k in [0,3) tx[] <- (u32)data_start[k];"
    )
    (out / "final.w").write_text(final)
    tools = module("source_tools", HERE.parents[1] / "build.py")
    env = dict(os.environ, PATH=str(tools.SDK / "bin") + ":" + os.environ["PATH"])

    def run(name, *args):
        with (dest / "compile.log").open("ab") as f:
            subprocess.run(
                [tools.SDK / "bin" / name, *args],
                cwd=dest,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=1200,
                check=True,
            )

    run(
        "paint",
        "source=src/program.paint",
        "lib=src",
        "arch=sdr",
        "annotate",
        "image=chain.obj",
        "map=chain.map",
    )
    roles = {}
    pair = []
    for role, fabric in [("simulator", (w, 4)), ("hardware", tools.direct.FABRIC)]:
        image = dest / role / "out"
        image.mkdir(parents=True)
        elf = image / "program.elf"
        run("mielf", "image=chain.obj", "elf=" + str(elf), f"w={fabric[0]}", f"h={fabric[1]}")
        pair.append(tools.direct.equivalence(elf, fabric, (w, 4), readelf))
        (image / "ranges.txt").rename(image.parent / "ranges.txt")
        shutil.copyfile(port, image / "out_port_map.json")
        roles[role] = dict(fabric=list(fabric), elf_sha256=device.sha(elf))
    if pair[0] != pair[1]:
        raise ValueError("application payload differs on the physical fabric")
    device.write(
        dest / "build.json",
        dict(
            first=cfg,
            final=data[7],
            roles=roles,
            first_sha256=device.sha(first),
            last_sha256=device.sha(last),
            port_sha256=device.sha(port),
            original_build_sha256=device.sha(reference / "build.json"),
            canonical_sha256=pair[0]["canonical_sha256"],
            map_sha256=device.sha(dest / "chain.map"),
            source_sha256={p.name: device.sha(p) for p in sorted(out.iterdir())},
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("first", "last", "reference", "port", "dest"):
        p.add_argument(n, type=Path)
    p.add_argument("--readelf", default="cs-readelf")
    a = p.parse_args()
    build(
        a.first.resolve(),
        a.last.resolve(),
        a.reference.resolve(),
        a.port.resolve(),
        a.dest.resolve(),
        a.readelf,
    )
