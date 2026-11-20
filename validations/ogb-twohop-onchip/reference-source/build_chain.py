"""Link first-layer owners to a final-layer PE through the existing gather row."""

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

from chain import HERE, device, prepare
from selector_ingress import render


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def build(first, last, dest, p, e, readelf):
    a, b, cfg, groups, owner, slot, tiles, final, setup = prepare(first, last, p, e)
    tools = module("first_builder", HERE.parent / "build.py")
    sys.path.insert(0, str(tools.DEVICE))
    route = module("ogb_owner_layout", tools.DEVICE / "build.py")
    src = dest / "src"
    src.mkdir(parents=True, exist_ok=False)
    layout = route.layout(dict(cfg, K=0)).replace(" K:0", "")
    lines = layout.splitlines()
    row = [i for i, s in enumerate(lines) if s.startswith("g0 : code(")]
    if len(row) != 1:
        raise ValueError("original collection row changed")
    params = " ".join(f"{k}:{v}" for k, v in final.items())
    lines[row[0]] = f"g0 : code('final [{params} local_rx:down right_rx:even tx:hout])"
    token = f" rx:hin tx:down]"
    lines = [
        s.replace(token, f" N:{final['N']} SETUP:{len(setup)}" + token) if " : code('owner " in s else s
        for s in lines
    ]
    (src / "program.paint").write_text("\n".join(lines) + "\n")
    owner_text = render((HERE.parent / "owner.w").read_text(), cfg, a, owner, slot, "deferred", False)

    def change(old, new):
        nonlocal owner_text
        if owner_text.count(old) != 1:
            raise ValueError("source of the first layer changed at " + old)
        owner_text = owner_text.replace(old, new)

    change("X,P,R,F,Q,E,RELU,FRAMES;", "X,P,R,F,Q,E,RELU,FRAMES,N,SETUP;")
    change(
        "    frame in [0,FRAMES)",
        """    if (X==0) { k in [0,SETUP) tx[] <- (u32)rx[]; }
    else { k in [0,SETUP) { u32 unused=(u32)rx[]; } }
    frame in [0,FRAMES)""",
    )
    change(
        "u32 tag=(u32)rx[];",
        """u32 tag=(u32)rx[];
        xp flags=(xp)(u32)rx[];
        xp echo=flags%2;
        if (flags>1) { u32 pad=(u32)rx[]; }
        if (X==0) { tx[] <- tag; tx[] <- (u32)echo; }""",
    )
    change(
        "k in [0,R*Q) tx[] <- out[k];",
        """k in [0,R*Q) tx[] <- out[k];
        if (echo!=0) {
            if (X==0) { k in [0,N*Q) tx[] <- (u32)rx[]; }
            else { k in [0,N*Q) { u32 unused=(u32)rx[]; } }
        }""",
    )
    (src / "owner.w").write_text(owner_text)
    (src / "collect.w").write_text(
        (tools.BASE / "collect.w").read_text().replace("frame in [0,3)", f"frame in [0,{cfg['FRAMES']})")
    )
    shutil.copyfile(HERE / "final.w", src / "final.w")
    env = dict(os.environ, PATH=str(tools.SDK / "bin") + ":" + os.environ["PATH"])

    def tool(name, *args):
        with (dest / "compile.log").open("ab") as log:
            subprocess.run(
                [tools.SDK / "bin" / name, *args],
                cwd=dest,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    tool(
        "paint",
        "source=src/program.paint",
        "lib=src",
        "arch=sdr",
        "annotate",
        "image=chain.obj",
        "map=chain.map",
    )
    port = tools.BASE / "evidence/out_port_map.json"
    roles = {}
    pairs = []
    for role, fabric in [("simulator", (p, 2)), ("hardware", tools.direct.FABRIC)]:
        out = dest / role / "out"
        out.mkdir(parents=True)
        elf = out / "program.elf"
        tool("mielf", "image=chain.obj", "elf=" + str(elf), f"w={fabric[0]}", f"h={fabric[1]}")
        pairs.append(tools.direct.equivalence(elf, fabric, (p, 2), readelf))
        (out / "ranges.txt").rename(out.parent / "ranges.txt")
        shutil.copyfile(port, out / "out_port_map.json")
        roles[role] = dict(fabric=list(fabric), elf_sha256=device.sha(elf))
    if pairs[0] != pairs[1]:
        raise ValueError("physical and local application payloads differ")
    device.write(
        dest / "build.json",
        dict(
            first=cfg,
            final=final,
            first_sha256=device.sha(first),
            last_sha256=device.sha(last),
            roles=roles,
            canonical_sha256=pairs[0]["canonical_sha256"],
            port_sha256=device.sha(port),
            source_sha256={s.name: device.sha(s) for s in src.iterdir()},
            map_sha256=device.sha(dest / "chain.map"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "last", "dest"):
        p.add_argument(name, type=Path)
    p.add_argument("--owners", type=int, required=True)
    p.add_argument("--entries", type=int, required=True)
    p.add_argument("--readelf", default="cs-readelf")
    a = p.parse_args()
    build(a.first.resolve(), a.last.resolve(), a.dest.resolve(), a.owners, a.entries, a.readelf)
