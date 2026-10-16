#!/usr/bin/env python3
"""Read a native16 capture directory or downloaded capture archive.

Verify the preserved hashes and API enter/exit pairs, then print numerical and
diagnostic carriers. This does not connect to a cluster. NumPy is loaded only
when inspecting a capture, so --help works with the Python standard library.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import zipfile

LIMITS = {"capture.json": 2 << 20, "calls.json": 2 << 20, "raw_device.npz": 32 << 20}


def read_capture(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("capture path must not be a symlink")
    payloads = {}
    if path.is_dir():
        for name, limit in LIMITS.items():
            member = path / name
            if not member.is_file() or member.is_symlink() or member.stat().st_size > limit:
                raise ValueError("missing, nonregular or oversized " + name)
            payloads[name] = member.read_bytes()
    else:
        if not path.is_file() or path.stat().st_size > 256 << 20:
            raise ValueError("capture archive must be a regular file of at most 256 MiB")
        with tarfile.open(path, "r:gz") as archive:
            count = 0
            total = 0
            for member in archive:
                count += 1
                total += member.size
                if count > 256:
                    raise ValueError("capture archive has too many members")
                if total > 256 << 20:
                    raise ValueError("capture archive exceeds expanded size limit")
                if not member.name.startswith("capture/"):
                    continue
                name = member.name[len("capture/"):]
                if name not in LIMITS:
                    continue
                if name in payloads or not member.isfile() or member.size > LIMITS[name]:
                    raise ValueError("duplicate, nonregular or oversized " + name)
                payloads[name] = archive.extractfile(member).read()
        if set(payloads) != set(LIMITS):
            raise ValueError("capture archive is missing required files")
    meta = json.loads(payloads["capture.json"])
    for name in ("calls.json", "raw_device.npz"):
        if hashlib.sha256(payloads[name]).hexdigest() != meta.get(name + "_sha256"):
            raise ValueError("hash mismatch: " + name)
    calls = json.loads(payloads["calls.json"])
    return meta, calls, payloads["raw_device.npz"]


def validate_calls(calls):
    if not isinstance(calls, list) or not calls or len(calls) % 2:
        raise ValueError("API log is empty or has an unclosed call")
    for index in range(0, len(calls), 2):
        enter, leave = calls[index:index + 2]
        if (enter.get("phase") != "enter" or leave.get("phase") != "exit" or
                any(enter.get(key) != leave.get(key) for key in ("index", "api", "stage", "detail")) or
                enter.get("index") != index // 2):
            raise ValueError(f"API enter/exit mismatch at pair {index // 2}")
    if calls[-1].get("api") != "stop":
        raise ValueError("API log does not end with a completed stop")


def validate_result(meta, calls):
    validate_calls(calls)
    panels = meta.get("panels")
    if (not isinstance(panels, list) or not 1 <= len(panels) <= 3 or
            any(not isinstance(panel, dict) or panel.get("panel") != index or
                not isinstance(panel.get("errors"), dict) for index, panel in enumerate(panels))):
        raise ValueError("capture has no valid sequential panel results")
    if meta.get("status") == "pass":
        if len(panels) != 3 or any(panel["errors"] for panel in panels):
            raise ValueError("pass capture must contain three successful A/B/A panels")
    elif meta.get("status") in ("observed_numeric_fail", "observed_payload_difference"):
        if not panels[-1]["errors"] or any(panel["errors"] for panel in panels[:-1]):
            raise ValueError("numeric failure must terminate at its first failing panel")
        if meta["status"] == "observed_payload_difference" and (
                meta.get("variant") != "payload" or set(panels[-1]["errors"]) != {"payload_stream"}):
            raise ValueError("payload difference requires the payload variant and only a payload_stream mismatch")
    else:
        raise ValueError("capture does not describe a completed numerical test")


def latest(raw, symbol):
    keys = sorted(name for name in raw.files if name.endswith("_d2h_" + symbol))
    if not keys:
        raise ValueError("missing raw symbol: " + symbol)
    return raw[keys[-1]]


def print_diagnostics(raw):
    states = latest(raw, "diag_state")
    payload = latest(raw, "rx_payload")
    events = latest(raw, "diag_events")
    if states.shape != (1, 2, 8) or payload.shape != (1, 2, 68) or events.shape != (1, 2, 128):
        raise ValueError("unexpected diagnostic carrier shape")
    print("diag_state fields: variant,payload_count,overflow,marker_count,event_count,event_overflow,done,reserved")
    print("both PE diag_state:", states[0].tolist())
    for pe in range(2):
        state = states[0, pe]
        count = min(int(state[1]), 64)
        print(f"PE {pe} payload guards:", [f"0x{int(value):08x}" for value in payload[0, pe, [0, 1, 66, 67]]])
        print(f"PE {pe} payload words:", [f"0x{int(value):08x}" for value in payload[0, pe, 2:2 + count]])
        count = min(int(state[4]), 16)
        rows = events[0, pe].reshape(16, 8)[:count]
        first = None
        decoded = []
        for event in rows:
            timestamp = sum((int(event[2 + k]) & 65535) << (16 * k) for k in range(3))
            if first is None:
                first = timestamp
            decoded.append(dict(event_id=int(event[0]), phase=int(event[1]),
                                timestamp=timestamp, cycles_from_first=(timestamp - first) % (1 << 48),
                                input_queue_raw=f"0x{int(event[5]):08x}",
                                output_queue_raw=f"0x{int(event[6]):08x}", reserved=int(event[7])))
        print(f"PE {pe} events:", json.dumps(decoded))


def inspect(path):
    meta, calls, raw_bytes = read_capture(path)
    print("config/status:", meta["config"], meta["status"])
    print("backend/variant:", meta.get("backend"), meta.get("variant", "baseline"))
    print("panels:", json.dumps(meta["panels"]))
    print("last calls:", json.dumps(calls[-2:]))
    validate_result(meta, calls)
    print("capture hashes and closed API calls: verified")
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
        members = archive.infolist()
        if (len(members) > 512 or len({member.filename for member in members}) != len(members) or
                sum(member.file_size for member in members) > 64 << 20 or
                any(member.file_size > 16 << 20 for member in members)):
            raise ValueError("raw array archive exceeds bounds or contains duplicates")
    import numpy as np
    with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as raw:
        length = meta["config"]["length"]
        if not isinstance(length, int) or not 0 <= length <= 16:
            raise ValueError("invalid capture length")
        rows = latest(raw, "received_rows")[0, 1, 2:2 + length]
        values = latest(raw, "received_values")[0, 1, 2:2 + length]
        print("received rows:", (rows & np.uint32(65535)).tolist())
        print("received f32:", values.tolist())
        print("f32 raw bits:", [f"0x{int(value):08x}" for value in values.view(np.uint32)])
        print("both PE status:", latest(raw, "status")[0].tolist())
        if meta.get("variant") in ("control", "events", "payload"):
            print_diagnostics(raw)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args(argv)
    try:
        return inspect(args.capture)
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
