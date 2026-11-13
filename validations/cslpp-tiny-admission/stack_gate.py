"""Fail-closed reference equivalence for the pinned E236 application programs.

This is deliberately not a general stack analyzer. The reference is the final
E236 program produced by CSL++ aa660748's two-pass resolver.
Independent total coroutine capacity remains unestablished because backend
call closure has not been decoded. The adapter preserves admission HOLD. It contains the
measured root frame and a source closure whose other functions were inlined.
The target compiler must reproduce all executable bytes and function locations;
a different code generation requires a new proof, even if SRAM still fits.
Only generated kernel ELF data is read. No SDK binary is inspected.
"""
import hashlib
import re
import struct


LIMIT = 2 << 20
ROOT = "compute_tiles"
FIBER = "__fiber_stack_" + ROOT
# Root argument/register save area: FiberStackResolver.cpp:474–483 in
# https://github.com/esa-tu-darmstadt/csl-pp/tree/aa6607484d78f2e74e18b9aae465a5a2d0a02db1
ARGUMENT_BYTES = 12
SOURCE_PINS = {
    "direct": ["4b97eba48e3659294756c6883187847ec666a225667d3a6c92fa8cab104449e5"] * 4,
    "frontend": [
        "5a615a7842f553b9e4b822f22de642c0168ea748ad321d34d7d49b43991c6b77",
        "b842f1275171f332343c3da26f64f5d412cc5e2a91ceb7710b48c0d3a04bb231",
        "e9e2f49be978ff6852a427987e42c2c65556f581383f7f300ec6dc41f700da97",
        "e12fd04eaaffa9b4630e4602dc47bd106e10bdee5fd82862761fdd5e265e5c50"],
    "async": [
        "7f72d742bb0c125980990c413abdb6c943d4236c19b75e28a14542db38124ea4",
        "fcd2c00da2abb8de882320dcc00926ba8a316136b2e078dfcd689b74d6a2edb9",
        "7c1bb40c4f1a445ee51dec8e9cc70f0359a0f7df9bf0d18b27145ce18e971887",
        "e025c722ed537c69925f74898a9ce838a0adc1b1c8055848847dddd652d0b2db"],
}
ELF_PINS = {
    "direct": [
        "7cbd389766602b391af11c961dd317149967824a996dcea538fefd13b48e1d11",
        "af8c22c2a3617f898dae86a0f135e7bace4f68c8a7ced01e212dc9912e4ef12d",
        "fdc77e4b7232f65e3e77e69a30bb46eacf6e6c1aac12dc1f2b9246369d7d569d",
        "596a2a3c89f49726bb87ee74d430ee33fec420cbdb336faa07f698408f8a50f0"],
    "frontend": [
        "822853dd10e4b140007e24562d5cc9663d0afffd8b9283387f7ae073ae489894",
        "ce0e0b145fcdae252d9c461da242715e26b7a8b453dddd6669bf3c65515ec777",
        "9e597403a964eb352e9241cafdf95a0674fc2c385f7c7450a98c0560e62b3e1e",
        "d0923ca7ec36a5e676337680deafa68590731c6c4cfcecf296d640c660ba3253"],
    "async": [
        "2e9e19175de49f73a84d4989dddde052c9c74255028fbec268e69812ed876688",
        "d8d0085263b1ef017cf1ae463b296696f7e15333ed0458b1812bba34832cbade",
        "6e90c1d40be5bd903cc3dfa87a83fdf4858c634a8976429ffbba9290e0b5c41b",
        "2d78b5f20a3da0d3d7859bacfec6c4ac9c546a4eb86a99e2d8a1d66080c3c531"],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _range(data, offset, length):
    require(0 <= offset <= len(data) and 0 <= length <= len(data) - offset,
            "bounded ELF range")
    return data[offset:offset + length]


def _string(table, offset):
    require(0 <= offset < len(table), "bounded ELF string")
    end = table.find(b"\0", offset)
    require(end >= offset, "terminated ELF string")
    try:
        return table[offset:end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("ELF string encoding") from error


def _sections(data):
    require(isinstance(data, bytes) and 64 <= len(data) <= LIMIT, "bounded ELF bytes")
    require(data[:7] == b"\x7fELF\x02\x01\x01", "ELF64 little-endian version one")
    require(struct.unpack_from("<HHI", data, 16) == (2, 248, 1), "Cerebras executable ELF")
    offset = struct.unpack_from("<Q", data, 40)[0]
    size, count, strings = struct.unpack_from("<HHH", data, 58)
    require(size == 64 and 1 <= count <= 256 and strings < count, "bounded section table")
    _range(data, offset, size * count)
    headers = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size) for i in range(count)]
    h = headers[strings]
    require(h[1] == 3, "section name string table")
    names = _range(data, h[4], h[5])
    sections = {}
    for h in headers:
        name = _string(names, h[0])
        require(name not in sections, "unique ELF sections")
        require(h[5] <= LIMIT, "bounded ELF section")
        if h[1] != 8:
            _range(data, h[4], h[5])
        sections[name] = h
    return headers, sections


def _frames(payload, addresses):
    frames, cursor = {}, 0
    while cursor < len(payload):
        require(cursor + 2 < len(payload), "truncated stack-size record")
        address = struct.unpack_from("<H", payload, cursor)[0]
        cursor += 2
        value = 0
        for shift in range(0, 35, 7):
            require(cursor < len(payload), "truncated stack-size ULEB128")
            byte = payload[cursor]
            cursor += 1
            value |= (byte & 127) << shift
            if byte < 128:
                break
        else:
            raise ValueError("oversized stack-size ULEB128")
        require(value <= 48 * 1024, "bounded frame size")
        require(address in addresses, "stack frame has function symbol")
        # Several aliases may name one address; all denote the same frame.
        for name in addresses[address]:
            require(name not in frames, "unique function stack frame")
            frames[name] = value
    require(frames, "nonempty stack-size metadata")
    return frames


def inspect(data):
    """Return a JSON-compatible bounded view of generated executable data."""
    headers, sections = _sections(data)
    require(".symtab" in sections, "function/stack symbols required")
    h = sections[".symtab"]
    require(h[1] == 2 and h[9] == 24 and h[5] % 24 == 0 and h[6] < len(headers),
            "bounded symbol table")
    st = headers[h[6]]
    require(st[1] == 3, "symbol name string table")
    strings = _range(data, st[4], st[5])
    functions, fibers, addresses = {}, {}, {}
    for pos in range(h[4], h[4] + h[5], 24):
        name, info, _, section, address, size = struct.unpack_from("<IBBHQQ", data, pos)
        name = _string(strings, name)
        if info & 15 == 2:
            require(name not in functions and 0 < section < len(headers), "unique defined function")
            sh = headers[section]
            require(sh[2] & 4 and sh[3] <= address and address + size <= sh[3] + sh[5],
                    "function belongs to executable section")
            functions[name] = dict(address=address, size=size)
            addresses.setdefault(address, []).append(name)
        if name.startswith("__fiber_stack_"):
            require(name not in fibers and info & 15 == 1 and 0 < section < len(headers),
                    "unique defined coroutine stack")
            sh = headers[section]
            require(sh[2] & 2 and 0 < size <= 48 * 1024
                    and sh[3] <= address and address + size <= sh[3] + sh[5],
                    "coroutine stack allocation")
            fibers[name] = dict(address=address, size=size)
    require(functions, "nonempty function symbols")
    # Cerebras also marks hardware configuration sections SHF_EXECINSTR. Those
    # high addresses are not instructions; physical routing may differ. Low
    # executable sections include both .text and the task-table trampolines.
    executable = {name: dict(address=h[3], size=h[5], sha256=digest(_range(data, h[4], h[5])))
                  for name, h in sections.items() if h[2] & 4 and h[3] < 48 * 1024 and h[5]}
    require(executable and all(v["size"] > 0 for v in executable.values()), "nonempty executable sections")
    storage = {name: dict(address=h[3], size=h[5], type=h[1],
                         sha256=None if h[1] == 8 else digest(_range(data, h[4], h[5])))
               for name, h in sections.items() if h[2] & 2 and h[3] < 48 * 1024}
    h = sections.get(".stack_sizes")
    frames = _frames(_range(data, h[4], h[5]), addresses) if h else None
    offset = struct.unpack_from("<Q", data, 32)[0]
    size, count = struct.unpack_from("<HH", data, 54)
    require(size == 56 and 1 <= count <= 256, "bounded program header table")
    _range(data, offset, size * count)
    tiles, loads, descriptors = set(), [], []
    for pos in range(offset, offset + size * count, size):
        h = struct.unpack_from("<IIQQQQQQ", data, pos)
        if h[0] == 1:
            payload = _range(data, h[2], h[5])
            require(h[5] <= h[6], "ELF load storage bound")
            tiles.add((h[4] >> 40) & 65535)
            descriptor = dict(flags=h[1], virtual_address=h[3],
                              physical_address=h[4] & ~(65535 << 40),
                              file_size=h[5], memory_size=h[6], alignment=h[7])
            descriptors.append(descriptor)
            if h[3] < 48 * 1024:
                require(h[3] + h[6] <= 48 * 1024, "low load stays within SRAM")
                # ELF file offsets may move as debug metadata changes. Compare
                # the bytes actually loaded there, not only section contents.
                # Only physical tile-index bits vary with fabric geometry.
                loads.append(dict(descriptor, payload_sha256=digest(payload)))
    require(tiles, "ELF application tile mapping")
    require(loads, "low-address ELF load segments")
    static_end = max(h[3] + h[5] for h in sections.values() if h[2] & 2 and h[3] < 48 * 1024)
    # File/table offsets are validated by parsing, but do not affect execution
    # when the loaded bytes match. Entry point and architecture flags do.
    execution_header = dict(ident=data[:16].hex(),
                            type=struct.unpack_from("<H", data, 16)[0],
                            machine=struct.unpack_from("<H", data, 18)[0],
                            version=struct.unpack_from("<I", data, 20)[0],
                            entry=struct.unpack_from("<Q", data, 24)[0],
                            flags=struct.unpack_from("<I", data, 48)[0],
                            header_size=struct.unpack_from("<H", data, 52)[0])
    return dict(sha256=digest(data), tile_indices=sorted(tiles), executable_sections=executable,
                low_allocated_sections=storage, functions=functions, fibers=fibers,
                frames=frames, low_load_segments=loads, load_descriptors=descriptors,
                execution_header=execution_header, static_sram_end=static_end)


def _root_closure(source):
    # This syntax recognizer is used only after exact E236 generated-source SHA
    # admission. It does not accept arbitrary CSL or infer indirect-call targets.
    text = source.decode("utf-8")
    functions = {}
    for match in re.finditer(r"^(?:export )?(?:fn|task) (\w+)\([^\n]*\)[^\n]*\{", text, re.M):
        cursor, depth = match.end(), 1
        while depth:
            require(cursor < len(text), "complete pinned function body")
            depth += (text[cursor] == "{") - (text[cursor] == "}")
            cursor += 1
        require(match[1] not in functions, "unique pinned source function")
        functions[match[1]] = text[match.end():cursor - 1]
    require(ROOT in functions, "pinned coroutine root")
    calls = {name: set(re.findall(r"\b(\w+)\(", body)) & functions.keys()
             for name, body in functions.items()}
    visited, active = set(), set()

    def visit(name):
        require(name not in active, "acyclic pinned coroutine closure")
        if name in visited:
            return
        active.add(name)
        for callee in calls[name]:
            visit(callee)
        active.remove(name)
        visited.add(name)

    visit(ROOT)
    return visited


def compare(reference, target, *, arm, reference_sha256, source):
    """Compare a target program only against its exact E236 reference evidence.

    The caller binds both programs to the same application coordinate and all
    original/generated source files. Physical tile indices may differ between
    simulator fabric and hardware fabric. No instruction relocation is relaxed.
    """
    require(arm in ELF_PINS and reference_sha256 in ELF_PINS[arm], "pinned E236 reference")
    require(digest(reference) == reference_sha256, "reference ELF identity")
    index = ELF_PINS[arm].index(reference_sha256)
    require(isinstance(source, bytes) and len(source) <= LIMIT
            and digest(source) == SOURCE_PINS[arm][index], "reference generated/source identity")
    before, after = inspect(reference), inspect(target)
    require(before["execution_header"] == after["execution_header"], "target executable header equivalence")
    require(before["executable_sections"] == after["executable_sections"], "target executable code equivalence")
    require(before["functions"] == after["functions"], "target function location equivalence")
    require(before["fibers"] == after["fibers"], "target coroutine allocation equivalence")
    require(before["low_allocated_sections"] == after["low_allocated_sections"],
            "target initialized data and static layout equivalence")
    require(before["low_load_segments"] == after["low_load_segments"],
            "target loaded program equivalence")
    require(before["load_descriptors"] == after["load_descriptors"],
            "target all load descriptors equivalence")
    stack = dict(resolver_estimated_bytes=0, reservation_bytes=0, mode="no-coroutine",
                 capacity_status="NOT_APPLICABLE")
    if arm == "async":
        closure = _root_closure(source)
        require(closure & before["functions"].keys() == {ROOT}, "only coroutine root survives out of line")
        require(before["frames"] is not None and before["frames"].get(ROOT) == 64,
                "measured pinned root frame")
        require(set(before["fibers"]) == {FIBER}, "single pinned coroutine stack")
        required = (before["frames"][ROOT] + ARGUMENT_BYTES + 3) // 4 * 4
        require(before["fibers"][FIBER]["size"] >= required, "reference coroutine capacity")
        require(after["fibers"][FIBER]["size"] >= required, "target coroutine capacity")
        if after["frames"] is not None:
            require(after["frames"] == before["frames"], "target stack metadata equivalence")
        stack = dict(root=ROOT, measured_root_frame_bytes=64, argument_area_bytes=ARGUMENT_BYTES,
                     resolver_estimated_bytes=required, reservation_bytes=after["fibers"][FIBER]["size"],
                     capacity_status="NOT_ESTABLISHED",
                     capacity_reason="backend-generated call closure unverified",
                     source_closure_functions=len(closure),
                     mode="target-code-equivalence-to-measured-reference",
                     target_frame_metadata_present=after["frames"] is not None)
    else:
        require(not before["fibers"] and not after["fibers"], "noncoroutine control arm")
    return dict(status="PASS", arm=arm, reference_sha256=reference_sha256,
                target_sha256=after["sha256"], source_sha256=digest(source),
                executable_code_equal=True, function_locations_equal=True, stack=stack,
                target_static_sram_end=after["static_sram_end"],
                scope="pinned tiny program equivalence and reservation evidence; independent coroutine capacity not established")
