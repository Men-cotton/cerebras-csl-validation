"""Mutation checks for the pinned E236 target-compiler admission proof."""
from pathlib import Path
import struct
import unittest

import stack_gate as gate


class StackGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        here = Path(__file__).resolve()
        if here.parent.name == "tests":
            root = here.parents[1]
            cls.references = root / "references"
            cls.generated = root / "generated"
        else:
            root = here.parents[5]
            cls.references = root / "research/experiments/data/E236/arms"
            cls.generated = None

    def fixture(self, arm="async", index=0):
        root = self.references / arm
        reference = next(p.read_bytes() for p in (root / "build/bin").glob("*.elf")
                         if gate.digest(p.read_bytes()) == gate.ELF_PINS[arm][index])
        sources = (self.generated / arm if self.generated else
                   root / ("source" if arm == "direct" else "generated"))
        source = next(p.read_bytes() for p in sources.glob("*.csl")
                      if gate.digest(p.read_bytes()) == gate.SOURCE_PINS[arm][index])
        return reference, dict(arm=arm, reference_sha256=gate.digest(reference), source=source)

    def test_all_pinned_programs_and_frame_requirement(self):
        for arm in gate.ELF_PINS:
            for index in range(4):
                with self.subTest(arm=arm, index=index):
                    reference, args = self.fixture(arm, index)
                    result = gate.compare(reference, reference, **args)
                    self.assertEqual(result["status"], "PASS")
                    self.assertEqual(result["stack"]["resolver_estimated_bytes"], 76 if arm == "async" else 0)
                    self.assertEqual(result["stack"]["capacity_status"], "NOT_ESTABLISHED" if arm == "async" else "NOT_APPLICABLE")

    def test_absent_target_metadata_needs_identical_code(self):
        reference, args = self.fixture()
        target = reference.replace(b".stack_sizes\0", b".stack_other\0", 1)
        self.assertIsNone(gate.inspect(target)["frames"])
        result = gate.compare(reference, target, **args)
        self.assertFalse(result["stack"]["target_frame_metadata_present"])
        self.assertEqual(result["stack"]["resolver_estimated_bytes"], 76)
        _, sections = gate._sections(target)
        mutated = bytearray(target)
        mutated[sections[".text"][4]] ^= 1
        with self.assertRaisesRegex(ValueError, "executable code equivalence"):
            gate.compare(reference, bytes(mutated), **args)

    def test_corrupted_target_stack_metadata_is_rejected(self):
        reference, args = self.fixture()
        _, sections = gate._sections(reference)
        target = bytearray(reference)
        h = sections[".stack_sizes"]
        target[h[4] + h[5] - 1] = 128
        with self.assertRaisesRegex(ValueError, "truncated stack-size ULEB128"):
            gate.compare(reference, bytes(target), **args)

    def test_undersized_target_reservation_is_rejected(self):
        reference, args = self.fixture()
        headers, sections = gate._sections(reference)
        h = sections[".symtab"]
        st = headers[h[6]]
        strings = reference[st[4]:st[4] + st[5]]
        target = bytearray(reference)
        for pos in range(h[4], h[4] + h[5], 24):
            name = struct.unpack_from("<I", reference, pos)[0]
            if gate._string(strings, name) == gate.FIBER:
                struct.pack_into("<Q", target, pos + 16, 72)
                break
        else:
            self.fail("fixture has no coroutine stack")
        with self.assertRaisesRegex(ValueError, "coroutine allocation equivalence"):
            gate.compare(reference, bytes(target), **args)

    def test_reference_source_and_elf_pins_cannot_be_relabelled(self):
        reference, args = self.fixture()
        with self.assertRaisesRegex(ValueError, "source identity"):
            gate.compare(reference, reference, **dict(args, source=args["source"] + b"\n"))
        with self.assertRaisesRegex(ValueError, "reference ELF identity"):
            gate.compare(reference + b"x", reference, **args)
        with self.assertRaisesRegex(ValueError, "pinned E236 reference"):
            gate.compare(reference, reference, **dict(args, arm="frontend"))

    def test_initialized_data_change_and_truncated_elf_rejected(self):
        reference, args = self.fixture()
        _, sections = gate._sections(reference)
        target = bytearray(reference)
        target[sections[".data"][4]] ^= 1
        with self.assertRaisesRegex(ValueError, "initialized data"):
            gate.compare(reference, bytes(target), **args)
        for malformed in (b"", reference[:63], reference[:100]):
            with self.subTest(size=len(malformed)), self.assertRaises(ValueError):
                gate.inspect(malformed)

    def test_invalid_frame_records_rejected(self):
        for malformed in (b"\0", b"\0\0", b"\0\0\x80", b"\0\0" + b"\x80" * 5):
            with self.subTest(record=malformed), self.assertRaises(ValueError):
                gate._frames(malformed, {0: ["root"]})

    def test_load_segment_mutations_rejected_despite_matching_sections(self):
        reference, args = self.fixture()
        offset = struct.unpack_from("<Q", reference, 32)[0]
        size, count = struct.unpack_from("<HH", reference, 54)
        low = next(pos for pos in range(offset, offset + size * count, size)
                   if struct.unpack_from("<I", reference, pos)[0] == 1
                   and struct.unpack_from("<Q", reference, pos + 16)[0] < 48 * 1024)
        for field, width in ((4, "I"), (8, "Q"), (16, "Q"), (24, "Q"), (48, "Q")):
            with self.subTest(field=field):
                target = bytearray(reference)
                value = struct.unpack_from("<" + width, target, low + field)[0]
                struct.pack_into("<" + width, target, low + field, value ^ 1)
                with self.assertRaisesRegex(ValueError, "loaded program equivalence"):
                    gate.compare(reference, bytes(target), **args)

    def test_physical_tile_relocation_preserves_stack_proof(self):
        reference, args = self.fixture()
        target = bytearray(reference)
        offset = struct.unpack_from("<Q", target, 32)[0]
        size, count = struct.unpack_from("<HH", target, 54)
        for pos in range(offset, offset + size * count, size):
            if struct.unpack_from("<I", target, pos)[0] == 1:
                old = struct.unpack_from("<Q", target, pos + 24)[0]
                struct.pack_into("<Q", target, pos + 24, (old & ~(65535 << 40)) | (767 << 40))
        self.assertEqual(gate.inspect(bytes(target))["tile_indices"], [767])
        self.assertEqual(gate.compare(reference, bytes(target), **args)["status"], "PASS")

    def test_high_configuration_load_descriptors_cannot_redirect_to_code(self):
        reference, args = self.fixture()
        offset = struct.unpack_from("<Q", reference, 32)[0]
        size, count = struct.unpack_from("<HH", reference, 54)
        high = next(pos for pos in range(offset, offset + size * count, size)
                    if struct.unpack_from("<I", reference, pos)[0] == 1
                    and struct.unpack_from("<Q", reference, pos + 16)[0] >= 48 * 1024)
        target = bytearray(reference)
        old = struct.unpack_from("<Q", target, high + 24)[0]
        struct.pack_into("<Q", target, high + 24, old ^ 1)
        with self.assertRaisesRegex(ValueError, "all load descriptors"):
            gate.compare(reference, bytes(target), **args)

    def test_entrypoint_and_architecture_flags_cannot_change(self):
        reference, args = self.fixture()
        for offset, fmt in ((24, "Q"), (48, "I")):
            with self.subTest(offset=offset):
                target = bytearray(reference)
                old = struct.unpack_from("<" + fmt, target, offset)[0]
                struct.pack_into("<" + fmt, target, offset, old ^ 1)
                with self.assertRaisesRegex(ValueError, "executable header"):
                    gate.compare(reference, bytes(target), **args)


if __name__ == "__main__":
    unittest.main()
