import struct
import sys
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent if HERE.name=="tests" else HERE
sys.path.insert(0,str(ROOT))
from physical_compare import load_digest


class LoadedImage(unittest.TestCase):
    def test_load_byte_change_and_container_only_change(self):
        roots=[ROOT/"csl-r16/out/bin",ROOT.parent/"cora-resident-device/csl-r16/out/bin"]
        image=next(p for root in roots if root.is_dir() for p in root.glob("*.elf"))
        original=image.read_bytes()
        expected=load_digest(original)
        # Additional non-load evidence can differ between compiler containers.
        self.assertEqual(load_digest(original+b"debug receipt"),expected)
        start=struct.unpack_from("<Q",original,32)[0]
        stride,count=struct.unpack_from("<HH",original,54)
        for j in range(count):
            kind,flags,off,virt,phys,size,span,align=struct.unpack_from("<IIQQQQQQ",original,start+j*stride)
            if kind==1 and size:
                changed=bytearray(original)
                changed[off]^=1
                self.assertNotEqual(load_digest(changed),expected)
                break
        else:
            self.fail("simulator fixture has no nonempty LOAD")


if __name__ == "__main__":
    unittest.main()
