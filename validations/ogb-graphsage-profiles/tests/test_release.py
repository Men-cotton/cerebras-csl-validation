import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np


HERE=Path(__file__).resolve().parent
BASE=HERE.parent if HERE.name=="tests" else HERE
sys.path.insert(0,str(BASE))
from contract import check_frame,load,pack
from appliance import checked_case,read


class FrozenCases(unittest.TestCase):
    def test_fixed_inputs_and_timing(self):
        for path in sorted((BASE/"cases").glob("*/release.json")):
            folder,release,native,direct=checked_case(path.parent.name)
            z=load(folder/"fixture.npz")
            self.assertEqual(z["fp64"].shape,z["gpu"].shape)
            self.assertEqual(native["params"],direct["params"])

    def test_modified_output_is_rejected(self):
        for path in sorted((BASE/"cases").glob("*/release.json")):
            case=path.parent;m=read(case/"native/build.json");z=load(case/"fixture.npz")
            params,b=pack(z,m["params"]["P"],m["placement"],m["edge_capacity"],m["params"].get("S",1))
            with np.load(case/"sim-csl/raw.npz",allow_pickle=False) as raw:
                bad=raw["reply1"].copy()
            bad[0,0]=123
            with self.assertRaises((ValueError,AssertionError)):check_frame(bad,z,b,params,1)


if __name__=="__main__":unittest.main()
