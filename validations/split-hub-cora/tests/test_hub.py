import json
from pathlib import Path
import sys
import unittest

import numpy as np

FAMILY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FAMILY))
import hub_contract
import saved_audit as native


class HubContract(unittest.TestCase):
    def test_ordered_relocation_and_original_predictions(self):
        package = json.loads((FAMILY / "package.json").read_text())
        totals = []
        for name in sorted(package["cases"]):
            case = FAMILY / "cases" / name
            m = native.load(case)
            expected, _ = native.independent_expected(case, m)
            words = expected["split_words"].sum(axis=0)
            c, r = m["params"]["C"], m["params"]["R"]
            hops = (46 if (c, r) == (16, 32) else 23) if name.endswith("-split") else 0
            np.testing.assert_array_equal(words, [hops, hops])
            totals.append(int(expected["stats"][:, 4:7].sum()))
        self.assertTrue(totals and len(set(totals)) == 1)

    def test_rejects_a_changed_helper_or_occurrence(self):
        for case in sorted((FAMILY / "cases").glob("*-split")):
            with self.subTest(case=case.name):
                m = native.load(case)
                nodes = native.load_npz(case / "node_ids.npz")["nodes"]
                original = native.native_inputs(case, m)
                arrays = native.load_npz(case / "inputs.npz")
                roles = arrays["split_control"][:, 0]
                helper = int(np.flatnonzero(roles == 2)[0])
                for pe in np.flatnonzero((roles == 2) | (roles == 3)):
                    broken = arrays["split_control"].copy()
                    broken[pe, 0] = 0
                    with self.assertRaises((AssertionError, ValueError)):
                        hub_contract.audit_uploads(m["params"], dict(arrays, split_control=broken), nodes, original)
                weights = arrays["edge_weight"].copy()
                begin = int(arrays["edge_ptr"][helper, -2])
                weights[helper, begin] += 1
                with self.assertRaises((AssertionError, ValueError)):
                    hub_contract.audit_uploads(m["params"], dict(arrays, edge_weight=weights), nodes, original)
