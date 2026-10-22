"""Replay a genuine LOCAL SIM capture and reject semantic/identity mutations."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
import mesh_audit as audit

ROOT = Path(__file__).resolve().parent


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.capture = self.root/'capture'
        shutil.copytree(ROOT/'mesh_local_evidence/capture', self.capture)
        shutil.copyfile(ROOT/'mesh_local_evidence/request.json',self.root/'request.json')
        self.request = audit.read_json(self.root/'request.json')
        self.package = ROOT/'mesh_cases'

    def run_audit(self):
        return audit.audit_capture(self.package,'small-colored',self.capture,self.request)

    def change_execution(self, change):
        p=self.capture/'execution.json'; e=audit.read_json(p);change(e);p.write_text(json.dumps(e)+'\n')

    def change_raw(self, field, change):
        p=self.capture/'raw.npz'; values={k:v.copy() for k,v in audit.arrays(p).items()}
        change(values[field]); np.savez(p,**values)
        self.change_execution(lambda e:e.update(raw_sha256=audit.digest(p)))

    def test_local_sim_capture_and_oracle(self):
        result=self.run_audit()
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['backend'],'local-sim')
        self.assertEqual(len(result['checkpoints']),2)
        self.assertFalse(list((self.package/'oracle').rglob('__pycache__')))

    def test_run_identity(self):
        self.change_execution(lambda e:e.update(run_id='00000000-0000-0000-0000-000000000000'))
        with self.assertRaisesRegex(ValueError,'capture binding: run_id'):self.run_audit()

    def test_request_bytes(self):
        with (self.root/'request.json').open('a') as f:f.write(' ')
        with self.assertRaisesRegex(ValueError,'request byte identity'):self.run_audit()

    def test_compiled_argument(self):
        p=self.capture/'out_rpc.json';r=audit.read_json(p)
        next(s for s in r['rpc_symbols'] if s['name']=='run')['inputs'][0]['type']='u32'
        p.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'single u16'):self.run_audit()

    def test_compiled_round_count(self):
        p=self.capture/'out.json';r=audit.read_json(p);r['params']['Q']='99';p.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'compiled parameters differ'):self.run_audit()

    def test_sender_numeric_after_rehash(self):
        self.change_raw('chunk0_sender_cache',lambda x:x.__setitem__((0,0),x[0,0]+10))
        with self.assertRaises(AssertionError):self.run_audit()

    def test_velocity_numeric_after_rehash(self):
        self.change_raw('chunk0_velocity',lambda x:x.__setitem__((0,0),x[0,0]+1))
        with self.assertRaises(AssertionError):self.run_audit()

    def test_comm_generation_after_rehash(self):
        self.change_raw('chunk0_comm_stats',lambda x:x.__setitem__((0,9),x[0,9]+1))
        with self.assertRaises((ValueError,AssertionError)):self.run_audit()

    def test_late_host_upload(self):
        self.change_execution(lambda e:e['api'].insert(-1,'h2d:velocity'))
        with self.assertRaises((ValueError,AssertionError)):self.run_audit()

    def change_package(self):
        self.package=self.root/'mesh_cases';shutil.copytree(ROOT/'mesh_cases',self.package)
        return audit.read_json(self.package/'manifest.json')

    def test_missing_oracle_dependency(self):
        m=self.change_package();m['files'].pop('oracle/models/meshgraphnet/colored_validation.py')
        (self.package/'manifest.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'exact frozen Python inventory'):audit.verify_package(self.package)

    def test_csl_origin_binding(self):
        m=self.change_package();case=self.package/'cases/small-colored/case.json';c=audit.read_json(case)
        c['sources']['kernels/communication/sparse-halo/pe.csl']='0'*64
        case.write_text(json.dumps(c));m['files']['cases/small-colored/case.json']=audit.digest(case)
        m['cases']['small-colored']['case_sha256']=audit.digest(case)
        (self.package/'manifest.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'prepared/source binding'):audit.verify_package(self.package)

    def test_matched_capture_comparison(self):
        report=self.run_audit()
        records=[dict(case_name='small-colored',run_id=name,capture_dir=self.capture,report=report) for name in ('a','b')]
        result=audit.compare_runs(self.package,records)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['comparisons'][0]['local_interval_ratio'],[1.,1.])
        self.assertFalse(result['complete_transport_set'])

    def test_matched_capture_rejects_changed_state(self):
        report=self.run_audit(); second=self.root/'second';shutil.copytree(self.capture,second)
        values={k:v.copy() for k,v in audit.arrays(second/'raw.npz').items()}
        values['chunk0_h'][0,0]+=1;np.savez(second/'raw.npz',**values)
        records=[dict(case_name='small-colored',run_id='a',capture_dir=self.capture,report=report),
                 dict(case_name='small-colored',run_id='b',capture_dir=second,report=report)]
        with self.assertRaises(AssertionError):audit.compare_runs(self.package,records)

    def test_request_schema(self):
        self.request['schema']=True
        with self.assertRaisesRegex(ValueError,'request schema'):self.run_audit()


if __name__=='__main__':unittest.main()
