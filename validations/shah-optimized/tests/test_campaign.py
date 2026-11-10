"""Check the batch's comparison contracts before it can request hardware."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import campaign
import driver
import appliance
import numpy as np
from argparse import Namespace

class CampaignTests(unittest.TestCase):
    def test_manifest_bindings_bounds_and_repeat_counts(self):
        total={k:0 for k in ('compile','run','audit')}
        for axis,cases in campaign.configurations().items():
            request=campaign.manifest(axis,cases)
            builds=set();outputs=set();runs={}
            for step in request['steps']:
                action,*args=step['args'];total[action]+=1
                parsed=appliance.parser().parse_args(step['args'])
                self.assertNotIn(str(parsed.out),outputs);outputs.add(str(parsed.out))
                if action=='compile':
                    appliance.admitted_config(parsed)
                    builds.add(str(parsed.out)+'/build.json')
                else:
                    self.assertIn(str(parsed.artifact_record),builds)
                    if action=='run':
                        key=(str(parsed.artifact_record),parsed.output_mode)
                        runs[key]=runs.get(key,0)+1
            self.assertTrue(all(n==3 for n in runs.values()))
        self.assertEqual(total,dict(compile=33,run=111,audit=111))

    def test_worker_cap_is_one_day(self):
        parsed=appliance.parser().parse_args(['run','--artifact-record','record.json','--out','capture'])
        self.assertEqual(parsed.host_timeout,86400)
        self.assertIn('86400s',appliance.host_command('hardware',parsed.host_timeout))
        with self.assertRaises(ValueError): appliance.host_command('hardware',86401)

    def test_dispatch_and_controlled_admission(self):
        for workload in ('balanced','paired-skew'):
            cfg=dict(R=4,F=2,V=4,Q=2,B=2,degree=2,seed=7,channels=4,empty=False,workload=workload)
            case=driver.make_case(appliance.admitted_config(Namespace(**cfg)))
            self.assertEqual(case['params']['K'],16)
            self.assertEqual(case['metrics']['max_to_mean_worker_work'],1 if workload=='balanced' else 2)
            for bad in (dict(empty=True),dict(degree=1000000),dict(R=3,workload='paired-skew')):
                with self.assertRaises(ValueError): appliance.admitted_config(Namespace(**{**cfg,**bad}))

if __name__=='__main__': unittest.main()
