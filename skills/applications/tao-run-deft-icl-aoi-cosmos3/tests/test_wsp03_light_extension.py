# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only fixtures: no provider, target model, Docker or experiment data."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import wsp03_light as light
import wsp03_light_final as final
from wsp03_statistics import paired,PAIRS


def asset(path,value):
    path.write_text(json.dumps(value));return {'path':str(path),'sha256':final.file_hash(path)}


class LightExtension(unittest.TestCase):
    def test_unlimited_scoped_policy_and_transition(self):
        self.assertTrue(light.token_limit_reached(5000000,light.budget_policy({},'owned')))
        self.assertFalse(light.token_limit_reached(99999999,light.budget_policy({'token_target':None},'owned')))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);override=root/'override.json';events=root/'events.jsonl'
            value={'job_id':'owned','token_target':None,'reason':'approved post-hoc change','observed_tokens_at_change':123}
            override.write_text(json.dumps(value));policy=light.budget_policy({'wall_hours':24},'owned',override)
            self.assertEqual(policy['wall_hours'],24)
            light.policy_event(events,light.budget_policy({},'owned'),0,1)
            light.policy_event(events,policy,140,2);light.policy_event(events,policy,160,3)
            rows=[json.loads(line) for line in events.read_text().splitlines()]
            self.assertEqual(len(rows),2);self.assertEqual(rows[-1]['cumulative_tokens_at_change'],123)
            for bad in (dict(value,job_id='other'),dict(value,wall_hours=48),dict(value,token_target=False)):
                override.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):light.budget_policy({},'owned',override)

    def test_four_freezes_and_stopped_editors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=asset(root/'r3.json',{'status':'FROZEN'});r4=asset(root/'r4.json',{'status':'FROZEN'})
            prereg=asset(root/'prereg.json',{'before_own_results':True})
            ext=asset(root/'ext.json',{'status':'FROZEN','preregistration_sha256':prereg['sha256']})
            control=asset(root/'r4c.json',{'status':'FROZEN','profile':'P01','patch_manifest_sha256':'old-patch'})
            cfg={'freezes':{'R3':base,'R4':r4},'r4_selection':{'freeze_sha256':r4['sha256'],'profile':'P02','patch_manifest_sha256':'old-patch'},
                 'editor_containers':{'R3':['one'],'R4':['two'],'R3-ext':['three']},'r4_c_freeze':control,'r3_ext':{'freeze':ext,'preregistration':prereg}}
            with patch.object(final.subprocess,'check_output',return_value='{"Running":false,"Status":"exited"}') as inspect:
                freezes,_=final.freeze_gate(cfg);self.assertEqual(set(freezes),{'R3','R4','R4-C','R3-ext'});self.assertEqual(inspect.call_count,3)
                del cfg['r4_c_freeze']
                with self.assertRaises(ValueError):final.freeze_gate(cfg)
                cfg['r4_c_freeze']=control
            with patch.object(final.subprocess,'check_output',return_value='{"Running":true,"Status":"running"}'):
                with self.assertRaises(RuntimeError):final.freeze_gate(cfg)
            cfg['r3_ext']['preregistration']=asset(root/'other.json',{'changed':True})
            with self.assertRaises(ValueError):final.freeze_gate(cfg)

    def test_distinct_base_relative_patch_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);exports={};freezes={};manifests={}
            for arm,name,count in [('R3','frozen-r3',1),('R3-ext','frozen-r3-ext',2)]:
                export=root/name;(export/'patch').mkdir(parents=True)
                meta=asset(export/'patch/manifest.json',{'valid':True,'changed_elements':count,'tensors':[]})
                exports[arm]=export;manifests[arm]=meta
                freezes[arm]={'reusable_exports':{'/work/'+name+'/patch/manifest.json':meta['sha256']}}
            cfg={'r3_export':str(exports['R3']),'r4_selection':{'patch_manifest_sha256':manifests['R3']['sha256']},
                 'r3_ext':{'export':str(exports['R3-ext']),'patch_manifest_sha256':manifests['R3-ext']['sha256']}}
            original=final.snapshot_patch(cfg,freezes['R3'],root/'original')
            extension=final.snapshot_patch(cfg,freezes['R3-ext'],root/'extension',arm='R3-ext')
            self.assertNotEqual(final.digest(original),final.digest(extension))
            self.assertEqual(json.loads((root/'patch-binding-r3-ext.json').read_text())['arms'],['R3-ext'])

    def test_aliases_keep_original_contract_and_extension_uses_own_patch(self):
        for selected in ('P00','P01','P02'):
            with self.subTest(profile=selected),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);calls=[]
                def evaluate(cfg,recipe,runtime,patch_dir,manifest,output,arm,profile,split):
                    calls.append((arm,profile,split,patch_dir));(output/'arms'/split/arm).mkdir(parents=True)
                    return {'arm':arm,'profile':profile,'split':split,'generation_debit':1,'patch':str(patch_dir)}
                cfg={'r4_selection':{'profile':selected},'r3_ext':{'configured':True}}
                with patch.object(final,'evaluate',side_effect=evaluate):
                    results=final.evaluate_arms(cfg,None,None,Path('original'),{},root,Path('extension'),{})
                self.assertEqual(len(results),8);self.assertEqual(len(json.loads((root/'results.json').read_text())),6)
                self.assertEqual(len(json.loads((root/'results-r3-ext.json').read_text())),2)
                self.assertEqual(len(calls),6 if selected in ('P00','P01') else 8)
                self.assertEqual(len({(arm,split) for arm,_,split,_ in calls}),len(calls))
                self.assertTrue(all(path==Path('extension') and profile=='P00' for arm,profile,_,path in calls if arm=='R3-ext'))
                self.assertTrue(all(path==Path('original') for arm,_,_,path in calls if arm!='R3-ext'))
                if selected in ('P00','P01'):
                    self.assertEqual(results['R4:benchmark']['generation_debit'],0)

    def test_extension_statistics_are_separate_and_default_pairs_unchanged(self):
        class Evaluator:
            @staticmethod
            def score(rows):
                value=sum(r['prediction'] for r in rows)/len(rows)
                return {key:value for key in ('mcc','macro_f1','balanced_accuracy')}
        identities={'a':{'object_name':'part','scan_id':'SCN_1'},'b':{'object_name':'part','scan_id':'SCN_2'}}
        arms={}
        for name,value in [('B',0),('C',.2),('R3',.1),('R4',.3),('R4-C',.2),('R3-ext',.4)]:
            predictions=[{'id':key,'truth':'Good','prediction':value} for key in identities]
            arms[name]={'predictions':predictions,'metrics':Evaluator.score(predictions)}
        original={k:v for k,v in arms.items() if k!='R3-ext'}
        unchanged=paired(original,identities,Evaluator,expected_groups=2)
        self.assertEqual(set(unchanged['pairs']),{a+'__minus__'+b for a,b in PAIRS})
        extension=paired({k:v for k,v in arms.items() if k in ('R3-ext','R3','B','C')},identities,Evaluator,expected_groups=2,
                         pairs=(('R3-ext','C'),('R3-ext','B'),('R3-ext','R3')))
        self.assertEqual(len(extension['pairs']),3)
        self.assertAlmostEqual(extension['pairs']['R3-ext__minus__R3']['mcc']['estimate'],.3)
        self.assertEqual(unchanged,paired(original,identities,Evaluator,expected_groups=2))

if __name__=='__main__':unittest.main()
