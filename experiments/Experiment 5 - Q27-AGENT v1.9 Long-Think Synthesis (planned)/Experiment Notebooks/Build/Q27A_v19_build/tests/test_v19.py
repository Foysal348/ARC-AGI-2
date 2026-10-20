import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'modules'))
import synth
import run_eval_v19
import submit_main_v19
from rescue import load_direct_ref

IDENT={'train':[{'input':[[1,2],[3,4]],'output':[[1,2],[3,4]]}],
       'test':[{'input':[[2,3],[4,5]]}]}
WRONG={'train':[{'input':[[1,2],[3,4]],'output':[[9,9],[9,9]]}],
       'test':[{'input':[[2,3],[4,5]]}]}
CODE='```python\ndef transform(grid):\n    return [row[:] for row in grid]\n```'
CFG={'agent_mode':'synth','synth_think':49152,'synth_answer':8192,'repair_turns':0,
     'base_url':'http://127.0.0.1:19235/v1','model':'qwen','max_model_len':81920,
     'deadline_epoch':time.time()+600,'synth_commit':True,
     'template_kwargs':{'reasoning_effort':'xhigh'},'thinking':True,'eval_direct':'ref',
     'server_ladder':[['--reasoning-config','{}']],
     'rescue_think':49152,'rescue_answer':8192,'rescue_min_think':49152,'rescue_effort':'xhigh'}

class TestV19(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import socket
        import urllib.request
        sock=socket.socket()
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
        sock.close()
        CFG['base_url']=f'http://127.0.0.1:{port}/v1'
        cls.server=subprocess.Popen([sys.executable,str(ROOT/'modules'/'fake_vllm.py'),str(port)],
                                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for _ in range(60):
            if cls.server.poll() is not None:
                raise RuntimeError(f'fake server exited: {cls.server.returncode}')
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=1)
                break
            except Exception:time.sleep(.1)
        else:
            cls.server.kill()
            raise RuntimeError('fake server not started')
    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        try:cls.server.wait(timeout=5)
        except subprocess.TimeoutExpired:cls.server.kill()

    def test_parse_fenced_and_no_prose(self):
        self.assertEqual(synth.extract_program(CODE)['status'],'program')
        self.assertEqual(synth.extract_program('I propose def transform(grid): return grid')['status'],'no_program')
        self.assertEqual(synth.extract_program('```python\ndef x():\n    def transform(grid):\n        return grid\n```')['status'],'no_program')
        self.assertEqual(synth.extract_program('```python\ndef transform(grid):\n    return\nsyntax ???\n```')['status'],'no_program')
        self.assertEqual(synth.extract_program('```python\ndef transform(grid):\n  return train_inputs[0]\n```')['status'],'nonportable')
        self.assertEqual(synth.extract_program('```python\nimport os\ndef transform(grid):\n  return grid\n```')['status'],'unsafe_program')
        self.assertEqual(synth.extract_program('```python\ndef transform(grid):\n  return open("/kaggle/input/solutions.json")\n```')['status'],'unsafe_program')

    def test_config_and_primary_contract(self):
        from check_config import validate
        self.assertTrue(validate(dict(CFG)))
        for key,value in [('synth_think',6144),('repair_turns',1),('force_submit_after',5)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate(dict(CFG,**{key:value}))
        p=synth.strict_payload('qwen',[{'role':'user','content':'x'}],CFG)
        self.assertEqual((p['thinking_token_budget'],p['max_tokens']),(49152,57344))
        self.assertEqual(p['chat_template_kwargs']['reasoning_effort'],'xhigh')
        self.assertNotIn('tools',p)
        self.assertNotIn('tool_choice',p)
        c=synth.strict_payload('qwen',[{'role':'user','content':'x'}],CFG,thinking=False)
        self.assertNotIn('thinking_token_budget',c)
        self.assertEqual(c['max_tokens'],8192)
        self.assertFalse(c['chat_template_kwargs']['enable_thinking'])

    def test_oversize_does_not_reduce_thinking(self):
        fn=lambda *a,**k:(30000,'forced_overflow')
        r=synth.synthesize_one('oversized',IDENT,CFG,count_fn=fn)
        self.assertEqual(r['status'],'context_too_large')
        self.assertEqual(r['gen_tokens'],0)

    def test_budget_rejection_never_retries_without_limit(self):
        n=[]
        def fake(url,payload,timeout):
            n.append(payload)
            return None,{'code':400,'message':'thinking_token_budget parameter is unavailable'}
        with patch.object(synth,'_post',side_effect=fake):
            r=synth._request('http://localhost:1/v1',synth.strict_payload('qwen',[{'role':'user','content':'x'}],CFG),10,retry_transient=True)
        self.assertEqual(r['status'],'budget_rejected')
        self.assertEqual(len(n),1)
        self.assertEqual(n[0]['thinking_token_budget'],49152)

    def test_fatal_500_only_one_retry(self):
        n=[]
        def fake(url,payload,timeout):
            n.append(1)
            return None,{'code':500,'message':'server error'}
        with patch.object(synth,'_post',side_effect=fake),patch.object(synth.time,'sleep',return_value=None):
            r=synth._request('http://localhost:1/v1',{'model':'qwen'},10,retry_transient=True)
        self.assertEqual(r['status'],'request_error')
        self.assertEqual(len(n),2)

    def test_real_fake_server_synth_verifies(self):
        r=synth.synthesize_one('identity',IDENT,CFG)
        self.assertEqual(r['status'],'verified_usable')
        self.assertEqual(r['n_ok'],1)
        self.assertEqual(r['test'],[[[2,3],[4,5]]])
        self.assertEqual(r['program_source'],'primary')

    def test_real_fake_server_wrong_train(self):
        r=synth.synthesize_one('wrong',WRONG,CFG)
        self.assertEqual(r['status'],'train_mismatch')
        self.assertEqual(r['n_ok'],0)
        self.assertFalse(r['train_verified'])

    def test_commit_fallback_once_and_no_budget(self):
        replies=[{'status':'ok','content':'No program','finish_reason':'length','completion_tokens':49152,'usage_known':True},
                 {'status':'ok','content':CODE,'finish_reason':'stop','completion_tokens':800,'usage_known':True}]
        with patch.object(synth,'_request',side_effect=replies) as rq:
            r=synth.synthesize_one('commit',IDENT,CFG,count_fn=lambda *a,**k:(250,'dummy'))
        self.assertEqual(r['status'],'verified_usable')
        self.assertEqual(r['program_source'],'commit')
        self.assertEqual(rq.call_count,2)
        second=rq.call_args_list[1].args[1]
        self.assertNotIn('thinking_token_budget',second)
        self.assertFalse(second['chat_template_kwargs']['enable_thinking'])

    def test_merge_preserves_recorded_direct(self):
        ref=load_direct_ref(ROOT/'modules'/'direct_ref_eval12.json','run7')
        task={'136b0064':IDENT}
        records=[{'task_id':'136b0064','status':'verified_usable','test':[[[8]]]}]
        _,direct,merged=run_eval_v19.make_submissions(task,records,ref)
        expected=ref['136b0064']['grids'][0]
        self.assertEqual(merged['136b0064'][0]['attempt_1'],expected)
        self.assertEqual(merged['136b0064'][0]['attempt_2'],[[8]])
        self.assertEqual(direct['136b0064'][0]['attempt_1'],expected)

    def test_report_end_to_end_no_solution_in_inference(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'source').mkdir()
            ids=['136b0064','247ef758','2b83f449']
            challenge={tid:IDENT for tid in ids}
            sol={tid:[[[2,3],[4,5]]] for tid in ids}
            (root/'source'/'arc-agi_evaluation_challenges.json').write_text(json.dumps(challenge))
            (root/'source'/'arc-agi_evaluation_solutions.json').write_text(json.dumps(sol))
            # Ensure scoring cannot start prior to all synthesis returns.
            config=dict(CFG,data_dir=str(root),eval_out=str(root/'eval'),eval_task_ids=ids,
                        metrics_every_s=60,concurrency=3)
            rep=run_eval_v19.evaluate(config)
            self.assertEqual(rep['tasks'],3)
            self.assertEqual(rep['test_outputs'],3)
            self.assertEqual(rep['agent_query_hits'],3)
            self.assertIn('official_score_pct',rep)
            self.assertEqual(len((root/'eval'/'attempts.jsonl').read_text().splitlines()),3)
            self.assertTrue((root/'eval'/'eval_submission.json').exists())
            self.assertTrue((root/'eval'/'progress.json').exists())

    def test_hidden_direct_only_fallback_path(self):
        from run_submit import main as hidden_main
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=root/'src';source.mkdir()
            ch={'136b0064':IDENT, '247ef758':IDENT}
            (source/'arc-agi_test_challenges.json').write_text(json.dumps(ch))
            sub=root/'submission.json'
            cfg=dict(CFG,eval=False,data_dir=str(root),sub_path=str(sub),
                     agent_disabled=True,rescue=True,rescue_min_start_s=1,
                     rescue_concurrency=2,concurrency=2,metrics_every_s=60,
                     rescue_log=str(root/'rescue_attempts.jsonl'),
                     metrics_path=str(root/'server_metrics.json'),
                     deadline_epoch=time.time()+300,interactive_tasks=2)
            hidden_main(cfg)
            data=json.loads(sub.read_text())
            self.assertEqual(set(data),set(ch))
            for tid,rows in data.items():
                self.assertEqual(len(rows),1)
                for k in ('attempt_1','attempt_2'):
                    self.assertIsInstance(rows[0][k],list)
            self.assertTrue((root/'rescue_attempts.jsonl').is_file())

    def test_server_contract_smoke(self):
        x=submit_main_v19.budget_contract_smoke(CFG)
        self.assertTrue(x['ok'],x)
        self.assertTrue(x['synth_think_retained'])

    # ---------------- v1.9.1 regression tests ----------------
    def test_v191_rescue_budget_guard(self):
        from check_config import validate
        from rescue import time_budget
        for key,value in [('rescue_min_think',4096),('rescue_think',32768),('rescue_answer',4096),('rescue_effort','low')]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate(dict(CFG,**{key:value}))
        now=time.time()
        # plenty of time -> the full budget; too little -> None (not started), never a smaller budget
        self.assertEqual(time_budget(49152,now+7200,15.0,4096,120.0,49152),49152)
        self.assertIsNone(time_budget(49152,now+3000,15.0,4096,120.0,49152))

    def test_v191_timeout_never_retried(self):
        n=[]
        def fake(url,payload,timeout):
            n.append(timeout)
            return None,{'code':None,'message':'TimeoutError: timed out'}
        with patch.object(synth,'_post',side_effect=fake),patch.object(synth.time,'sleep',return_value=None):
            r=synth._request('http://localhost:1/v1',{'model':'qwen'},3000,retry_transient=True,deadline=time.time()+3010)
        self.assertEqual(r['status'],'request_timeout')
        self.assertEqual(len(n),1)

    def test_v191_retry_bounded_by_deadline(self):
        n=[]
        def fake(url,payload,timeout):
            n.append(timeout)
            return None,{'code':503,'message':'busy'}
        with patch.object(synth,'_post',side_effect=fake),patch.object(synth.time,'sleep',return_value=None):
            synth._request('http://localhost:1/v1',{'model':'qwen'},3000,retry_transient=True,deadline=time.time()+1000)
            self.assertEqual(len(n),2)
            self.assertLessEqual(n[1],1000)
            n.clear()
            r=synth._request('http://localhost:1/v1',{'model':'qwen'},3000,retry_transient=True,deadline=time.time()+40)
            self.assertEqual(len(n),1)
            self.assertEqual(r.get('retry_skipped'),'deadline')

    def test_v191_think_accounting(self):
        a=synth.think_accounting({'reasoning_tokens':None,'completion_tokens':50000,'content':'x'},'u','m',
                                 count_text=lambda *a:848)
        self.assertEqual((a['think_tokens'],a['think_source'],a['think_hit_budget']),(49152,'derived',True))
        b=synth.think_accounting({'reasoning_tokens':None,'completion_tokens':30000,'content':'x'},'u','m',
                                 count_text=lambda *a:1000)
        self.assertEqual((b['think_tokens'],b['think_hit_budget']),(29000,False))
        c=synth.think_accounting({'reasoning_tokens':None,'completion_tokens':30000,'content':'x'},'u','m',
                                 count_text=lambda *a:None)
        self.assertEqual((c['think_tokens'],c['think_source']),(None,'unknown'))

    def test_v191_official_is_micro(self):
        # task A: 2 test outputs, 1 right; task B: 1 output, wrong -> micro 1/3 = 33.33, task average 25.0
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'s').mkdir()
            A={'train':IDENT['train'],'test':[{'input':[[2,3],[4,5]]},{'input':[[1]]}]}
            ch={'136b0064':A,'247ef758':IDENT}
            sol={'136b0064':[[[2,3],[4,5]],[[7]]],'247ef758':[[[9,9],[9,9]]]}
            (root/'s'/'arc-agi_evaluation_challenges.json').write_text(json.dumps(ch))
            (root/'s'/'arc-agi_evaluation_solutions.json').write_text(json.dumps(sol))
            with patch.object(run_eval_v19,'load_direct_ref',return_value={}):
                rep=run_eval_v19.evaluate(dict(CFG,data_dir=str(root),eval_out=str(root/'e'),
                                               eval_task_ids=list(ch),metrics_every_s=60,concurrency=2))
            self.assertEqual(rep['test_outputs'],3)
            self.assertEqual(rep['agent_query_hits'],1)
            self.assertAlmostEqual(rep['agent_only_official_pct'],33.33,places=2)
            self.assertAlmostEqual(rep['agent_task_macro_pct'],25.0,places=2)
            self.assertIn('think',rep)

if __name__=='__main__':unittest.main(verbosity=2)
