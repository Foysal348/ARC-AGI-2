import json,ast,runpy,time,tempfile,os,types,sys,hashlib,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb'
NB=ROOT/'Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R3.ipynb'
base=json.loads(BASE.read_text());n=json.loads(NB.read_text());assert len(n['cells'])==9
for i,c in enumerate(n['cells']):
 if c['cell_type']=='code':ast.parse(''.join(c['source']))
assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='530cb119e302899426a2e3f24c5e100279e7167aef77525b6cb93b90a7fc10c0'
assert ''.join(n['cells'][1]['source'])==''.join(base['cells'][1]['source'])
assert ''.join(n['cells'][3]['source'])==''.join(base['cells'][3]['source'])
assert ''.join(n['cells'][6]['source'])==''.join(base['cells'][6]['source'])
assert '77,824' in ''.join(n['cells'][0]['source'])
print('PASS base freeze/source parity, 9-cell notebook syntax')
core=runpy.run_path(str(ROOT/'poetiq_core.py'))
train={'train':[{'input':[[1,0],[1,1]],'output':[[2,0],[2,2]]}],
       'test':[{'input':[[1,1],[0,0]]},{'input':[[0,1],[1,0]]}]}
assert 'Challenge #2' in core['poetiq_format_problem'](train)
assert core['poetiq_parse_code']('hello\n```python\nimport numpy as np\ndef transform(grid):return grid\n```')
assert core['poetiq_parse_code']('not code') is None
assert all('PROBLEM' in p for p in core['POETIQ_SOLVER_PROMPTS'].values())
assert 'EXISTING PARTIAL' in core['POETIQ_FEEDBACK_PROMPT']
print('PASS Poetiq task-level formatting, prompt variants and extraction')
BAD='import numpy as np\ndef transform(grid):\n    return grid\n'
GOOD='import numpy as np\ndef transform(grid):\n    return np.where(grid==1,2,grid)\n'
resbad=core['poetiq_verify_code'](BAD,train,5.0)
assert resbad['passed']==False and resbad['score']<1 and 'prediction/correct' in resbad['feedback']
resgood=core['poetiq_verify_code'](GOOD,train,5.0)
assert resgood['passed']==True and resgood['score']==1
assert resgood['test_results'][0]['grid']==[[2,2],[0,0]]
assert core['poetiq_sandbox_run']('def transform(grid):\n    raise RuntimeError("oops")',[[1]],4.0)[0] is False
assert core['poetiq_sandbox_run']('def transform(grid):\n    return [[1,2],[3]]',[[1]],4.0)[0] is True
print('PASS actual subprocess program execution, error and train-grid feedback')
R=lambda passed,soft,grids,expert: {'passed':passed,'train_results':[{'soft_score':soft}],
    'test_results':[{'grid':g} for g in grids],'iteration':expert,'expert':expert}
r1=R(True,1.,[[[1]],[[2]]],1);r2=R(True,1.,[[[1]],[[2]]],2)
r3=R(True,1.,[[[5]],[[2]]],3);r4=R(False,.8,[[[4]],[[2]]],4)
ranked=core['poetiq_vote']([r1,r2,r3,r4]);assert ranked[0] is r1 and ranked[1] is r3
at=core['poetiq_two_attempts'](train,ranked)
assert at[0]['attempt_1']==[[1]] and at[0]['attempt_2']==[[5]]
assert all(set(row)=={'attempt_1','attempt_2'} for row in at)
print('PASS Poetiq votes, diversity-first ranking, two-attempt mapping')

# Fake vLLM end-to-end with real Poetiq subprocess evaluator.
class MockTokenizer:
 def encode(self,text,add_special_tokens=False):return [4 if 'EXISTING PARTIAL/INCORRECT' in text else 3,5,6]
 def apply_chat_template(self,msgs,**kwargs):return msgs[0]['content']
 def decode(self,ids,skip_special_tokens=True):
  if 778 in ids:return '```python\n'+GOOD+'\n```'
  if 777 in ids:return '```python\n'+BAD+'\n```'
  return ''
class MockOutput:
 def __init__(self,rid,tokens):
  self.request_id=rid;self.finished=True
  self.outputs=[types.SimpleNamespace(token_ids=tokens,finish_reason='stop')]
class MockEngine:
 def __init__(self):self.pending={};self.ids=[]
 def add_request(self,rid,prompt,params):
  if rid in self.pending:raise AssertionError('Duplicate live request')
  self.pending[rid]=prompt;self.ids.append(rid)
 def step(self):
  if not self.pending:return []
  rid,prompt=next(iter(self.pending.items()));self.pending.pop(rid)
  return [MockOutput(rid,[11] if rid.endswith('|1') else [778 if '~i1|' in rid else 777])]
 def abort_request(self,ids):
  for rid in ids:self.pending.pop(rid,None)
with tempfile.TemporaryDirectory() as td:
 g={'OUT':Path(td),'WORKING':Path(td),'SUBMISSION_PATH':Path(td)/'submission.json',
    'test_challenges':{'taskA':train},'CHALLENGES':{'taskA':train},'SOLUTIONS':{'taskA':[[[2,2],[0,0]],[[0,2],[2,0]]]},
    'TASK_IDS':['taskA'],'NOTEBOOK_T0':time.perf_counter(),'CONCURRENCY':2,'SEED':10,
    'POETIQ_SEED':0,'POETIQ_NUM_EXPERTS':1,'POETIQ_MAX_ITERATIONS':2,'POETIQ_MAX_SOLUTIONS':5,
    'POETIQ_SOLVER_VARIANTS':(1,),'POETIQ_SELECTION_PROB':1.,'POETIQ_IMPROVING_ORDER':True,
    'POETIQ_SHUFFLE_EXAMPLES':True,'POETIQ_SANDBOX_TIMEOUT':5.,'POETIQ_SANDBOX_WORKERS':2,
    'POETIQ_SOLVER_TEMPERATURE':1.,'POETIQ_USE_NEW_VOTING':True,
    'POETIQ_COUNT_FAILED_MATCHES':True,'POETIQ_ITERS_TIEBREAK':False,'POETIQ_LOW_TO_HIGH_ITERS':False,
    'SAMPLING':{'top_p':.95,'top_k':20},'THINK_CLOSE_ID':99,'STOP_IDS':[98,99],'EOS_IDS':set(),
    'MAX_MODEL_LEN':106496,'THINK_BUDGET':77824,'ANSWER_MAX_TOKENS':8192,
    'DRAIN_ANSWER_MAX_TOKENS':8192,'MIN_THINK_BUDGET':4096,
    'REASONING_EFFORT':'xhigh','RERUN':True,'ENGINE_READY':True,
    'HARD_STOP':time.perf_counter()+40,'DRAIN_DEADLINE':time.perf_counter()+35,
    'ADMIT_DEADLINE':time.perf_counter()+30,
    'tokenizer':MockTokenizer(),'SamplingParams':lambda **kw:kw,
    'RequestOutputKind':types.SimpleNamespace(DELTA='delta'),'engine':MockEngine(),
    'identity_submission':lambda z:{k:[{'attempt_1':v['input'],'attempt_2':v['input']} for v in task['test']] for k,task in z.items()},
    'write_json_atomic':lambda p,obj:Path(p).write_text(json.dumps(obj)),
    'metrics_snapshot':lambda:[],
    'phase1_cap':lambda prompt_len:77824,
    'time':time,'json':json,'re':__import__('re'),'traceback':__import__('traceback'),
    'os':os,'sys':sys,'np':core['np'], 'SamplingParams':lambda **kw:kw}
 g.update({k:v for k,v in core.items() if not k.startswith('__')})
 exec((ROOT/'poetiq_scheduler.py').read_text(),g)
 state=g['E10_STATES'][('taskA',0)]
 assert state['iteration']==2, state
 assert state['status']=='train_verified',state
 assert g['E10_FINAL']['taskA'][0]['attempt_1']==[[2,2],[0,0]],g['E10_FINAL']
 assert g['E10_FINAL']['taskA'][1]['attempt_1']==[[0,2],[2,0]],g['E10_FINAL']
 assert g['E10_SCHED']['train_passed']==1
 assert len(g['engine'].ids)==4, g['engine'].ids
 assert json.loads((Path(td)/'submission.json').read_text())['taskA']==g['E10_FINAL']['taskA']
 print('PASS fake-engine iterative generation→train feedback→refinement→verified grids→submission')
print('ALL E10 CPU TESTS PASSED')
