"""R3: user-approved test-validity refinement, historical attempt-2 and OS sandbox.

Run after build_e10.py. No GPU or Kaggle dependencies. All checks are adversarial.
"""
import ast
import hashlib
import json
import os
import runpy
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
core = runpy.run_path(str(HERE / 'poetiq_core.py'))
run = core['poetiq_sandbox_run']
verify = core['poetiq_verify_code']
assert len(core['POETIQ_SOLVER_PROMPTS']) == 3

task = {'train': [{'input': [[0]], 'output': [[1]]}],
        'test': [{'input': [[9]]}]}

# 1. Valid train output, exception on unseen test input. No test ground truth.
bad_test = '''def transform(grid):
 if int(grid[0][0]) == 9:
  raise ValueError('unhandled size/pattern')
 return [[1]]
'''
res = verify(bad_test, task, 5)
assert res['passed'] and not res['test_usable'] and not res['verified_usable']
assert 'TEST EXECUTION / FORMAT ERRORS' in res['feedback']
assert 'Challenge #1' in res['feedback']
print('PASS train-pass/test-exception -> refinement feedback')

# 2. Exactly train-pass, invalid 31-row test grid.
invalid_shape = '''def transform(grid):
 if int(grid[0][0]) == 9:
  return [[1] for _ in range(31)]
 return [[1]]
'''
res = verify(invalid_shape, task, 5)
assert res['passed'] and not res['test_usable']
assert 'Challenge #1' in res['feedback']
print('PASS train-pass/31-row test output -> refinement feedback')

# 3. Success must retain original Poetiq early stop when ALL outputs are valid.
good = '''def transform(grid):
 return [[1]]
'''
res = verify(good, task, 5)
assert res['passed'] and res['test_usable'] and res['verified_usable']
print('PASS train-pass/usable-test -> original early success')

# 4. Linux namespaces + chroot prohibit host data access even with model code.
escape = Path('/mnt/data/E10_R3_OUTSIDE_SANDBOX_MARKER')
if escape.exists(): escape.unlink()
attacks = [
 ('outside-write', f'''def transform(grid):
 open({str(escape)!r},'w').write('escaped')
 return [[1]]'''),
 ('outside-read', '''def transform(grid):
 open('/mnt/data/README_E10_R2_AUDIT.md').read()
 return [[1]]'''),
 ('readonly-runtime-write', '''def transform(grid):
 open('/usr/lib/E10_R3_ESCAPE_MARKER','w').write('bad')
 return [[1]]'''),
 ('child-process', '''import subprocess
def transform(grid):
 subprocess.check_output(['/usr/bin/id'])
 return [[1]]'''),
 ('network', '''import socket
def transform(grid):
 socket.socket()
 return [[1]]'''),
]
for label, program in attacks:
    ok, response = run(program, [[0]], 5)
    assert not ok, (label, response)
    assert ('sandbox unavailable' not in response), (label, response)
    print('PASS blocked:', label)
assert not escape.exists(), 'SANDBOX ESCAPE! host file modified'

# Runaway program bounded by parent + process group timeout.
t0 = time.perf_counter()
ok, msg = run('def transform(grid):\n while True: pass\n', [[0]], 1.4)
assert not ok and ('timeout' in msg or 'sandbox process failure' in msg), msg
assert time.perf_counter() - t0 < 6, 'process group did not terminate promptly'
print('PASS runaway CPU/process-group termination')

# Stdout debug must be safe and deterministic.
ok, msg = run('import numpy as np\nprint("debug")\ndef transform(grid):\n print("still debugging")\n return np.array([[1]])', [[0]], 6)
assert ok and json.loads(msg)==[[1]], msg
print('PASS NumPy/debug output in sandbox')

# 5. Full notebook has exactly the two intentionally changed code cells.
old = json.loads((HERE/'Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R2.ipynb').read_text())
new = json.loads((HERE/'Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R3.ipynb').read_text())
base = json.loads((HERE/'Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb').read_text())
assert len(new['cells'])==9
changed=[]
for idx,(o,n) in enumerate(zip(old['cells'],new['cells'])):
    if o['source'] != n['source']:changed.append(idx)
    if n['cell_type']=='code':ast.parse(''.join(n['source']))
assert changed==[5,7],changed
for index in (1,3,6):
    assert new['cells'][index]['source']==base['cells'][index]['source']
for token in ('77824','8192','106496'):
    assert token in ''.join(new['cells'][2]['source'])
assert 'E10_POETIQ_EXPERTS' in ''.join(new['cells'][2]['source'])
assert 'E10_POETIQ_MAX_ITERATIONS' in ''.join(new['cells'][2]['source'])
assert hashlib.sha256((HERE/'Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb').read_bytes()).hexdigest()== '530cb119e302899426a2e3f24c5e100279e7167aef77525b6cb93b90a7fc10c0'
print('PASS notebook only cell 5 & 7 changed, 86K/base parity preserved')

# 6. One expert: best result remains first, previous differing valid program
#    supplies second attempt; no new second reasoning request added.
oldr={'passed':False,'test_usable':True,'train_results':[{'soft_score':0.0}],
    'test_results':[{'grid':[[9]]}],'iteration':1,'expert':0}
newr={'passed':True,'test_usable':True,'train_results':[{'soft_score':1.0}],
    'test_results':[{'grid':[[1]]}],'iteration':2,'expert':0}
S={('task',0):{'best':newr,'candidate_history':[oldr,newr]}}
scheduler_src=(HERE/'poetiq_scheduler.py').read_text()
start=scheduler_src.index('def e10_rank_task(tid):')
end=scheduler_src.index('\ndef e10_update_task_submission',start)
ns={'E10_STATES':S,'poetiq_vote':core['poetiq_vote'],
'POETIQ_USE_NEW_VOTING':True,'POETIQ_COUNT_FAILED_MATCHES':True,
'POETIQ_ITERS_TIEBREAK':False,'POETIQ_LOW_TO_HIGH_ITERS':False}
exec(scheduler_src[start:end],ns)
ranked=ns['e10_rank_task']('task')
assert ranked[0] is newr and len(ranked)==2, ranked
at=core['poetiq_two_attempts'](task, ranked)
assert at==[{'attempt_1':[[1]],'attempt_2':[[9]]}],at
print('PASS one-expert history yields distinct attempt_2 without disturbing best program')

# 7. Full task-level fake-engine: train-perfect but test-exception MUST
#    generate another iteration containing the test diagnostics.
import tempfile, types, traceback
from collections import deque
FIRST_PROGRAM = "def transform(grid):\n if int(grid[0][0])==9: raise ValueError('cannot solve unseen form')\n return [[1]]\n"
SECOND_PROGRAM = 'def transform(grid):\n return [[1]]\n'
class ProbeTokenizer:
    def __init__(self):self.messages=[]
    def apply_chat_template(self,messages,**kwargs):
        prompt=messages[0]['content'];self.messages.append(prompt);return prompt
    def encode(self,text,add_special_tokens=False):return [2,3,4]
    def decode(self,ids,skip_special_tokens=True):
        p=FIRST_PROGRAM if 777 in ids else SECOND_PROGRAM if 778 in ids else ''
        return '```python\n'+p+'\n```' if p else ''
class EngineOutput:
    def __init__(self,rid,ids):
        self.request_id=rid;self.finished=True
        self.outputs=[types.SimpleNamespace(token_ids=ids,finish_reason='stop')]
class Engine:
    def __init__(self):self.pending={}
    def add_request(self,rid,prompt,params):
        assert rid not in self.pending,rid
        self.pending[rid]=prompt
    def step(self):
        rid=next(iter(self.pending));self.pending.pop(rid)
        if rid.endswith('|1'):return [EngineOutput(rid,[11])]
        return [EngineOutput(rid,[778 if '~i1|' in rid else 777])]
    def abort_request(self,ids):
        for rid in ids:self.pending.pop(rid,None)
with tempfile.TemporaryDirectory() as tmp:
    now=time.perf_counter()
    tok=ProbeTokenizer()
    task2={'train':[{'input':[[0]],'output':[[1]]}], 'test':[{'input':[[9]]}]}
    fake={'OUT':Path(tmp),'WORKING':Path(tmp),'SUBMISSION_PATH':Path(tmp)/'submission.json',
          'CHALLENGES':{'taskX':task2},'test_challenges':{'taskX':task2},'TASK_IDS':['taskX'],
          'NOTEBOOK_T0':now,'CONCURRENCY':2,'SEED':10,
          'POETIQ_SEED':0,'POETIQ_NUM_EXPERTS':1,'POETIQ_MAX_ITERATIONS':2,'POETIQ_MAX_SOLUTIONS':5,
          'POETIQ_SOLVER_VARIANTS':(1,),'POETIQ_SELECTION_PROB':1.,'POETIQ_IMPROVING_ORDER':True,
          'POETIQ_SHUFFLE_EXAMPLES':True,'POETIQ_SANDBOX_TIMEOUT':6.,'POETIQ_SANDBOX_WORKERS':2,
          'POETIQ_SOLVER_TEMPERATURE':1.,'POETIQ_USE_NEW_VOTING':True,
          'POETIQ_COUNT_FAILED_MATCHES':True,'POETIQ_ITERS_TIEBREAK':False,'POETIQ_LOW_TO_HIGH_ITERS':False,
          'SAMPLING':{'top_p':.95,'top_k':20},'THINK_CLOSE_ID':99,'STOP_IDS':[98,99],'EOS_IDS':set(),
          'MAX_MODEL_LEN':106496,'THINK_BUDGET':77824,'ANSWER_MAX_TOKENS':8192,
          'DRAIN_ANSWER_MAX_TOKENS':8192,'MIN_THINK_BUDGET':4096,
          'REASONING_EFFORT':'xhigh','RERUN':True,'ENGINE_READY':True,
          'HARD_STOP':now+50,'DRAIN_DEADLINE':now+45,'ADMIT_DEADLINE':now+40,'LOG_EVERY_SEC':50,
          'tokenizer':tok,'SamplingParams':lambda **kw:kw,
          'RequestOutputKind':types.SimpleNamespace(DELTA='delta'),'engine':Engine(),
          'identity_submission':lambda z:{k:[{'attempt_1':v['input'],'attempt_2':v['input']}
                for v in task['test']] for k,task in z.items()},
          'write_json_atomic':lambda path,data:Path(path).write_text(json.dumps(data)),
          'metrics_snapshot':lambda:[], 'phase1_cap':lambda prompt_len:77824,
          'time':time,'json':json,'traceback':traceback,'re':__import__('re'),
          'np':core['np'],'os':os,'sys':__import__('sys')}
    fake.update({k:v for k,v in core.items() if not k.startswith('__')})
    exec((HERE/'poetiq_scheduler.py').read_text(),fake)
    final_state=fake['E10_STATES'][('taskX',0)]
    assert final_state['iteration']==2 and final_state['status']=='train_verified',final_state
    assert fake['E10_SCHED']['train_passed']==1
    assert any('TEST EXECUTION / FORMAT ERRORS' in msg for msg in tok.messages[1:]),tok.messages
    assert fake['E10_FINAL']['taskX'][0]['attempt_1']==[[1]]
    print('PASS full fake-engine train-perfect/test-invalid -> feedback -> valid next program')

print('ALL E10 R3 APPROVED REGRESSIONS PASS')
