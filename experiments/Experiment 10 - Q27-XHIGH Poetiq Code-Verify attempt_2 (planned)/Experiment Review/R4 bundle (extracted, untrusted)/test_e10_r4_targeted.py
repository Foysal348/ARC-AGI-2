"""Independent R4 regressions for audited Kaggle-blocking defects.

Run against the real R4 core/scheduler, without GPU or Kaggle APIs.
"""
import ast, json, runpy, tempfile, os, time, math
from pathlib import Path
R=Path(__file__).resolve().parent
core=runpy.run_path(str(R/'poetiq_core.py'))
assert core['E10_SANDBOX_SELFTEST']['ok']
assert core['poetiq_normalize_grid']([[1, 1.0]]) == [[1,1]]
assert core['poetiq_normalize_grid']([[1.999]]) is None
assert core['poetiq_normalize_grid']([[True]]) is None
assert core['poetiq_normalize_grid']([['1']]) is None
assert core['poetiq_normalize_grid']([[-1]]) is None
assert core['poetiq_normalize_grid']([[float('nan')]]) is None
print('PASS lossless numeric normalization, rejects lossy conversions')

task={'train':[{'input':[[1]],'output':[[0]]}], 'test':[{'input':[[3]]}]}
res=core['poetiq_verify_code']('import numpy as np\ndef transform(grid):\n return np.zeros((1,1))',task,8)
assert res['verified_usable'] and res['passed'] and res['test_results'][0]['grid']==[[0]]
print('PASS numpy.float64 integral output trains and yields valid ARC test grid')

for label, prog in {
  'subprocess':'import subprocess\ndef transform(grid):\n subprocess.check_output(["/bin/id"])\n return grid',
  'socket':'import socket\ndef transform(grid):\n socket.socket()\n return grid',
  'fork':'import os\ndef transform(grid):\n os.fork()\n return grid',
  'outside-write':'def transform(grid):\n open("/mnt/data/e10_r4_should_not_write", "w").write("x")\n return grid',
}.items():
  ok,msg=core['poetiq_sandbox_run'](prog,[[2]],8)
  assert not ok,(label,msg)
  print('PASS sandbox rejected',label)
assert not Path('/mnt/data/e10_r4_should_not_write').exists()

start=time.perf_counter();ok,msg=core['poetiq_sandbox_run']('def transform(grid):\n while True: pass',[[0]],1.7)
assert not ok and time.perf_counter()-start < 7,(ok,msg)
print('PASS parent-enforced timeout')

# Verify actual notebook cell source, not independent injected globals.
nb=json.loads((R/'Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R4.ipynb').read_text())
src=[''.join(c['source']) for c in nb['cells']]
for i in range(9):
  if nb['cells'][i]['cell_type']=='code': ast.parse(src[i])
assert 'LOG_EVERY_SEC = 120.0' in src[7]
assert 'LOG_EVERY_SEC = 120.0' not in src[2]
assert "cmd=['unshare'" not in src[5] and "subprocess.Popen(['unshare'" not in src[5]
assert 'E10_SANDBOX_SELFTEST=poetiq_sandbox_selftest()' in src[5]
print('PASS notebook has real logger definition and sandbox self-test before GPU engine')

# The exact notebook's 86K configuration is present; no silent reduction.
assert '77824' in src[2] and '8192' in src[2] and '106496' in src[2]
assert 'if cap < 49152:' in src[7]
assert src[1]==''.join(json.loads((R/'Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb').read_text())['cells'][1]['source'])
assert src[6]==''.join(json.loads((R/'Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb').read_text())['cells'][6]['source'])
print('PASS model/engine/base fidelity and min-thinking static gate')

# Test the real e10_render_request/e10_admit functions with a synthetic token budget.
mod=ast.parse(src[7]); funcs={n.name:n for n in mod.body if isinstance(n,ast.FunctionDef)}
selected=ast.Module(body=[funcs[k] for k in ('e10_render_request','e10_admit')],type_ignores=[])
class FakeTok:
 def __init__(self):self.last=''
 def apply_chat_template(self,msg,**kw):self.last=msg[0]['content'];return self.last
 def encode(self,msg,add_special_tokens=False):return [1]*len(msg)
class FakeEngine:
 def __init__(self):self.requests=[]
 def add_request(self,rid,src,params):self.requests.append((rid,params))
seed={'OUT':None,'CHALLENGES':{'t':{'train':[{'input':[[1]],'output':[[2]]}], 'test':[{'input':[[1]]}]}},
      'POETIQ_SOLVER_VARIANTS':(1,),'POETIQ_SELECTION_PROB':1.,'POETIQ_MAX_SOLUTIONS':5,
      'POETIQ_IMPROVING_ORDER':True,'POETIQ_SHUFFLE_EXAMPLES':True,'POETIQ_SEED':0,
      'REASONING_EFFORT':'xhigh','tokenizer':FakeTok(),'E10_STATES':{},
      'E10_ACTIVE':{}, 'E10_SCHED':{'skipped_context':0,'feedback_trims':0,'requests':0},
      'THINK_BUDGET':77824,'MIN_THINK_BUDGET':4096,
      'phase1_cap':lambda n:max(0,min(77824,106496-n-8192-256)),
      'engine':FakeEngine(),'time':time,'SEED':42,
      'SamplingParams':lambda **kw:kw,'RequestOutputKind':type('X',(),{'DELTA':'delta'}),
      'SAMPLING':{'top_p':.95,'top_k':20},
      'POETIQ_SOLVER_TEMPERATURE':1.0,
      'e10_req':lambda st,phase:f"{st['task_id']}~e{st['expert']}~i{st['iteration']}|{phase}",
      'e10_make_params':lambda cap,seed:{'max_tokens':cap,'seed':seed},
      'e10_seed':lambda st,phase:42}
seed.update({k:core[k] for k in ('poetiq_build_message',)})
exec(compile(selected,'<E10-R4-functions>','exec'),seed)
st={'task_id':'t','expert':0,'task_index':0,'iteration':2,'status':'queued','finished':False,
    'history':[], 'last_error':None}
# The bottom-scoring feedback is a huge middle-score text; exact token count
# is synthetic and conservative (one character per token).
for k in range(5):
 st['history'].append({'code':'def transform(g): return g\n'+'#x'*11500,
                       'feedback':('diagnostic '*1400),'score':(k+1)/10})
seed['E10_STATES'][('t',0)]=st
out=seed['e10_admit'](('t',0))
assert out=='admitted',out
assert seed['E10_ACTIVE'][('t',0)]['p1_cap']>=49152
assert seed['E10_SCHED']['feedback_trims']==1
assert len(st['history'])==5
print('PASS feedback pruning preserves >=49,152 allowance, and retains original history')
# No feedback and oversized base prompt => explicit skip, never shortened admission.
st2={'task_id':'t','expert':0,'task_index':0,'iteration':0,'status':'queued','finished':False,
     'history':[],'last_error':None}
seed['E10_STATES'][('t',0)]=st2
seed['tokenizer'].apply_chat_template=lambda msg,**kw: 'z'*50000
out=seed['e10_admit'](('t',0))
assert out=='skipped' and seed['E10_SCHED']['skipped_context']==1,out
print('PASS underfloor base prompt is skipped explicitly (not run with reduced thinking)')
print('ALL R4 TARGETED REGRESSIONS PASS')
