"""Reproducible E8 R2 CPU/fake-engine regression suite. No Kaggle GPU needed."""
import ast, json, os, sys, tempfile, types, time, math, traceback, hashlib, zipfile, copy
from pathlib import Path
import numpy as np
import nbformat

ROOT = Path('/mnt/data')
P = ROOT/'Q27_E8_EXACT_ARCH_R2/Q27-XHIGH-v1_4_4_E8_EXACT_ARCH_FIXED_R2.ipynb'
BASE = ROOT/'Q27-XHIGH-v1_4_4_E7_ORDERING_ONLY.ipynb'
DONOR = ROOT/'arc-2026d-with-codex (24)(1).ipynb'
nb = nbformat.read(P,as_version=4)
base = nbformat.read(BASE,as_version=4)
nbformat.validate(nb)
assert len(nb.cells)==12
for c in nb.cells:
    if c.cell_type=='code': ast.parse(c.source)
for i in [0,1,3,4,5,6,7]:
    assert nb.cells[i].source == base.cells[i].source, ('E7 primary changed unexpectedly',i)
assert nb.cells[2].source.replace(' # INVALID',' # INVALID') != base.cells[2].source  # explicit scheduling budget change
print('PASS notebook/schema/syntax and E7 primary freeze')

# Donor loader semantic parity (uses same AST for core donor augment and geometry)
donor_src = '\n'.join(''.join(c.get('source',[])) for c in json.load(open(DONOR))['cells'])
our_src = nb.cells[9].source
for func in ['augment','mod','mod_single','shuffle_ex','permute_mod','permute_rnd_all_']:
    assert f'def {func}(' in donor_src and f'def {func}(' in our_src,func
print('PASS donor operation source presence')

class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(x) for x in text]
    def decode(self, ids, skip_special_tokens=True):
        return ''.join(map(chr,ids))

class FakeLogProb:
    def __init__(self,v): self.logprob=v

class FakeSamplingParams:
    def __init__(self,**kw): self.kw=kw

class FakeOutputKind:
    FINAL_ONLY='final'

OUTD=Path(tempfile.mkdtemp(prefix='q27_e8_test_'))
CKPT=OUTD/'submission.json'
ns = {
    '__name__':'__main__','np':np, 'math':math,'time':time,
    'json':json,'hashlib':hashlib,'os':os,'zipfile':zipfile,'Path':Path,
    'SamplingParams':FakeSamplingParams,'RequestOutputKind':FakeOutputKind,
    'MAX_MODEL_LEN':81920,'STOP_IDS':[17],
    'THINK_BUDGET':49152,'ANSWER_MAX_TOKENS':8192,
    'tokenizer':FakeTokenizer(), 'OUT':OUTD,'SUBMISSION_PATH':CKPT,
    'SamplingParams':FakeSamplingParams,
}
exec(compile(nb.cells[9].source,'donor_loader','exec'),ns)
# Trim functions only, avoid config assertions? config has E8_GLOBAL values etc no issue.
src=nb.cells[10].source
# Keep all defs but do NOT execute notebook hooks. The helper cell only contains defs and configuration.
ns.update({
 'HARD_STOP':time.perf_counter()+1800,
 'E8_STAGE_DEADLINE':time.perf_counter()+1200,
 'E8_RESERVED_SEC':1800,
 'SOLUTIONS':None,
 'RERUN':True,'ENGINE_READY':True,
 'SCHED_ERROR':None,'SCHED':{'aborted_at_hard_stop':0},
 'OUT':OUTD,
})
exec(compile(src,'e8_helpers','exec'),ns)
valid_grid=lambda g: isinstance(g,list) and 1<=len(g)<=30 and all(isinstance(r,list) and 1<=len(r)<=30 and all(type(x)==int and 0<=x<=9 for x in r) for r in g) and len({len(r) for r in g})==1
ns['valid_grid']=valid_grid
ns['write_json_atomic']=lambda p,x: (Path(str(p)+'.tmp').write_text(json.dumps(x)),os.replace(str(p)+'.tmp',p))
ns['_load_json']=lambda p: json.loads(Path(p).read_text())
ns['identity_submission']=lambda ch: {tid:[{'attempt_1':x['input'],'attempt_2':x['input']} for x in task['test']] for tid,task in ch.items()}

# F1: valid 1/2/3 test input and invalid shapes
for n in [1,2,3]:
    challenges={'task':{'train':[],'test':[{'input':[[1]]} for _ in range(n)]}}
    sub={'task':[{'attempt_1':[[1]],'attempt_2':[[2]]} for _ in range(n)]}
    assert ns['e8_validate_submission_obj'](sub,challenges) is True
    try:
        ns['e8_validate_submission_obj']({'task':sub['task'][:-1]},challenges)
        raise AssertionError('missing sample accepted')
    except ValueError: pass
print('PASS F1 submission count')

# Setup rotation puzzle
X=[[1,2],[3,4]]
Y=np.rot90(np.array(X),1).tolist()
CHALLENGES={'task':{'train':[{'input':X,'output':Y}],'test':[{'input':X}]}}
ns.update({'CHALLENGES':CHALLENGES,'test_challenges':CHALLENGES,'UNIT_TASK':{'task#0':'task'},'UNIT_Q':{'task#0':0},'ORDERED_UNITS':['task#0'],'UNITS':['task#0']})
wrong=[[9]]
fallback=X
ns['FINAL']={'task#0':{'attempt_1':wrong,'attempt_2':fallback}}
ns['SOLUTIONS']={'task':[Y]}
ns['score_unit']=lambda uid, att:{'hit':int(Y==att['attempt_1'] or Y==att['attempt_2']),'pass1':int(Y==att['attempt_1'])}
ns['write_submission']=lambda: ns['write_json_atomic'](CKPT, ns['identity_submission'](CHALLENGES) if not ns['RERUN'] else {tid:[{'attempt_1':ns['FINAL'][f'{tid}#{i}']['attempt_1'],'attempt_2':ns['FINAL'][f'{tid}#{i}']['attempt_2']} for i,_ in enumerate(ch['test'])] for tid,ch in CHALLENGES.items()})

# F2 ensure rotation and supported color map are not removed by legacy guess order
pool=ns['e8_build_candidates']('task#0',wrong,fallback,4)
assert any(c['grid']==Y for c in pool), pool
assert len(pool)>=3
print('PASS F2 supported candidate survives cap')

# Donor 8 views exact and deterministic, preserve outer NumPy RNG
np.random.seed(104)
state=np.random.get_state()
views=ns['e8_make_views']('task#0',Y)
assert len(views)==8
assert np.array_equal(np.random.get_state()[1],state[1])
assert views==ns['e8_make_views']('task#0',Y)
print('PASS 8 view deterministic RNG and shape')

# F5 verify scoring span matches chosen answer tokens
v=views[0]
request=ns['e8_prepare_score_request']('task#0',pool[0],v,0)
assert request['prompt_ids'][request['answer_start']:] == request['answer_ids']
positions=[None]*len(request['prompt_ids'])
for j, tok in enumerate(request['answer_ids']):
    pos=request['answer_start']+j
    positions[pos]={tok:FakeLogProb(-0.25)}
score=ns['e8_prompt_nll'](positions,request['prompt_ids'],request['answer_start'],request['answer_ids'])
assert math.isclose(score,len(request['answer_ids'])*.25)
print('PASS F5 answer chosen-token position and NLL')

class FakeEngine:
    def __init__(self,fail_after_register=False,abort_raises=False):
        self.active={};self.fail_after_register=fail_after_register;self.abort_raises=abort_raises
        self.aborted=[]
    def add_request(self,rid,req,params):
        self.active[rid]=req['prompt_token_ids']
        if self.fail_after_register: raise RuntimeError('injected partial registration')
    def abort_request(self,ids):
        self.aborted.extend(ids)
        if self.abort_raises: raise RuntimeError('injected abort failure')
        for id_ in ids:self.active.pop(id_,None)
    def step(self):
        out=[]
        for rid,ids in list(self.active.items()):
            p=[None]*len(ids)
            for ix,tok in enumerate(ids):
                p[ix]={tok:FakeLogProb(-.3)}
            out.append(types.SimpleNamespace(request_id=rid,finished=True,prompt_logprobs=p))
            self.active.pop(rid)
        return out
    def has_unfinished_requests(self):return bool(self.active)

# F6: partial registration cleanup success and fail stop
f=FakeEngine(fail_after_register=True)
try: ns['e8_score_batch'](f,[request],time.perf_counter()+100)
except RuntimeError:pass
else:raise AssertionError('failed to signal registration error')
assert not f.active and f.aborted, f.active
f2=FakeEngine(fail_after_register=True,abort_raises=True)
try: ns['e8_score_batch'](f2,[request],time.perf_counter()+100)
except RuntimeError as e: assert 'cleanup_failed' in str(e)
else:raise AssertionError('ambiguous cleanup not flagged')
print('PASS F6 partial registration cleanup and failure escalation')

# Batch fake scorer
f=FakeEngine()
val=ns['e8_score_batch'](f,[request],time.perf_counter()+10)
assert len(val)==1 and not f.active
print('PASS fake vLLM scoring batch')

# F4 EVAL final generated pair and updated score
ns['RERUN']=False
ns['SOLUTIONS']={'task':[Y]}
ns['FINAL']={'task#0':{'attempt_1':wrong,'attempt_2':Y}}
sub=ns['e8_make_eval_generated_submission']()
assert sub['task'][0]['attempt_2']==Y
assert ns['e8_actual_eval_results']()['eval_e8_hits']==1
ns['FINAL']['task#0']['attempt_2']=fallback
assert ns['e8_actual_eval_results']()['eval_e8_hits']==0
print('PASS F4 true final EVAL correctness')

# Integration exact architecture: run E8 before final report, assert active scoring and changed attempt_2
ns['RERUN']=True
ns['FINAL']={'task#0':{'attempt_1':wrong,'attempt_2':fallback}}
ns['ENGINE_READY']=True
ns['engine']=FakeEngine()
ns['E8_STAGE_DEADLINE']=time.perf_counter()+200
ns['SCHED_ERROR']=None
ns['SOLUTIONS']={'task':[Y]}
ns['SCHED']={'aborted_at_hard_stop':0}
# Fake NLL score prefers train-supported rotated candidate (score 0.01)
original_score_batch=ns['e8_score_batch']
def score_batch(engine_obj, requests,deadline):
    out={}
    for r in requests:
        low=(r['candidate_key']==ns['e8_key'](Y))
        out[(r['uid'],r['candidate_key'],r['view'])]=((0.01 if low else 5.0)*len(r['answer_ids']),len(r['answer_ids']))
    return out
ns['e8_score_batch']=score_batch
ns['write_submission']()
res=ns['e8_run_stage']()
assert res['checkpoint_written'],res
assert res['units_scored']==1,res
assert res['attempt2_replaced']==1,res
assert ns['FINAL']['task#0']['attempt_1']==wrong
assert ns['FINAL']['task#0']['attempt_2']==Y,res
assert ns['_load_json'](OUTD/'e8_baseline_submission.json')['task'][0]['attempt_2']==fallback
assert ns['_load_json'](CKPT)['task'][0]['attempt_2']==Y
assert res['eval_e8_net_gain']==1,res
print('PASS integrated E8 candidate score APPLY and baseline checkpoint')

# F7 diagnostic I/O failure isolated from final submission
save=ns['write_json_atomic']
def fail_report(p,v):
    if str(p).endswith('e8_augmented_scoring_report.json'):
        raise OSError('disk fail')
    return save(p,v)
ns['write_json_atomic']=fail_report
ns['FINAL']={'task#0':{'attempt_1':wrong,'attempt_2':fallback}}
ns['write_submission']()
res=ns['e8_run_stage']()
assert res['attempt2_replaced']==1
assert ns['_load_json'](CKPT)['task'][0]['attempt_2']==Y
print('PASS F7 diagnostic I/O failure does not block submission')
ns['write_json_atomic']=save

# F3 allocation existence / primary cutoff order
cfg=nb.cells[2].source
assert 'E8_RESERVED_SEC' in cfg and 'PRIMARY_HARD_STOP' in cfg
assert 'HARD_STOP = PRIMARY_HARD_STOP' in cfg
assert 'E8_STAGE_DEADLINE = E8_GLOBAL_HARD_STOP - E8_FINALIZE_SEC' in cfg
assert 'stage_deadline = E8_STAGE_DEADLINE' in nb.cells[10].source
print('PASS F3 explicit reserved scoring deadline plumbing')

print('ALL E8 R2 TESTS PASS')

# Additional acceptance checks: donor AST identity, explicit deadline monotonicity,
# rollback integrity under submission write failure.
donor_cells=json.load(open(DONOR))['cells']
donor_loader=''.join(donor_cells[3]['source']).split('\n',1)[1]
donor_ast=ast.parse(donor_loader)
vendor_ast=ast.parse(nb.cells[9].source)
def node_by_name(tree,name):
    return next(n for n in ast.walk(tree) if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
for fun in ['ArcDataset','augment','mod','mod_single','shuffle_ex','permute_mod','permute_rnd_all_']:
    assert ast.dump(node_by_name(donor_ast,fun),include_attributes=False)==ast.dump(node_by_name(vendor_ast,fun),include_attributes=False),fun
print('PASS exact donor ArcDataset + augmentation AST parity')

budget_source=nb.cells[2].source
b=budget_source.index('# E8 architecture: guarantee')
e=budget_source.index('assert NOTEBOOK_T0 < ADMIT_DEADLINE < DRAIN_DEADLINE',b)
code=budget_source[b:e]
for label,budget,hard,drain,admit,P in [
    ('eval',10800,9000,8900,8500,{'drain_frac':.9,'admit_frac':.8}),
    ('hidden',42600,42240,41760,41640,{'drain_margin_sec':480,'admit_margin_sec':120}),
]:
    t={
        'os':os,'NOTEBOOK_T0':0.,'TOTAL_BUDGET_SEC':budget,
        'HARD_STOP':hard,'DRAIN_DEADLINE':drain,'ADMIT_DEADLINE':admit,
        'VERIFY_DEADLINE':budget-180,'P':P,
    }
    exec(code,t)
    assert 0<t['ADMIT_DEADLINE']<t['DRAIN_DEADLINE']<t['PRIMARY_HARD_STOP']<t['E8_STAGE_DEADLINE']<t['E8_GLOBAL_HARD_STOP']<t['VERIFY_DEADLINE']
    assert t['E8_STAGE_DEADLINE']-t['PRIMARY_HARD_STOP']==1680, t
print('PASS real 3h/12h deadline ordering and 28-minute active scoring window')

# F6/F7 failure AFTER valid checkpoint: force a failure during final submission,
# and confirm attempt_2 and file both recover to original baseline.
ns['write_json_atomic']=save
ns['FINAL']={'task#0':{'attempt_1':wrong,'attempt_2':fallback}}
ns['write_submission']()
old=ns['_load_json'](CKPT)
original_writer=ns['write_submission']
writes={'n':0}
def injected_write_submission():
    writes['n']+=1
    if writes['n']==2: # checkpoint is first, selection is second
        # Simulate partially written file followed by exception.
        CKPT.write_text('{}')
        raise OSError('injected submission write failure')
    return original_writer()
ns['write_submission']=injected_write_submission
res=ns['e8_run_stage']()
assert res['error'] is not None or res.get('error_recovery'),res
assert ns['FINAL']['task#0']['attempt_2']==fallback
assert ns['_load_json'](CKPT)==old, (ns['_load_json'](CKPT),old)
print('PASS interrupted final submission restores validated checkpoint and original attempt_2')

# Primary cutoff may legitimately abort ongoing primary requests. E8 MUST STILL run
# on already finished units, instead of treating this expected cutoff as failure.
ns['write_submission']=original_writer
ns['FINAL']={'task#0':{'attempt_1':wrong,'attempt_2':fallback}}
ns['SCHED']={'aborted_at_hard_stop':7}
ns['ENGINE_READY']=True
ns['SCHED_ERROR']=None
ns['engine']=FakeEngine()
ns['write_submission']()
r=ns['e8_run_stage']()
assert r['units_scored']==1 and r['attempt2_replaced']==1, r
assert r['primary_hardstop_aborts']==7, r
print('PASS expected primary cutoff does not disable E8 scoring')

print('ALL EXTENDED E8 R2 TESTS PASS')
