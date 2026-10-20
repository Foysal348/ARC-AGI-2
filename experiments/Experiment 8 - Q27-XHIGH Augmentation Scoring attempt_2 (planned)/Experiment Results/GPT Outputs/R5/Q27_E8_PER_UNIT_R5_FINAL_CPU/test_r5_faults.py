import runpy,time,copy,types,json,ast
from pathlib import Path
import nbformat
z=runpy.run_path(str(Path(__file__).resolve().parent/'test_r5_cpu.py'))
ns=z['ns']; G7=z['G7']; G8=z['G8']; P1=z['P1']; LP=z['LP']; Params=z['Params']
NB=z['NB'];BASE=z['BASE'];val=z['val']

# Reproduce adversarial scorer failures without touching primary outputs.
ns['FINAL']={'a#0':{'attempt_1':P1,'attempt_2':[[0]],'src_1':'model','src_2':'test_input'}}
ns['E8_BASELINE']={'a#0':copy.deepcopy(ns['FINAL']['a#0'])}
ns['E8_JOBS'].clear();ns['E8_CURRENT']=None;ns['E8_ACTIVE']=None;ns['E8_REPORT']['scorer_disabled_reason']=None
ns['E8_REPORT']['errors']=[]
ns['E8_JOBS'].append({'uid':'a#0','candidates':[{'grid':G7,'source':'think'},{'grid':G8,'source':'think'}],
                    'index':0,'requests':None,'scores':{}})
class AddFails:
    def __init__(self):self.active={};self.aborted=[]
    def add_request(self,rid,data,params):self.active[rid]=data;raise RuntimeError('failed after register')
    def abort_request(self,ids):
        self.aborted.extend(ids)
        for rid in ids:self.active.pop(rid,None)
ns['engine']=AddFails()
res=ns['e8_try_admit'](0,time.perf_counter())
assert res is False
assert not ns['engine'].active and ns['engine'].aborted
assert ns['FINAL']['a#0']['attempt_1']==P1 and ns['FINAL']['a#0']['attempt_2']==[[0]]
assert ns['E8_REPORT']['scorer_disabled_reason'] is not None
print('PASS partial registration failure aborts scorer and preserves baseline')

# logprob missing chosen token: disable only scorer, no mutation of primary.
ns['E8_REPORT']['scorer_disabled_reason']=None
ns['E8_CURRENT']={'uid':'a#0','candidates':[{'grid':G7,'source':'think'},{'grid':G8,'source':'think'}],
 'scores':{},'index':0,'requests':[]}
ns['E8_ACTIVE']={'rid':'e8|123','request':{'uid':'a#0','key':ns['e8_key'](G7),'view':0,
 'tokens':[1,2,3],'answer':[3], 'answer_start':2}}
ns['engine']=AddFails()
ns['e8_on_score_output'](types.SimpleNamespace(request_id='e8|123',finished=True,prompt_logprobs=[None,None,{}]))
assert ns['E8_REPORT']['scorer_disabled_reason']=='scorer_logprob_failure'
assert ns['FINAL']['a#0']['attempt_2']==[[0]]
print('PASS missing vLLM logprob triggers fallback-only fail-stop')

# Source-level equality all four deadlines: original E7 config retained verbatim.
assert NB.cells[2].source == BASE.cells[2].source
for s in ('ADMIT_DEADLINE','DRAIN_DEADLINE','HARD_STOP','VERIFY_DEADLINE'):
    assert s in NB.cells[2].source
assert 'E8_RESERVED_SEC' not in NB.cells[2].source
assert 'E8_STAGE_DEADLINE' not in NB.cells[-1].source
print('PASS primary deadlines identical to E7; no 30-min reserve')

# Original submitted E7 attempts are referenced by per-unit checkpoint, not modified on failures.
assert ns['e8_validate_submission_obj']({'a':[{'attempt_1':P1,'attempt_2':[[0]]}]},
 {'a':ns['CHALLENGES']['a']})
print('PASS partial failure submission validity')

# EVAL report must use separate initial RECORDS and final FINAL.
assert 'baseline_by_uid' in NB.cells[-1].source
assert 'final_by_uid' in NB.cells[-1].source
assert "report['projection_valid'] = False" in NB.cells[-1].source
print('PASS baseline EVAL, final E8, contention-aware projection separate')
print('ALL R5 BASELINE FAULT CHECKS PASS')
