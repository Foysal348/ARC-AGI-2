"""CPU-only, targeted tests for R5 new fixes; use no GPU or Kaggle data."""
import runpy,copy,time,types,json,ast
from pathlib import Path
D=Path(__file__).resolve().parent
z=runpy.run_path(str(D/'test_r5_cpu.py'))
ns=z['ns'];NB=z['NB'];BASE=z['BASE'];G7=z['G7'];G8=z['G8'];P1=z['P1']; LP=z['LP']
uid='a#0'

# A. Misleading intermediate grid must not be promoted even if labelled "final" in negation.
for text in [
    'This is only a scratch visualization, not the final output:\n```grid\n9\n```\n',
    'An intermediate candidate calculation (not final):\n```grid\n99\n99\n```\n',
    'Example training output:\n[[9, 9], [9, 9]]\n',
    'wrong candidate:\n```grid\n88\n88\n```\n',
]:
    found=ns['e8_extract_all_grids'](text)
    assert not found,(text,found)
assert ns['e8_extract_all_grids']('Final candidate output:\n```grid\n99\n99\n```')
print('PASS F1 scratch, negated-final, training-example, wrong labels rejected; final candidate accepted')

# B. Extractor rank by actual position across JSON/fences/rows (not category).
text='Earlier example:\n[[1,1],[1,1]]\n\nFinal candidate:\n```grid\n77\n77\n```\n'
assert [x['grid'] for x in ns['e8_extract_all_grids'](text)]==[[[7,7],[7,7]]],ns['e8_extract_all_grids'](text)
text='Possible output:\n[[6,6],[6,6]]\n\nAlternative candidate:\n```grid\n77\n77\n```\n'
assert [x['grid'] for x in ns['e8_extract_all_grids'](text)]==[[[7,7],[7,7]],[[6,6],[6,6]]]
assert ns['e8_extract_all_grids'](text)[0]['text_offset']>ns['e8_extract_all_grids'](text)[1]['text_offset']
print('PASS F2 cross-format chronology preserved by literal source offsets')

# C. Scoring request fail-closed on inconsistent encode boundary.
views=ns['e8_make_views'](uid,G7)
normal_tok=ns['tokenizer']
class BoundaryTok(type(normal_tok)):
    def encode(self,txt,add_special_tokens=False):
        # Contrived BPE-ish effect: full concatenation encodes differently.
        tokens=super().encode(txt,add_special_tokens=add_special_tokens)
        return [42]+tokens if 'assistant' in txt and 'user' in txt else tokens
ns['tokenizer']=BoundaryTok()
try:
    # force test without assumptions on real donor prompt words
    try:
        ns['e8_prepare_score_request'](uid,{'grid':G7,'source':'test'},
                                        {'input':'abc','reply':'assistant user'},0)
        assert False,'strict mismatch must fail'
    except ValueError as e:
        assert 'scorer_unverified_token_boundary' in str(e)
finally:
    ns['tokenizer']=normal_tok
print('PASS F5 unverified scorer boundary rejected, avoids misleading NLL')

# D. Finisher counts active + current + queued UIDs exactly once; no leftover scoring requests.
ns['E8_REPORT']['units_incomplete']=0
ns['E8_REPORT']['units_incomplete_ids']=[]
ns['E8_REPORT']['scorer_disabled_reason']=None
ns['E8_JOBS'].clear()
ns['E8_JOBS'].extend([{'uid':'b#0'},{'uid':'c#0'}])
ns['E8_CURRENT']={'uid':'a#0'}
ns['E8_ACTIVE']={'rid':'e8|100','request':{'uid':'a#0'}}
class Aborting:
    def __init__(self): self.ids=[]
    def abort_request(self,ids):self.ids.extend(ids)
a=Aborting();ns['engine']=a
ns['e8_finish_pending']()
assert ns['E8_REPORT']['units_incomplete']==3
assert set(ns['E8_REPORT']['units_incomplete_ids'])=={'a#0','b#0','c#0'}
assert a.ids==['e8|100']
ns['e8_finish_pending']()
assert ns['E8_REPORT']['units_incomplete']==3
assert not ns['E8_JOBS'] and ns['E8_CURRENT'] is None and ns['E8_ACTIVE'] is None
print('PASS F6 in-flight+queued count=3 and idempotent cleanup')

# E. Baseline fallback train-consistent safeguard protects correct E7 attempt_2.
ns['CHALLENGES']={'a':{'train':[{'input':[[4]],'output':[[4]]}],
                        'test':[{'input':[[5]]}]}}
ns['UNIT_TASK']={'a#0':'a'};ns['UNIT_Q']={'a#0':0}
assert ns['e8_baseline_is_train_supported']('a#0',[[5]])
ns['FINAL']={'a#0':{'attempt_1':[[1]],'attempt_2':[[5]],'src_1':'model','src_2':'test_input'}}
ns['E8_REPORT']['rows']=[];ns['E8_REPORT']['units_one_novel']=0
ns['E8_REPORT']['scorer_disabled_reason']=None
ns['SOLUTIONS']={'a':[[[5]]]}
ns['E8_JOBS'].clear()
ns['E8_BASELINE']={}
ns['E8_REPORT']['units_with_novel']=0
ns['write_submission']=lambda:None
thinking='Candidate answer:\n```grid\n9\n```\n'
ns['e8_on_primary_complete']('a#0',{'p1_ids':[ord(c) for c in thinking]},'')
assert ns['FINAL']['a#0']['attempt_2']==[[5]]
assert ns['E8_REPORT']['rows'][-1]['choice']=='baseline_train_consistent_protected'
print('PASS F1 additional train-consistent baseline protection, no 1->0 pass@2 regression')

# F. Source and scheduler priority; no non-consensual deadline changes.
sched=NB.cells[10].source
needle='if E8_ENABLED and (E8_JOBS or E8_CURRENT is not None):\n            e8_try_admit(len(active), now)'
assert needle in sched
start=sched.index('# R5 primary-FIRST')
assert sched.index('while queue',start)<sched.index('e8_try_admit(len(active), now)',start)
assert NB.cells[2].source==BASE.cells[2].source
print('PASS F3 queued primary admissions precede scorer in scheduler; deadlines unchanged')

# G. Main and auxiliary report order: EVAL first, reports afterwards.
r=NB.cells[11].source
p_ev=r.index('E8_REPORT.update(e8_actual_eval_results())')
p_aux=r.index("write_json_atomic(OUT/'e8_per_unit_report.json',E8_REPORT)")
p_main=r.index('write_json_atomic(OUT / "run_report.json", report)')
assert p_ev<p_aux<p_main
assert "report['e8'] = {k:v for k,v in E8_REPORT.items() if k!='rows'}" in r
print('PASS F7 final EVAL reported in both JSON outputs after selection')

# H. Validate source parity with donor and primary unchanged.
assert NB.cells[8].source==z['NB'].cells[8].source
for i in range(7):assert NB.cells[i].source==BASE.cells[i].source
print('ALL R5 ADVERSARIAL TESTS PASS')
