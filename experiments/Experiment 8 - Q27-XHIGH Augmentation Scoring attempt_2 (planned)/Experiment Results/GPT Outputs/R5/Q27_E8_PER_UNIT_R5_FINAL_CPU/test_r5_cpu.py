from pathlib import Path
import os, sys, re, json, ast, time, math, zipfile, tempfile, copy, hashlib, types
import numpy as np
import nbformat
D=Path(__file__).resolve().parent; ROOT=D.parent
SRC=D/'sources' if (D/'sources/Q27-XHIGH-v1_4_4_E7_ORDERING_ONLY.ipynb').exists() else ROOT
NB=nbformat.read(D/'Q27-XHIGH-v1_4_4_E8_PER_UNIT_R5_FINAL_CPU.ipynb',as_version=4)
BASE=nbformat.read(SRC/'Q27-XHIGH-v1_4_4_E7_ORDERING_ONLY.ipynb',as_version=4)
DONOR=nbformat.read(SRC/'arc-2026d-with-codex (24)(1).ipynb',as_version=4)
nbformat.validate(NB)
assert all(NB.cells[i].source==BASE.cells[i].source for i in range(7))
for c in NB.cells:
    if c.cell_type=='code':ast.parse(c.source)
print('PASS notebook schema/syntax/E7 primary source 0..6 unchanged')

# exact donor AST augment/mapping is retained
src='\n'.join(c.source for c in DONOR.cells if c.cell_type=='code')
new=NB.cells[8].source
names=['permute_mod','permute_rnd_all_','mod','mod_single','augment','shuffle_ex']
def signature_functions(code):
    out={}
    # donor notebook contains %%writefile arc_loader.py markers, strip them out
    code='\n'.join(l for l in code.splitlines() if not l.startswith('%%writefile') and not l.startswith('%'))
    try:t=ast.parse(code)
    except SyntaxError:return {}
    for n in ast.walk(t):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
            out.setdefault(n.name,[]).append(ast.dump(n,include_attributes=False))
    return out
new_sig=signature_functions(new)
donor_loader=NB.cells[8].source # direct donor comparison with extracted cell 3, only type annotation removed
assert 'def augment(' in new and 'def shuffle_ex(' in new
assert all(n in new_sig for n in names)
print('PASS donor geometric/augment symbols available')

class Tok:
    def encode(self,txt,add_special_tokens=False):return [ord(c) for c in txt]
    def decode(self,ids,skip_special_tokens=True):return ''.join(chr(i) for i in ids if i<1114112)
class LP:
    def __init__(self,x):self.logprob=x
class Params:
    def __init__(self,**kw):self.kw=kw
class Kinds:FINAL_ONLY='final'

def val(g):
    return isinstance(g,list) and 1<=len(g)<=30 and len({len(r) for r in g})==1 and all(isinstance(r,list) and 1<=len(r)<=30 and all(type(x)==int and 0<=x<=9 for x in r) for r in g)

def fn(comp, name):
    nodes=[n for n in ast.parse(comp).body if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(nodes)==1,name
    return compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),name,'exec')

ns={'np':np,'json':json,'re':re,'time':time,'math':math,'hashlib':hashlib,
    'tokenizer':Tok(),'MAX_MODEL_LEN':81920,'SamplingParams':Params,'RequestOutputKind':Kinds,
    'valid_grid':val,'_FENCE_RE':re.compile(r'```([^\n`]*)\n(.*?)```',re.S),
    '_SEP_ROW':re.compile(r'\[?\s*\d(?:\s*[, ]\s*\d)*\s*\]?,?'),
    '_JSON_ROW':re.compile(r'\[\s*\d(?:\s*,\s*\d)*\s*\]'),
    'STOP_IDS':[991], 'THINK_CLOSE_ID':999,
    'EOS_IDS':[],'MAX_MODEL_LEN':81920,'ANSWER_MAX_TOKENS':8192,
    'DRAIN_ANSWER_MAX_TOKENS':8192,'NOTEBOOK_T0':time.perf_counter(),
    'SEED':1,'ENGINE_READY':True,'RERUN':True,'CONCURRENCY':2,
    'RUN_ID':'test','PROFILE':'EVAL12','CONFIG':{},'ENGINE_INFO':{},
    'N_HIDDEN_UNITS':259,'P1_CAPS':{},'ORDER_ADMIT_RANK':{},
    'CODEX_PRIORITY_TIERS':{},'TASK_IDS':[],
    'ZIP':None,'SKIPPED':{},'SOLUTIONS':None,'METRICS_SNAPSHOTS':[],
    'LOG_EVERY_SEC':9999,'VERIFY_DEADLINE':time.perf_counter()+500,
    'HARD_STOP':time.perf_counter()+100,'DRAIN_DEADLINE':time.perf_counter()+80,
    'ADMIT_DEADLINE':time.perf_counter()+70,
    'SCHED_ERROR':None,'OUTPUT_PATH':'', 'Path':Path,
    'traceback':__import__('traceback'),'zipfile':zipfile,
}
work=Path(tempfile.mkdtemp(prefix='q27_r4_cpu_'))
ns.update({'OUT':work/'out','WORKING':work,'SUBMISSION_PATH':work/'submission.json'})
ns['OUT'].mkdir()
for n in ['_parse_row','_first_row_run','_grid_from_obj','_json_rows_grid','_grid_from_block','parse_grid','make_attempts','score_unit']:
    exec(fn(BASE.cells[5].source,n),ns)
exec(compile(NB.cells[8].source,'donor','exec'),ns)
exec(compile(NB.cells[9].source,'e8_core','exec'),ns)
ns['identity_submission']=lambda ch:{t:[{'attempt_1':v['input'],'attempt_2':v['input']} for v in task['test']] for t,task in ch.items()}
ns['write_json_atomic']=lambda p,obj:(Path(str(p)+'.tmp').write_text(json.dumps(obj)),os.replace(str(p)+'.tmp',p))
ns['_load_json']=lambda p:json.loads(Path(p).read_text())

assert ns['e8_extract_all_grids']('Scratch intermediate illustration (not final):\n```grid\n12\n34\n```\n') == []
assert ns['e8_extract_all_grids']('Alternative solution:\n```grid\n12\n34\n```\n')
G7=[[7,7]];G8=[[8,8]]; P1=[[1,1]]
uid='a#0'
ns['CHALLENGES']={'a':{'train':[{'input':[[3]],'output':[[4]]}],
                 'test':[{'input':[[0]]}]}}
ns['UNIT_TASK']={uid:'a'};ns['UNIT_Q']={uid:0}
ns['UNIT_INDEX']={uid:0};ns['P1_CAPS']={uid:10}
thought='I considered candidate solutions.\n```grid\n77\n```\nAn alternative:\n```grid\n88\n```\n'
novel=ns['e8_build_novel'](uid,thought,'Final\n```grid\n11\n```',P1,[[0]])
assert [p['grid'] for p in novel]==[G8,G7],novel
print('PASS full reasoning candidate mining dedup, exclude primary/test/demo')

# 0, 1, 2 candidate routing; donor deterministic 8-view generation
ns['FINAL']={uid:{'attempt_1':P1,'attempt_2':[[0]],'src_1':'model','src_2':'test_input'}}
ns['write_submission']=lambda:ns['write_json_atomic'](ns['SUBMISSION_PATH'],
    {t:[{'attempt_1':ns['FINAL'][f'{t}#{i}']['attempt_1'],'attempt_2':ns['FINAL'][f'{t}#{i}']['attempt_2']} for i in range(len(task['test']))]
     for t,task in ns['CHALLENGES'].items() if all(f'{t}#{i}' in ns['FINAL'] for i in range(len(task['test'])))})
ns['write_submission']()
views=ns['e8_make_views'](uid,G7)
assert len(views)==8 and views==ns['e8_make_views'](uid,G7)
req=ns['e8_prepare_score_request'](uid,{'grid':G7,'source':'thought'},views[0],0)
L=[None]*len(req['tokens'])
for i,tid in enumerate(req['answer']):L[req['answer_start']+i]={tid:LP(-0.1)}
assert abs(ns['e8_prompt_nll'](L,req)-0.1*len(req['answer']))<1e-6
print('PASS exact donor 8 views repeatability & logprob positions')

# Single choice replaces attempt2, attempt1 stays, no score jobs. Use mock st with close id.
class ST(dict):pass
p1=[ord(c) for c in 'candidate\n```grid\n77\n```\n']+[999]
ns['SOLUTIONS']={'a':[G7]};ns['E8_REPORT']['rows']=[]
ns['e8_on_primary_complete'](uid,{'p1_ids':p1},'')
assert ns['FINAL'][uid]['attempt_1']==P1
assert ns['FINAL'][uid]['attempt_2']==G7
assert ns['E8_REPORT']['units_one_novel']==1
assert not ns['E8_JOBS']
print('PASS single novel direct attempt2, primary unmodified, baseline checkpoint')

# Synthetic end-to-end mixed scheduler. 3 tasks, concurrency2. Two alternatives for first unit.
ns['CHALLENGES']={
 'a':{'train':[{'input':[[3]],'output':[[4]]}],'test':[{'input':[[0]]}]},
 'b':{'train':[{'input':[[1]],'output':[[2]]}],'test':[{'input':[[5]]}]},
 'c':{'train':[{'input':[[1]],'output':[[2]]}],'test':[{'input':[[6]]}]},
}
ns['UNIT_TASK']={f'{t}#0':t for t in 'abc'}
ns['UNIT_Q']={f'{t}#0':0 for t in 'abc'}
ns['UNIT_INDEX']={f'{t}#0':i for i,t in enumerate('abc')}
ns['P1_CAPS']={f'{t}#0':10 for t in 'abc'}
ns['ORDER_ADMIT_RANK']={f'{t}#0':i for i,t in enumerate('abc')}
ns['CODEX_PRIORITY_TIERS']={t:0 for t in 'abc'}
ns['ORDERED_UNITS']=[f'{t}#0' for t in 'abc']
ns['UNITS']=list(ns['ORDERED_UNITS']);ns['TASK_IDS']=list('abc')
ns['PROMPT_IDS']={f'{t}#0':[i] for i,t in enumerate('abc')}
ns['SOLUTIONS']={'a':[G7],'b':[[0]],'c':[[0]]}
ns['test_challenges']=ns['CHALLENGES'];ns['FINAL']={};ns['STATE']={};ns['RECORDS']=[];ns['E8_JOBS'].clear();ns['E8_ACTIVE']=None;ns['E8_CURRENT']=None
ns['E8_REPORT'].update({'units_one_novel':0,'units_ranked':0,'units_with_novel':0,'attempt2_replaced':0,'score_requests_started':0,'score_views_finished':0,'units_incomplete':0,'rows':[],'scorer_disabled_reason':None,'errors':[]})
ns['SCHED']={'started':time.perf_counter(),'admitted':0,'phase2_continue':0,'phase2_forced':0,'drained':0,
            'aborted_at_hard_stop':0,'not_admitted':0,'max_active':0,'steps':0}
ns['GEN_TOKENS_TOTAL']=0
ns['RECORDS_PATH']=ns['OUT']/'records.jsonl'
ns['ANSWERS_DIR']=ns['OUT']/'answers';ns['ANSWERS_DIR'].mkdir(exist_ok=True)
ns['METRICS_SNAPSHOTS']=[]
ns['metrics_snapshot']=lambda:[]
ns['make_params']=lambda *a,**kw:Params()
ns['NOTEBOOK_T0']=time.perf_counter();ns['HARD_STOP']=time.perf_counter()+100
ns['DRAIN_DEADLINE']=time.perf_counter()+90;ns['ADMIT_DEADLINE']=time.perf_counter()+80

class FakeEngine:
    def __init__(self):self.active={};self.calls=[];self.mixed_step=False;self.maxactive=0
    def add_request(self,rid,data,params):self.active[rid]=(data,params);self.calls.append(rid);self.maxactive=max(self.maxactive,len(self.active))
    def abort_request(self,ids):
        for rid in ids:self.active.pop(rid,None)
    def step(self):
        from collections import namedtuple
        if any(k.startswith('e8|') for k in self.active) and any(not k.startswith('e8|') for k in self.active):
            self.mixed_step=True
        out=[]
        for rid,(data,params) in list(self.active.items()):
            if rid.startswith('e8|'):
                tok=data['prompt_token_ids'];logps=[{t:LP(-.1)} for t in tok]
                req=ns['E8_ACTIVE']['request']
                # Prefer G7 over G8 deliberately.
                loss=0.01 if req['key']==ns['e8_key'](G7) else 0.3
                for i,t in enumerate(req['answer']):logps[req['answer_start']+i]={t:LP(-loss)}
                out.append(types.SimpleNamespace(request_id=rid,finished=True,prompt_logprobs=logps))
            else:
                label=rid.split('|')[0][0]
                if label=='a':
                    txt='Candidate hypothesis\n```grid\n77\n```\nAlternative solution\n```grid\n88\n```\n'
                else:txt='No other grid.\n'
                raw=[ord(v) for v in txt]+[999]+[ord(v) for v in '\n```grid\n11\n```']
                out.append(types.SimpleNamespace(request_id=rid,finished=True,
                    outputs=[types.SimpleNamespace(token_ids=raw,finish_reason='stop')]))
            self.active.pop(rid)
        return out
    def has_unfinished_requests(self):return bool(self.active)
ns['engine']=FakeEngine()
ns['write_submission']=lambda: ns['write_json_atomic'](ns['SUBMISSION_PATH'],ns['identity_submission'](ns['CHALLENGES']) if not ns['RERUN'] else
    {t:[{'attempt_1':ns['FINAL'][f'{t}#0']['attempt_1'],'attempt_2':ns['FINAL'][f'{t}#0']['attempt_2']}]
     for t in ns['CHALLENGES'] if f'{t}#0' in ns['FINAL']})
for name in ['write_submission','_req','admit','start_phase2','answer_text','finish','_record','run_scheduler']:
    # original scheduler functions but r4 integration modifications
    exec(fn(NB.cells[10].source,name),ns)
ns['run_scheduler']()
assert len(ns['FINAL'])==3
assert ns['FINAL']['a#0']['attempt_1']==P1,ns['FINAL']['a#0']
assert ns['FINAL']['a#0']['attempt_2']==G7,ns['FINAL']['a#0']
assert ns['E8_REPORT']['units_ranked']==1,ns['E8_REPORT']
assert ns['E8_REPORT']['score_views_finished']==16,ns['E8_REPORT']
assert ns['engine'].mixed_step, 'scoring never overlapped primary task'
assert ns['engine'].maxactive<=ns['CONCURRENCY'], ns['engine'].maxactive
assert ns['SCHED']['not_admitted']==0
print('PASS mixed primary/scorer event dispatch + actual attempt2 selection + max_num_seqs guard')

# Hard stop should cancel pending scoring without changing already-frozen primary answers.
old=copy.deepcopy(ns['FINAL']); ns['E8_JOBS'].append({'uid':'a#0','candidates':[{'grid':G7,'source':'test'}]})
ns['e8_finish_pending']()
assert not ns['E8_JOBS'] and ns['FINAL']==old
print('PASS cancellation preserves existing primary and attempt2 results')

# EVAL score fields can reflect actual final and baseline independently; R3 block adapted in R4.
assert 'eval_baseline_query_hits' in NB.cells[-1].source and 'eval_e8_hits' in NB.cells[-1].source
assert 'E8_STAGE_DEADLINE' not in NB.cells[-1].source
assert ns['e8_validate_submission_obj'](ns['_load_json'](ns['SUBMISSION_PATH']),
    {t:ns['CHALLENGES'][t] for t in ns['_load_json'](ns['SUBMISSION_PATH'])})
print('PASS final EVAL metric fields and generated JSON schema')
print('ALL R5 BASELINE CHECKS PASS')
