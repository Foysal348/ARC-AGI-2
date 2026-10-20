import json,hashlib,ast,copy,pathlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb'
DEST=ROOT/'Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R4.ipynb'
source=json.loads(BASE.read_text())
assert len(source['cells'])==9
assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='530cb119e302899426a2e3f24c5e100279e7167aef77525b6cb93b90a7fc10c0'

def code(src):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':src.splitlines(True)}
def md(src):
    return {'cell_type':'markdown','metadata':{},'source':src.splitlines(True)}
def src(i):return ''.join(source['cells'][i]['source'])
config_append='''
# ---- E10 Poetiq donor configurations; source: poetiq-ai/poetiq-arc-agi-solver/config.py
# Original 86K context, GPU recipe, deadlines and original puzzle order above are unchanged.
POETIQ_NUM_EXPERTS = int(os.environ.get('E10_POETIQ_EXPERTS','1'))
POETIQ_MAX_ITERATIONS = int(os.environ.get('E10_POETIQ_MAX_ITERATIONS','10'))
POETIQ_MAX_SOLUTIONS = int(os.environ.get('E10_POETIQ_MAX_SOLUTIONS','5'))
POETIQ_SOLVER_TEMPERATURE = float(os.environ.get('E10_POETIQ_TEMPERATURE','1.0'))
POETIQ_SOLVER_VARIANTS = tuple(int(v) for v in os.environ.get('E10_POETIQ_PROMPT_VARIANTS','1').split(','))
POETIQ_SELECTION_PROB = float(os.environ.get('E10_POETIQ_SELECTION_PROBABILITY','1.0'))
POETIQ_SEED = int(os.environ.get('E10_POETIQ_SEED','0'))
POETIQ_SHUFFLE_EXAMPLES = os.environ.get('E10_POETIQ_SHUFFLE','1')=='1'
POETIQ_IMPROVING_ORDER = os.environ.get('E10_POETIQ_IMPROVING_ORDER','1')=='1'
POETIQ_USE_NEW_VOTING = os.environ.get('E10_POETIQ_NEW_VOTING','1')=='1'
POETIQ_COUNT_FAILED_MATCHES = os.environ.get('E10_POETIQ_COUNT_FAILED_MATCHES','1')=='1'
POETIQ_ITERS_TIEBREAK = os.environ.get('E10_POETIQ_ITERS_TIEBREAK','0')=='1'
POETIQ_LOW_TO_HIGH_ITERS = os.environ.get('E10_POETIQ_LOW_TO_HIGH_ITERS','0')=='1'
POETIQ_SANDBOX_TIMEOUT = float(os.environ.get('E10_POETIQ_SANDBOX_TIMEOUT','5'))
POETIQ_SANDBOX_WORKERS = int(os.environ.get('E10_POETIQ_SANDBOX_WORKERS','4'))
assert POETIQ_NUM_EXPERTS >= 1 and POETIQ_MAX_ITERATIONS >= 1
assert POETIQ_SOLVER_VARIANTS and all(v in (1,2,3) for v in POETIQ_SOLVER_VARIANTS)
assert POETIQ_MAX_SOLUTIONS >= 1 and 0 <= POETIQ_SELECTION_PROB <= 1
CONFIG['poetiq']={k:globals()[k] for k in (
    'POETIQ_NUM_EXPERTS','POETIQ_MAX_ITERATIONS','POETIQ_MAX_SOLUTIONS',
    'POETIQ_SOLVER_TEMPERATURE','POETIQ_SOLVER_VARIANTS','POETIQ_SELECTION_PROB',
    'POETIQ_SANDBOX_TIMEOUT','POETIQ_SANDBOX_WORKERS','POETIQ_USE_NEW_VOTING')}
'''
tasks='''# E10: original task-order and evaluation profiles, but one Poetiq program per task/expert/iteration.
if P['source']=='eval':
    _ch=_load_json(ARC_ROOT/'arc-agi_evaluation_challenges.json')
    _sol=_load_json(ARC_ROOT/'arc-agi_evaluation_solutions.json')
    if P.get('task_list'):
        assert all(t in _ch for t in P['task_list']), 'Evaluation task ID missing'
        TASK_IDS=sorted(P['task_list'])
    else:
        TASK_IDS=sorted(random.Random(TASK_SAMPLE_SEED).sample(sorted(_ch),P['n_tasks']))
    CHALLENGES={t:_ch[t] for t in TASK_IDS}
    SOLUTIONS={t:_sol[t] for t in TASK_IDS}
else:
    CHALLENGES=dict(test_challenges)
    TASK_IDS=sorted(CHALLENGES) # no Codex ordering
    SOLUTIONS=None
N_HIDDEN_UNITS=sum(len(t['test']) for t in test_challenges.values())
print('E10 Poetiq:',len(TASK_IDS),'tasks,',sum(len(CHALLENGES[t]['test']) for t in TASK_IDS),
      'queries | experts',POETIQ_NUM_EXPERTS,'iterations per expert',POETIQ_MAX_ITERATIONS)
'''
report='''# E10 — Poetiq program synthesis, train-verification and task-level output report
try:
    for tid in TASK_IDS:
        if tid not in E10_FINAL:
            E10_FINAL[tid]=poetiq_two_attempts(CHALLENGES[tid],[])
    generated={tid:E10_FINAL[tid] for tid in TASK_IDS}
    write_json_atomic(OUT/'poetiq_eval_generated_submission.json',generated)
    report={'run_id':RUN_ID,'profile':PROFILE,'rerun':RERUN,
            'source_base_sha256':'530cb119e302899426a2e3f24c5e100279e7167aef77525b6cb93b90a7fc10c0',
            'donor':'https://github.com/poetiq-ai/poetiq-arc-agi-solver',
            'configuration':CONFIG,'engine':ENGINE_INFO,'scheduler':E10_SCHED,
            'tasks_planned':len(TASK_IDS),'queries_planned':sum(len(CHALLENGES[t]['test']) for t in TASK_IDS),
            'experts_planned':len(E10_STATES),'experts_train_verified':sum(s['status']=='train_verified' for s in E10_STATES.values()),
            'programs_generated':E10_SCHED['programs_parsed'],
            'expert_states':{f'{t}~e{e}':{'iteration':s['iteration'],'status':s['status'],
                                       'best_score':s['best_score']} for (t,e),s in E10_STATES.items()},
            'task_ranked_programs':{tid:len(e10_rank_task(tid)) for tid in TASK_IDS},
            'generated_attempts_path':'q27x/poetiq_eval_generated_submission.json'}
    if SOLUTIONS is not None:
        hits=0;pass1=0;pass2=0;total=0;records=[]
        for tid in TASK_IDS:
            for qi,att in enumerate(E10_FINAL[tid]):
                gold=SOLUTIONS[tid][qi];a=(att['attempt_1']==gold);b=(att['attempt_2']==gold)
                hits+=a or b;pass1+=a;pass2+=b;total+=1
                records.append({'task':tid,'query':qi,'pass1':bool(a),'pass2':bool(b),'hit':bool(a or b)})
        report.update(eval_query_hits=hits,eval_query_total=total,eval_score_pct=round(100*hits/max(1,total),2),
                      eval_pass1_hits=pass1,eval_pass2_hits=pass2,eval_unit_records=records)
    write_json_atomic(OUT/'poetiq_run_report.json',report)
    print('E10 RUN REPORT:',str(OUT/'poetiq_run_report.json'),
          'eval',report.get('eval_query_hits'),'/ ',report.get('eval_query_total'),flush=True)
    # Last: ensure the actual Kaggle submission and bundled version are identical.
    e10_write_submission()
    _pack=WORKING/'Q27X_E10_POETIQ_BUNDLE.zip'
    with zipfile.ZipFile(_pack,'w',zipfile.ZIP_DEFLATED) as z:
        for fp in (SUBMISSION_PATH, OUT/'poetiq_run_report.json',OUT/'poetiq_final_generated.json',
                   OUT/'poetiq_eval_generated_submission.json',OUT/'poetiq_iterations.jsonl',
                   OUT/'poetiq_vllm_metrics.json'):
            if fp.exists():z.write(fp,arcname=fp.name)
    print('E10 bundle:',_pack,flush=True)
except BaseException:
    print('E10 reporting error (submission.json still exists):',traceback.format_exc()[-3000:],flush=True)
'''
notebook={'cells':[
    md('''# Q27 E10 — 86K × Poetiq Train-Verified Program Solver

**Base:** User-supplied Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb (SHA256 `530cb119...`).

**Frozen architecture:** 86K Q27 local vLLM → Poetiq task-level program synthesis → isolated Python execution on visible train pairs → exact/soft feedback → iterative refinement → expert voting/diversity → two ranked attempt grids.

**User decisions:** 77,824 thinking + 8,192 answer; original task order (no Codex ordering); no E9 tail resample; original 4×L4 FP8-KV/MTP3 engine initialization; do not insert old Q27 grid ranker. Poetiq donor default: one expert, 10 iterations, 5 feedback examples, temperature 1.0. Config supports 2/8+ experts and prompt variants 1/2/3 without editing code.

**Source:** [poetiq-ai/poetiq-arc-agi-solver](https://github.com/poetiq-ai/poetiq-arc-agi-solver), MIT. This is a local-vLLM adaptation, not a byte-for-byte Poetiq clone or a measured Poetiq accuracy claim. Kaggle has no internet; all runtime code is embedded here.

**STATUS:** CPU-validated prototype; requires 4×L4 GPU validation. No GPU or Kaggle submission run by the authoring assistant.
'''),
    copy.deepcopy(source['cells'][1]),
    code(src(2)+'\n'+config_append),
    copy.deepcopy(source['cells'][3]),
    code(tasks),
    code((ROOT/'poetiq_core.py').read_text()),
    copy.deepcopy(source['cells'][6]),
    code((ROOT/'poetiq_scheduler.py').read_text()),
    code(report)
], 'metadata':copy.deepcopy(source['metadata']),'nbformat':4,'nbformat_minor':5}
for i,c in enumerate(notebook['cells']):
    c['id']=hashlib.sha256((str(i)+''.join(c['source'])).encode()).hexdigest()[:8]
    if c['cell_type']=='code':ast.parse(''.join(c['source']))
try:
    import nbformat
    nbformat.validate(nbformat.from_dict(notebook))
except ImportError:pass
DEST.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
print('WROTE',DEST,'SHA256',hashlib.sha256(DEST.read_bytes()).hexdigest(), 'cells',len(notebook['cells']))
