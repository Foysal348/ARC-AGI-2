#!/usr/bin/env python3
"""CPU-only tests for Q27X R5_SAFE_PATCH. No model/GPU needed. No public GT injected."""
from pathlib import Path
import ast, collections, json, re, sys, tempfile, time, types, zipfile

ROOT = Path(__file__).resolve().parent
NB = ROOT / 'Q27X_v1_4_R5_SAFE_PATCH.ipynb'
ORIG = ROOT / 'Q27X_v1_4_RUN4_FOR_GPT_2026-10-06.zip'

def src(nb,i):return ''.join(nb['cells'][i]['source'])
def function_fragment(code, first, last_marker):
    start=code.index(first)
    end=code.index(last_marker,start)
    return code[start:end]

def load_helper_definitions(code):
    tree=ast.parse(code)
    names={'valid_grid','_parse_row','_first_row_run','_grid_from_obj','_json_rows_grid','_grid_from_block',
           'parse_grid','make_attempts','score_unit','raw_grid_diagnostics','merge_second_candidate'}
    defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    src_txt='\n'.join(ast.get_source_segment(code,n) for n in defs)
    ctx={'re':re,'json':json,'_FENCE_RE':re.compile(r'```([^\n`]*)\n(.*?)```',re.S),
         '_SEP_ROW':re.compile(r'\[?\s*\d(?:\s*[, ]\s*\d)*\s*\]?,?'),
         '_JSON_ROW':re.compile(r'\[\s*\d(?:\s*,\s*\d)*\s*\]'),
         'FINAL':{}}
    exec(compile(src_txt,'<helper-definitions>','exec'),ctx)
    return ctx

def main():
    nb=json.loads(NB.read_text(encoding='utf-8'))
    assert len(nb['cells'])==9
    for i in range(1,9): compile(src(nb,i),f'notebook_cell_{i}','exec')
    print('PASS 1: all 8 notebook code cells compile')
    main_src=src(nb,5)
    c=load_helper_definitions(main_src)
    vg,parse,raw,merge=(c[k] for k in ('valid_grid','parse_grid','raw_grid_diagnostics','merge_second_candidate'))
    with zipfile.ZipFile(ORIG) as z:
        for uid, exp_h in [('247ef758_q0',15),('6e4f6532_q0',31)]:
            s=z.read(f'run4/output/q27x/answers/{uid}.txt').decode().split('### ANSWER',1)[-1]
            diag=raw(s)
            assert not vg(parse(s)) if parse(s) is not None else True
            assert diag['parseable_rows']==exp_h,(uid,diag)
            assert diag['ragged'],(uid,diag)
            if exp_h==31: assert diag['too_tall']
    print('PASS 2: both archived malformed grids diagnosed, including the 31-row violation')
    for s,expected in [('```grid\n1 0\n3 4\n```',[[1,0],[3,4]]),
                       ('```grid\n9 9\n9\n```',None),
                       ('```json\n[[1,0],[3,4]]\n```',[[1,0],[3,4]])]:
        assert parse(s)==expected,(s,parse(s))
    print('PASS 3: parser keeps valid grids and rejects ragged grids')
    uid='abc#0'; first=[[1,2],[3,4]]; fallback=[[3,4],[1,2]]; alt=[[9,8],[7,6]]
    c['FINAL'][uid]={'attempt_1':first,'attempt_2':fallback,'src_1':'model','src_2':'test_input'}
    assert not merge(uid,None)
    assert not merge(uid,[[1,10]])
    assert not merge(uid,first)
    assert not merge(uid,fallback)
    assert merge(uid,alt)
    assert not merge(uid,alt)
    assert c['FINAL'][uid]['attempt_1']==first
    assert c['FINAL'][uid]['attempt_2']==alt
    print('PASS 4: candidate-bank merge is non-destructive and rejects invalid/duplicate grids')

    # Test write_submission and real EVAL diagnostics without loading the GPU engine.
    code=src(nb,7)
    tree=ast.parse(code)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='write_submission')
    with tempfile.TemporaryDirectory() as t:
        root=Path(t); out=root/'q27x';out.mkdir(); submission=root/'submission.json'
        challenges={'test1':{'test':[{'input':[[0,1]]},{'input':[[2,3]]}]}}
        eval_ch={'ev01':{'test':[{'input':[[1,2]]}]}}
        def identity(ch):
            return {tid:[{'attempt_1':q['input'],'attempt_2':q['input']} for q in d['test']] for tid,d in ch.items()}
        def atomic(path,obj): Path(path).write_text(json.dumps(obj))
        ctx={'RERUN':False,'CHALLENGES':eval_ch,'test_challenges':challenges,'TASK_IDS':['ev01'],
             'FINAL':{'ev01#0':{'attempt_1':[[9,9]],'attempt_2':[[8,8]]}},
             'UNIT_TASK':{'ev01#0':'ev01'},'UNIT_Q':{'ev01#0':0},
             'OUT':out,'SUBMISSION_PATH':submission,'identity_submission':identity,'write_json_atomic':atomic}
        exec(compile(ast.get_source_segment(code,fn),'<write_submission>','exec'),ctx)
        ctx['write_submission']()
        sub=json.loads(submission.read_text()); model=json.loads((out/'eval_predictions.json').read_text())
        assert sub==identity(challenges)
        assert model['ev01'][0]['attempt_1']==[[9,9]]
        ctx.update(RERUN=True, FINAL={'test1#1':{'attempt_1':[[4,5]],'attempt_2':[[6,7]]}},
                   UNIT_TASK={'test1#1':'test1'},UNIT_Q={'test1#1':1})
        ctx['write_submission']()
        sub=json.loads(submission.read_text())
        assert sub['test1'][0]['attempt_1']==[[0,1]]
        assert sub['test1'][1]['attempt_1']==[[4,5]]
        assert sub['test1'][1]['attempt_2']==[[6,7]]
    print('PASS 5: non-rerun submission is labeled by separate eval predictions; rerun merges real outputs')

    # Simulated vLLM delta stream drives review phase3/phase4; ensure it changes only second attempt.
    code=src(nb,7)
    t=ast.parse(code)
    funcs={n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
    review_src='\n\n'.join(ast.get_source_segment(code,funcs[k]) for k in ('_req','_review_prompt','run_optional_audit'))
    class StubTokenizer:
        def encode(self,*a,**k): return [77,78]
        def apply_chat_template(self,*a,**k):return 'STUB'
        def decode(self,ids,**k): return ''.join({90:'```grid\n4 5\n6 7\n```'}.get(x,'') for x in ids)
    class StubEngine:
        def __init__(self):self.queue=[]
        def add_request(self,uid,prompt,params):
            self.queue.append(uid)
        def step(self):
            uid=self.queue.pop(0)
            ph=int(uid.rsplit('|',1)[-1])
            toks=[1] if ph==3 else [90]
            c=types.SimpleNamespace(token_ids=toks,finish_reason='length' if ph==3 else 'stop')
            return [types.SimpleNamespace(request_id=uid,outputs=[c],finished=True)]
        def abort_request(self,ids):pass
    tok=StubTokenizer(); engine=StubEngine()
    now=time.perf_counter()
    out_dir=ROOT/'_r5_test_temp'
    out_dir.mkdir(exist_ok=True)
    ctx2={**c,'__builtins__':__builtins__,'AUDIT_ENABLED':True,'ENGINE_READY':True,'AUDIT_MAX_QUERIES':1,
          'CONCURRENCY':4,'AUDIT_CONCURRENCY':1,'HARD_STOP':now+300,
          'UNITS':[uid],'UNIT_INDEX':{uid:0},'UNIT_TASK':{uid:'abc'},'UNIT_Q':{uid:0},
          'CHALLENGES':{'abc':{'train':[{'input':[[0]],'output':[[1]]}], 'test':[{'input':[[3,4],[1,2]]}]}},
          'AUDIT_STATS':{'enabled':True,'started':0,'completed':0,'valid_distinct':0,'aborted_at_deadline':0,'candidates_planned':0},
          'AUDIT_RECORDS_PATH':out_dir/'audit_records.jsonl',
          'engine':engine,'tokenizer':tok,'THINK_CLOSE_ID':1,'EOS_IDS':{99},'STOP_IDS':[2],
          'AUDIT_THINK_BUDGET':128,'AUDIT_ANSWER_MAX':256,'MAX_MODEL_LEN':2048,'CONTEXT_SAFETY':10,
          'SEED':13,'render_grid':lambda rows:'\n'.join(' '.join(map(str,row)) for row in rows),
          'build_user_message':lambda train,grid:'stub',
          'make_params':lambda cap,seed: {'cap':cap,'seed':seed},'write_submission':lambda:None,
          'time':time,'json':json,'tokenizer_encode':None}
    exec(compile(review_src,'<review>', 'exec'),ctx2)
    ctx2['run_optional_audit']()
    assert ctx2['FINAL'][uid]['attempt_1']==first
    assert ctx2['FINAL'][uid]['attempt_2']==[[4,5],[6,7]]
    assert ctx2['FINAL'][uid]['src_2']=='model_review'
    assert ctx2['AUDIT_STATS']['valid_distinct']==1
    print('PASS 6: simulated two-phase review promotes a distinct valid grid without changing first attempt')
    try:
        (out_dir/'audit_records.jsonl').unlink();out_dir.rmdir()
    except OSError: pass
    print('OVERALL: 6/6 CPU checks PASS; NO GPU/model inference or leaderboard score was measured')

if __name__=='__main__':main()
