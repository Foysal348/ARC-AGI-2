"""Q27 v1.9 measured synthesis-only EVAL; challenge-only inference before scoring.
Public solutions are loaded only AFTER generated programs have been executed.
"""
import glob
import json
import os
import time
import threading
from collections import Counter
from pathlib import Path
from synth import solve_synthesis
from agent import score
from rescue import load_direct_ref, hits, direct_grid, server_metrics, MetricsSampler


def find(name, root='/kaggle/input'):
    matches=glob.glob(f'{root}/**/{name}',recursive=True)
    if not matches:
        raise FileNotFoundError(name)
    return matches[0]


def atomic_json(path, obj):
    path = Path(path)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,default=str))
    os.replace(tmp,path)


def _valid(g):
    from rescue import valid_grid
    return valid_grid(g)


def make_submissions(tasks, records, direct_refs):
    by_id={r['task_id']:r for r in records}
    agent, direct, merged={}, {},{}
    for tid,task in tasks.items():
        rec=by_id.get(tid,{})
        good=rec.get('status')=='verified_usable'
        arows=[]; drows=[]; mrows=[]
        for j,p in enumerate(task['test']):
            base=p['input']
            ag=(rec.get('test') or [None]*len(task['test']))[j] if good else None
            if not _valid(ag):
                ag=None
            dr=direct_grid(direct_refs.get(tid),j)
            if not _valid(dr):
                dr=None
            # Preserve recorded DIRECT primary; if two recorded baseline candidates exist,
            # use both before injecting a verified agent candidate.
            second_ref=None
            units=(direct_refs.get(tid) or {}).get('units', {})
            if j in units:
                u=units[j]
                if isinstance(u,dict):
                    c2=u.get('attempt_2')
                    if _valid(c2):second_ref=c2
            arows.append({'attempt_1':ag or base,'attempt_2':base})
            drows.append({'attempt_1':dr or base,'attempt_2':second_ref or base})
            if dr:
                # Preserve both baseline attempts if they differ, else include AGENT.
                c2=second_ref if second_ref is not None and second_ref != dr else (ag if ag and ag!=dr else base)
                mrows.append({'attempt_1':dr,'attempt_2':c2})
            else:
                mrows.append({'attempt_1':ag or base,'attempt_2':base})
        agent[tid]=arows;direct[tid]=drows;merged[tid]=mrows
    return agent,direct,merged


def evaluate(cfg):
    t0=time.time()
    output=Path(cfg.get('eval_out','/kaggle/working/agent_eval'))
    output.mkdir(parents=True,exist_ok=True)
    root=cfg.get('data_dir','/kaggle/input')
    # NEVER load solution labels until after synthesis is complete.
    ch=json.loads(Path(find('arc-agi_evaluation_challenges.json',root)).read_text())
    ids=[t for t in cfg['eval_task_ids'] if t in ch]
    if not ids:raise ValueError('No selected ARC evaluation IDs found')
    tasks={tid:ch[tid] for tid in ids}
    lock=threading.Lock();records=[]
    (output/'attempts.jsonl').write_text('')
    sampler=MetricsSampler(cfg.get('base_url','http://127.0.0.1:1234/v1'),
                           float(cfg.get('metrics_every_s',30))).start()
    def on_done(rec):
        with lock:
            records.append(rec)
            with (output/'attempts.jsonl').open('a') as f:
                f.write(json.dumps(rec,default=str)+'\n'); f.flush();os.fsync(f.fileno())
            atomic_json(output/'progress.json',{
                'attempts_done':len(records), 'tasks_total':len(tasks),
                'statuses':dict(Counter(r['status'] for r in records)),
                'generated_tokens':sum(int(r.get('gen_tokens',0)) for r in records),
                'elapsed_s':round(time.time()-t0,2)})
        print('[synth]',rec['task_id'],rec['status'],rec.get('gen_tokens'), 'tokens',flush=True)
    try:
        solve_synthesis(tasks,cfg,on_done=on_done)
    finally:
        try:
            atomic_json(output/'server_metrics.json',{
                'final':server_metrics(cfg.get('base_url','http://127.0.0.1:1234/v1')),
                'timeline':sampler.stop()})
        except Exception as e:
            atomic_json(output/'server_metrics.json',{'error':repr(e)})
    # Model-originated executable code is done; score in reporting phase only.
    ref=load_direct_ref(Path(__file__).with_name('direct_ref_eval12.json'),cfg.get('eval_direct_run','run7'))
    direct_refs={t:ref[t] for t in ids if t in ref}
    agent,direct,hybrid=make_submissions(tasks,records,direct_refs)
    atomic_json(output/'agent_submission.json',agent)
    atomic_json(output/'direct_reference_submission.json',direct)
    atomic_json(output/'eval_submission.json',hybrid)
    sol=json.loads(Path(find('arc-agi_evaluation_solutions.json',root)).read_text())
    sol_ids={tid:sol[tid] for tid in ids}
    n_outputs=sum(len(sol_ids[t]) for t in ids)
    ah,dh,hh=[hits(sub,sol_ids,ids) for sub in (agent,direct,hybrid)]
    a_total,a_macro,a_per=score(agent,sol_ids)
    d_total,d_macro,d_per=score(direct,sol_ids)
    h_total,h_macro,h_per=score(hybrid,sol_ids)
    keyed={r['task_id']:r for r in records}
    verified=[tid for tid in ids if keyed.get(tid,{}).get('status')=='verified_usable']
    wrong_verified=[tid for tid in verified if a_per.get(tid,0)<1]
    summary={
        'version':'v1.9.1','architecture':'longthink_program_verify','repair_turns':0,
        'tasks':len(ids),'test_outputs':n_outputs,
        # v1.9.1: official = Kaggle's metric = correct test outputs / all test outputs (as in v1.7-v1.8.3);
        # the task average is reported separately and is NOT comparable with the earlier runs' official numbers
        'agent_only_official_pct':round(100.0*sum(ah.values())/max(n_outputs,1),2),
        'direct_only_official_pct':round(100.0*sum(dh.values())/max(n_outputs,1),2),
        'official_score_pct':round(100.0*sum(hh.values())/max(n_outputs,1),2),
        'agent_query_hits':sum(ah.values()),
        'direct_query_hits':sum(dh.values()),'query_hits':sum(hh.values()),
        'task_macro_score_pct':round(h_macro,2),'agent_task_macro_pct':round(a_macro,2),
        'direct_task_macro_pct':round(d_macro,2),
        'think':{tid:{k:keyed.get(tid,{}).get(k) for k in ('think_tokens','think_source','think_hit_budget',
                                                            'gen_tokens','finish','program_source','request_wall_s')}
                 for tid in ids},
        'verified_usable_tasks':verified,'tasks_with_verified_program':verified,
        'verified_but_wrong_tasks':wrong_verified,
        'status_counts':dict(Counter(r['status'] for r in records)),
        'complementarity':{
            'agent_only':sum(ah[k] and not dh[k] for k in ah),
            'direct_only':sum(dh[k] and not ah[k] for k in dh),
            'both':sum(ah[k] and dh[k] for k in ah),
            'lost_vs_direct':sum(dh[k] and not hh[k] for k in dh),
            'oracle_union_hits':sum(ah[k] or dh[k] for k in ah),
            'raw_hybrid_gain_hits':sum(hh.values())-sum(dh.values()),
            'net_gain_is_heuristic':True,'split_decision_ready':False,
            'direct_source':'recorded: Q27-XHIGH v1.4.4 run 7 LB 26.94',
        },
        'gen_tokens_total':sum(int(r.get('gen_tokens',0)) for r in records),
        'wall_s':round(time.time()-t0,1),
        'per_task':{tid:{'status':keyed.get(tid,{}).get('status','missing'),
                          'agent_correct_frac':a_per.get(tid),
                          'direct_correct_frac':d_per.get(tid),
                          'hybrid_correct_frac':h_per.get(tid)} for tid in ids},
    }
    atomic_json(output/'eval_report.json',summary)
    print('v1.9 evaluation:',json.dumps({k:summary[k] for k in
        ('tasks','test_outputs','status_counts','agent_query_hits','direct_query_hits','query_hits','official_score_pct','agent_only_official_pct','direct_only_official_pct','wall_s')},default=str),flush=True)
    return summary

if __name__=='__main__':
    import sys
    evaluate(json.loads(sys.argv[1]))
