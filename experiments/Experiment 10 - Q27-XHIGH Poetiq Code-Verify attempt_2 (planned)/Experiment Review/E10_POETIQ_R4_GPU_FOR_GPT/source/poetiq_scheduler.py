# E10 — Poetiq iterative program-synthesis solver on the existing Q27 86K vLLM engine.
import concurrent.futures as futures
from collections import deque

# Reinstated from the original E10 base; required at the first engine step.
LOG_EVERY_SEC = 120.0

E10_RECORDS = []
E10_RESULT_PATH = OUT / 'poetiq_iterations.jsonl'
E10_PROGRAM_DIR = OUT / 'poetiq_programs'; E10_PROGRAM_DIR.mkdir(parents=True,exist_ok=True)
E10_FINAL = {} # task -> list of Kaggle attempt pairs; EVAL written separately
E10_STATES = {}
E10_QUEUE = deque()
E10_ACTIVE = {} # (tid, expert) -> running phase state
E10_VERIFY = {} # future -> (tid, expert, result)
E10_TOTAL_GEN = 0
E10_SCHED = {'started':time.perf_counter(),'requests':0,'iterations_finished':0,'programs_parsed':0,
             'train_passed':0,'expert_count':POETIQ_NUM_EXPERTS,'max_iterations':POETIQ_MAX_ITERATIONS,
             'max_active':0,'aborted':0,'steps':0,'skipped_context':0,'deadline_not_started':0,
             'sandbox_verifications':0,'voting_updates':0,'feedback_trims':0,'sandbox_startup':E10_SANDBOX_SELFTEST}

for _task_idx, _tid in enumerate(TASK_IDS):
    for _expert in range(POETIQ_NUM_EXPERTS):
        _key=(_tid,_expert)
        E10_STATES[_key] = {'task_id':_tid,'expert':_expert,'task_index':_task_idx,
                            'iteration':0,'finished':False,'history':[], 'candidate_history':[], 'best':None,
                            'best_score':-1.0,'last_error':None,'status':'queued'}
        E10_QUEUE.append(_key)

def e10_render_request(st, max_feedback=None):
    it=st['iteration']
    idx = POETIQ_SOLVER_VARIANTS[st['expert'] % len(POETIQ_SOLVER_VARIANTS)]
    limit = POETIQ_MAX_SOLUTIONS if max_feedback is None else max_feedback
    msg = poetiq_build_message(CHALLENGES[st['task_id']],it,st['expert'],st['history'],
           variant=idx,selection_probability=POETIQ_SELECTION_PROB,
           max_solutions=limit,improving_order=POETIQ_IMPROVING_ORDER,
           shuffle=POETIQ_SHUFFLE_EXAMPLES,seed=POETIQ_SEED+st['task_index']*1000+st['expert']*100)
    chat=[{'role':'user','content':msg}]
    return tokenizer.apply_chat_template(chat,tokenize=False,add_generation_prompt=True,
                        enable_thinking=True,reasoning_effort=REASONING_EFFORT)

def e10_seed(st, phase):
    return SEED + st['task_index']*1000 + st['expert']*100 + st['iteration']*2 + int(phase)

def e10_make_params(cap,seed):
    return SamplingParams(max_tokens=int(cap),seed=int(seed),
                          detokenize=False,skip_special_tokens=False,
                          output_kind=RequestOutputKind.DELTA,
                          temperature=POETIQ_SOLVER_TEMPERATURE,
                          top_p=SAMPLING['top_p'],top_k=SAMPLING['top_k'])

def e10_req(st,phase):
    return f"{st['task_id']}~e{st['expert']}~i{st['iteration']}|{phase}"

def e10_admit(key):
    st=E10_STATES[key]
    rendered=e10_render_request(st)
    ids=tokenizer.encode(rendered,add_special_tokens=False)
    cap=phase1_cap(len(ids)) # preserve original 86K configured allowance
    # Never admit below the user's 49,152 minimum. Remove only low-score
    # historical feedback examples, retaining Poetiq's best-score ordering.
    # Entire history remains intact for future iterations and voting.
    trimmed_feedback=False
    if cap < 49152 and st['history']:
        for keep in range(min(POETIQ_MAX_SOLUTIONS,len(st['history']))-1,-1,-1):
            rendered=e10_render_request(st,max_feedback=keep)
            ids=tokenizer.encode(rendered,add_special_tokens=False)
            cap=phase1_cap(len(ids))
            if cap >= 49152:
                trimmed_feedback=True
                break
    if trimmed_feedback:
        E10_SCHED['feedback_trims']+=1
    if cap < 49152:
        st['finished']=True; st['status']='insufficient_context'
        st['last_error']=f'prompt {len(ids)} tokens leaves {cap} thinking room'
        E10_SCHED['skipped_context']+=1
        return 'skipped'
    req={'key':key,'uid':e10_req(st,1),'phase':1,'prompt_ids':ids,'p1_cap':cap,
          'p1_ids':[],'p2_ids':[],'p2_prefix':[],'p1_finish':None,'p2_finish':None,
          't_admit':time.perf_counter(),'mode':None}
    # Commit active state only when the vLLM request is registered.
    engine.add_request(e10_req(st,1),{'prompt_token_ids':list(ids)},e10_make_params(cap,e10_seed(st,1)))
    E10_ACTIVE[key]=req;st['status']='running'
    E10_SCHED['requests']+=1
    return 'admitted'

def e10_start_phase2(req,answer_cap=ANSWER_MAX_TOKENS):
    st=E10_STATES[req['key']]
    closed=THINK_CLOSE_ID in req['p1_ids']
    if not closed:
        while req['p1_ids'] and req['p1_ids'][-1] in EOS_IDS: req['p1_ids'].pop()
    pref=list(req['prompt_ids'])+list(req['p1_ids'])+([] if closed else list(STOP_IDS))
    cap=min(int(answer_cap),MAX_MODEL_LEN-len(pref)-8)
    if cap<64:return False
    # Do not advance phase before successful registration.
    engine.add_request(e10_req(st,2),{'prompt_token_ids':pref},e10_make_params(cap,e10_seed(st,2)))
    req['phase']=2;req['p2_prefix']=[] if closed else list(STOP_IDS)
    req['mode']='continue' if closed else 'forced_close'
    return True

def e10_answer_text(req):
    ids=list(req['p1_ids'])+list(req['p2_prefix'])+list(req['p2_ids'])
    if THINK_CLOSE_ID not in ids:
        return '',False
    idx=ids.index(THINK_CLOSE_ID)
    return tokenizer.decode(ids[idx+1:],skip_special_tokens=True), True

def e10_after_decode(req,reason,pool):
    key=req['key'];st=E10_STATES[key]
    E10_ACTIVE.pop(key,None)
    text, closed=e10_answer_text(req)
    code=poetiq_parse_code(text)
    E10_SCHED['iterations_finished']+=1
    base_rec={'task_id':st['task_id'],'expert':st['expert'],'iteration':st['iteration']+1,
              'mode':req['mode'] or 'natural','finish':reason,
              'p1_tokens':len(req['p1_ids']),'p2_tokens':len(req['p2_ids']),
              'prompt_tokens':len(req['prompt_ids']),'p1_cap':req['p1_cap'],
              'sec':round(time.perf_counter()-req['t_admit'],2),'has_python':bool(code)}
    if code:
        E10_SCHED['programs_parsed']+=1
        try:
            (E10_PROGRAM_DIR / f"{st['task_id']}_e{st['expert']}_i{st['iteration']+1}.py").write_text(code,encoding='utf-8')
        except Exception:pass
        try:
            future=pool.submit(poetiq_verify_code,code,CHALLENGES[st['task_id']],POETIQ_SANDBOX_TIMEOUT)
            E10_VERIFY[future]=(key,code,base_rec)
            st['status']='verifying'
            return
        except Exception as exc:
            base_rec['verification_submit_error']=repr(exc)
    st['last_error']='missing code block' if not code else 'verification scheduling failed'
    e10_commit_iteration(key,None,None,base_rec)

def e10_commit_iteration(key,code,verification,base_rec):
    st=E10_STATES[key]
    result=None
    if verification:
        result=dict(verification)
        result.update({'task_id':st['task_id'],'expert':st['expert'],
                       'iteration':st['iteration']+1,'code':code})
        feedback=result['feedback'];score=result['score']
        st['history'].append({'code':code,'feedback':feedback,'score':score})
        # All executed programs stay available for a second diverse attempt.
        # Best/expert voting remains the original primary selection mechanism.
        st['candidate_history'].append(result)
        usable=bool(result.get('test_usable',False))
        if usable and score >= st['best_score']:
            st['best_score']=score;st['best']=result
        if result['passed'] and usable:
            st['finished']=True;st['status']='train_verified'
            E10_SCHED['train_passed']+=1
        elif result['passed'] and not usable:
            st['status']='train_pass_test_unusable'
    base_rec['score']=result['score'] if result else None
    base_rec['train_passed']=bool(result and result['passed'])
    E10_RECORDS.append(base_rec)
    with open(E10_RESULT_PATH,'a',encoding='utf-8') as log:log.write(json.dumps(base_rec)+'\n')
    st['iteration']+=1
    if st['iteration']>=POETIQ_MAX_ITERATIONS:st['finished']=True
    if not st['finished']:
        st['status']='queued';E10_QUEUE.append(key)
    elif st['status']!='train_verified':st['status']='iterations_exhausted'
    e10_update_task_submission(st['task_id'])
    print(f"[E10] {st['task_id']} expert={st['expert']} iter={st['iteration']} "
          f"score={base_rec['score']} passed={base_rec['train_passed']} status={st['status']}",flush=True)

def e10_poll_verifications():
    for fut in list(E10_VERIFY):
        if not fut.done():continue
        key,code,rec=E10_VERIFY.pop(fut)
        try:
            result=fut.result()
        except Exception as err:
            result=None;rec['verify_error']=repr(err)[:1000]
        E10_SCHED['sandbox_verifications']+=1
        e10_commit_iteration(key,code,result,rec)

def e10_rank_task(tid):
    # Preserve Poetiq's main expert-best voting ranking. Append earlier valid
    # (possibly partial-train) program histories only as extra distinct
    # candidates for attempt_2, never ahead of the donor's top experts.
    expert_bests=[s['best'] for (task,_),s in E10_STATES.items() if task==tid and s['best']]
    voted=poetiq_vote(expert_bests,use_new_voting=POETIQ_USE_NEW_VOTING,
          count_failed_matches=POETIQ_COUNT_FAILED_MATCHES,
          iters_tiebreak=POETIQ_ITERS_TIEBREAK,
          low_to_high_iters=POETIQ_LOW_TO_HIGH_ITERS)
    already={id(r) for r in voted}
    history=[r for (task,_),s in E10_STATES.items() if task==tid
             for r in s['candidate_history']
             if id(r) not in already and r.get('test_usable')]
    if history:
        # Same donor voting/soft-score semantics for historical candidates;
        # no synthetic identity/mirror candidates are introduced.
        voted.extend(poetiq_vote(history,use_new_voting=POETIQ_USE_NEW_VOTING,
            count_failed_matches=POETIQ_COUNT_FAILED_MATCHES,
            iters_tiebreak=POETIQ_ITERS_TIEBREAK,
            low_to_high_iters=POETIQ_LOW_TO_HIGH_ITERS))
    return voted

def e10_update_task_submission(tid):
    ranked=e10_rank_task(tid)
    E10_FINAL[tid]=poetiq_two_attempts(CHALLENGES[tid],ranked)
    E10_SCHED['voting_updates']+=1
    e10_write_submission()

def e10_write_submission():
    submission=identity_submission(test_challenges)
    if RERUN:
        for tid,rows in E10_FINAL.items():
            if tid in submission and len(submission[tid])==len(rows):submission[tid]=rows
    # In evaluation, preserve the competition-format test fallback.
    write_json_atomic(SUBMISSION_PATH,submission)

E10_METRICS=[]
E10_SCHED_ERROR=None

def e10_run_solver():
    global E10_TOTAL_GEN
    workers=max(1,POETIQ_SANDBOX_WORKERS)
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        last_log=time.perf_counter();drained=False
        while True:
            now=time.perf_counter()
            e10_poll_verifications()
            if now>=HARD_STOP:
                ids=[e10_req(E10_STATES[key],r['phase']) for key,r in E10_ACTIVE.items()]
                if ids:engine.abort_request(ids)
                E10_SCHED['aborted']+=len(ids)
                for req in list(E10_ACTIVE.values()):e10_after_decode(req,'hard_stop',pool)
                break
            if now>=DRAIN_DEADLINE and not drained:
                drained=True
                p1=[r for r in E10_ACTIVE.values() if r['phase']==1]
                if p1:engine.abort_request([e10_req(E10_STATES[r['key']],1) for r in p1])
                for req in p1:
                    req['p1_finish']='drained'
                    if not e10_start_phase2(req,DRAIN_ANSWER_MAX_TOKENS):e10_after_decode(req,'drain_no_room',pool)
            while E10_QUEUE and len(E10_ACTIVE)<CONCURRENCY and now<ADMIT_DEADLINE and not drained:
                key=E10_QUEUE[0]
                if E10_STATES[key]['finished']:
                    E10_QUEUE.popleft();continue
                outcome=e10_admit(key)
                if outcome in ('admitted','skipped'):
                    E10_QUEUE.popleft()
                now=time.perf_counter()
            E10_SCHED['max_active']=max(E10_SCHED['max_active'],len(E10_ACTIVE))
            if not E10_ACTIVE:
                if E10_VERIFY:
                    time.sleep(.05);continue
                if not E10_QUEUE or now>=ADMIT_DEADLINE or drained:break
                time.sleep(.01);continue
            outs=engine.step();E10_SCHED['steps']+=1
            for o in outs:
                match=re.fullmatch(r'(.+)~e(\d+)~i(\d+)\|([12])',o.request_id)
                if not match:raise RuntimeError(f'Unexpected request ID: {o.request_id}')
                tid,ex,iteration,phase=match.group(1),int(match.group(2)),int(match.group(3)),int(match.group(4))
                key=(tid,ex)
                req=E10_ACTIVE.get(key)
                if req is None or req['phase']!=phase:continue
                if E10_STATES[key]['iteration']!=iteration:continue
                comp=o.outputs[0] if o.outputs else None
                if comp is not None:
                    token_ids=list(comp.token_ids);E10_TOTAL_GEN+=len(token_ids)
                    (req['p1_ids'] if phase==1 else req['p2_ids']).extend(token_ids)
                if not o.finished:continue
                reason=comp.finish_reason if comp is not None else None
                if phase==1:
                    req['p1_finish']=reason
                    if THINK_CLOSE_ID in req['p1_ids'] and reason!='length':
                        e10_after_decode(req,'natural',pool)
                    elif not e10_start_phase2(req):
                        e10_after_decode(req,'no_room_for_code',pool)
                else:
                    req['p2_finish']=reason
                    e10_after_decode(req,req['mode'],pool)
            if time.perf_counter()-last_log >= LOG_EVERY_SEC:
                last_log=time.perf_counter()
                E10_METRICS.append({'t_sec':round(last_log-NOTEBOOK_T0,1),
                                    'active':len(E10_ACTIVE),'queue':len(E10_QUEUE),
                                    'verifying':len(E10_VERIFY),'metrics':metrics_snapshot()})
                print(f"[E10] {(last_log-NOTEBOOK_T0)/60:.1f} min active={len(E10_ACTIVE)} "
                      f"pending={len(E10_QUEUE)} verify={len(E10_VERIFY)} "
                      f"verified_programs={E10_SCHED['train_passed']} tokens={E10_TOTAL_GEN}",flush=True)
        # Harvest verifications that have already finished without launching new iterations.
        endwait=min(HARD_STOP,time.perf_counter()+POETIQ_SANDBOX_TIMEOUT*2)
        while E10_VERIFY and time.perf_counter()<endwait:
            e10_poll_verifications()
            if E10_VERIFY:time.sleep(.05)
        E10_SCHED['remaining_verifications']=len(E10_VERIFY)
    E10_SCHED['not_started']=sum(1 for s in E10_STATES.values() if s['iteration']==0 and not s['finished'])
    E10_SCHED['wall_sec']=round(time.perf_counter()-E10_SCHED['started'],1)
    E10_SCHED['tokens']=E10_TOTAL_GEN
    E10_SCHED['tok_per_sec']=round(E10_TOTAL_GEN/max(1,E10_SCHED['wall_sec']),1)

try:
    if ENGINE_READY:
        e10_run_solver()
    else:
        E10_SCHED['skipped']='engine_unavailable'
except BaseException:
    E10_SCHED_ERROR=traceback.format_exc()
    print('[E10] solver error:',E10_SCHED_ERROR[-3500:],flush=True)
    try:
        _ids=[e10_req(E10_STATES[key],r['phase']) for key,r in E10_ACTIVE.items()]
        if _ids:engine.abort_request(_ids)
    except BaseException:pass
finally:
    for _tid in TASK_IDS:
        e10_update_task_submission(_tid)
    E10_SCHED['error']=E10_SCHED_ERROR[-4000:] if E10_SCHED_ERROR else None
    E10_SCHED.setdefault('wall_sec',round(time.perf_counter()-E10_SCHED['started'],1))
    E10_SCHED.setdefault('tokens',E10_TOTAL_GEN)
    try:write_json_atomic(OUT/'poetiq_vllm_metrics.json',E10_METRICS)
    except Exception:pass
    try:write_json_atomic(OUT/'poetiq_final_generated.json',E10_FINAL)
    except Exception:pass
    e10_write_submission()
print('[E10] solver complete',json.dumps({k:v for k,v in E10_SCHED.items() if k!='error'}),flush=True)
