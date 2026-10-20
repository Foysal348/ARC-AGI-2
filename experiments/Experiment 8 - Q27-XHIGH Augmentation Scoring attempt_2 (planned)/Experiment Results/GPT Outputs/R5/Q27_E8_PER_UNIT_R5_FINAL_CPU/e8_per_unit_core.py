# E8 per-unit candidate mining + donor eight-view scoring (no extra model instance)
# Primary time configuration remains exactly the original E7 configuration.
import json as _e8_json
import math as _e8_math
import hashlib as _e8_hashlib
import re as _e8_re
import copy as _e8_copy
from collections import deque as _e8_deque

# These are experiment features, not diagnostic-only placeholders.
E8_ENABLED = True
E8_SAVE_FULL_THINKING = True
E8_AUG_COLOR_POLICY = 'donor'
E8_RANK_METRIC = 'mean_nll'
E8_MAX_NOVEL_CANDIDATES = 4
E8_MAX_SCORER_INFLIGHT = 1
E8_MAX_SCORE_REQUEST_TOKENS = MAX_MODEL_LEN - 8
E8_SHORT_NEW_COMPLETIONS_ENABLED = False  # separately validated option, not part of this prototype
# Fail closed when token-space prefix+reply cannot be decoded to the requested text.
# No unmeasured inference or primary vLLM memory settings are changed.
E8_STRICT_SCORING_TEXT_ALIGNMENT = True
E8_PRIMARY_ADMISSION_PRIORITY = True
assert E8_MAX_SCORER_INFLIGHT == 1, 'single scorer request concurrency preserves primary request routing'

E8_JOBS = _e8_deque()
E8_ACTIVE = None
E8_SERIAL = 0
E8_CURRENT = None
E8_BASELINE = {}
E8_REPORT = {'version':'E8_per_unit_R4', 'enabled':E8_ENABLED,
             'candidate_source':'full_thinking_and_answer', 'direct_generation':'not_enabled_unvalidated',
             'primary_deadlines':'exact_original_E7', 'score_requests_started':0,
             'score_views_finished':0, 'units_with_novel':0,
             'units_no_novel':0, 'units_one_novel':0, 'units_ranked':0,
             'units_incomplete':0, 'attempt2_replaced':0, 'errors':[], 'rows':[],
             'oracle_new_hits':0, 'scorer_disabled_reason':None,
             'gpu_logprobs_smoke_tested':False, 'boundary_mismatch_views':0,
             'score_request_prompt_tokens':[], 'score_request_answer_tokens':[],
             'primary_priority_enabled': E8_PRIMARY_ADMISSION_PRIORITY,
             'novel_context_rejections':0, 'novel_context_accepted':0,
             'scoring_skipped_unverified_alignment':0}


def e8_key(g):
    return tuple(tuple(int(v) for v in row) for row in g)


def e8_matrix(g):
    return [list(map(int,row)) for row in g]


def e8_d4_transform(grid, transpose, turns):
    arr = np.asarray(grid, dtype=int)
    if transpose: arr = arr.T
    return np.rot90(arr, k=turns).tolist()


def e8_is_train_supported(task, transpose, turns):
    return all(e8_d4_transform(p['input'], transpose, turns) == p['output'] for p in task['train'])


def e8_mapping_candidate(task, inp):
    mapping = {}
    for p in task['train']:
        a, b = p['input'], p['output']
        if len(a)!=len(b) or any(len(x)!=len(y) for x,y in zip(a,b)):
            return None
        for x,y in zip(a,b):
            for v,w in zip(x,y):
                if v in mapping and mapping[v]!=w: return None
                mapping[v]=w
    if any(v not in mapping for row in inp for v in row): return None
    return [[mapping[v] for v in row] for row in inp]


def e8_aug_seed(uid):
    return int.from_bytes(_e8_hashlib.sha256((uid+':donor_aug').encode()).digest()[:4], 'big')


def e8_zero_fixed_perm(_query):
    return 'permute0'+''.join(map(str, np.random.permutation(np.arange(1,10)).tolist()))


def e8_make_views(uid, candidate, policy='donor'):
    tid, qi = UNIT_TASK[uid], UNIT_Q[uid]
    task = CHALLENGES[tid]
    key = f'{tid}_{qi}'
    ds = ArcDataset(queries={key:{'train':task['train'], 'test':[task['test'][qi]]}},
                    replies={key:[e8_matrix(candidate)]}, keys=[key])
    state = np.random.get_state()
    try:
        if policy=='donor':
            views=ds.augment(seed=e8_aug_seed(uid))
        elif policy=='zero_preserving':
            np.random.seed(e8_aug_seed(uid))
            views=ds.mod(np.transpose,keep=True)
            views=views.mod(np.rot90,n=3,keep=True)
            views=views.mod(permute_mod,e8_zero_fixed_perm,n=1,shuffle=False,keep=False)
            views=views.shuffle_ex()
        else: raise ValueError(policy)
    finally:
        np.random.set_state(state)
    assert len(views.keys)==8
    return views.as_list(QwenFormatter(tokenizer))


def e8_prepare_score_request(uid, candidate, view, view_id):
    prefix = tokenizer.encode(view['input'], add_special_tokens=False)
    answer = tokenizer.encode(view['reply'], add_special_tokens=False)
    tokens=prefix+answer
    if not prefix or not answer: raise ValueError('empty_answer_or_prefix')
    if len(tokens)>E8_MAX_SCORE_REQUEST_TOKENS: raise ValueError('scoring_context_exceeded')
    # Token-space concatenation follows donor's exact scoring convention.
    joint=tokenizer.encode(view['input']+view['reply'],add_special_tokens=False)
    boundary_equal = (joint == tokens)
    # Donor scoring uses this exact forced token sequence (prefix+answer). An
    # independent joint encoding can differ at subword boundaries, but decoding
    # MUST still reconstruct the exact scoring text, or NLL means something else.
    decoded_equal = (tokenizer.decode(tokens,skip_special_tokens=False) == view['input']+view['reply'])
    if E8_STRICT_SCORING_TEXT_ALIGNMENT and (not boundary_equal or not decoded_equal):
        raise ValueError('scorer_unverified_token_boundary_or_decode')
    return {'uid':uid, 'key':e8_key(candidate['grid']), 'view':view_id,
            'tokens':tokens, 'answer':answer, 'answer_start':len(prefix),
            'boundary_equal':boundary_equal, 'decoded_equal':decoded_equal,
            'source':candidate['source']}


def e8_prompt_nll(prompt_logprobs, request):
    if prompt_logprobs is None or len(prompt_logprobs)!=len(request['tokens']):
        raise ValueError('scorer_missing_prompt_logprobs')
    score=0.0
    for off,tid in enumerate(request['answer']):
        pos=request['answer_start']+off
        item=prompt_logprobs[pos]
        if not isinstance(item,dict): raise ValueError('scorer_misaligned_prompt_logprobs')
        lp=item.get(tid, item.get(str(tid)))
        if lp is None: raise ValueError('scorer_missing_chosen_token')
        v=float(getattr(lp,'logprob',lp))
        if not _e8_math.isfinite(v) or v>0.001: raise ValueError('scorer_invalid_logprob')
        score-=v
    return score


# Context-aware, offset-preserving extraction. This is an empirical hypothesis
# filter, NOT a correctness oracle. It deliberately rejects unlabelled scratch
# grids to avoid promoting intermediate calculations as an output.
_E8_POSITIVE = _e8_re.compile(
    r'\b(?:candidate|alternative|hypothesis|possible|plausible|prediction|predicted|'
    r'proposed|final|answer|solution|output|result|guess)\b', _e8_re.I)
_E8_NEGATIVE = _e8_re.compile(
    r'\b(?:intermediate|scratch|visualization|illustration|example|demonstration|'
    r'training|input|incorrect|wrong|reject(?:ed)?|discard(?:ed)?|not\s+(?:the\s+)?final|'
    r'not\s+(?:an?\s+)?(?:answer|solution|output|candidate))\b', _e8_re.I)


def e8_context_tag(text, start, end, source):
    """Inspect the nearest explicit cue instead of global mentions of 'output'.

    A line/paragraph gap prevents an old example label leaking into a later grid.
    Negative explanations such as 'scratch ... not the final output' override
    positive words. Unlabelled reasoning grids remain diagnostic, not candidates.
    """
    pre = text[max(0,start-260):start]
    # Favor the closest heading or clause, stopping at an earlier grid boundary.
    pre = pre.split('```')[-1]
    pre = pre.split('\n\n')[-1]
    # A reasoning heading may span 2 short lines. Do not read unrelated paragraphs.
    pre = '\n'.join(pre.splitlines()[-3:])
    post = text[end:end+105]
    # Trailing explicit rejection ('... this is wrong') also disqualifies.
    tail = post.split('\n\n',1)[0].split('```',1)[0].lower()
    negative_pre = list(_E8_NEGATIVE.finditer(pre))
    positive_pre = list(_E8_POSITIVE.finditer(pre))
    if _e8_re.search(r'\b(?:wrong|incorrect|not\s+(?:the\s+)?(?:answer|solution|final))\b',tail):
        return 'reject_trailing_negation'
    # Negative cues on the SAME heading line always win. A later positive
    # hypothesis heading is allowed only if separated by a newline.
    if not positive_pre:
        return 'reject_no_output_hypothesis_cue'
    if negative_pre:
        neg=negative_pre[-1]
        pos=positive_pre[-1]
        if pos.start() < neg.end():
            return 'reject_intermediate_or_example'
        if '\n' not in pre[neg.end():pos.start()]:
            return 'reject_intermediate_or_example'
    return 'answer_hypothesis'


def e8_extract_all_grids(text):
    """Return provenance-preserving candidate hypotheses newest-first.

    Extractors (fenced, JSON, row-runs) overlap. Prefer fenced parsing and actual
    character offsets; discard input/training/scratch/unlabelled intermediate
    grids. No gold labels are accessed. An extracted grid is not known correct.
    """
    if not text: return []
    found = []
    # source kind precedence for two parses covering the identical text span.
    priority = {'fence':0,'json':1,'rows':2}
    spans=[]
    for m in _FENCE_RE.finditer(text):
        g = _grid_from_block(m.group(2))
        if valid_grid(g):
            spans.append((m.start(),m.end(),'fence',g))
    # JSON decoder: reject JSON representations inside known fenced blocks.
    decoder = _e8_json.JSONDecoder()
    for j,m in enumerate(_e8_re.finditer(r'\[\s*\[',text)):
        if j>=300:break
        try:
            g,end_offset=decoder.raw_decode(text[m.start():])
            g=_grid_from_obj(g)
            if valid_grid(g): spans.append((m.start(),m.start()+end_offset,'json',g))
        except (ValueError,TypeError,IndexError): pass
    lines=text.splitlines(keepends=True)
    run=[];run_start=0;offset=0
    # Explicitly ignore grid rows inside fences; they are parsed above.
    in_fence=False
    for line in lines+['']:
        stripped=line.strip()
        if stripped.startswith('```'):
            if run:
                spans.append((run_start,offset,'rows',run))
                run=[]
            in_fence=not in_fence
            offset+=len(line)
            continue
        row=_parse_row(line) if not in_fence else None
        if row is not None:
            if not run:run_start=offset
            run.append(row)
            if len(run)>30:run=[]
        else:
            if run:
                spans.append((run_start,offset,'rows',run))
                run=[]
        offset+=len(line)
    seen=set()
    # True reverse chronology; within a position prefer fenced > JSON > rows.
    for start,end,kind,g in sorted(spans,key=lambda v:(-v[0],priority[v[2]])):
        if not valid_grid(g):continue
        # Unlabelled singleton numeric fragments are never output hypotheses.
        if kind=='rows' and len(g)==1 and len(g[0])==1:continue
        tag=e8_context_tag(text,start,end,kind)
        if tag!='answer_hypothesis':
            E8_REPORT['novel_context_rejections']+=1
            continue
        key=e8_key(g)
        if key in seen:continue
        seen.add(key)
        found.append({'grid':e8_matrix(g),'where':kind,
                      'text_offset':start,'context':tag})
    return found


def e8_build_novel(uid,thinking,answer,attempt1,attempt2):
    task=CHALLENGES[UNIT_TASK[uid]]
    inp=task['test'][UNIT_Q[uid]]['input']
    mirror=[list(reversed(r)) for r in inp]
    excluded={e8_key(g) for g in (attempt1,attempt2,inp,mirror) if valid_grid(g)}
    for p in task['train']:
        excluded.add(e8_key(p['input']));excluded.add(e8_key(p['output']))
    sources=[];seen=set(excluded)
    def add(g,source,offset=None):
        if not valid_grid(g):return
        key=e8_key(g)
        if key in seen:return
        seen.add(key)
        sources.append({'grid':e8_matrix(g),'source':source,'text_offset':offset})
    # Within each source, hypotheses are sorted by TRUE text offset newest first.
    # Answer text precedes thinking; valid alternative grids only, not scratch.
    for v in e8_extract_all_grids(answer):
        add(v['grid'],'answer_'+v['where'],v['text_offset'])
    for v in e8_extract_all_grids(thinking):
        add(v['grid'],'thinking_'+v['where'],v['text_offset'])
    # Preserve cheap exact train-supported hypotheses as in the agreed architecture.
    for trans in (False,True):
        for turns in range(4):
            if e8_is_train_supported(task,trans,turns):
                add(e8_d4_transform(inp,trans,turns),f'train_d4_{int(trans)}_{turns}')
    mapped=e8_mapping_candidate(task,inp)
    if mapped is not None:add(mapped,'train_color_map')
    return sources

def e8_baseline_is_train_supported(uid, baseline_grid):
    task=CHALLENGES[UNIT_TASK[uid]]
    inp=task['test'][UNIT_Q[uid]]['input']
    if baseline_grid==inp:
        return all(p['input']==p['output'] for p in task['train'])
    mirror=[list(reversed(row)) for row in inp]
    if baseline_grid==mirror:
        return all([list(reversed(row)) for row in p['input']]==p['output'] for p in task['train'])
    return False


def e8_capture_thinking(uid, st, answer_text):
    ids=list(st['p1_ids'])
    if THINK_CLOSE_ID in ids:
        ids=ids[:ids.index(THINK_CLOSE_ID)]
    thinking=tokenizer.decode(ids, skip_special_tokens=True)
    if E8_SAVE_FULL_THINKING:
        d=OUT/'e8_full_thinking';d.mkdir(exist_ok=True,parents=True)
        (d/f'{UNIT_TASK[uid]}_q{UNIT_Q[uid]}.txt').write_text(thinking,encoding='utf-8')
    return thinking


def e8_on_primary_complete(uid, st, answer):
    """Always checkpoint BEFORE starting or applying the secondary attempt."""
    att=FINAL[uid]
    E8_BASELINE[uid]=_e8_copy.deepcopy(att)
    try:
        with (OUT/'e8_primary_baseline.jsonl').open('a',encoding='utf8') as f:
            f.write(_e8_json.dumps({'uid':uid,'attempt_1':att['attempt_1'],
                                    'attempt_2':att['attempt_2']})+'\n')
    except Exception as e:
        E8_REPORT['errors'].append({'uid':uid,'checkpoint':repr(e)[:150]})
        return  # never replace an uncheckpointed primary result
    try:
        thinking=e8_capture_thinking(uid,st,answer)
        novel=e8_build_novel(uid,thinking,answer,att['attempt_1'],att['attempt_2'])
    except Exception as e:
        E8_REPORT['errors'].append({'uid':uid,'candidate_builder':repr(e)[:150]})
        return
    E8_REPORT['units_with_novel']+=bool(novel)
    E8_REPORT['novel_context_accepted']+=sum(1 for v in novel if v['source'].startswith(('thinking_','answer_')))
    row={'uid':uid,'novel_count':len(novel),'sources':[v['source'] for v in novel[:E8_MAX_NOVEL_CANDIDATES]],
         'baseline_src_2':att['src_2'],'choice':'baseline'}
    if SOLUTIONS is not None:
        truth=SOLUTIONS[UNIT_TASK[uid]][UNIT_Q[uid]]
        row['base_hit']=bool(truth==att['attempt_1'] or truth==att['attempt_2'])
        row['oracle_new_hit']=any(v['grid']==truth for v in novel)
        row['oracle_within_cap_hit']=any(v['grid']==truth for v in novel[:E8_MAX_NOVEL_CANDIDATES])
        E8_REPORT['oracle_new_hits']+=bool(row['oracle_new_hit'] and not row['base_hit'])
        E8_REPORT.setdefault('oracle_new_hits_within_cap',0)
        E8_REPORT['oracle_new_hits_within_cap']+=bool(row['oracle_within_cap_hit'] and not row['base_hit'])
    E8_REPORT['rows'].append(row)
    if not novel:
        E8_REPORT['units_no_novel']+=1;return
    if len(novel)==1:
        E8_REPORT['units_one_novel']+=1
        # The single-novel route remains direct as user-approved, EXCEPT when
        # the original fallback itself exactly fits all training examples.
        # We cannot know hidden correctness; this is a deterministic safeguard.
        if e8_baseline_is_train_supported(uid,att['attempt_2']):
            row['choice']='baseline_train_consistent_protected'
            E8_REPORT.setdefault('units_baseline_train_protected',0)
            E8_REPORT['units_baseline_train_protected']+=1
            return
        e8_commit_second(uid,novel[0]['grid'],'one_novel_direct')
        return
    # Ranking is conditional: only 2+ novel candidates trigger GPU NLL scoring.
    E8_JOBS.append({'uid':uid,'candidates':novel[:E8_MAX_NOVEL_CANDIDATES],
                    'scores':{}, 'index':0,'requests':None})
    row['choice']='queued_for_8view_scoring'


def e8_commit_second(uid,grid,source):
    if not valid_grid(grid):raise ValueError('e8_invalid_secondary')
    original=E8_BASELINE[uid]
    if FINAL[uid]['attempt_1']!=original['attempt_1']:
        raise RuntimeError('e8_modified_primary')
    if grid==FINAL[uid]['attempt_1']:
        raise ValueError('e8_second_equals_first')
    prev=_e8_copy.deepcopy(FINAL[uid])
    try:
        FINAL[uid]['attempt_2']=e8_matrix(grid)
        FINAL[uid]['src_2']='e8_'+source
        write_submission()  # original E7 atomic writer in RERUN; EVAL remains diagnostic-only
    except Exception:
        FINAL[uid]=prev
        write_submission()
        raise
    if prev['attempt_2']!=grid:E8_REPORT['attempt2_replaced']+=1
    for row in reversed(E8_REPORT['rows']):
        if row['uid']==uid:
            row['choice']=source;row['changed']=prev['attempt_2']!=grid;break


def e8_pending_unit_ids():
    """One task counted once even when present both as current and queued."""
    ids={j['uid'] for j in E8_JOBS}
    if E8_CURRENT is not None:ids.add(E8_CURRENT['uid'])
    if E8_ACTIVE is not None:ids.add(E8_ACTIVE['request']['uid'])
    return ids


def e8_disable_scoring(reason):
    global E8_ACTIVE,E8_CURRENT
    if E8_REPORT['scorer_disabled_reason'] is None:
        E8_REPORT['scorer_disabled_reason']=str(reason)[:250]
    pending=e8_pending_unit_ids()
    E8_REPORT['units_incomplete']+=len(pending)
    E8_REPORT.setdefault('units_incomplete_ids',[])
    for uid in sorted(pending):
        if uid not in E8_REPORT['units_incomplete_ids']:
            E8_REPORT['units_incomplete_ids'].append(uid)
    if E8_ACTIVE is not None:
        rid=E8_ACTIVE['rid']
        try:engine.abort_request([rid])
        except Exception as e:E8_REPORT['errors'].append({'abort_failed':repr(e)[:200]})
    E8_ACTIVE=None;E8_CURRENT=None
    E8_JOBS.clear()

def e8_try_admit(active_primary_count,now):
    """Use no more than one scorer request and share the existing MAX_NUM_SEQS/CONCURRENCY.

    DOES NOT claim primary GPU throughput is unchanged; scoring prefill competes
    with primary decode despite identical admission/drain/hard-stop timestamps.
    """
    global E8_ACTIVE,E8_CURRENT,E8_SERIAL
    if not E8_ENABLED or E8_REPORT['scorer_disabled_reason'] or E8_ACTIVE is not None:
        return False
    if now>=HARD_STOP or active_primary_count>=CONCURRENCY:return False
    if E8_CURRENT is None:
        if not E8_JOBS:return False
        E8_CURRENT=E8_JOBS.popleft()
    job=E8_CURRENT
    try:
        if job['requests'] is None:
            reqs=[]
            for c in job['candidates']:
                for i,v in enumerate(e8_make_views(job['uid'],c['grid'],E8_AUG_COLOR_POLICY)):
                    reqs.append(e8_prepare_score_request(job['uid'],c,v,i))
            job['requests']=reqs
        if job['index']>=len(job['requests']):
            e8_complete_scoring(job)
            E8_CURRENT=None
            return e8_try_admit(active_primary_count,now)
        req=job['requests'][job['index']]
        E8_SERIAL+=1
        rid=f'e8|{E8_SERIAL}'
        params=SamplingParams(max_tokens=1, temperature=0.0,prompt_logprobs=1,
             detokenize=False,skip_special_tokens=False,
             output_kind=RequestOutputKind.FINAL_ONLY)
        try:
            engine.add_request(rid,{'prompt_token_ids':req['tokens']},params)
        except Exception:
            # Covers partial registration: abort even if add_request raised.
            try:engine.abort_request([rid])
            except Exception as err:raise RuntimeError('e8_partial_registration_cleanup_failed') from err
            raise
        E8_ACTIVE={'rid':rid,'request':req,'t_admit':now}
        E8_REPORT['score_requests_started']+=1
        E8_REPORT['score_request_prompt_tokens'].append(len(req['tokens']))
        E8_REPORT['score_request_answer_tokens'].append(len(req['answer']))
        E8_REPORT.setdefault('boundary_mismatch_views',0)
        E8_REPORT['boundary_mismatch_views']+=int(not req['boundary_equal'])
        return True
    except Exception as exc:
        E8_REPORT['errors'].append({'scorer_admit':repr(exc)[:300]})
        if 'scorer_unverified_token_boundary' in str(exc):
            E8_REPORT['scoring_skipped_unverified_alignment']+=1
        e8_disable_scoring('scorer_admission_or_preparation_failed')
        return False


def e8_on_score_output(output):
    """Call from main engine.step() dispatcher, never from a nested step loop."""
    global E8_ACTIVE,E8_CURRENT
    if E8_ACTIVE is None or output.request_id!=E8_ACTIVE['rid']:
        E8_REPORT['errors'].append({'unknown_score_request':str(output.request_id)[:100]})
        return
    if not output.finished:return
    req=E8_ACTIVE['request']
    try:
        nll=e8_prompt_nll(output.prompt_logprobs,req)
        E8_CURRENT['scores'][(req['key'],req['view'])]=(nll,len(req['answer']))
        E8_CURRENT['index']+=1
        E8_REPORT['score_views_finished']+=1
        if E8_CURRENT['index']==len(E8_CURRENT['requests']):
            e8_complete_scoring(E8_CURRENT)
            E8_CURRENT=None
    except Exception as exc:
        E8_REPORT['errors'].append({'uid':req['uid'],'scorer_output':repr(exc)[:300]})
        e8_disable_scoring('scorer_logprob_failure')
    finally:
        E8_ACTIVE=None


def e8_complete_scoring(job):
    rows=[]
    for c in job['candidates']:
        ck=e8_key(c['grid'])
        scores=[job['scores'].get((ck,i)) for i in range(8)]
        if any(v is None for v in scores):
            E8_REPORT['units_incomplete']+=1;return
        total=sum(v[0] for v in scores)
        mean=sum(v[0]/v[1] for v in scores)/8
        rows.append((mean if E8_RANK_METRIC=='mean_nll' else total,c,mean,total))
    rows.sort(key=lambda v:v[0])
    winner=rows[0][1]
    baseline=E8_BASELINE[job['uid']]['attempt_2']
    if e8_baseline_is_train_supported(job['uid'],baseline):
        for r in reversed(E8_REPORT['rows']):
            if r['uid']==job['uid']:
                r['choice']='baseline_train_consistent_protected';break
    else:
        e8_commit_second(job['uid'],winner['grid'],'ranked_8view')
    E8_REPORT['units_ranked']+=1
    for row in reversed(E8_REPORT['rows']):
        if row['uid']==job['uid']:
            row['ranked']=[{'source':c['source'],'mean_nll':round(mean,5),'sum_nll':round(total,3)}
                           for _,c,mean,total in rows]
            break


def e8_finish_pending():
    """At primary stop, cancel all incomplete scorer jobs exactly once."""
    if E8_ACTIVE is not None or E8_CURRENT is not None or E8_JOBS:
        e8_disable_scoring('primary_hard_stop')

def e8_final_report():
    report={**E8_REPORT}
    report['units_primary_finished']=len(FINAL)
    report['total_primary_queued']=len(UNITS)
    report['scorer_contention_steps']=SCHED.get('e8_mixed_primary_scorer_steps',0)
    report['scorer_max_concurrent']=SCHED.get('e8_max_active',0)
    report['primary_deadline_offsets']={'admit':round((ADMIT_DEADLINE-NOTEBOOK_T0)/60,2),
       'drain':round((DRAIN_DEADLINE-NOTEBOOK_T0)/60,2),
       'hard_stop':round((HARD_STOP-NOTEBOOK_T0)/60,2)}
    return report


def e8_validate_submission_obj(sub,challenges=None):
    challenges=test_challenges if challenges is None else challenges
    if not isinstance(sub,dict) or set(sub)!=set(challenges):
        raise ValueError('submission_task_set_mismatch')
    for tid,t in challenges.items():
        if not isinstance(sub[tid],list) or len(sub[tid])!=len(t['test']):
            raise ValueError(f'bad_task_len_{tid}')
        for i,item in enumerate(sub[tid]):
            if not isinstance(item,dict):raise ValueError('invalid_query_item')
            if not valid_grid(item.get('attempt_1')) or not valid_grid(item.get('attempt_2')):
                raise ValueError(f'invalid_grid_{tid}_{i}')
    return True


def e8_make_eval_generated_submission():
    sub=identity_submission(CHALLENGES)
    for uid,att in FINAL.items():
        tid,qi=UNIT_TASK[uid],UNIT_Q[uid]
        sub[tid][qi]={'attempt_1':e8_matrix(att['attempt_1']),'attempt_2':e8_matrix(att['attempt_2'])}
    e8_validate_submission_obj(sub,CHALLENGES)
    return sub


def e8_actual_eval_results():
    if SOLUTIONS is None:return {}
    hits={uid:int(score_unit(uid,att)['hit']) for uid,att in FINAL.items()}
    return {'eval_e8_hits':sum(hits.values()), 'eval_e8_total':len(UNITS),
            'eval_e8_hit_units':sorted(uid for uid,v in hits.items() if v)}
