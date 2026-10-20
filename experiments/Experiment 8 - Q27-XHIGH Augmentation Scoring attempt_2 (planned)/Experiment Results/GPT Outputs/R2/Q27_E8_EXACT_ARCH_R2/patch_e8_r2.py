from pathlib import Path
import json, re, ast, hashlib, copy
import nbformat
ROOT=Path('/mnt/data')
IN=ROOT/'Q27_E8_EXACT_ARCH/Q27-XHIGH-v1_4_4_E8_EXACT_ARCH.ipynb'
DIR=ROOT/'Q27_E8_EXACT_ARCH_R2'
DIR.mkdir(exist_ok=True)
base=nbformat.read(ROOT/'Q27-XHIGH-v1_4_4_E7_ORDERING_ONLY.ipynb',as_version=4)
nb=nbformat.read(IN,as_version=4)
assert len(nb.cells)==12
cfg=nb.cells[2].source
needle='assert NOTEBOOK_T0 < ADMIT_DEADLINE < DRAIN_DEADLINE < HARD_STOP < VERIFY_DEADLINE < NOTEBOOK_T0 + TOTAL_BUDGET_SEC'
assert needle in cfg
budget='''# E8 architecture: guarantee a RESERVED post-primary window (not leftover runtime).
# Experimental allocation: 30 minutes before the original E7 hard stop; change explicitly when approved.
E8_RESERVED_SEC = float(os.environ.get("Q27X_E8_RESERVED_SEC", "1800"))
E8_FINALIZE_SEC = float(os.environ.get("Q27X_E8_FINALIZE_SEC", "120"))
assert 0 < E8_FINALIZE_SEC < E8_RESERVED_SEC
E8_GLOBAL_HARD_STOP = HARD_STOP
E8_STAGE_DEADLINE = E8_GLOBAL_HARD_STOP - E8_FINALIZE_SEC
PRIMARY_HARD_STOP = E8_GLOBAL_HARD_STOP - E8_RESERVED_SEC
assert NOTEBOOK_T0 < PRIMARY_HARD_STOP < E8_STAGE_DEADLINE < E8_GLOBAL_HARD_STOP < VERIFY_DEADLINE
HARD_STOP = PRIMARY_HARD_STOP  # Existing E7 scheduler uses HARD_STOP unchanged.
# Adjust only admission/drain windows, preserving each profile's original internal buffers.
if "drain_margin_sec" in P:
    DRAIN_DEADLINE = HARD_STOP - P["drain_margin_sec"]
    ADMIT_DEADLINE = DRAIN_DEADLINE - P["admit_margin_sec"]
else:
    DRAIN_DEADLINE = min(DRAIN_DEADLINE, HARD_STOP - 8 * 60)
    ADMIT_DEADLINE = min(ADMIT_DEADLINE, DRAIN_DEADLINE - 7 * 60)
'''
cfg=cfg.replace(needle,budget+'\n'+needle)
cfg=cfg.replace('CONFIG["profile_row"] = P','CONFIG["profile_row"] = P\nCONFIG["e8_reserved_sec"] = E8_RESERVED_SEC\nCONFIG["e8_finalize_sec"] = E8_FINALIZE_SEC\nCONFIG["e8_stage_deadline_min"] = round((E8_STAGE_DEADLINE-NOTEBOOK_T0)/60,1)')
nb.cells[2].source=cfg

sc=nb.cells[10].source
# feature active by default and config still exact
assert 'E8_STAGE_GUARD_SEC = 30.0' in sc
sc=sc.replace('E8_STAGE_GUARD_SEC = 30.0         # keep margin before HARD_STOP for final validation/reporting',
              'E8_STAGE_GUARD_SEC = 0.0          # Stage deadline set explicitly in Cell 1; no extra silent cuts')

# F1 correct validator & helper that covers EVAL selected challenge, hidden schema
assert 'exp_n = len(test_challenges[tid])' in sc
sc=sc.replace('''def e8_validate_submission_obj(sub):
    if not isinstance(sub, dict):
        raise ValueError('submission_not_dict')
    for tid in test_challenges:
        if tid not in sub or not isinstance(sub[tid], list):
            raise ValueError(f'missing_task_{tid}')
        exp_n = len(test_challenges[tid])''', '''def e8_validate_submission_obj(sub, challenges=None):
    challenges = test_challenges if challenges is None else challenges
    if not isinstance(sub, dict) or set(sub) != set(challenges):
        raise ValueError('submission_task_set_mismatch')
    for tid in challenges:
        if tid not in sub or not isinstance(sub[tid], list):
            raise ValueError(f'missing_task_{tid}')
        exp_n = len(challenges[tid]['test'])''')
assert 'exp_n = len(challenges[tid][\'test\'])' in sc

# F2 pool order
before='''    add(inp, "legacy_identity")
    add([list(reversed(r)) for r in inp], "legacy_mirror")
    for transpose in (False, True):
        for turns in range(4):
            if e8_is_train_supported(task, transpose, turns):
                add(e8_d4_transform(inp, transpose, turns), f"train_d4_t{int(transpose)}_r{turns}")
    mapped = e8_mapping_candidate(task, inp)
    if mapped is not None:
        add(mapped, "train_color_mapping")
    cap = max(int(max_candidates), baseline_count)
    return pool[:cap]'''
after='''    # F2: rank train-consistent solutions before legacy guesses; bounded cap is respected.
    for transpose in (False, True):
        for turns in range(4):
            if e8_is_train_supported(task, transpose, turns):
                add(e8_d4_transform(inp, transpose, turns), f"train_d4_t{int(transpose)}_r{turns}")
    mapped = e8_mapping_candidate(task, inp)
    if mapped is not None:
        add(mapped, "train_color_mapping")
    supported_count = len(pool)  # Never truncate any train-supported hypothesis.
    add(inp, "legacy_identity")
    add([list(reversed(r)) for r in inp], "legacy_mirror")
    cap = max(int(max_candidates), supported_count)
    return pool[:cap]'''
assert before in sc
sc=sc.replace(before,after)

# F5 tokenization explicit remove special duplication, chosen token alignment + donor parity tag
before='''    prefix_ids = tokenizer.encode(view['input'])
    answer_ids = tokenizer.encode(view['reply'])
    if not prefix_ids or not answer_ids:
        raise ValueError('empty_score_tokens')
    ids = prefix_ids + answer_ids'''
after='''    # Donor calc_scores also uses two tokenized pieces concatenated in token space.
    # Disable accidental extra BOS/EOS; these are already explicit in QwenFormatter text.
    prefix_ids = tokenizer.encode(view['input'], add_special_tokens=False)
    answer_ids = tokenizer.encode(view['reply'], add_special_tokens=False)
    if not prefix_ids or not answer_ids:
        raise ValueError('empty_score_tokens')
    ids = prefix_ids + answer_ids
    # The scored suffix is *exactly* the supplied answer token sequence (donor semantics).
    # Joint text tokenization may differ at a BPE boundary; record it as diagnostic,
    # never silently shift chosen-token logprob positions.
    joined_ids = tokenizer.encode(view['input'] + view['reply'], add_special_tokens=False)
    boundary_equal = joined_ids == ids
    assert ids[len(prefix_ids):] == answer_ids'''
assert before in sc
sc=sc.replace(before,after)
sc=sc.replace("'answer_start': len(prefix_ids), 'answer_ids': answer_ids,", "'answer_start': len(prefix_ids), 'answer_ids': answer_ids,\n        'boundary_equal': boundary_equal,")

# F6 request cleanup ambiguous registrations
before='''        except Exception:
            if pending:
                try: engine_obj.abort_request(list(pending))
                except Exception: pass
            raise
        pending[rid] = r'''
after='''        except Exception as add_exc:
            # add_request may register a request ID BEFORE raising. Abort that ID too.
            cleanup_ids = list(pending) + [rid]
            try:
                engine_obj.abort_request(cleanup_ids)
            except Exception as cleanup_exc:
                raise RuntimeError('e8_ambiguous_registration_cleanup_failed') from cleanup_exc
            raise
        pending[rid] = r'''
assert before in sc
sc=sc.replace(before,after)
# ensure abort in finally can fail closed (report error, don't continue stages)
sc=sc.replace('''            try: engine_obj.abort_request(list(pending))
            except Exception: pass
    return received''', '''            try:
                engine_obj.abort_request(list(pending))
            except Exception as cleanup_exc:
                raise RuntimeError('e8_pending_request_cleanup_failed') from cleanup_exc
    return received''')

# F4 EVAL selected submissions built from real results and scored after selection
start=sc.index('def e8_validate_submission_file(path):')
end=sc.index('def e8_write_baseline_checkpoint():')
sc=sc[:start]+'''def e8_validate_submission_file(path, challenges=None):
    obj = _load_json(path)
    e8_validate_submission_obj(obj, challenges)
    return obj


def e8_make_eval_generated_submission():
    """EVAL-only diagnostic predictions, separate from the hidden submission file."""
    sub = identity_submission(CHALLENGES)
    for uid, att in FINAL.items():
        tid, qi = UNIT_TASK[uid], UNIT_Q[uid]
        sub[tid][qi] = {
            'attempt_1': e8_matrix(att['attempt_1']),
            'attempt_2': e8_matrix(att['attempt_2']),
        }
    e8_validate_submission_obj(sub, CHALLENGES)
    return sub


def e8_actual_eval_results():
    if SOLUTIONS is None:
        return {}
    hits = {uid: int(score_unit(uid, FINAL[uid])['hit']) for uid in FINAL}
    return {
        'eval_e8_hits': int(sum(hits.values())),
        'eval_e8_total': len(UNITS),
        'eval_e8_hit_units': sorted(uid for uid, hit in hits.items() if hit),
    }


'''+sc[end:]
# checkpoint baseline generated EVAL
sc=sc.replace('''    write_json_atomic(ck, sub)
    return ck''', '''    write_json_atomic(ck, sub)
    if SOLUTIONS is not None:
        write_json_atomic(OUT / 'e8_eval_baseline_generated_submission.json',
                          e8_make_eval_generated_submission())
    return ck''')

# stage deadline explicit
sc=sc.replace('''        stage_deadline = HARD_STOP - E8_STAGE_GUARD_SEC''','''        stage_deadline = E8_STAGE_DEADLINE''')
# Final re-audit: Primary hard stop is now an expected boundary, not a reason to skip E8.
# Finished units may be scored after unfinished primary requests are aborted by the moved-up deadline.
sc=sc.replace("if SCHED_ERROR is not None or SCHED.get('aborted_at_hard_stop', 0) > 0:",
              "if SCHED_ERROR is not None:")
sc=sc.replace("skip('scheduler_error_or_hardstop')", "skip('scheduler_error')")
sc=sc.replace("        stage_deadline = E8_STAGE_DEADLINE", "        report['primary_hardstop_aborts'] = SCHED.get('aborted_at_hard_stop', 0)\n        stage_deadline = E8_STAGE_DEADLINE")

# stage run scheduler error gating? partial primary accepted. main error skip okay

# remove global UID hack e8_select_attempt2 pass uid, no extra source dependency
sc=sc.replace('''def e8_select_attempt2(frozen_attempt1, frozen_attempt2, ranked):
    protected = e8_baseline_is_train_supported(_E8_CURRENT_UID, frozen_attempt2)''', '''def e8_select_attempt2(uid, frozen_attempt1, frozen_attempt2, ranked):
    protected = e8_baseline_is_train_supported(uid, frozen_attempt2)''')
sc=sc.replace('''    global _E8_CURRENT_UID
    t0 = time.perf_counter()''','''    t0 = time.perf_counter()''')
sc=sc.replace('''                _E8_CURRENT_UID = uid
                selected2, reason, protected = e8_select_attempt2(baseline1, baseline2, ranked)''','''                selected2, reason, protected = e8_select_attempt2(uid, baseline1, baseline2, ranked)''')

# F4 assert attempt 1 frozen & EVAL scoring delta
sc=sc.replace("        selection_map = {}", "        selection_map = {}")
sc=sc.replace('''        stage_deadline = E8_STAGE_DEADLINE''', '''        stage_deadline = E8_STAGE_DEADLINE''')
# use immutable primary snapshot before apply
sc=sc.replace('''        # Apply protected attempt_2 selections atomically only after the stage has produced its map.
        original_attempt2 = {uid: _e8_deepcopy(FINAL[uid]['attempt_2']) for uid in selection_map}''', '''        # Apply protected attempt_2 selections only after stage has produced its map.
        primary_snapshot = {uid: _e8_deepcopy(a['attempt_1']) for uid, a in FINAL.items()}
        original_attempt2 = {uid: _e8_deepcopy(FINAL[uid]['attempt_2']) for uid in selection_map}''')
sc=sc.replace('''            write_submission()
            e8_validate_submission_file(SUBMISSION_PATH)
        except Exception:
            for uid, old in original_attempt2.items():
                FINAL[uid]['attempt_2'] = old
            write_submission()
            e8_validate_submission_file(SUBMISSION_PATH)
            raise''', '''            assert all(FINAL[uid]['attempt_1'] == frozen for uid, frozen in primary_snapshot.items()), \\
                'e8_modified_attempt_1'
            # EVAL's original write_submission() intentionally writes a test-set identity fallback.
            # Record true EVAL generated pairs in a separate file, and compute their actual hit rate.
            if SOLUTIONS is not None:
                write_json_atomic(OUT / 'e8_eval_final_generated_submission.json',
                                  e8_make_eval_generated_submission())
                report.update(e8_actual_eval_results())
                report['eval_e8_net_gain'] = (
                    report['eval_e8_hits'] - report['oracle_base_hits']
                )
            write_submission()
            e8_validate_submission_file(SUBMISSION_PATH)
        except Exception:
            for uid, old in original_attempt2.items():
                FINAL[uid]['attempt_2'] = old
            # Real rollback is from checkpoint, not from possibly modified live in-memory values.
            write_json_atomic(SUBMISSION_PATH, _load_json(ck))
            e8_validate_submission_file(SUBMISSION_PATH)
            if SOLUTIONS is not None:
                write_json_atomic(OUT / 'e8_eval_final_generated_submission.json',
                                  e8_make_eval_generated_submission())
            raise''')
# IMPORTANT: issue initial oracle_base_hits computed? oracle_base counted all eligible, yes.
sc=sc.replace('''        # Ensure a valid baseline submission still exists.
        try:
            if not SUBMISSION_PATH.exists():
                write_submission()
            e8_validate_submission_file(SUBMISSION_PATH)
        except Exception as exc2:
            report['error_recovery'] = repr(exc2)[:500]''', '''        # Ensure the baseline checkpoint is the recovery authority on any systemic failure.
        try:
            if report['checkpoint_written'] and report['checkpoint_path']:
                write_json_atomic(SUBMISSION_PATH, _load_json(Path(report['checkpoint_path'])))
            elif not SUBMISSION_PATH.exists():
                write_submission()
            e8_validate_submission_file(SUBMISSION_PATH)
        except Exception as exc2:
            report['error_recovery'] = repr(exc2)[:500]''')
# F7 isolate report I/O
sc=sc.replace('''        write_json_atomic(OUT / 'e8_augmented_scoring_report.json', report)
    return report''', '''        try:
            write_json_atomic(OUT / 'e8_augmented_scoring_report.json', report)
        except Exception as report_exc:
            print('E8 diagnostic write failed (submission retained):', repr(report_exc)[:300])
    return report''')
# Additional diagnostics for tokenizer boundary consistency and candidate-pool width.
sc=sc.replace("'units_scored': 0,", "'units_scored': 0,\n        'boundary_mismatch_views': 0,\n        'max_candidate_count': 0,")
sc=sc.replace("            candidates = e8_build_candidates(uid, baseline1, baseline2, E8_MAX_CANDIDATES)\n            if len(candidates) < 2:",
              "            candidates = e8_build_candidates(uid, baseline1, baseline2, E8_MAX_CANDIDATES)\n            report['max_candidate_count'] = max(report['max_candidate_count'], len(candidates))\n            if len(candidates) < 2:")
sc=sc.replace("                    requests = [e8_prepare_score_request(uid, c, v, i) for i, v in enumerate(views)]\n                    for i in range(0, len(requests), E8_BATCH_SIZE):",
              "                    requests = [e8_prepare_score_request(uid, c, v, i) for i, v in enumerate(views)]\n                    report['boundary_mismatch_views'] += sum(not r['boundary_equal'] for r in requests)\n                    for i in range(0, len(requests), E8_BATCH_SIZE):")

nb.cells[10].source=sc

# final report inject e8 current values and must run independent of diagnostic write
rep=nb.cells[11].source
rep=rep.replace('''    e8_validate_submission_file(SUBMISSION_PATH)
    rows = RECORDS''', '''    e8_validate_submission_file(SUBMISSION_PATH)
    if SOLUTIONS is not None:
        # Validate actual final EVAL generated pairs, not the original E7 RECORDS scores.
        E8_REPORT.update(e8_actual_eval_results())
        E8_REPORT.setdefault('oracle_base_hits', sum(
            r['score']['hit'] for r in RECORDS if r.get('score') is not None
        ))
        E8_REPORT['eval_e8_net_gain'] = E8_REPORT['eval_e8_hits'] - E8_REPORT['oracle_base_hits']
        write_json_atomic(OUT / 'e8_eval_final_generated_submission.json',
                          e8_make_eval_generated_submission())
    rows = RECORDS''')
# make report final scores clear separate fields
rep=rep.replace('''    if SOLUTIONS is not None:
        # Kaggle metric''','''    if SOLUTIONS is not None:
        report.update(e8_actual_eval_results())
        report['eval_baseline_hits'] = E8_REPORT.get('oracle_base_hits')
        report['eval_e8_net_gain'] = E8_REPORT.get('eval_e8_net_gain')
        # Kaggle metric''')
rep=rep.replace('''"deadlines_min":''','''"deadlines_min":''')
# ensure report contains both primary and secondary deadlines
rep=rep.replace('''"verify": VERIFY_DEADLINE}.items()}''','''"verify": VERIFY_DEADLINE, "e8_stage": E8_STAGE_DEADLINE,
                                "e8_original_hard": E8_GLOBAL_HARD_STOP}.items()}''')
# F4 console output must reflect actual selected E8 pair, with baseline shown separately.
rep=rep.replace('''    if SOLUTIONS is not None:
        print(f"EVAL SCORE (Kaggle metric, per query): {report['eval_query_hits']} / {report['eval_query_total']} queries = "
              f"{report['eval_score_pct']}% | task-normalised {report['eval_task_normalized_pct']}% | grids parsed "''', '''    if SOLUTIONS is not None:
        report['eval_e8_score_pct'] = round(100.0 * report['eval_e8_hits'] / max(len(UNITS), 1), 2)
        print(f"E8 FINAL EVAL SCORE (per query): {report['eval_e8_hits']} / {report['eval_query_total']} = "
              f"{report['eval_e8_score_pct']}% | baseline {report['eval_baseline_hits']} / {report['eval_query_total']} "
              f"| E8 net gain {report['eval_e8_net_gain']} | task-normalised baseline {report['eval_task_normalized_pct']}% | grids parsed "''')

nb.cells[11].source=rep

# markdown annotate time budget
nb.cells[8].source = nb.cells[8].source.replace('''The donor augmentation utilities''','''**Time allocation:** This experiment reserves 1,800 seconds before the original E7 hard stop (configurable via `Q27X_E8_RESERVED_SEC`), and retains 120 seconds for finalisation. The primary scheduler's admission/drain/hard-stop offsets are moved earlier; model/prompt/token caps remain unchanged. This allocation changes primary throughput by design, not silently.\n\nThe donor augmentation utilities''')

# smoke syntax and schema; assign stable notebook cell IDs without touching code source.
for i, c in enumerate(nb.cells):
    if not c.get('id'):
        c['id'] = f'q27-e8-r2-{i:02d}'
for i,c in enumerate(nb.cells):
    if c.cell_type=='code':
        try: ast.parse(c.source)
        except SyntaxError as e:
            print('SYNTAX ERROR',i,e)
            raise
nbformat.validate(nb)
P=DIR/'Q27-XHIGH-v1_4_4_E8_EXACT_ARCH_FIXED_R2.ipynb'
nbformat.write(nb,P)
print(P,'size',P.stat().st_size)
