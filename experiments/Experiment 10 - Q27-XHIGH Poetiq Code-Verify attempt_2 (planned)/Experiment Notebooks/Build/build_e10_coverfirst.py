"""E10 ORDER2 + COVER-FIRST (Claude, 2026-10-15, user request). Deterministic.

Why: E9 (DIRECT, 86k thinking for every unit, file order) scored LB 22.92, below v1.4.4 at 49k (26.94) and E7 49k +
ordering (27.92), although 86k beats 49k on EVAL120 at full coverage (36.9 % vs ~25 %). Most likely the hidden run could
not finish every unit in 12 h. Cover-first = coverage before depth.

Source: Q27-XHIGH-v1_4_4_E10_86K_ORDER2.ipynb (sha b62ae567...; GPU EVAL120 58/172). ONE change:
  * pass 1: every unit thinks up to PASS1_THINK = 49,152 tokens (the user's floor), ORDER2 order, as before otherwise
  * depth pass: once every unit has been admitted, units whose pass-1 thinking reached 49,152 without closing are resumed
    from their own thinking (prompt + pass-1 thinking ids, token-exact) for the rest of THINK_BUDGET = 77,824, ORDER2 rank
    first, while slots, time (admit deadline) and KV allow. A resumed trace closes naturally or gets the usual forced close.
  * KV guard: a resumed trace is admitted only if every running request can still reach its maximum length without
    preemption (Poetiq R5 hit 99.5 % KV with 9 preemptions)
  * answers: the deeper answer becomes attempt_1 and the pass-1 answer attempt_2 - a unit never loses its pass-1 answer
Everything else unchanged (prompt, ORDER2, 16 sequences, deadlines, EVAL120 commit default, hidden rerun). LOOPSTOP is not
included (one change at a time).
usage: python build_e10_coverfirst.py
"""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "Q27-XHIGH-v1_4_4_E10_86K_ORDER2.ipynb"
SRC_SHA = "b62ae567aa4d8af26c652dcfcfc16fdf8d1d4b8e1a32ba04c6e3892b71d0b6b2"
OUT = HERE.parent / "Q27-XHIGH-v1_4_4_E10_ORDER2_COVERFIRST.ipynb"


def once(text, old, new, where):
    if text.count(old) != 1:
        sys.exit(f"replacement not unique ({text.count(old)}) in {where}: {old[:80]!r}")
    return text.replace(old, new)


raw = SRC.read_bytes()
assert hashlib.sha256(raw).hexdigest() == SRC_SHA, "ORDER2 source changed"
nb = json.loads(raw)
c = nb["cells"]
src = lambda i: "".join(c[i]["source"])
put = lambda i, t: c[i].__setitem__("source", t.splitlines(keepends=True))

# ---------------------------------------------------------------- cell 1: configuration
t = src(2)
t = once(t, 'CONFIG = {k: globals()[k] for k in ["PROFILE"',
         '# COVER-FIRST (2026-10-15): E9 (86k thinking for every unit) scored LB 22.92 < 26.94 at 49k - most likely the hidden run\n'
         '# could not finish every unit. Pass 1 gives every unit 49,152 thinking tokens (coverage). Once every unit has been\n'
         '# admitted, traces that reached 49,152 without closing are resumed from their own thinking up to THINK_BUDGET (depth),\n'
         '# ORDER2 rank first, while slots, time and KV allow. The pass-1 answer is kept as the other attempt.\n'
         'COVER_FIRST = os.environ.get("Q27X_COVER_FIRST", "1") == "1"\n'
         'PASS1_THINK = 49152\n'
         'DEEPEN_MIN_TOKENS = 2048          # resume only when at least this much thinking budget is left after pass 1\n'
         'CONFIG = {k: globals()[k] for k in ["PROFILE"', "config block")
t = once(t, '"TOTAL_BUDGET_SEC", "MIN_THINK_BUDGET"]}',
         '"TOTAL_BUDGET_SEC", "MIN_THINK_BUDGET", "COVER_FIRST", "PASS1_THINK", "DEEPEN_MIN_TOKENS"]}', "config keys")
put(2, t)

# ---------------------------------------------------------------- cell 6: scheduler
t = src(7)
t = once(t, '''def admit(uid):
    st = {"uid": uid, "phase": 1, "prompt_ids": PROMPT_IDS[uid], "p1_ids": [], "p2_ids": [], "p2_prefix": [],
          "p1_finish": None, "p2_finish": None, "t_admit": time.perf_counter(), "t_done": None, "mode": None}
    STATE[uid] = st
    engine.add_request(_req(uid, 1), {"prompt_token_ids": list(st["prompt_ids"])},
                       make_params(P1_CAPS[uid], SEED + 10 * UNIT_INDEX[uid] + 1))''',
         '''def admit(uid):
    cap1 = min(PASS1_THINK, P1_CAPS[uid]) if COVER_FIRST else P1_CAPS[uid]     # COVER-FIRST: pass 1 = 49,152 thinking
    st = {"uid": uid, "phase": 1, "prompt_ids": PROMPT_IDS[uid], "p1_ids": [], "p2_ids": [], "p2_prefix": [],
          "p1_finish": None, "p2_finish": None, "t_admit": time.perf_counter(), "t_done": None, "mode": None, "cap1": cap1}
    STATE[uid] = st
    engine.add_request(_req(uid, 1), {"prompt_token_ids": list(st["prompt_ids"])},
                       make_params(cap1, SEED + 10 * UNIT_INDEX[uid] + 1))''', "admit")
t = once(t, '(ANSWERS_DIR / f"{UNIT_TASK[uid]}_q{UNIT_Q[uid]}.txt").write_text(',
         '(ANSWERS_DIR / f"{UNIT_TASK[uid]}_q{UNIT_Q[uid]}{\'_deep\' if st.get(\'deepen_of\') else \'\'}.txt").write_text(', "answer file")
t = once(t, '''    _record(uid, st, make_attempts(uid, grid))

def _record(uid, st, att):''',
         '''    if st.get("deepen_of"):
        _record_deepen(uid, st, grid); return
    _record(uid, st, make_attempts(uid, grid))
    # COVER-FIRST: a pass-1 trace that reached its 49,152 cap without closing is a depth candidate
    if (COVER_FIRST and reason == "forced_close" and st["p1_finish"] == "length"
            and P1_CAPS[uid] - len(st["p1_ids"]) >= DEEPEN_MIN_TOKENS):
        DEEPEN_CAND[uid] = list(st["p1_ids"]); SCHED["deepen_candidates"] += 1

def _record(uid, st, att):''', "finish dispatch")
t = once(t, '"order_score": ORDER_SCORE[UNIT_TASK[uid]], "admit_rank": ORDER_ADMIT_RANK[uid]}',
         '"order_score": ORDER_SCORE[UNIT_TASK[uid]], "admit_rank": ORDER_ADMIT_RANK[uid],\n'
         '           "cap1": st.get("cap1")}', "record cap1")
t = once(t, 'def run_scheduler():\n', '''# ==================================================================== COVER-FIRST depth pass (2026-10-15)
DEEPEN_SUFFIX = "~d"
DEEPEN_CAND = {}                  # base uid -> pass-1 thinking ids (reached the pass-1 cap without closing)
DEEP_RECORDS = []
DEEP_RECORDS_PATH = OUT / "deep_records.jsonl"
SCHED.update({"deepen_candidates": 0, "deepen_admitted": 0, "deepen_finished": 0, "deepen_changed_attempt1": 0,
              "deepen_not_started": 0})

def _reserve(st):
    """worst-case KV tokens a running request may still need: prompt + its whole generation budget + the answer"""
    cap = st.get("cap_resume") or st.get("cap1") or P1_CAPS.get(st["uid"], THINK_BUDGET)
    return len(st["prompt_ids"]) + cap + ANSWER_MAX_TOKENS

def deepen_fits(base, active):
    """COVER-FIRST KV guard: admit a resumed trace only if every running request can still reach its maximum length
    without preemption (Poetiq R5: 99.5 % KV, 9 preemptions with 16 long requests). One request always fits."""
    kv = ENGINE_INFO.get("kv_tokens")
    if not kv or not active:
        return True
    return sum(_reserve(s) for s in active) + len(PROMPT_IDS[base]) + P1_CAPS[base] + ANSWER_MAX_TOKENS <= kv

def admit_deepen(base):
    """resume the unit's own pass-1 thinking (token-exact continuation) for the rest of its thinking budget"""
    rid = base + DEEPEN_SUFFIX
    think = DEEPEN_CAND.pop(base)
    UNIT_TASK[rid], UNIT_Q[rid] = UNIT_TASK[base], UNIT_Q[base]
    UNIT_INDEX[rid] = UNIT_INDEX[base] + 200000
    remaining = P1_CAPS[base] - len(think)
    st = {"uid": rid, "deepen_of": base, "phase": 1, "prompt_ids": list(PROMPT_IDS[base]) + think, "p1_ids": [],
          "p2_ids": [], "p2_prefix": [], "p1_finish": None, "p2_finish": None, "t_admit": time.perf_counter(),
          "t_done": None, "mode": None, "cap_resume": remaining, "pass1_think": len(think)}
    STATE[rid] = st
    engine.add_request(_req(rid, 1), {"prompt_token_ids": list(st["prompt_ids"])},
                       make_params(remaining, SEED + 10 * UNIT_INDEX[rid] + 1))
    SCHED["deepen_admitted"] += 1

def _record_deepen(rid, st, grid):
    base = st["deepen_of"]
    before = dict(FINAL[base])
    changed = bool(valid_grid(grid) and grid != before["attempt_1"])
    if changed:       # the deeper answer leads; the pass-1 answer stays as the other attempt (nothing is lost)
        FINAL[base] = {"attempt_1": grid, "src_1": "deep", "attempt_2": before["attempt_1"], "src_2": before["src_1"]}
        SCHED["deepen_changed_attempt1"] += 1
    SCHED["deepen_finished"] += 1
    sc, sb = score_unit(base, FINAL[base]), score_unit(base, before)
    total_think = st["pass1_think"] + st["think_tokens"]
    rec = {"unit": rid, "deepen_of": base, "task_id": UNIT_TASK[base], "query": UNIT_Q[base],
           "mode": st["mode"] or "natural", "end_reason": st["end_reason"], "prompt_tokens": len(st["prompt_ids"]),
           "pass1_think": st["pass1_think"], "p1_tokens": len(st["p1_ids"]), "p2_tokens": len(st["p2_ids"]),
           "think_tokens_total": total_think, "think_closed": st["think_closed"], "p1_finish": st["p1_finish"],
           "p2_finish": st["p2_finish"], "sec": round(st["t_done"] - st["t_admit"], 1), "has_grid": st["has_grid"],
           "changed_attempt1": changed, "score_before": sb, "score_after": sc,
           "t_admit_min": round((st["t_admit"] - NOTEBOOK_T0) / 60, 2), "t_done_min": round((st["t_done"] - NOTEBOOK_T0) / 60, 2)}
    DEEP_RECORDS.append(rec)
    for r in RECORDS:                 # the unit's own record carries its final attempts and score
        if r["unit"] == base:
            r.update({"score": sc, "src": [FINAL[base]["src_1"], FINAL[base]["src_2"]], "deepened": True,
                      "deep_mode": rec["mode"], "deep_think_total": total_think})
    try:
        with open(DEEP_RECORDS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\\n")
    except Exception as e:
        print("deep record write failed:", rid, repr(e)[:200])
    print(f"[{(time.perf_counter() - NOTEBOOK_T0) / 60:6.1f} min] {rid:16s} DEEP {rec['mode']:12s} think {total_think:6d} "
          f"closed={rec['think_closed']} grid={rec['has_grid']} attempt_1 changed={changed}"
          + (f" hit {sc['hit']} (before {sb['hit']})" if sc else ""), flush=True)
    write_submission()

def run_scheduler():
''', "deepen block")
t = once(t, '''        while queue and len(active) < CONCURRENCY and now < ADMIT_DEADLINE and not drained:
            admit(queue.pop(0)); active = [s for s in STATE.values() if s["phase"] in (1, 2)]
''', '''        while queue and len(active) < CONCURRENCY and now < ADMIT_DEADLINE and not drained:
            admit(queue.pop(0)); active = [s for s in STATE.values() if s["phase"] in (1, 2)]
        # COVER-FIRST: depth only after every unit has been admitted; the most promising (ORDER2 rank) first
        while COVER_FIRST and not queue and DEEPEN_CAND and len(active) < CONCURRENCY and now < ADMIT_DEADLINE and not drained:
            _base = min(DEEPEN_CAND, key=lambda u: ORDER_ADMIT_RANK[u])
            if not deepen_fits(_base, active):
                break
            admit_deepen(_base); active = [s for s in STATE.values() if s["phase"] in (1, 2)]
''', "deepen admission")
t = once(t, '''        if not active:
            if not queue or now >= ADMIT_DEADLINE or drained:
                break''', '''        if not active:
            if (not queue and not (COVER_FIRST and DEEPEN_CAND)) or now >= ADMIT_DEADLINE or drained:
                break''', "idle break")
t = once(t, '    SCHED["not_admitted"] = len(queue)\n',
         '    SCHED["not_admitted"] = len(queue)\n    SCHED["deepen_not_started"] = len(DEEPEN_CAND)\n', "not started")
t = once(t, '    write_submission()\nprint("scheduler done:"',
         '''    write_submission()
    try:   # COVER-FIRST: records.jsonl ends with every unit's final attempts and score (deepened units updated)
        if DEEP_RECORDS:
            _tmp = RECORDS_PATH.with_suffix(".tmp")
            _tmp.write_text("".join(json.dumps(r) + "\\n" for r in RECORDS), encoding="utf-8")
            os.replace(_tmp, RECORDS_PATH)
    except Exception as _e:
        print("records rewrite failed:", repr(_e)[:200])
print("scheduler done:"''', "records rewrite")
put(7, t)

# ---------------------------------------------------------------- cell 7: report
t = src(8)
t = once(t, '    _live = sorted(r["prompt_tokens"] + r["p1_tokens"] + r["p2_tokens"] for r in rows)\n',
         '''    report["cover_first"] = {"enabled": COVER_FIRST, "pass1_think": PASS1_THINK, "think_budget": THINK_BUDGET,
                             "candidates": SCHED.get("deepen_candidates"), "admitted": SCHED.get("deepen_admitted"),
                             "finished": SCHED.get("deepen_finished"), "changed_attempt1": SCHED.get("deepen_changed_attempt1"),
                             "not_started": SCHED.get("deepen_not_started"),
                             "deep_modes": {m: sum(r["mode"] == m for r in DEEP_RECORDS) for m in ("natural", "continue", "forced_close")},
                             "deep_gen_tokens": sum(r["p1_tokens"] + r["p2_tokens"] for r in DEEP_RECORDS)}
    if SOLUTIONS is not None and DEEP_RECORDS:
        report["cover_first"]["deepened_hits_before"] = sum(r["score_before"]["hit"] for r in DEEP_RECORDS)
        report["cover_first"]["deepened_hits_after"] = sum(r["score_after"]["hit"] for r in DEEP_RECORDS)
    _live = sorted(r["prompt_tokens"] + r["p1_tokens"] + r["p2_tokens"] for r in rows)
''', "report block")
put(8, t)

c[0]["source"] = (src(0) + "\n\n**COVER-FIRST (2026-10-15):** E9 (86k thinking for every unit) scored LB 22.92 < 26.94 at 49k - the hidden "
                  "run most likely could not finish every unit. Pass 1 now gives every unit **49,152** thinking tokens (coverage). Once "
                  "every unit has been admitted, traces that reached 49,152 without closing are **resumed from their own thinking** up "
                  "to 77,824 (depth), ORDER2 rank first, while slots, time and KV allow (a KV guard prevents preemption). The deeper "
                  "answer becomes attempt_1 and the pass-1 answer attempt_2, so no unit loses its pass-1 answer. `deep_records.jsonl` "
                  "logs the depth pass; the report has `cover_first`. `Q27X_COVER_FIRST=0` gives ORDER2 behaviour.\n").splitlines(keepends=True)
for cell in c:
    if cell["cell_type"] == "code":
        cell["outputs"] = []; cell["execution_count"] = None
data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
OUT.write_bytes(data)
print("BUILT", OUT.name, len(data), "bytes sha256", hashlib.sha256(data).hexdigest())
