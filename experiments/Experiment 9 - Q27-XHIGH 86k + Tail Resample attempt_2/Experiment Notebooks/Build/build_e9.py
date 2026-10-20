"""Experiment 9 builder (Claude, 2026-10-13). Deterministic; every replacement must hit exactly once.

Base: v1.4.4 DIRECT (LB 26.94), source/Q27-XHIGH-v1_4_4_LB26.94.ipynb. NO puzzle ordering (original v1.4.4 order).
Changes:
  1. 86k budget: THINK_BUDGET 49,152 -> 77,824 (answer 8,192 unchanged), MAX_MODEL_LEN 81,920 -> 106,496,
     EXPECTED_LIVE_TOKENS_PER_SEQ 30,000 -> 43,000, phase1_cap asserts for the new values.
  2. Eval profile PROXY56 (available; commit-run default is EVAL12 per the user, 2026-10-13): PROXY48 + the 8 EVAL12 tasks not in PROXY48 = 56 tasks / 72 test inputs,
     6 h budget, admit/drain fractions 0.80/0.90 as EVAL12. Hidden rerun (RERUN_ALL) deadlines unchanged.
  3. Tail resample for attempt_2: once the primary queue is empty, free slots (< CONCURRENCY, before ADMIT_DEADLINE,
     not drained) run ONE independent full sample (new seed) of a finished unit, priority:
       0 = no valid grid (attempt_1 is the test-input fallback)
       1 = forced close (thinking hit the cap) / no room for an answer
       2 = natural close with long thinking (>= 0.9 x THINK_BUDGET)
     A valid resample grid that differs from attempt_1 becomes attempt_2. attempt_1 is never changed.
  5. Audit fixes (2026-10-13): task-normalised score from the final pairs; projection_valid=False when resamples ran.
  4. Records carry t_admit_min / t_done_min and resample fields; the report scores the FINAL pairs (after resamples) and
     keeps the primary-only (baseline) score separately.
usage: python build_e9.py
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "source" / "Q27-XHIGH-v1_4_4_LB26.94.ipynb"
SRC_SHA = "959959806170960c"
OUT = HERE.parent / "Q27-XHIGH-v1_4_4_E9_86K_RESAMPLE.ipynb"

PROXY48 = ["0934a4d8", "20a9e565", "4c7dc4dd", "89565ca0", "136b0064", "291dc1e1", "16b78196", "271d71e2",
           "28a6681f", "2b83f449", "2c181942", "31f7f899", "35ab12c3", "3dc255db", "409aa875", "4c416de3",
           "64efde09", "6e453dd6", "7491f3cf", "7b0280bc", "7b80bb43", "7c66cb00", "1818057f", "332f06d7",
           "53fb4810", "7666fa5d", "da515329", "8f3a5a89", "db695cfb", "97d7923e", "de809cff", "c7f57c3e",
           "cb2d8a2c", "58490d8a", "dbff022c", "5961cc34", "80a900e0", "269e22fb", "8b9c3697", "e376de54",
           "b99e7126", "c4d067a0", "135a2760", "4a21e3da", "71e489b6", "800d221b", "dd6b8c4b", "36a08778"]
EVAL12_EXTRA = ["21897d95", "247ef758", "4c3d4a41", "6e4f6532", "88bcf3b4", "898e7135", "8b7bacbf", "b6f77b65"]
PROXY56 = PROXY48 + EVAL12_EXTRA
assert len(set(PROXY56)) == 56

BANNER = ("**Experiment 9 (86k + tail resample):** v1.4.4 DIRECT (LB 26.94), original order (no puzzle ordering). "
          "Thinking budget **77,824** + answer 8,192 (MAX_MODEL_LEN 106,496). When the primary queue is empty, free slots "
          "run one independent new-seed sample of finished units (no grid > forced close > long natural close); a valid "
          "different grid becomes **attempt_2** (attempt_1 never changes). Commit run = EVAL12 (12 tasks / 22 test inputs, 3 h); "
          "profile PROXY56 (56 tasks / 72 inputs, 6 h) is available. Hidden rerun deadlines unchanged.")

CELL7_RESAMPLE = r'''
# ==================================================================== E9 tail resample (attempt_2 from a new independent sample)
E9_RESAMPLE = True
E9_LONG_NATURAL = int(0.9 * THINK_BUDGET)          # natural closes this long are 'uncertain'
E9_SUFFIX = "~r1"                                    # one resample per unit
RESAMPLED = set()                                    # base uids already resampled
SCHED.update({"resample_admitted": 0, "resample_finished": 0, "resample_changed_attempt2": 0})

def _e9_priority(uid):
    """lower = more likely wrong; None = not a resample candidate"""
    rec = next((r for r in reversed(RECORDS) if r["unit"] == uid), None)
    att = FINAL.get(uid)
    if rec is None or att is None:
        return None
    if att.get("src_1") != "model":
        return 0
    if rec["mode"] == "forced_close" or rec["end_reason"] in ("no_room_for_answer", "drain_no_room"):
        return 1
    if rec["mode"] == "natural" and rec["think_tokens"] >= E9_LONG_NATURAL:
        return 2
    return None

def _e9_next_candidate():
    best = None
    for i, r in enumerate(RECORDS):
        u = r["unit"]
        if r.get("resample_of") or u in RESAMPLED:
            continue
        p = _e9_priority(u)
        if p is not None and (best is None or (p, i) < best[0]):
            best = ((p, i), u)
    return None if best is None else best[1]

def admit_resample(base):
    rid = base + E9_SUFFIX
    UNIT_TASK[rid], UNIT_Q[rid] = UNIT_TASK[base], UNIT_Q[base]
    UNIT_INDEX[rid] = UNIT_INDEX[base] + 100000      # distinct seeds: SEED + 10*(idx+100000) + 1/2
    st = {"uid": rid, "resample_of": base, "phase": 1, "prompt_ids": PROMPT_IDS[base], "p1_ids": [], "p2_ids": [],
          "p2_prefix": [], "p1_finish": None, "p2_finish": None, "t_admit": time.perf_counter(), "t_done": None,
          "mode": None}
    STATE[rid] = st
    RESAMPLED.add(base)
    engine.add_request(_req(rid, 1), {"prompt_token_ids": list(st["prompt_ids"])},
                       make_params(P1_CAPS[base], SEED + 10 * UNIT_INDEX[rid] + 1))
    SCHED["resample_admitted"] += 1

def _record_resample(rid, st, grid):
    base = st["resample_of"]
    before = dict(FINAL[base])
    changed = bool(valid_grid(grid) and grid != before["attempt_1"] and grid != before["attempt_2"])
    if changed:
        FINAL[base] = {**before, "attempt_2": grid, "src_2": "resample"}
        SCHED["resample_changed_attempt2"] += 1
    SCHED["resample_finished"] += 1
    sc = score_unit(base, FINAL[base])
    rec = {"unit": rid, "resample_of": base, "task_id": UNIT_TASK[base], "query": UNIT_Q[base],
           "mode": st["mode"] or "natural", "end_reason": st["end_reason"], "prompt_tokens": len(st["prompt_ids"]),
           "p1_tokens": len(st["p1_ids"]), "p2_tokens": len(st["p2_ids"]), "think_tokens": st["think_tokens"],
           "think_closed": st["think_closed"], "p1_finish": st["p1_finish"], "p2_finish": st["p2_finish"],
           "sec": round(st["t_done"] - st["t_admit"], 1), "answer_chars": st["answer_chars"], "has_grid": st["has_grid"],
           "changed_attempt2": changed, "same_as_attempt1": bool(grid is not None and grid == before["attempt_1"]),
           "score_after": sc, "score_before": score_unit(base, before),
           "t_admit_min": round((st["t_admit"] - NOTEBOOK_T0) / 60, 2), "t_done_min": round((st["t_done"] - NOTEBOOK_T0) / 60, 2)}
    RECORDS.append(rec)
    try:
        with open(RECORDS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        print("records write failed:", rid, repr(e)[:200])
    print(f"[{(time.perf_counter() - NOTEBOOK_T0) / 60:6.1f} min] {rid:18s} RESAMPLE {rec['mode']:12s} think "
          f"{rec['think_tokens']:6d} grid={rec['has_grid']} attempt_2 changed={changed}"
          + (f" hit {sc['hit']} (before {rec['score_before']['hit']})" if sc else ""), flush=True)
    write_submission()
'''


def once(text, old, new, where):
    n = text.count(old)
    if n != 1:
        sys.exit(f"replacement not unique ({n}) in {where}: {old[:80]!r}")
    return text.replace(old, new)


def src_of(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def build():
    raw = SRC.read_bytes()
    assert hashlib.sha256(raw).hexdigest()[:16] == SRC_SHA, "base changed"
    nb = json.loads(raw)
    c = nb["cells"]
    set_src(c[0], BANNER + "\n\n" + src_of(c[0]))
    # ---- cell 2: budgets, context, profile
    t = src_of(c[2])
    t = once(t, 'THINK_BUDGET = int(os.environ.get("Q27X_THINK_BUDGET", 49152))',
             'THINK_BUDGET = int(os.environ.get("Q27X_THINK_BUDGET", 77824))   # E9: 86k total (77,824 + 8,192); was 49,152',
             "think budget")
    t = once(t, 'MAX_MODEL_LEN = int(os.environ.get("Q27X_MAX_MODEL_LEN", 81920))',
             'MAX_MODEL_LEN = int(os.environ.get("Q27X_MAX_MODEL_LEN", 106496))   # E9: prompt (<= ~20k) + 77,824 + 8,192 fits; was 81,920',
             "max model len")
    t = once(t, 'if (MAX_MODEL_LEN, THINK_BUDGET, ANSWER_MAX_TOKENS) == (81920, 49152, 8192):\n',
             'if (MAX_MODEL_LEN, THINK_BUDGET, ANSWER_MAX_TOKENS) == (106496, 77824, 8192):   # E9\n'
             '    assert phase1_cap(12800) == 77824 and phase1_cap(20000) == 77824 and phase1_cap(30000) == 68048\n'
             'if (MAX_MODEL_LEN, THINK_BUDGET, ANSWER_MAX_TOKENS) == (81920, 49152, 8192):\n', "asserts")
    t = once(t, 'EXPECTED_LIVE_TOKENS_PER_SEQ = 30000 ',
             'EXPECTED_LIVE_TOKENS_PER_SEQ = 43000 ', "expected live")
    # commit-run default stays EVAL12 (user 2026-10-13: evaluate on EVAL12 first); PROXY56 kept in the table
    t = once(t, '    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},\n',
             '    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},\n'
             '    # E9: PROXY48 (EDA LB-mirror) + the 8 EVAL12 tasks not in it = 56 tasks / 72 test inputs\n'
             f'    "PROXY56":   {{"source": "eval", "task_list": {json.dumps(PROXY56)}, "time_budget_sec": 6.0 * 3600,\n'
             '                  "admit_frac": 0.80, "drain_frac": 0.90},\n', "profile table")
    set_src(c[2], t)
    # ---- cell 4: fixed task list
    t = src_of(c[4])
    t = once(t, '    TASK_IDS = sorted(random.Random(TASK_SAMPLE_SEED).sample(sorted(_ch), P["n_tasks"]))\n',
             '    if P.get("task_list"):   # E9: fixed list (every id must exist in the evaluation file)\n'
             '        assert all(t in _ch for t in P["task_list"]), "task list id missing from evaluation file"\n'
             '        TASK_IDS = sorted(P["task_list"])\n'
             '    else:\n'
             '        TASK_IDS = sorted(random.Random(TASK_SAMPLE_SEED).sample(sorted(_ch), P["n_tasks"]))\n', "task list")
    set_src(c[4], t)
    # ---- cell 7: scheduler
    t = src_of(c[7])
    t = once(t, 'def run_scheduler():\n', CELL7_RESAMPLE.lstrip("\n") + '\ndef run_scheduler():\n', "resample block")
    t = once(t, '        (ANSWERS_DIR / f"{UNIT_TASK[uid]}_q{UNIT_Q[uid]}.txt").write_text(',
             '        (ANSWERS_DIR / f"{UNIT_TASK[uid]}_q{UNIT_Q[uid]}{\'_r1\' if st.get(\'resample_of\') else \'\'}.txt").write_text(',
             "answer file name")
    t = once(t, '    _record(uid, st, make_attempts(uid, grid))\n',
             '    if st.get("resample_of"):\n'
             '        _record_resample(uid, st, grid)\n'
             '    else:\n'
             '        _record(uid, st, make_attempts(uid, grid))\n', "finish dispatch")
    t = once(t, '"score": sc, "p1_cap": P1_CAPS.get(uid)}',
             '"score": sc, "p1_cap": P1_CAPS.get(uid),\n'
             '           "t_admit_min": round((st["t_admit"] - NOTEBOOK_T0) / 60, 2),\n'
             '           "t_done_min": round((st["t_done"] - NOTEBOOK_T0) / 60, 2)}', "record times")
    t = once(t, '''        while queue and len(active) < CONCURRENCY and now < ADMIT_DEADLINE and not drained:
            admit(queue.pop(0)); active = [s for s in STATE.values() if s["phase"] in (1, 2)]
''', '''        while queue and len(active) < CONCURRENCY and now < ADMIT_DEADLINE and not drained:
            admit(queue.pop(0)); active = [s for s in STATE.values() if s["phase"] in (1, 2)]
        # E9: tail resample - only when every primary unit has been admitted; same deadlines as primary units
        while E9_RESAMPLE and not queue and len(active) < CONCURRENCY and now < ADMIT_DEADLINE and not drained:
            _cand = _e9_next_candidate()
            if _cand is None:
                break
            admit_resample(_cand); active = [s for s in STATE.values() if s["phase"] in (1, 2)]
''', "tail admission")
    set_src(c[7], t)
    # ---- cell 8: report scores the FINAL pairs; primary-only baseline kept separately
    t = src_of(c[8])
    t = once(t, '    rows = RECORDS\n',
             '    rows = [r for r in RECORDS if not r.get("resample_of")]          # E9: primary units only\n'
             '    rs_rows = [r for r in RECORDS if r.get("resample_of")]\n', "rows")
    t = once(t, '''        _hits = [r["score"]["hit"] for r in rows if r["score"]]
''', '''        _hits = [r["score"]["hit"] for r in rows if r["score"]]            # primary-only (baseline) pairs
        report["eval_primary_only_hits"] = int(sum(_hits))
        _final = {u: score_unit(u, FINAL[u]) for u in FINAL if u in UNIT_SET_E9}   # E9: final pairs after resamples
        _hits = [s["hit"] for s in _final.values() if s]
        report["eval_resample_gain"] = int(sum(_hits)) - report["eval_primary_only_hits"]
''', "final hits")
    t = once(t, '''        report["eval_hit_units"] = [r["unit"] for r in rows if r["score"] and r["score"]["hit"]]
''', '''        report["eval_hit_units"] = sorted(u for u, s in _final.items() if s and s["hit"])
        report["eval_resample_hit_units"] = sorted(r["resample_of"] for r in rs_rows
                                                   if r["score_after"] and r["score_after"]["hit"]
                                                   and not r["score_before"]["hit"])
''', "hit units")
    # audit fix 5: task-normalised score from the FINAL pairs; the primary-only figure is kept under its own name
    t = once(t, '''        _by = {}
        for r in rows:
            if r["score"]: _by.setdefault(r["task_id"], []).append(r["score"]["hit"])
''', '''        _by0 = {}
        for r in rows:
            if r["score"]: _by0.setdefault(r["task_id"], []).append(r["score"]["hit"])
        report["eval_task_normalized_primary_only_pct"] = round(
            100 * sum(sum(v) / len(CHALLENGES[t]["test"]) for t, v in _by0.items()) / max(len(TASK_IDS), 1), 2)
        _by = {}                                                          # E9 audit fix 5: final pairs
        for _u, _s in _final.items():
            if _s: _by.setdefault(UNIT_TASK[_u], []).append(_s["hit"])
''', "fix5 task-normalised")
    # audit fix 6: no hidden-time projection once tail resamples ran (primary-only tokens vs mixed throughput)
    t = once(t, '''    if not report["projection_valid"]:
        report["projection_reason"] = "incomplete commit run, engine failure, scheduler error, skipped / unadmitted / drained / hard-stopped queries"
''', '''    if not report["projection_valid"]:
        report["projection_reason"] = "incomplete commit run, engine failure, scheduler error, skipped / unadmitted / drained / hard-stopped queries"
    if SCHED.get("resample_admitted", 0) > 0:                             # E9 audit fix 6
        report["projection_valid"] = False
        report["projection_reason"] = ("tail resamples ran: the primary-only token mean and the mixed primary+resample "
                                       "throughput are not comparable; hidden-time projection not calibrated")
''', "fix6 projection")
    t = once(t, '    try:   # v1.3: MTP acceptance from the final vLLM metrics snapshot\n',
             '    report["resample"] = {"admitted": SCHED.get("resample_admitted", 0), "finished": SCHED.get("resample_finished", 0),\n'
             '                          "changed_attempt2": SCHED.get("resample_changed_attempt2", 0),\n'
             '                          "modes": {m: sum(r["mode"] == m for r in rs_rows) for m in ("natural", "continue", "forced_close")},\n'
             '                          "same_as_attempt1": sum(bool(r.get("same_as_attempt1")) for r in rs_rows)}\n'
             '    try:   # v1.3: MTP acceptance from the final vLLM metrics snapshot\n', "resample report")
    t = "UNIT_SET_E9 = set(UNITS)\n" + t
    set_src(c[8], t)
    for cell in c:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    OUT.write_bytes(data)
    print("BUILT", OUT.name, len(data), "bytes sha256", hashlib.sha256(data).hexdigest()[:16])


if __name__ == "__main__":
    build()
