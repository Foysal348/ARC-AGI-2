"""E29 rev 2 DIRECT (user 2026-11-02, fixed spec): build-time helpers of build_e29_final.py (stdlib only).

DIRECT becomes the SECOND BRANCH (attempt_2) of E29_DFLASH_D3S, behind the CELL 1 switch E29_DIRECT2:
  - e29_direct.py   = E31's solver e31_direct.py (sha 9967a6e0, reused verbatim) + exact-once edits:
                      N1 (the open audit item: a phase-2 resend keeps the already streamed phase-2 ids: prompt = prefix +
                      p2_ids, cap = the remaining answer tokens) and task_index (the ORDER2 seed index of a task in the FULL
                      task set: the scheduler calls the solver one task at a time);
  - e29d_sched_patch.py = the DIRECT diff on arc2_duck.py (the E29 post text d8603109 -> the E29 DIRECT post text): phase C
                      of the E28 A / B / C scheduler runs ONE DIRECT request per task in its worker slot (C's priority order
                      unchanged) through the notebook hook E12A.E29D_RUN; the answer rule (attempt_1 never changes;
                      attempt_2 = the DIRECT grid of that test when valid and != attempt_1, else today's attempt_2).
                      rev 2d (audit F1 / F1+, A4 H1): a phase-C item goes to DIRECT only when the second notebook hook
                      E12A.E29D_CAN_START(deadline) returns True (the notebook: now + 60 s + drain_min * 60 < deadline);
                      no hook / False / an exception -> the ORIGINAL E28 Duck C session for that item (a late C is never
                      a DIRECT no-op: E28 started C as late as 21,291 s of 21,600 s).
Both go to the payload dataset (the notebook loads them by name + sha256; the 400-kB commit limit).
Standalone (rev 2d, the scheduler agent): python e29_direct_build.py --regen-sched  -> writes Build/e29d_sched_patch.py exactly
as build_e29_final.build_direct() does (same inputs, same checks) and prints its pre / post / patch / module shas.
"""
import difflib
import hashlib
from pathlib import Path

E31_DIRECT = Path(r"D:/ARC AGI 2/experiments/Experiment 31 - Flash-Next Direct measurement (planned)/Experiment Notebooks/Build/e31_direct.py")
E31_DIRECT_SHA = "9967a6e0c83067e9d4a4bb3cf262614d356b693acc26a573c946faa630362528"


def sha(t):
    return hashlib.sha256(t if isinstance(t, bytes) else t.encode("utf-8")).hexdigest()


def once(text, old, new, label):
    assert text.count(old) == 1, (label, text.count(old))
    return text.replace(old, new)


# ------------------------------------------------------------------ e29_direct.py
DIRECT_EDITS = [
    ("D0 doc", '"""E31 Flash-Next DIRECT (user 2026-11-01): ',
     '"""E29 DIRECT (Experiment 29 rev 2, 2026-11-02) = E31\'s e31_direct.py (sha 9967a6e0) VERBATIM + two marked "E29" edits: N1 (a\n'
     'phase-2 resend keeps the streamed phase-2 ids) and task_index (ORDER2 seeds over the full task set; the E28 scheduler calls\n'
     'solve_all_direct for ONE task per phase-C worker slot). The original docstring follows.\n'
     'E31 Flash-Next DIRECT (user 2026-11-01): '),
    ("D1 version", 'LIB_VERSION = "e31_direct 1.0 (2026-11-01)"',
     'LIB_VERSION = "e29_direct 1.0 (2026-11-02) = e31_direct 1.0 (9967a6e0) + E29 N1 + task_index"'),
    ("D2 signature", "                     max_model_len=MAX_MODEL_LEN, health=None, max_error_retries=2, log=print):\n",
     "                     max_model_len=MAX_MODEL_LEN, health=None, max_error_retries=2, log=print, task_index=None):   # E29\n"),
    ("D3 seed index", "    TASK_INDEX = {t: i for i, t in enumerate(task_ids)}            # seeds do not depend on timing (ORDER2 F13)\n",
     "    TASK_INDEX = {t: int((task_index or {}).get(t, i)) for i, t in enumerate(task_ids)}   # ORDER2 F13; E29: full-set index\n"),
    ("D4 N1", '''        prefix = list(st["prompt_ids"]) + list(st["p1_ids"]) + list(st["p2_prefix"])
        room = max_model_len - len(prefix) - 8
        cap = min(int(answer_cap(st["tid"])), room)
        if cap < 64:
            return False
        st["phase"] = 2; st["p2_cap"] = cap; st["p2_ids"] = []; st["p2_prefix_len"] = len(prefix)
''', '''        kept = list(st["p2_ids"]) if resend else []    # E29 N1: a phase-2 resend keeps the streamed phase-2 ids
        while kept and kept[-1] in EOS_IDS:
            kept.pop()
        prefix = list(st["prompt_ids"]) + list(st["p1_ids"]) + list(st["p2_prefix"]) + kept
        room = max_model_len - len(prefix) - 8
        cap = min(int(answer_cap(st["tid"])) - len(kept), room)    # E29 N1: cap = the remaining answer tokens
        if cap < 64:
            return False
        st["phase"] = 2; st["p2_cap"] = cap; st["p2_ids"] = kept; st["p2_prefix_len"] = len(prefix)
        st["p2_kept"] = int(st.get("p2_kept") or 0) + len(kept)       # E29 N1 (recorded)
'''),
    ("D5 N1 doc", "        (after a phase-2 request error) sends the same prefix again.\"\"\"\n",
     "        (after a phase-2 request error) sends the same prefix + the streamed phase-2 ids (E29 N1).\"\"\"\n"),
    ("D6 N1 record", '"seed_p2": task_seed(TASK_INDEX[tid], 2) if st["p2_cap"] else None, "p2_prefix_len": st.get("p2_prefix_len"),',
     '"seed_p2": task_seed(TASK_INDEX[tid], 2) if st["p2_cap"] else None, "p2_prefix_len": st.get("p2_prefix_len"),\n'
     '               "p2_kept": int(st.get("p2_kept") or 0),   # E29 N1'),
]


def make_direct():
    raw = E31_DIRECT.read_bytes()
    assert sha(raw) == E31_DIRECT_SHA and b"\r" not in raw, ("e31_direct.py is not the pinned 9967a6e0", sha(raw))
    t = raw.decode("utf-8")
    for lab, old, new in DIRECT_EDITS:
        t = once(t, old, new, lab)
    compile(t, "e29_direct.py", "exec")
    return t


# ------------------------------------------------------------------ the DIRECT diff on arc2_duck.py (E29 post -> E29 DIRECT post)
DUCK_MARKER = "# E29 DIRECT: DIRECT as the second branch (attempt_2) of the E28 scheduler: env-gated (E28_SCHED=1 + E29_DIRECT=1 + hook)."
DUCK_EDITS = [
    ("K1 hook", 'E28_RELEASE = None           # notebook hook (N4): (key) -> None, frees a reservation (idempotent)\n',
     'E28_RELEASE = None           # notebook hook (N4): (key) -> None, frees a reservation (idempotent)\n'
     'E29D_RUN = None              # E29 DIRECT hook: (task, task dict, plan, deadline, base_url, model_id, out_dir) -> record | None\n'
     'E29D_CAN_START = None        # E29 DIRECT hook (F1): (deadline) -> bool; DIRECT only when True, else the original Duck C\n'
     'E29D_PHASE_ENV = "E29_DIRECT_PHASE"   # E29 rev 2f: "C1" (default) = DIRECT only on train-fail C items; "C" = every C item\n'),
    ("K2 select", "    if _e29_on():      # E29 SUPPLY: the A / B / S best-train rule (S attempts = branches >= 3)\n",
     "    if any(_e29d_is(r) for r in branch_records):   # E29 DIRECT: attempt_2 = the DIRECT grid (the rule below on the rest)\n"
     "        return select_attempts_e29d(task, branch_records, _slots_out)\n"
     "    if _e29_on():      # E29 SUPPLY: the A / B / S best-train rule (S attempts = branches >= 3)\n"),
    ("K3 take", '''                        qlen = [len(qa), len(qb), len(qc)]
                        e28["pop_seq"] += 1
                        start_seq = e28["pop_seq"]
                        try:
                            s = sess(tid, p, qpos)
''', '''                        qlen = [len(qa), len(qb), len(qc)]
                        e28["pop_seq"] += 1
                        start_seq = e28["pop_seq"]
                        if p["phase"] == "C" and _e29d_on():   # E29 DIRECT: branch 1 = ONE DIRECT request, no Duck session
                            go_, why_ = _e29d_can_start(deadline, p)   # F1 + rev 2f scope: C1 only (default), while admissible
                            if go_:
                                ph, s, t_start = "C", None, _sec()
                                e28["started"]["C"] += 1
                                e28["first_start_sec"].setdefault("C", _sec())
                                e28["last_start_sec"]["C"] = _sec()
                                state["running"] += 1
                                active.add((tid, 1))
                                break
                            _e29d_duck_c(e28, tid, why_, _sec())   # F1: late / no hook / hook error -> the original Duck C
                        try:
                            s = sess(tid, p, qpos)
'''),
    ("K6 early set", """    meta: dict = {}                     # (task, branch) -> admission wait bookkeeping of a queued B / C item
    active: set = set()
""", """    meta: dict = {}                     # (task, branch) -> admission wait bookkeeping of a queued B / C item
    active: set = set()
    e29d_early: set = set()             # E29 rev 2g: tasks whose branch 1 = an EARLY DIRECT item (queued at the A end)
"""),
    ("K7 candidates", """        bq = sorted(qb)
        wb2 = lambda e: e[3]["phase"] == "B2" and meta.get((e[2], 2), {}).get("waiting")
        return ([("B", e, e[2], e[3]) for e in bq if wb2(e)] + [("A", tid, tid, None) for tid in qa] +
                [("B", e, e[2], e[3]) for e in bq if not wb2(e)] + [("C", e, e[2], e[3]) for e in sorted(qc)])
""", """        bq = sorted(qb)
        wb2 = lambda e: e[3]["phase"] == "B2" and meta.get((e[2], 2), {}).get("waiting")
        ce = lambda e: e[3].get("c_from") == E29D_EARLY_FROM    # E29 rev 2g: an early DIRECT item goes right after A
        cq = sorted(qc)
        return ([("B", e, e[2], e[3]) for e in bq if wb2(e)] + [("A", tid, tid, None) for tid in qa] +
                [("C", e, e[2], e[3]) for e in cq if ce(e)] +
                [("B", e, e[2], e[3]) for e in bq if not wb2(e)] + [("C", e, e[2], e[3]) for e in cq if not ce(e)])
"""),
    ("K8 route", """            if full:
                push_c(tid, True, cls["tokens"], "A")
            elif e28_b_phase(cls["kind"]):
                push_b(tid, e28_b_phase(cls["kind"]), cls)
            # TIME: unfinished phase-A work (the solve window ended); no re-run
        elif ph in ("B1", "B2"):
            e28["b_class"][tid] = {k: cls.get(k) for k in ("kind", "end_reason", "tokens")}
            push_c(tid, full, cls["tokens"], ph)
""", """            if full:
                push_c(tid, True, cls["tokens"], "A")
            elif e28_b_phase(cls["kind"]):
                push_b(tid, e28_b_phase(cls["kind"]), cls)
                if _e29d_early_on() and _e29d_can_start(deadline, {"c_priority": 1})[0]:   # E29 rev 2g: DIRECT early
                    # on this train-fail task (branch 1, queued now) only while DIRECT is still admissible; else the E28 path
                    push_c(tid, False, cls["tokens"], E29D_EARLY_FROM)
                    e29d_early.add(tid)
                    _e29d_stats(e28)["early_queued"] += 1
            # TIME: unfinished phase-A work (the solve window ended); no re-run
        elif ph in ("B1", "B2"):
            e28["b_class"][tid] = {k: cls.get(k) for k in ("kind", "end_reason", "tokens")}
            if tid in e29d_early:              # E29 rev 2g: branch 1 is the early DIRECT item: no second C item
                _e29d_stats(e28)["early_b_end_no_c"] += 1
            else:
                push_c(tid, full, cls["tokens"], ph)
"""),
    ("K4 run", '''            t0 = time.monotonic()
            tag = {"stage": {"A": 1, "C": 2}.get(ph, 3), "e28_phase": ph}
''', '''            t0 = time.monotonic()
            if s is None:                      # E29 DIRECT: the phase-C item runs its DIRECT request in this worker slot
                _e29d_item(tid, p, wi, t_start, qlen, start_seq, t0)
                continue
            tag = {"stage": {"A": 1, "C": 2}.get(ph, 3), "e28_phase": ph}
'''),
    ("K5 item", '''    def worker(wi: int):
        while True:
            try:
                return _worker_body(wi)
''', '''    def _e29d_item(tid: str, p: dict, wi: int, t_start, qlen, start_seq, t0: float):
        """E29 DIRECT: this task's phase-C item = ONE DIRECT request (the notebook's E29D_RUN: e29_direct.py on the same
        server), published as branch 1 (submissions[{test_outputs}], accepted False, train_exact []). No record (an error,
        or not admitted after the DIRECT drain point) = nothing published: the E28 attempt_2 rule stands for that task."""
        rec, err = None, None
        try:
            rec = E29D_RUN(tid, tasks[tid], dict(p), deadline, base_url, model_id, Path(out_dir) / "direct")
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("%s E29 DIRECT failed: %s", tid, err)
        if E28_RELEASE is not None:
            try:
                E28_RELEASE((tid, 1))
            except Exception:
                pass
        x = (rec or {}).get("e31") or {}
        m = meta.get((tid, 1)) or {}
        row = {"task": tid, "branch": 1, "phase": "C", "e29_direct": True, "c_priority": p.get("c_priority"),
               "c_key_tokens": p.get("c_key_tokens"), "budget": p.get("budget"), "worker_slot": wi,
               "tokens": int((rec or {}).get("generated_tokens") or 0), "start_sec": t_start, "end_sec": _sec(),
               "status": "done" if rec else "no_record", "queues_at_start": qlen, "start_seq": start_seq, "error": err,
               "kind": "DIRECT" if rec else "DIRECT_NO_RECORD",
               "end_reason": x.get("end_reason") or ("error" if err else "not_admitted"), "queued_sec": m.get("queued_sec"),
               "wait_sec": (round(t_start - m["first_tried_sec"], 1) if m.get("first_tried_sec") is not None else None),
               **{k: x.get(k) for k in ("mode", "think_tokens", "p2_tokens", "mapping_source", "has_grid", "score")}}
        if rec is not None:
            try:
                rec = dict(rec, branch=1, e29_direct=True, e28=dict(row))
                rec["submissions"] = [dict(s_, branch=1) for s_ in rec.get("submissions") or []]
                publish(rec)
            except Exception as exc:
                row["error"] = f"publish {type(exc).__name__}: {exc}"[:300]
        with cond:
            d = _e29d_stats(e28)
            d["started"] += 1
            d["records" if rec is not None else "no_record"] += 1
            d["errors"] += int(bool(row.get("error")))
            d["by_task"][tid] = {k: row.get(k) for k in ("kind", "mode", "end_reason", "tokens", "think_tokens", "p2_tokens",
                                                         "mapping_source", "has_grid", "score", "start_sec", "end_sec")}
            e28["rows"][f"{tid}_b1"] = row
            e28["done"]["C"] += 1
            state["running"] -= 1
            state["last_free_t"] = time.monotonic()
            active.discard((tid, 1))
            state["durations"].append(round(time.monotonic() - t0, 1))
            cond.notify_all()

    def worker(wi: int):
        while True:
            try:
                return _worker_body(wi)
'''),
    ("K6 report", '                           **({"supply": _e29_report(sup, e28, rows_, slots, order)} if sup is not None else {}),\n',
     '                           **({"supply": _e29_report(sup, e28, rows_, slots, order)} if sup is not None else {}),\n'
     '                           **({"direct": e28["direct"]} if e28.get("direct") else {}),   # E29 DIRECT\n'),
    ("K7 slot src", '''    if s in src:
        return src[s]
    if isinstance(s, str) and s[:1] == "S" and s[1:].isdigit():
''', '''    if s == "D":                       # E29 DIRECT: the DIRECT answer (branch 1)
        return 1
    if s in src:
        return src[s]
    if isinstance(s, str) and s[:1] == "S" and s[1:].isdigit():
'''),
]
DUCK_TAIL = '''

# ==================================================================== E29 DIRECT (Experiment 29 rev 2, user 2026-11-02)
''' + DUCK_MARKER + '''
# On: a phase-C item (C's priority order unchanged: train-fail tasks first, then the accepted ones by ascending attempt-1
# tokens) runs ONE DIRECT request in its worker slot (the notebook's E29D_RUN = e29_direct.py, the E31 solver) instead of a
# Duck attempt; A / B / supply / the deadline guard are unchanged. Answers: attempt_1 = today's rule on the Duck records (it
# never changes); attempt_2 = the DIRECT grid of that test when valid and != attempt_1, else today's attempt_2.
# Off (E29_DIRECT unset, or no hook): the code above runs exactly as the E29 post file (no DIRECT record exists).
# F1 (rev 2d): a phase-C item goes to DIRECT only when E29D_CAN_START(deadline) is True (the notebook: now + 60 s + drain_min
# * 60 < deadline, i.e. the solver can still admit it); otherwise that SAME item runs the original Duck C session (one C
# start, one C completion, one reservation release), counted in e28["direct"]["duck_c"] / ["duck_c_why"] (record only).
# rev 2f (user 2026-11-02; E28 K100: all 7 attempt_2-only hits came from Duck C on train-pass tasks): scope "C1" (the
# default) = DIRECT only on train-fail C items (c_priority 1); a C2 item (full train match) runs the original Duck C session
# through the same F1 path (why "c2_duck"). Scope "C" (env E29_DIRECT_PHASE=C) = rev 2e (every C item).
# rev 2g (user 2026-11-02; E28 K100: C1 items started at 308-352 min of 360, after B2): under scope C1 a train-fail task's
# DIRECT item is queued the moment its A attempt ends (next to its B re-run) and admitted right after the A queue, ahead
# of B, so DIRECT gets its full thinking time; that task's B end then queues no second C item (branch 1 = DIRECT).
def _e29d_on() -> bool:
    return _e28_on() and os.environ.get("E29_DIRECT", "").strip() == "1" and E29D_RUN is not None


E29D_EARLY_FROM = "A_DIRECT_EARLY"     # rev 2g: the c_from of an early DIRECT item


def _e29d_early_on() -> bool:
    """rev 2g (user 2026-11-02: DIRECT gets its full thinking time on train-fail tasks, never time on tasks Duck passed): scope C1
    + DIRECT on + the admission hook set -> a train-fail task's DIRECT item is queued at its A end (ahead of B). Off / no hook =
    the E28 order exactly (no early item)."""
    return _e29d_on() and _e29d_scope() == "C1" and callable(E29D_CAN_START)


def _e29d_scope() -> str:
    """rev 2f (user 2026-11-02): the DIRECT scope from the notebook's E29_DIRECT2['phase'] (env E29_DIRECT_PHASE): "C" = every
    phase-C item; anything else, unset included, = "C1" (train-fail C items only, c_priority 1)."""
    return "C" if os.environ.get(E29D_PHASE_ENV, "").strip() == "C" else "C1"


def _e29d_can_start(deadline, p=None) -> tuple:
    """F1: (go, why). go only when the notebook hook E29D_CAN_START is callable and returns True for the scheduler's
    deadline; no hook -> "no_hook", False -> "late", an exception -> "hook_error" (never raises). rev 2f: scope C1 and a
    C2 item (a task with a full train match, c_priority != 1) -> "c2_duck": that item runs the original Duck C session."""
    h = E29D_CAN_START
    if not callable(h):
        return False, "no_hook"
    if _e29d_scope() == "C1" and (p or {}).get("c_priority") != 1:
        return False, "c2_duck"
    try:
        return (True, "ok") if bool(h(deadline)) else (False, "late")
    except Exception as exc:
        return False, f"hook_error {type(exc).__name__}"[:80]


def _e29d_stats(e28: dict) -> dict:
    d = e28.setdefault("direct", {"started": 0, "records": 0, "no_record": 0, "errors": 0, "by_task": {}})
    d.setdefault("duck_c", 0)
    d.setdefault("duck_c_why", {})
    d.setdefault("duck_c_tasks", {})
    d.setdefault("early_queued", 0)       # rev 2g
    d.setdefault("early_b_end_no_c", 0)   # rev 2g
    return d


def _e29d_duck_c(e28: dict, tid: str, why: str, sec) -> None:
    """F1 (record only): this phase-C item runs the original Duck C session instead of DIRECT (caller holds the lock)."""
    d = _e29d_stats(e28)
    d["duck_c"] += 1
    d["duck_c_why"][why] = d["duck_c_why"].get(why, 0) + 1
    d["duck_c_tasks"][tid] = {"why": why, "sec": sec}


def _e29d_is(rec) -> bool:
    return isinstance(rec, dict) and bool(rec.get("e29_direct"))


def select_attempts_e29d(task: dict, branch_records: list[dict], _slots_out: list | None = None) -> list[dict]:
    """E29 DIRECT answer slots (same output format): attempt_1 / the fallbacks = select_attempts_e28 on the Duck records
    (unchanged); attempt_2 = the latest DIRECT record's grid for that test input (grid K -> test K) when it is a valid grid
    and differs from attempt_1, else the Duck rule's attempt_2. Slot source "D" = the DIRECT answer."""
    rows: list = []
    out = select_attempts_e28(task, [r for r in branch_records if not _e29d_is(r)], rows)
    ds = [r for r in branch_records if _e29d_is(r)]
    d = max(ds, key=lambda r: int(r.get("publish_seq") or 0)) if ds else None
    outs = (((d.get("submissions") or [{}])[0] or {}).get("test_outputs") or []) if d else []
    for qi, o in enumerate(out):
        g = outs[qi] if qi < len(outs) else None
        use = valid_grid(g) and g != o["attempt_1"]
        if use:
            o["attempt_2"] = g
        if qi < len(rows) and isinstance(rows[qi], dict):
            rows[qi]["e29_direct"] = "attempt_2" if use else ("same_as_attempt_1" if valid_grid(g) else "no_grid")
            if use:
                rows[qi]["attempt_2"] = "D"
    if _slots_out is not None:
        _slots_out.extend(rows)
    return out
'''

MODULE_TMPL = r'''"""E29 DIRECT scheduler patch (Experiment 29 rev 2, user 2026-11-02; generated by Build/e29_direct_build.py, do not edit).

DIRECT = the SECOND BRANCH (attempt_2) of the E28 A / B / C scheduler: a strict unified diff on arc2_duck.py from the E29
post text (e29_sched_patch.py, sha __PRE__) to the E29 DIRECT post text (sha __POST__).
Runtime (E28_SCHED=1 + E29_DIRECT=1 + the notebook hook arc2_duck.E29D_RUN): a phase-C item runs ONE DIRECT request
(e29_direct.py) in its worker slot instead of a Duck attempt, recorded as branch 1; the answer rule select_attempts_e29d
(attempt_1 never changes; attempt_2 = the DIRECT grid when valid and != attempt_1, else today's attempt_2).
rev 2d (F1 / F1+): DIRECT only when the second hook arc2_duck.E29D_CAN_START(deadline) returns True (the notebook sets it
to now + 60 s + drain_min * 60 < deadline and clears it with E29D_RUN); no hook / False / an exception -> that phase-C item
runs the original Duck C session (recorded in e28["direct"]["duck_c"]).
rev 2f (user 2026-11-02): scope env E29_DIRECT_PHASE (the notebook's E29_DIRECT2['phase']): "C1" (default, unset included)
= DIRECT only on train-fail C items (c_priority 1), a C2 item runs the original Duck C session (why "c2_duck"); "C" = rev 2e.
rev 2g: under C1 the DIRECT item of a train-fail task is queued at its A end and admitted right after the A queue (ahead of B);
the B end of that task queues no second C item.

    import e29d_sched_patch as E29DS    # e28_sched_patch.py importable (the applier)
    rec = E29DS.apply_direct(E12_DIR)   # E29 post -> DIRECT post (dir / file / text); DIRECT post -> no-op; else raises
    ok, why = E29DS.verify_direct(E12_DIR)
    rec = E29DS.revert_direct(E12_DIR)  # DIRECT post -> the exact E29 post bytes; E29 post -> no-op
    E29DS.tree_state(E12_DIR)           # "E29_POST" | "DIRECT_PATCHED" | "OTHER"
"""
import hashlib
import json
from pathlib import Path

import e28_sched_patch as E28S

DUCK_REL = "arc2_duck.py"
PRE = "__PRE__"                 # the E29 post text (e29_sched_patch.E29_POST)
POST = "__POST__"
MARKER = __MARKER__
ENV_DIRECT = "E29_DIRECT"
PATCH = r__TQ__
apply_unified, reverse_unified = E28S.apply_unified, E28S.reverse_unified
SHA = {"pre": PRE, "post": POST, "patch": hashlib.sha256(PATCH.encode("utf-8")).hexdigest(), "applier": E28S.E28_SCHED_SHA["applier"]}


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _state_of(data):
    h = _sha(data)
    if h == POST and MARKER.encode() in data:
        return "DIRECT_PATCHED", h
    if h == PRE:
        return "E29_POST", h
    return "OTHER", h


def _file(target):
    p = Path(target)
    return p / DUCK_REL if p.is_dir() else p


def _is_text(target):
    return isinstance(target, str) and "\n" in target


def tree_state(target):
    if _is_text(target):
        return _state_of(target.encode("utf-8"))[0]
    f = _file(target)
    return _state_of(f.read_bytes())[0] if f.is_file() else "OTHER"


def verify_direct(target):
    if _is_text(target):
        st, h = _state_of(target.encode("utf-8"))
    else:
        f = _file(target)
        if not f.is_file():
            return False, f"OTHER missing {f}"
        st, h = _state_of(f.read_bytes())
    return st == "DIRECT_PATCHED", f"{st} sha256={h} (pre {PRE[:12]}, post {POST[:12]})"


def _move(data, forward):
    out = apply_unified(data.decode("utf-8"), PATCH if forward else reverse_unified(PATCH)).encode("utf-8")
    want = POST if forward else PRE
    if _sha(out) != want:
        raise RuntimeError(f"E29_DIRECT_{'POST' if forward else 'REVERT'}_SHA_MISMATCH {DUCK_REL}: {_sha(out)}")
    return out


def _do(target, forward):
    key = "applied" if forward else "reverted"
    done, start = ("DIRECT_PATCHED", "E29_POST") if forward else ("E29_POST", "DIRECT_PATCHED")
    if _is_text(target):
        data = target.encode("utf-8")
        st, h = _state_of(data)
        rec = {"mode": "text", "before": st, "sha_before": h, "sha": SHA}
        if st == done:
            rec.update({key: "already" if forward else "not_applied", "text": target})
        elif st == start:
            rec.update({key: key, "text": _move(data, forward).decode("utf-8")})
        else:
            raise RuntimeError(f"E29_DIRECT_STATE_MISMATCH {DUCK_REL}: {st} {h}")
        rec["after"] = _state_of(rec["text"].encode("utf-8"))[0]
        rec["ok"] = rec["after"] == done
        return rec
    f = _file(target)
    data = f.read_bytes() if f.is_file() else b""
    st, h = _state_of(data)
    rec = {"mode": "file", "file": str(f), "before": st, "sha_before": h, "sha": SHA}
    if st == done:
        rec[key] = "already" if forward else "not_applied"
    elif st == start:
        E28S._write_atomic(f, _move(data, forward))
        rec[key] = key
    else:
        raise RuntimeError(f"E29_DIRECT_STATE_MISMATCH {f}: {st} {h}")
    E28S._drop_pyc(f)
    rec["after"], rec["sha_after"] = _state_of(f.read_bytes())
    rec["py_compile"] = E28S._py_compile(f)
    if rec["after"] != done or rec["py_compile"]["rc"]:
        raise RuntimeError("E29_DIRECT_VERIFY_FAILED: " + json.dumps({k: rec.get(k) for k in ("after", "sha_after", "py_compile")}))
    rec["ok"] = True
    return rec


def apply_direct(target):
    """The E29 post arc2_duck.py -> the DIRECT post (atomic, pyc dropped, py_compile + post sha + marker verified)."""
    return _do(target, True)


def revert_direct(target):
    """The DIRECT post -> the exact E29 post bytes (the E29 / E28 reverts then work as before)."""
    return _do(target, False)


def cell3(e12_dir, sched_rec, rec):
    """The notebook's CELL 3 step (E29 rev 2), never raises: only on the E29 scheduler (sched_rec = E28_SCHED_REC applied,
    path E29); E29_DIRECT=1 only after apply + verify; any failure -> reverted to the E29 post file (no DIRECT), recorded."""
    import os
    rec = dict(rec)
    os.environ.pop(ENV_DIRECT, None)
    if not (sched_rec.get("applied") and sched_rec.get("path") == "E29"):
        rec["error"] = "not the E29 scheduler (path %s): no DIRECT" % sched_rec.get("path")
        return rec
    try:
        rec.update({k: v for k, v in apply_direct(e12_dir).items() if k != "text"})
        ok, why = verify_direct(e12_dir)
        if not ok:
            raise RuntimeError("E29_DIRECT_VERIFY_FAILED: " + why)
        os.environ[ENV_DIRECT] = "1"
        rec.update(ok=True, verify_why=why)
    except Exception as e:
        os.environ.pop(ENV_DIRECT, None)
        rec["error"] = f"{type(e).__name__}: {e}"[:600]
        try:
            rec["revert"] = {k: v for k, v in revert_direct(e12_dir).items() if k != "text"}
        except Exception as e2:
            rec["revert_error"] = f"{type(e2).__name__}: {e2}"[:300]
        print("!!!!!!!!!! E29 DIRECT diff not applied -> the E29 scheduler without DIRECT:", rec["error"], flush=True)
    return rec
'''


def make_dsched(e29_post_text, e29_post_sha):
    """-> (post_text, patch_text, module_text). The diff is strict (n=3 context), checked forward and backward with the
    E28 applier by the caller."""
    assert sha(e29_post_text) == e29_post_sha
    t = e29_post_text
    for lab, old, new in DUCK_EDITS:
        t = once(t, old, new, lab)
    assert t.endswith("    return out\n")
    post = t + DUCK_TAIL
    compile(post, "arc2_duck.py", "exec")
    a, b = e29_post_text.splitlines(True), post.splitlines(True)
    body = "".join(difflib.unified_diff(a, b, "a/arc2_duck.py", "b/arc2_duck.py", n=3))
    patch = ("# E29 DIRECT onto arc2_duck.py (the E29 SCHED post text).\n# pre-patch sha256  " + e29_post_sha
             + "\n# post-patch sha256 " + sha(post) + "\n" + body)
    assert "'''" not in patch and "\r" not in patch and not patch.rstrip("\n").endswith("\\")
    mod = (MODULE_TMPL.replace("__PRE__", e29_post_sha).replace("__POST__", sha(post))
           .replace("__MARKER__", repr(DUCK_MARKER)).replace("__TQ__", "'''" + patch + "'''"))
    compile(mod, "e29d_sched_patch.py", "exec")
    return post, patch, mod


# ------------------------------------------------------------------ standalone: regenerate Build/e29d_sched_patch.py (rev 2d)
E28B_DIR = Path(r"D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Build")
DUCK_PAYLOAD = Path(r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Notebooks"
                    r"/Build/payload/q27x_e15b_duck_9e4b89285136.bin")
DUCK_PAYLOAD_SHA = "9e4b89285136616d523028ac042dcca6afe82175081a1481d584c998ab3ee566"


def regen_sched(write=True):
    """= build_e29_final.build_direct() for e29d_sched_patch.py only: the E29 post arc2_duck.py (the pinned Duck payload +
    e29_sched_patch) -> make_dsched, checked forward / backward with the E28 applier and through the generated module."""
    import sys
    import zipfile
    here = Path(__file__).resolve().parent
    for p_ in (str(E28B_DIR), str(here)):
        if p_ not in sys.path:
            sys.path.insert(0, p_)
    import e29_sched_patch as E29S
    raw = DUCK_PAYLOAD.read_bytes()
    assert sha(raw) == DUCK_PAYLOAD_SHA, "the Duck payload is not 9e4b8928"
    with zipfile.ZipFile(DUCK_PAYLOAD) as z:
        duck = z.read("arc2_duck.py").decode("utf-8")
    p = E29S.apply_e29_sched(duck)
    assert p["ok"] and sha(p["text"]) == E29S.E29_SCHED_SHA["post"]
    post, patch, mod = make_dsched(p["text"], E29S.E29_SCHED_SHA["post"])
    assert E29S.apply_unified(p["text"], patch) == post and E29S.apply_unified(post, E29S.reverse_unified(patch)) == p["text"]
    ns = {"__name__": "e29d_sched_patch_b"}
    exec(compile(mod, "e29d_sched_patch.py", "exec"), ns)
    assert ns["apply_direct"](p["text"])["text"] == post and ns["revert_direct"](post)["text"] == p["text"]
    assert ns["verify_direct"](post)[0] and ns["tree_state"](post) == "DIRECT_PATCHED" and ns["tree_state"](p["text"]) == "E29_POST"
    if write:
        (here / "e29d_sched_patch.py").write_bytes(mod.encode("utf-8"))
    return {"e29d_sched_patch.py": sha(mod), "pre": E29S.E29_SCHED_SHA["post"], "post": sha(post), "patch": sha(patch)}


if __name__ == "__main__":
    import json
    import sys
    if "--regen-sched" in sys.argv:
        print(json.dumps(regen_sched(write="--dry" not in sys.argv), indent=1))
