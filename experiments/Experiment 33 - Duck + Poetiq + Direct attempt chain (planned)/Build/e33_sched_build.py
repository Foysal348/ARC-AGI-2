"""E33 ATTEMPT CHAIN (Experiment 33, user 2026-11-02, fixed spec DESIGN_E33_ATTEMPT_CHAIN_2026-11-02.md sections 3 + 4):
the build-time generator of Build/e33_sched_patch.py (stdlib only).

e33_sched_patch.py = the E33 scheduler diff on arc2_duck.py: the E29 post text (d8603109, e29_sched_patch.apply_e29_sched on the
pinned Duck payload 9e4b8928) -> the E33 post text. It is an ALTERNATIVE to the E29 DIRECT diff (e29d_sched_patch.py, same pre
text), never on top of it. Model: Experiment 29 Build/e29_direct_build.py (DUCK_EDITS = once-only text edits, DUCK_TAIL,
a strict n=3 unified diff checked forward and backward with the E28 applier, MODULE_TMPL with apply / verify / revert /
tree_state / cell3 + shas and a marker; notebook hooks = module globals).

    python e33_sched_build.py            -> writes Build/e33_sched_patch.py, prints the pre / post / patch / module shas
    python e33_sched_build.py --dry      -> the same checks, writes nothing

Reads (never writes): the Duck payload (Experiment 15), the E28 Build folder (e28_sched_patch.py, the applier) and the E29 Build
folder (e29_sched_patch.py). Bytecode writing is off (no __pycache__ in those folders).
"""
import sys

sys.dont_write_bytecode = True

import difflib      # noqa: E402
import hashlib      # noqa: E402
from pathlib import Path   # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = Path(r"D:/ARC AGI 2")
E28B_DIR = ROOT / "experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Build"
E29B_DIR = ROOT / "experiments/Experiment 29 - Final production K100 (planned)/Build"
DUCK_PAYLOAD = (ROOT / "experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Notebooks/Build/payload"
                / "q27x_e15b_duck_9e4b89285136.bin")
DUCK_PAYLOAD_SHA = "9e4b89285136616d523028ac042dcca6afe82175081a1481d584c998ab3ee566"
E29_POST_SHA = "d8603109f9530896e9e4a56e52806202fc2cd4c58bd664d9a25222fe6f7592c2"


def sha(t):
    return hashlib.sha256(t if isinstance(t, bytes) else t.encode("utf-8")).hexdigest()


def once(text, old, new, label):
    assert text.count(old) == 1, (label, text.count(old))
    return text.replace(old, new)


# ------------------------------------------------------------------ the edits on the E29 post arc2_duck.py (inside _e28_solve_all
# and select_attempts_e28 only; every inserted line is inert while e33 is None / qe is empty / no record carries an E33 tag)
DUCK_MARKER = ("# E33 ATTEMPT CHAIN: Duck + Poetiq + Direct attempt chains on the E28 scheduler: env-gated "
               "(E28_SCHED=1 + E33_SCHED=1).")
DUCK_EDITS = [
    ("K1 state", '''    sup = _e29_supply_state(e28) if _e29_on() else None   # E29 SUPPLY: attempts supply (None = off, = E28)
''', '''    sup = _e29_supply_state(e28) if _e29_on() else None   # E29 SUPPLY: attempts supply (None = off, = E28)
    qe: list = []                       # E33: the attempt-chain queue (every E33 item; qa / qb / qc stay empty under E33)
    e33 = _e33_state(e28, order, sup) if _e33_on() else None   # E33 ATTEMPT CHAIN (None = off = the E29 post file)
'''),
    ("K2 candidates", '''        bq = sorted(qb)
        wb2 = lambda e: e[3]["phase"] == "B2" and meta.get((e[2], 2), {}).get("waiting")
''', '''        if e33 is not None:                 # E33: attempt 1 before attempt 2; waiting B2, then A, B, C, D, E; group, task
            return _e33_candidates(qe, meta, e33)
        bq = sorted(qb)
        wb2 = lambda e: e[3]["phase"] == "B2" and meta.get((e[2], 2), {}).get("waiting")
'''),
    ("K3 take", '''    def take(where, entry, tid):
        if where == "A":
''', '''    def take(where, entry, tid):
        if where == "E33":                  # E33: a chain item (its own branch, reservation key and record)
            return e33_take(entry, tid)
        if where == "A":
'''),
    ("K4 route", '''            _e29_note_end(sup, tid, ph, cls)
        if ph == "A":
''', '''            _e29_note_end(sup, tid, ph, cls)
        if e33 is not None and p.get("e33_attempt"):   # E33: the attempt chain decides the next stage (or ends)
            return e33_route(tid, p, cls)
        if ph == "A":
'''),
    ("K5 has", '''                    has = bool(qa or qb or qc)
''', '''                    has = bool(qa or qb or qc or qe)       # E33: + the attempt-chain queue (always empty when off)
'''),
    ("K6 admit", '''                        qlen = [len(qa), len(qb), len(qc)]
                        e28["pop_seq"] += 1
                        start_seq = e28["pop_seq"]
                        try:
                            s = sess(tid, p, qpos)
''', '''                        qlen = [len(qa), len(qb), len(qc)] + ([len(qe)] if e33 is not None else [])   # E33: + qe
                        e28["pop_seq"] += 1
                        start_seq = e28["pop_seq"]
                        if p.get("e33_kind") in ("poetiq", "direct"):   # E33: a Poetiq / Direct stage, no Duck session
                            go_, why_ = _e33_can_start(p, deadline)   # hooks set + E33_CAN_START(stage, deadline)
                            if go_:
                                ph, s, t_start = p["phase"], None, _sec()
                                e28["started"][ph] += 1
                                e28["first_start_sec"].setdefault(ph, _sec())
                                e28["last_start_sec"][ph] = _sec()
                                state["running"] += 1
                                active.add((tid, int(p["branch"])))
                                break
                            e33_skip(tid, p, why_, qlen, start_seq)   # no hook / late / hook error: skipped, chain ends
                            cond.notify_all()
                            continue
                        try:
                            s = sess(tid, p, qpos)
'''),
    ("K7 run", '''            t0 = time.monotonic()
            tag = {"stage": {"A": 1, "C": 2}.get(ph, 3), "e28_phase": ph}
''', '''            t0 = time.monotonic()
            if s is None:                      # E33: the Poetiq / Direct stage runs its notebook hook in this worker slot
                _e33_item(tid, p, wi, t_start, qlen, start_seq, t0)
                continue
            tag = {"stage": {"A": 1, "C": 2}.get(ph, 3), "e28_phase": ph}
'''),
    ("K8 closures", '''    def worker(wi: int):
        while True:
            try:
                return _worker_body(wi)
''', '''    # ---------------------------------------------------------------- E33 ATTEMPT CHAIN closures (called only when e33 is set)
    def push_e33(tid: str, attempt: int, stage: str, b_phase=None, group=None, why: str = "", extra=None):
        """E33, under the lock: queue the next stage of the (task, attempt) chain (one queued-or-running item per pair).
        Duck stages get the E28 plan of their phase (A / B1 / B2 budgets, windows, timeouts) on their E33 branch; a
        Poetiq / Direct stage gets the plan budget of its KV reservation (E33_POETIQ_BUDGET / E33_DIRECT_BUDGET)."""
        name = b_phase if stage == "B" else E33_NAME[stage]
        label, b, kind = f"{name}:{attempt}", E33_BRANCH[(attempt, stage)], E33_KIND[stage]
        if group is not None:
            e33["group"][(tid, attempt)] = list(group)
        grp = e33["group"][(tid, attempt)]
        if kind == "duck":
            p = plan(tid, name, branch=b)
        else:
            p = plan(tid, "A", branch=b, window=None,
                     budget=int(e33["cfg"]["poetiq_budget" if kind == "poetiq" else "direct_budget"]))
        p["phase"] = label                 # the E33 phase label (plan() took the E28 base phase for budget / window)
        p.update(extra or {})
        p.update(e33_attempt=attempt, e33_stage=stage, e33_name=name, e33_kind=kind, e33_group=list(grp), e33_from=why,
                 attempt=attempt)          # the hooks: plan["attempt"] (Poetiq attempt 2 = its own seeds) + plan["branch"]
        seq[0] += 1
        qe.append(((attempt, E33_STAGE_RANK[stage]) + tuple(grp), seq[0], tid, p))
        e28["queued"][label] += 1
        e33["live"][(tid, attempt)] = ["queued", label]
        if not (attempt == 1 and stage == "A"):
            meta[(tid, b)] = _meta()

    def e33_seed():
        """E33: the attempt-1 A item of every task (task order) replaces the E28 phase-A FIFO (qa stays empty)."""
        with cond:
            qa.clear()
            e28["queued"]["A"] = 0
            for t_ in order:
                push_e33(t_, 1, "A", group=(0, 0, idx[t_]), why="start")

    def e33_take(entry, tid: str):
        """E33, under the lock (pick): take this chain item -> (task, plan, Duck queue position)."""
        qe.remove(entry)
        p = entry[3]
        a, st = int(p["e33_attempt"]), p["e33_stage"]
        e33["busy"].add((tid, a))
        e33["live"][(tid, a)] = ["running", p["phase"]]
        e33["n_taken"] += 1
        p["e33_q1_left"] = sum(1 for e in qe if int(e[3]["e33_attempt"]) == 1)   # attempt-1 items still queued
        _e33_stat(e33, p["phase"])["taken"] += 1
        if sup is not None:
            sup["c"].add(tid)          # E33: no phase C is ever queued (the E29 supply cap must not count one to come)
        if not qe:
            e28["queue_empty_sec"]["E33"] = _sec()
        if p["e33_kind"] != "duck":
            return tid, p, 400000 + e33["n_taken"]
        if a == 1:
            return tid, p, (idx[tid] if st == "A" else 100000 + idx[tid])
        return tid, p, 200000 + e33["n_taken"]

    def e33_route(tid: str, p: dict, cls: dict):
        """E33, under the lock (route): a chain item ended -> the next stage of its chain, or the chain ends. The
        attempt-1 A end also queues attempt 2 (not on TIME): group 1 = no train pass, group 2 = by its tokens."""
        a, st, label = int(p["e33_attempt"]), p["e33_stage"], p["phase"]
        kind, tok = cls.get("kind"), int(cls.get("tokens") or 0)
        e33["busy"].discard((tid, a))
        e33["live"].pop((tid, a), None)
        d = _e33_stat(e33, label)
        d["ended"] += 1
        d["tokens"] += tok
        d["train_pass"] += int(kind == "TRAIN_PASS")
        d["kinds"][str(kind)] = d["kinds"].get(str(kind), 0) + 1
        e33["reached"].setdefault(tid, {}).setdefault(str(a), []).append([label, str(kind)])
        if a == 1 and st == "A":
            e28["a_class"][tid] = {k: cls.get(k) for k in ("kind", "end_reason", "tokens")}
            if kind != "TIME" and e33["cfg"]["attempt2"]:
                g = e28_c_key(kind == "TRAIN_PASS", tok, idx[tid])
                e33["a2_queued"].append([tid, list(g)])
                push_e33(tid, 2, "A", group=g, why=f"{label} {kind}")
        elif a == 1 and st == "B":
            e28["b_class"][tid] = {k: cls.get(k) for k in ("kind", "end_reason", "tokens")}
        if st == "C" and cls.get("record") is not None:
            e33["prior"][(tid, a)] = cls["record"]
        nxt = _e33_next(e33["cfg"], st, kind)
        if nxt is None:
            e33["end"][f"{tid}:{a}"] = f"{kind} at {label}"
            return
        push_e33(tid, a, nxt[0], b_phase=nxt[1], why=f"{label} {kind}",
                 extra=({"b_class": kind, "b_reason": cls.get("end_reason")} if nxt[0] == "B" else None))

    def e33_skip(tid: str, p: dict, why: str, qlen, start_seq, release: bool = True):
        """E33, under the lock: a Poetiq / Direct stage that may not start (no hook / E33_CAN_START False / a hook error)
        or that the solver did not run (its record's status is not "done" / poetiq.admitted False; release=False: the
        item's own end released it): recorded, nothing published; the chain of this (task, attempt) ends there."""
        a, b, label = int(p["e33_attempt"]), int(p["branch"]), p["phase"]
        if release and E28_RELEASE is not None:
            try:
                E28_RELEASE((tid, b))
            except Exception:
                pass
        m = meta.get((tid, b)) or {}
        e28["rows"][f"{tid}_b{b}"] = dict({"task": tid, "branch": b, "phase": label, "budget": p.get("budget")},
                                          **{k: p.get(k) for k in E33_ROW_KEYS},
                                          status="skipped", kind="SKIPPED", end_reason=why, tokens=0, start_sec=None,
                                          end_sec=_sec(), queues_at_start=qlen, start_seq=start_seq,
                                          queued_sec=m.get("queued_sec"))
        d = _e33_stat(e33, label)
        d["skipped"][why] = d["skipped"].get(why, 0) + 1
        e33["busy"].discard((tid, a))
        e33["live"].pop((tid, a), None)
        e33["reached"].setdefault(tid, {}).setdefault(str(a), []).append([label, "SKIPPED " + why])
        e33["end"][f"{tid}:{a}"] = f"skipped ({why}) at {label}"
        if len(e33["skips"]) < 2000:
            e33["skips"].append([tid, label, why, _sec()])

    def _e33_item(tid: str, p: dict, wi: int, t_start, qlen, start_seq, t0: float):
        """E33: a Poetiq (C = iteration 1, D = iteration 2 with C's record) or Direct (E) stage in this worker slot = the
        notebook hook; the reservation released; the record published as this stage's branch (its submissions
        [{test_outputs, accepted, train_exact}], the solver's poetiq / e31 blocks, an e33 row); then the chain routed.
        No record (None / False / not a dict) or an exception = nothing published, the chain ends there."""
        a, st, label, b, kind_ = int(p["e33_attempt"]), p["e33_stage"], p["phase"], int(p["branch"]), p["e33_kind"]
        it = {"C": 1, "D": 2}.get(st)
        sub = Path(out_dir) / ("poetiq" if kind_ == "poetiq" else "direct") / (f"a{a}_it{it}" if it else f"a{a}")
        rec, err = None, None
        try:
            if kind_ == "poetiq":
                with cond:
                    prior = copy.deepcopy(e33["prior"].get((tid, a))) if st == "D" else None
                rec = E33_RUN_POETIQ(tid, tasks[tid], dict(p), deadline, base_url, model_id, sub, it, prior)
            else:
                rec = E33_RUN_DIRECT(tid, tasks[tid], dict(p), deadline, base_url, model_id, sub)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("%s %s: E33 hook failed: %s", tid, label, err)
        if E28_RELEASE is not None:
            try:
                E28_RELEASE((tid, b))
            except Exception:
                pass
        rec = rec if isinstance(rec, dict) else None
        skip_why = None                        # the solver's own "not run": status "not_admitted" / "not_run", admitted False
        if rec is not None:
            st_ = rec.get("status")
            adm = (rec.get("poetiq") or {}).get("admitted") if isinstance(rec.get("poetiq"), dict) else None
            if st_ not in (None, "done") or adm is False:
                skip_why = str(st_) if st_ not in (None, "done") else "not_admitted"
        raw = copy.deepcopy(rec) if rec is not None else None
        tp = kind_ == "poetiq" and rec is not None and _e33_train_pass(rec)
        kind = "NO_RECORD" if rec is None else ("TRAIN_PASS" if tp else kind_.upper())
        x = (rec or {}).get("e31") if isinstance((rec or {}).get("e31"), dict) else {}
        pq = (rec or {}).get("poetiq") if isinstance((rec or {}).get("poetiq"), dict) else {}
        m = meta.get((tid, b)) or {}
        row = dict({"task": tid, "branch": b, "phase": label}, **{k: p.get(k) for k in E33_ROW_KEYS},
                   budget=p.get("budget"), iteration=it, worker_slot=wi,
                   tokens=int((rec or {}).get("generated_tokens") or 0), start_sec=t_start, end_sec=_sec(),
                   status="done" if rec is not None else "no_record", queues_at_start=qlen, start_seq=start_seq,
                   error=err, kind=kind,
                   end_reason=((x.get("end_reason") or pq.get("end_reason") or ("train_pass" if tp else kind_))
                               if rec is not None else ("error" if err else "no_record")),
                   train_pass=bool(tp), train_score=(_e33_score(rec) if rec is not None else None),
                   best_train=(_e33_best_train(rec) if rec is not None else None), queued_sec=m.get("queued_sec"),
                   wait_sec=(round(t_start - m["first_tried_sec"], 1) if m.get("first_tried_sec") is not None else None))
        if rec is not None and skip_why is None:
            try:
                rec = dict(rec, task_id=tid, branch=b, e33=dict(row), e28=dict(row))
                rec["submissions"] = [dict(s_, branch=b) for s_ in rec.get("submissions") or [] if isinstance(s_, dict)]
                publish(rec)
            except Exception as exc:
                row["error"] = f"publish {type(exc).__name__}: {exc}"[:300]
        with cond:
            e28["done"][label] += 1
            d = _e33_stat(e33, label)
            state["running"] -= 1
            state["last_free_t"] = time.monotonic()
            active.discard((tid, b))
            state["durations"].append(round(time.monotonic() - t0, 1))
            if skip_why is not None:          # the solver did not run this stage: a skip (chain end), not an attempt
                e33_skip(tid, p, skip_why, qlen, start_seq, release=False)
                e28["rows"][f"{tid}_b{b}"].update(start_sec=t_start, worker_slot=wi, tokens=row["tokens"],
                                                  solver_status=raw.get("status"), skipped_after_start=True)
                d["skipped_after_start"] += 1
                cond.notify_all()
                return
            e28["rows"][f"{tid}_b{b}"] = row
            d["no_record"] += int(rec is None)
            d["errors"] += int(bool(row.get("error")))
            try:
                route(tid, p, {"kind": kind, "end_reason": row["end_reason"], "tokens": row["tokens"], "train": None,
                               "record": raw})
            except Exception as exc:
                state["stage2_errors"] = state.get("stage2_errors", 0) + 1
                state["events"].append({"t": round(time.monotonic() - started), "kind": "e28_route_error", "task": tid,
                                        "error": f"{type(exc).__name__}: {exc}"[:200]})
            cond.notify_all()

    def worker(wi: int):
        while True:
            try:
                return _worker_body(wi)
'''),
    ("K9 row tags", '''        if p.get("phase") in ("S1", "S2"):     # E29 SUPPLY
            r.update(s_index=p.get("s_index"), s_reason=p.get("s_reason"), view_plan=p.get("view"))
''', '''        if p.get("phase") in ("S1", "S2"):     # E29 SUPPLY
            r.update(s_index=p.get("s_index"), s_reason=p.get("s_reason"), view_plan=p.get("view"))
        if p.get("e33_attempt"):               # E33: the attempt / stage tags of a Duck stage (the answer rule reads them)
            r.update({k: p.get(k) for k in E33_ROW_KEYS})
'''),
    ("K10 select", '''    if _e29_on():      # E29 SUPPLY: the A / B / S best-train rule (S attempts = branches >= 3)
        return select_attempts_e29(task, branch_records, _slots_out)
''', '''    if any(_e33_is(r) for r in branch_records):   # E33: the attempt-chain answer rule (E33-tagged records only)
        return select_attempts_e33(task, branch_records, _slots_out)
    if _e29_on():      # E29 SUPPLY: the A / B / S best-train rule (S attempts = branches >= 3)
        return select_attempts_e29(task, branch_records, _slots_out)
'''),
    ("K11 seed", '''    if capacity_probe is not None:
        probe_once()
    probe = threading.Thread(target=probe_loop, daemon=True)
    probe.start()
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(max(1, int(workers)))]
''', '''    if e33 is not None:                # E33: the attempt-1 A items (task order) before the first admission
        e33_seed()
    if capacity_probe is not None:
        probe_once()
    probe = threading.Thread(target=probe_loop, daemon=True)
    probe.start()
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(max(1, int(workers)))]
'''),
    ("K12 gauge", '''                                        len(qa) + len(qb) + len(qc)])
''', '''                                        len(qa) + len(qb) + len(qc) + len(qe)])   # E33: + qe
'''),
    ("K13 slot src", '''            src = {"A": 0, "C": 1, "B": 2}
''', '''            src = {"A": 0, "C": 1, "B": 2}
            if e33 is not None:                # E33: the stage labels (A:1, B2:1, C-POETIQ1:2, ...) -> their branches
                src.update(E33_SRC)
'''),
    ("K14 heap left", '''                      stage1_left=len(qa), heap_left=len(qb) + len(qc), shares=[], p90_skipped=0,
''', '''                      stage1_left=len(qa), heap_left=len(qb) + len(qc) + len(qe), shares=[], p90_skipped=0,
'''),
    ("K15 report", '''                           **({"supply": _e29_report(sup, e28, rows_, slots, order)} if sup is not None else {}),
''', '''                           **({"supply": _e29_report(sup, e28, rows_, slots, order)} if sup is not None else {}),
                           **({"e33": _e33_report(e33, e28, qe, order)} if e33 is not None else {}),   # E33
'''),
]

DUCK_TAIL = '''

# ==================================================================== E33 ATTEMPT CHAIN (Experiment 33, user 2026-11-02)
''' + DUCK_MARKER + '''
# Fixed spec: DESIGN_E33_ATTEMPT_CHAIN_2026-11-02.md sections 3 (the scheduler) and 4 (the answers). An alternative to the E29
# DIRECT diff on the same E29 post text. Off (E33_SCHED unset / not "1", or E28_SCHED off) = the code above runs exactly as the
# E29 post file: qe stays empty, e33 is None, no record carries an E33 tag (so select_attempts_e33 is never reached).
# On, every task runs two attempt chains, each STRICTLY sequential (the next stage is queued only when the previous stage
# ended without a train pass; the chain stops at the first train pass; a TIME end queues nothing):
#   attempt 1: A (Duck, E28_T1) -> B (a fresh Duck re-run, E28 rule: B1 = E28_T1 for the B1 / FAIL kinds, B2 = E28_B2_TOKENS +
#              E28_B2_WINDOW for B2; any other non-pass A kind goes straight to C) -> C (Poetiq iteration 1) -> D (Poetiq
#              iteration 2 = the feedback iteration on C's record) -> E (Direct);
#   attempt 2: the same stages on their own branches, queued when the task's attempt-1 A ended (not on TIME): group 1 = the
#              attempt-1 A was not a train pass (task order), group 2 = the rest by ascending attempt-1 A tokens (e28_c_key).
# Admission (the E28 fit hook E28_TRY_RESERVE, backfill and force-when-idle unchanged): every attempt-1 item before any
# attempt-2 item; inside an attempt: B2 items that already waited, then A, B, C, D, E (earlier stage first); inside a stage:
# group, then task order. Attempt-2 items fill the slots attempt 1 cannot use. Supply (E29_SUPPLY=1) only when the chain queue
# is empty (as before: A / B / C empty); its branches start at E33_SUPPLY_B0 under E33.
# Train pass: Duck = the kind TRAIN_PASS (e28_classify); Poetiq = rec["poetiq"]["train_pass"] or rec["accepted"]; Direct =
# never (E is the last stage).
# Switches (env, the notebook sets them, cell3 / server_env of e33_sched_patch.py): E33_POETIQ=1 (C + D), E33_DIRECT=1 (E),
# E33_ATTEMPT2=1 (attempt 2); any other value, unset included = off (that stage is not part of the chain; e.g. Poetiq off:
# A -> B -> E). KV reservation plan budgets: E33_POETIQ_BUDGET (default 60,000 + 8,192), E33_DIRECT_BUDGET (104,000 + 8,192).
# Hooks (module globals the notebook sets; read at call time):
#   E33_RUN_POETIQ(tid, task, plan, deadline, base_url, model_id, out_dir, iteration, prior_record) -> record | None
#       (iteration 1 = stage C, prior_record None; iteration 2 = stage D, prior_record = a copy of C's record as returned);
#   E33_RUN_DIRECT(tid, task, plan, deadline, base_url, model_id, out_dir) -> record | None;
#   E33_CAN_START(stage, deadline) -> bool, stage = "C" / "D" / "E" (the 60,000-floor admission; a (go, ...) tuple is read
#       as its first item). Asked once per Poetiq / Direct item at its admission; never for Duck stages.
#   No hook, False, an exception, or no record = that stage is skipped for that task (recorded in e28["e33"] and its row),
#   and its chain ends there. Never a Duck fallback, never a crash. A record whose status is not "done" (e33_poetiq:
#   "not_admitted" / "not_run") or whose poetiq.admitted is False is the solver's own skip: not published, recorded as
#   skipped (after start), the chain ends. Plans passed to the hooks carry "attempt" (1 / 2) and "branch".
# Branch ids (one per (attempt, stage): its own record, reservation key (task, branch) and row):
#   attempt 1: A 0 (= E28 A, facts view), B 2 (= E28 B), C 113, D 114, E 115;
#   attempt 2: A 1 (= E28's attempt-2 branch), B 122, C 123, D 124, E 125;
#   supply: 200, 201, ... per task (E33_SUPPLY_B0; E29 supply alone starts at 3 and only grows, so the two never meet).
# Phase labels (records, rows, LIVE): A:1, B1:1, B2:1, C-POETIQ1:1, D-POETIQ2:1, E-DIRECT:1 and the same with :2.
# Answers (select_attempts_e33, user accepted 2026-11-02), per test input (grid K -> test K, valid 0-9 grids only):
#   attempt_1 = from the attempt-1 chain: 1 the first train-pass answer (stage order A, B, C, D, then supply by branch);
#   2 else the best train match among those records (the E28 / E29 best-train rule: full beats non-full, else a strictly
#   higher train fraction, ties keep the earlier; Poetiq's train score is its match); 3 else the attempt-1 Direct grid;
#   4 else the E28 fallbacks (select_attempts_r5 over the Duck records); 5 else the test input.
#   attempt_2 = the attempt-2 chain's answer by the same rule (1-3) when it differs from attempt_1; else the next distinct
#   attempt-1 candidate (the rest by (full, fraction, earlier), then its Direct grid); else the E28 fallbacks; else the test
#   input. The slot source of every query is recorded (e28_slot_table / SCHED_LAST["e28"]["slots"]).
# E33 A1 ORDER (user decision 2026-11-02, after the first build; FIX-A1ORDER): env E33_A1_ORDER = "direct_first" (the
#   default: unset or any unknown value) or "e28" (the rule above, exactly as first built). direct_first orders one attempt
#   chain as: 1 the train-pass candidates (chain order); 2 the attempt's Direct grid; 3 the partial-match Duck / Poetiq
#   candidates with a train fraction > 0 (higher first, ties: earlier in the chain); 4 the zero-match ones (chain order);
#   then the E28 fallbacks, then the test input. attempt_2 is built from these orders as above (its own chain head when it
#   differs from attempt_1, else attempt 1's next distinct candidate, then the fallbacks, then the input). The effective
#   order is recorded in every slot row ("a1_order") and in the e28["e33"] report.
E33_ENV = "E33_SCHED"
E33_ENV_POETIQ, E33_ENV_DIRECT, E33_ENV_ATTEMPT2 = "E33_POETIQ", "E33_DIRECT", "E33_ATTEMPT2"
E33_ENV_POETIQ_BUDGET, E33_ENV_DIRECT_BUDGET = "E33_POETIQ_BUDGET", "E33_DIRECT_BUDGET"
E33_ENV_A1_ORDER = "E33_A1_ORDER"     # FIX-A1ORDER: "direct_first" (default) | "e28" (the first-build rule)
E33_ENV_NAMES = (E33_ENV, E33_ENV_POETIQ, E33_ENV_DIRECT, E33_ENV_ATTEMPT2, E33_ENV_POETIQ_BUDGET, E33_ENV_DIRECT_BUDGET,
                 E33_ENV_A1_ORDER)
E33_RUN_POETIQ = None        # notebook hook: (tid, task, plan, deadline, base_url, model_id, out_dir, iteration, prior_record)
E33_RUN_DIRECT = None        # notebook hook: (tid, task, plan, deadline, base_url, model_id, out_dir) -> record | None
E33_CAN_START = None         # notebook hook: (stage "C" / "D" / "E", deadline) -> bool (the 60k-floor admission)
E33_STAGES = ("A", "B", "C", "D", "E")
E33_STAGE_RANK = {s_: i_ for i_, s_ in enumerate(E33_STAGES)}
E33_KIND = {"A": "duck", "B": "duck", "C": "poetiq", "D": "poetiq", "E": "direct"}
E33_NAME = {"A": "A", "C": "C-POETIQ1", "D": "D-POETIQ2", "E": "E-DIRECT"}       # B: its E28 phase name, B1 or B2
E33_BRANCH = {(1, "A"): 0, (1, "B"): 2, (1, "C"): 113, (1, "D"): 114, (1, "E"): 115,
              (2, "A"): 1, (2, "B"): 122, (2, "C"): 123, (2, "D"): 124, (2, "E"): 125}
E33_SUPPLY_B0 = 200
E33_PHASES = tuple(f"{n_}:{a_}" for a_ in (1, 2) for n_ in ("A", "B1", "B2", "C-POETIQ1", "D-POETIQ2", "E-DIRECT"))
E33_SRC = {ph_: E33_BRANCH[(int(ph_.split(":")[1]), ph_[0])] for ph_ in E33_PHASES}
E33_ROW_KEYS = ("e33_attempt", "e33_stage", "e33_name", "e33_kind", "e33_group", "e33_from", "e33_q1_left")


def _e33_on() -> bool:
    return _e28_on() and os.environ.get(E33_ENV, "").strip() == "1"


def _e33_a1_order() -> str:
    """FIX-A1ORDER: the answer order of every attempt chain: "e28" only for exactly that value (case and spaces ignored);
    unset or anything else -> "direct_first" (user decision 2026-11-02)."""
    return "e28" if os.environ.get(E33_ENV_A1_ORDER, "").strip().lower() == "e28" else "direct_first"


def e33_config() -> dict:
    """The E33 switches (read once per solve call): a stage is part of the chains only when its env value is exactly "1"."""
    on = lambda n: os.environ.get(n, "").strip() == "1"
    return {"poetiq": on(E33_ENV_POETIQ), "direct": on(E33_ENV_DIRECT), "attempt2": on(E33_ENV_ATTEMPT2),
            "poetiq_budget": _env_int(E33_ENV_POETIQ_BUDGET, 60000 + 8192),
            "direct_budget": _env_int(E33_ENV_DIRECT_BUDGET, 104000 + 8192),
            "a1_order": _e33_a1_order()}           # FIX-A1ORDER (record only; the answer rule reads the env itself)


def _e33_stat0() -> dict:
    return {"taken": 0, "ended": 0, "train_pass": 0, "no_record": 0, "errors": 0, "tokens": 0, "skipped": {}, "kinds": {},
            "skipped_after_start": 0}


def _e33_stat(e33: dict, label: str) -> dict:
    return e33["stats"].setdefault(label, _e33_stat0())


def _e33_state(e28: dict, order: list, sup) -> dict:
    """The E33 bookkeeping; adds the phase-label counters (started / done / queued) and moves the supply branches to
    E33_SUPPLY_B0 (an E29 supply branch never takes an E33 stage branch)."""
    for k in ("started", "done", "queued"):
        e28[k].update({ph: 0 for ph in E33_PHASES})
    if sup is not None:
        for t in order:
            sup["next_b"][t] = max(int(sup["next_b"].get(t, 3)), E33_SUPPLY_B0)
    return {"cfg": e33_config(), "stats": {ph: _e33_stat0() for ph in E33_PHASES}, "reached": {}, "end": {},
            "busy": set(), "live": {}, "group": {}, "prior": {}, "a2_queued": [], "skips": [], "n_taken": 0}


def _e33_candidates(qe: list, meta: dict, e33: dict) -> list:
    """Every queued chain item in admission order as (where, entry, task, plan): attempt 1 before attempt 2; inside an
    attempt the B2 items that already waited (did not fit before), then stage A, B, C, D, E; inside a stage the group key
    (attempt 1: task order; attempt 2: e28_c_key), then the queue order. A (task, attempt) already running is never
    offered (strictly sequential; by construction it has no queued item)."""
    def rank(e):
        p = e[3]
        w = p.get("e33_name") == "B2" and bool((meta.get((e[2], int(p["branch"]))) or {}).get("waiting"))
        return (int(p["e33_attempt"]), 0 if w else 1, E33_STAGE_RANK[p["e33_stage"]], tuple(p["e33_group"]), e[1])
    return [("E33", e, e[2], e[3]) for e in sorted(qe, key=rank) if (e[2], int(e[3]["e33_attempt"])) not in e33["busy"]]


def _e33_next(cfg: dict, stage: str, kind) -> tuple | None:
    """The chain's next stage after `stage` ended with `kind` -> (stage, B phase or None) | None (the chain ends). A train
    pass, a TIME end, a skipped stage or a stage without a record end the chain; A -> B (the E28 B kinds) or straight to
    C; B -> C; C -> D; D -> E; E is the last stage. A stage switched off is left out (C / D: E33_POETIQ, E: E33_DIRECT)."""
    if kind in ("TRAIN_PASS", "TIME", "NO_RECORD", "SKIPPED"):
        return None
    if stage == "A" and e28_b_phase(kind):
        return "B", e28_b_phase(kind)
    if stage in ("A", "B"):
        if cfg.get("poetiq"):
            return "C", None
        return ("E", None) if cfg.get("direct") else None
    if stage == "C":
        return "D", None
    if stage == "D":
        return ("E", None) if cfg.get("direct") else None
    return None


def _e33_can_start(p: dict, deadline) -> tuple:
    """(go, why) for a Poetiq / Direct item, never raises: its run hook callable, E33_CAN_START callable and True for
    (stage, deadline). no hook -> "no_hook", no admission hook -> "no_can_start_hook", False -> "late", an exception ->
    "hook_error <type>". A (go, why, need) tuple (e33_lib.can_start) is read as its first item."""
    run = E33_RUN_POETIQ if p.get("e33_kind") == "poetiq" else E33_RUN_DIRECT
    if not callable(run):
        return False, "no_hook"
    h = E33_CAN_START
    if not callable(h):
        return False, "no_can_start_hook"
    try:
        r = h(p.get("e33_stage"), deadline)
        if isinstance(r, (tuple, list)):
            r = r[0] if r else False
        return (True, "ok") if bool(r) else (False, "late")
    except Exception as exc:
        return False, f"hook_error {type(exc).__name__}"[:80]


def _e33_train_pass(rec: dict) -> bool:
    """Poetiq train pass: the transform program is exact on every train example (rec["poetiq"]["train_pass"] / accepted)."""
    pq = rec.get("poetiq") if isinstance(rec.get("poetiq"), dict) else {}
    return bool(pq.get("train_pass")) or bool(rec.get("accepted"))


def _e33_score(rec: dict):
    """Poetiq's train score (rec["poetiq"]["train_score"], else ["score"]) clamped to [0, 1], else None."""
    pq = rec.get("poetiq") if isinstance(rec.get("poetiq"), dict) else {}
    for k in ("train_score", "score"):
        v = pq.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v:
            return max(0.0, min(1.0, float(v)))
    return None


def _e33_best_train(rec: dict) -> dict:
    """The row's best_train of a Poetiq / Direct record (LIVE reads k / n): the first accepted submission, else the first."""
    subs = [s for s in rec.get("submissions") or [] if isinstance(s, dict)]
    s = next((x for x in subs if x.get("accepted")), subs[0] if subs else {})
    te = s.get("train_exact") if isinstance(s.get("train_exact"), list) else []
    full = _e33_train_pass(rec)            # Direct: never (no poetiq block, accepted False)
    k = len(te) if full else sum(bool(v) for v in te)
    return {"full": bool(full), "k": k, "n": len(te), "frac": 1.0 if full else (round(k / len(te), 4) if te else
                                                                                    (_e33_score(rec) or 0.0))}


def _e33_tag(rec):
    """(attempt, stage, phase label) of an E33 record (its e28 row carries the tags), else None."""
    e = rec.get("e28") if isinstance(rec, dict) and isinstance(rec.get("e28"), dict) else {}
    a, st = e.get("e33_attempt"), e.get("e33_stage")
    if not a or st not in E33_STAGE_RANK:
        return None
    return int(a), str(st), str(e.get("phase") or f"{E33_NAME.get(st, st)}:{a}")


def _e33_is(rec) -> bool:
    return _e33_tag(rec) is not None


def _e33_cand(rec: dict, qi: int):
    """This record's candidate for test input qi: {grid, full, k, n, frac} | None. Duck (A / B / supply) = _e28_cand. Poetiq
    (C / D): accepted submissions first, then the rest in order; the first valid grid; full = that submission accepted, or
    the record's train pass for its first submission; frac = 1.0 if full, else its train_exact fraction, else the Poetiq
    train score. Direct (E): the first valid grid, no train match (full False, frac 0)."""
    tag = _e33_tag(rec)
    st = tag[1] if tag else "S"
    if st in ("A", "B", "S"):
        return _e28_cand(rec, qi)
    subs = [s for s in rec.get("submissions") or [] if isinstance(s, dict)]

    def _g(s):
        outs = s.get("test_outputs") or []
        g = outs[qi] if qi < len(outs) else None
        return g if valid_grid(g) else None
    if st == "E":
        g = next((x for x in (_g(s) for s in subs) if x is not None), None)
        return None if g is None else {"grid": g, "full": False, "k": 0, "n": 0, "frac": 0.0}
    tp, score = _e33_train_pass(rec), _e33_score(rec)
    for i, s in enumerate(sorted(subs, key=lambda s_: not s_.get("accepted"))):
        g = _g(s)
        if g is None:
            continue
        te = s.get("train_exact") if isinstance(s.get("train_exact"), list) else []
        if s.get("accepted") or (tp and i == 0):
            return {"grid": g, "full": True, "k": len(te), "n": len(te), "frac": 1.0}
        k, n = sum(bool(v) for v in te), len(te)
        return {"grid": g, "full": False, "k": k, "n": n, "frac": (k / n) if n else float(score or 0.0)}
    return None


def _e33_chain(recs: dict, qi: int, extra=(), order: str = "e28") -> list:
    """One attempt chain's ordered candidates for test input qi: [(source label, candidate, step)]. recs = {stage: record};
    extra = supply records (attempt 1). First the rule's pick among the A / B / C / D (+ supply) candidates (step 1 = a
    train pass, 2 = the best train match; the E28 / E29 best-train loop), then the rest by (full, fraction, earlier),
    then the Direct grid (step 3)."""
    m = []
    for st in ("A", "B", "C", "D"):
        r = recs.get(st)
        c = _e33_cand(r, qi) if r is not None else None
        if c is not None:
            m.append((_e33_tag(r)[2], c))
    for r in extra:
        c = _e28_cand(r, qi)
        if c is not None:
            m.append((f"S{int(r.get('branch') or 0)}", c))
    if order != "e28":                 # FIX-A1ORDER direct_first: pass, Direct, partial (> 0, higher first), zero
        r = recs.get("E")
        ce = _e33_cand(r, qi) if r is not None else None
        out = [(s, c, 1) for s, c in m if c["full"]]
        if ce is not None:
            out.append((_e33_tag(r)[2], ce, 2))
        nf = [(i, s, c) for i, (s, c) in enumerate(m) if not c["full"]]
        out += [(s, c, 3) for _, s, c in sorted([x for x in nf if x[2]["frac"] > 0], key=lambda x: (-x[2]["frac"], x[0]))]
        out += [(s, c, 4) for _, s, c in nf if not c["frac"] > 0]
        return out
    best = None
    for i, (s, c) in enumerate(m):
        if best is None or (c["full"] and not best[2]["full"]) or (
                not c["full"] and not best[2]["full"] and c["frac"] > best[2]["frac"]):
            best = (i, s, c)
    out = [(best[1], best[2], 1 if best[2]["full"] else 2)] if best is not None else []
    rest = sorted([(i, s, c) for i, (s, c) in enumerate(m) if best is None or i != best[0]],
                  key=lambda x: (x[2]["full"], x[2]["frac"], -x[0]), reverse=True)
    out += [(s, c, 1 if c["full"] else 2) for _, s, c in rest]
    r = recs.get("E")
    c = _e33_cand(r, qi) if r is not None else None
    if c is not None:
        out.append((_e33_tag(r)[2], c, 3))
    return out


def select_attempts_e33(task: dict, branch_records: list[dict], _slots_out: list | None = None) -> list[dict]:
    """E33 answer slots (design section 4, user accepted 2026-11-02; the select_attempts_r5 output format). Per test input:
    attempt_1 = the attempt-1 chain's rule (1 first train pass, 2 best train match, 3 its Direct grid), else 4 the E28
    fallbacks (select_attempts_r5 over every Duck record), else 5 the test input. attempt_2 = the attempt-2 chain's rule
    (1-3) when its grid differs from attempt_1, else the next distinct attempt-1 candidate, else the next distinct E28
    fallback, else the test input, else attempt_1 ("same"). Slot rows record the source label and the step."""
    by: dict = {1: {}, 2: {}}
    supply, duck = [], []
    for r in branch_records:
        if not isinstance(r, dict):
            continue
        tag = _e33_tag(r)
        if tag is None:
            duck.append(r)
            if _e29_phase_of(r) == "S":
                supply.append(r)
            continue
        a, st, _ = tag
        cur = by.setdefault(a, {}).get(st)
        if cur is None or int(r.get("publish_seq") or 0) >= int(cur.get("publish_seq") or 0):
            by[a][st] = r
        if st in ("A", "B"):
            duck.append(r)
    supply.sort(key=lambda r: int(r.get("branch") or 0))
    lists: list = []
    select_attempts_r5(task, duck, _e28_lists=lists)
    order = _e33_a1_order()            # FIX-A1ORDER
    out = []
    for qi, q in enumerate(task["test"]):
        o1, o2 = _e33_chain(by.get(1) or {}, qi, supply, order), _e33_chain(by.get(2) or {}, qi, (), order)
        fb = [(g, "input" if g == q["input"] else "r5") for g in (lists[qi] if qi < len(lists) else [])]
        fb = fb + [(q["input"], "input")]
        steps = ({1: "1-pass", 2: "2-best", 3: "3-direct"} if order == "e28" else
                 {1: "1-pass", 2: "2-direct", 3: "3-partial", 4: "4-zero"})
        if o1:
            s1, c1, k1 = o1[0]
            a1, st1, tr1 = c1["grid"], steps[k1], {k: c1[k] for k in ("full", "k", "n")}
        else:
            a1, s1 = fb[0]
            st1, tr1 = ("5-input" if s1 == "input" else "4-fallback"), None
        tr2 = None
        if o2 and o2[0][1]["grid"] != a1:
            s2, c2, k2 = o2[0]
            a2, st2, tr2 = c2["grid"], "a2:" + steps[k2], {k: c2[k] for k in ("full", "k", "n")}
        else:
            nxt = next(((s, c) for s, c, _ in o1 if c["grid"] != a1), None)
            if nxt is not None:
                a2, s2, st2, tr2 = nxt[1]["grid"], nxt[0], "a1-next", {k: nxt[1][k] for k in ("full", "k", "n")}
            else:
                g2 = next(((g, s) for g, s in fb if g != a1), None)
                if g2 is not None:
                    a2, s2 = g2
                    st2 = "5-input" if s2 == "input" else "4-fallback"
                else:
                    a2, s2, st2 = a1, "same", "same"
        out.append({"attempt_1": a1, "attempt_2": a2})
        if _slots_out is not None:
            _slots_out.append({"qi": qi, "attempt_1": s1, "attempt_2": s2, "a1_step": st1, "a2_step": st2,
                               "a1_train": tr1, "a2_train": tr2, "a1_chain": [x[0] for x in o1],
                               "a2_chain": [x[0] for x in o2], "e33": True, "a1_order": order})
    return out


def _e33_report(e33: dict, e28: dict, qe: list, order: list) -> dict:
    """SCHED_LAST["e28"]["e33"]: the switches, hooks, branch map, per-stage counts (queued / taken / started / done /
    skipped by reason / skipped for time / no record / train pass / tokens / left in the queue at the end), the stage
    reached per task and attempt, the attempt-2 queue keys, the skips."""
    left: dict = {}
    for e in qe:
        left[e[3]["phase"]] = left.get(e[3]["phase"], 0) + 1
    stages = {}
    for ph in E33_PHASES:
        d = dict(e33["stats"].get(ph) or _e33_stat0())
        d.update(queued=e28["queued"].get(ph, 0), started=e28["started"].get(ph, 0), done=e28["done"].get(ph, 0),
                 left=left.get(ph, 0))
        d["skipped_for_time"] = int(d["skipped"].get("late", 0)) + int(d["skipped"].get("not_admitted", 0)) + d["left"]
        stages[ph] = d
    reached = {}
    for t in order:
        if t not in e33["reached"] and not any(k[0] == t for k in e33["live"]):
            continue
        row = {}
        for a in (1, 2):
            h = (e33["reached"].get(t) or {}).get(str(a)) or []
            lv = e33["live"].get((t, a))
            if h or lv:
                row[f"a{a}"] = {"stages": h, "end": e33["end"].get(f"{t}:{a}"),
                                "open": (f"{lv[0]} {lv[1]}" if lv else None),
                                "reached": (lv[1] if lv else (h[-1][0] if h else None))}
        reached[t] = row
    cfg = e33["cfg"]
    return {"enabled": True, "switches": {k: bool(cfg[k]) for k in ("poetiq", "direct", "attempt2")},
            "a1_order": _e33_a1_order(),          # FIX-A1ORDER: the order the slot table at the end was built with
            "budgets": {"poetiq": cfg["poetiq_budget"], "direct": cfg["direct_budget"]},
            "hooks": {"poetiq": callable(E33_RUN_POETIQ), "direct": callable(E33_RUN_DIRECT),
                      "can_start": callable(E33_CAN_START)},
            "branch_map": dict(E33_SRC), "supply_branch_base": E33_SUPPLY_B0, "phases": list(E33_PHASES),
            "stages": stages, "tokens": {ph: stages[ph]["tokens"] for ph in E33_PHASES},
            "train_pass": {ph: stages[ph]["train_pass"] for ph in E33_PHASES},
            "reached": reached, "a2_queued": e33["a2_queued"], "skips": e33["skips"], "taken": e33["n_taken"]}
'''

MODULE_TMPL = r'''"""E33 ATTEMPT CHAIN scheduler patch (Experiment 33, user 2026-11-02; generated by Build/e33_sched_build.py, do not edit).

A strict unified diff on arc2_duck.py from the E29 post text (e29_sched_patch.py, sha __PRE__) to the E33 post text (sha
__POST__). An ALTERNATIVE to the E29 DIRECT diff (e29d_sched_patch.py, same pre text): never applied on top of it.
Runtime (E28_SCHED=1 + E33_SCHED=1; the sub-switches E33_POETIQ / E33_DIRECT / E33_ATTEMPT2 = "1"; the notebook hooks
arc2_duck.E33_RUN_POETIQ / E33_RUN_DIRECT / E33_CAN_START): two strictly sequential attempt chains per task (attempt 1:
A -> B -> C Poetiq it1 -> D Poetiq it2 -> E Direct; attempt 2: the same on branches 1 / 122-125, queued at the attempt-1 A
end), the E33 priority, the E33 answer rule select_attempts_e33 and the report SCHED_LAST["e28"]["e33"]. Off (E33_SCHED
unset) = the E29 post file's behaviour exactly. Spec: DESIGN_E33_ATTEMPT_CHAIN_2026-11-02.md sections 3-4.

    import e33_sched_patch as E33S     # e28_sched_patch.py importable (the applier)
    rec = E33S.apply_e33(E12_DIR)      # E29 post -> E33 post (dir / file / text); E33 post -> no-op; else raises
    ok, why = E33S.verify_e33(E12_DIR)
    rec = E33S.revert_e33(E12_DIR)     # E33 post -> the exact E29 post bytes; E29 post -> no-op
    E33S.tree_state(E12_DIR)           # "E29_POST" | "E33_PATCHED" | "OTHER"
    E33S.server_env(chain)             # the E33 env names for the checked CELL 1 chain dict (e33_lib.check_chain)
    rec = E33S.cell3(E12_DIR, E28_SCHED_REC, rec, chain)   # the notebook step, never raises
"""
import hashlib
import json
import os
from pathlib import Path

import e28_sched_patch as E28S

DUCK_REL = "arc2_duck.py"
PRE = "__PRE__"                 # the E29 post text (e29_sched_patch.E29_POST)
POST = "__POST__"
MARKER = __MARKER__
ENV_E33 = "E33_SCHED"
ENV_NAMES = ("E33_SCHED", "E33_POETIQ", "E33_DIRECT", "E33_ATTEMPT2", "E33_POETIQ_BUDGET", "E33_DIRECT_BUDGET",
             "E33_A1_ORDER")          # FIX-A1ORDER: cell3 clears it with the others
A1_ORDERS = ("direct_first", "e28")   # FIX-A1ORDER: E33_A1_ORDER values (default direct_first; unknown = direct_first)
BRANCH_MAP = {"A:1": 0, "B1:1": 2, "B2:1": 2, "C-POETIQ1:1": 113, "D-POETIQ2:1": 114, "E-DIRECT:1": 115,
              "A:2": 1, "B1:2": 122, "B2:2": 122, "C-POETIQ1:2": 123, "D-POETIQ2:2": 124, "E-DIRECT:2": 125}
SUPPLY_BRANCH_BASE = 200
PATCH = r__TQ__
apply_unified, reverse_unified = E28S.apply_unified, E28S.reverse_unified
SHA = {"pre": PRE, "post": POST, "patch": hashlib.sha256(PATCH.encode("utf-8")).hexdigest(), "applier": E28S.E28_SCHED_SHA["applier"]}


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _state_of(data):
    h = _sha(data)
    if h == POST and MARKER.encode() in data:
        return "E33_PATCHED", h
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


def verify_e33(target):
    if _is_text(target):
        st, h = _state_of(target.encode("utf-8"))
    else:
        f = _file(target)
        if not f.is_file():
            return False, f"OTHER missing {f}"
        st, h = _state_of(f.read_bytes())
    return st == "E33_PATCHED", f"{st} sha256={h} (pre {PRE[:12]}, post {POST[:12]})"


def _move(data, forward):
    out = apply_unified(data.decode("utf-8"), PATCH if forward else reverse_unified(PATCH)).encode("utf-8")
    want = POST if forward else PRE
    if _sha(out) != want:
        raise RuntimeError(f"E33_{'POST' if forward else 'REVERT'}_SHA_MISMATCH {DUCK_REL}: {_sha(out)}")
    return out


def _do(target, forward):
    key = "applied" if forward else "reverted"
    done, start = ("E33_PATCHED", "E29_POST") if forward else ("E29_POST", "E33_PATCHED")
    if _is_text(target):
        data = target.encode("utf-8")
        st, h = _state_of(data)
        rec = {"mode": "text", "before": st, "sha_before": h, "sha": SHA}
        if st == done:
            rec.update({key: "already" if forward else "not_applied", "text": target})
        elif st == start:
            rec.update({key: key, "text": _move(data, forward).decode("utf-8")})
        else:
            raise RuntimeError(f"E33_STATE_MISMATCH {DUCK_REL}: {st} {h}")
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
        raise RuntimeError(f"E33_STATE_MISMATCH {f}: {st} {h}")
    E28S._drop_pyc(f)
    rec["after"], rec["sha_after"] = _state_of(f.read_bytes())
    rec["py_compile"] = E28S._py_compile(f)
    if rec["after"] != done or rec["py_compile"]["rc"]:
        raise RuntimeError("E33_VERIFY_FAILED: " + json.dumps({k: rec.get(k) for k in ("after", "sha_after", "py_compile")}))
    rec["ok"] = True
    return rec


def apply_e33(target):
    """The E29 post arc2_duck.py -> the E33 post (atomic, pyc dropped, py_compile + post sha + marker verified)."""
    return _do(target, True)


def revert_e33(target):
    """The E33 post -> the exact E29 post bytes (the E29 / E28 reverts then work as before)."""
    return _do(target, False)


def server_env(chain=None):
    """The E33 env names for a checked chain dict (e33_lib.check_chain keys on / poetiq / direct / attempt2 / poetiq_think
    / poetiq_answer / direct_think / direct_answer): E33_SCHED=1 (only when chain is None or chain["on"]), each sub-switch
    "1" / "0", the reservation budgets think + answer when both are ints. chain None -> E33_SCHED=1 alone (sub-switches
    unset = off). chain["on"] falsy -> {} (E33 off). FIX-A1ORDER: a chain key "a1_order" sets E33_A1_ORDER ("e28" for
    exactly that value, anything else "direct_first"); without the key the name stays unset (= direct_first)."""
    if chain is None:
        return {ENV_E33: "1"}
    if not chain.get("on"):
        return {}
    env = {ENV_E33: "1"}
    for k, name in (("poetiq", "E33_POETIQ"), ("direct", "E33_DIRECT"), ("attempt2", "E33_ATTEMPT2")):
        env[name] = "1" if chain.get(k) is True else "0"
    for s, name in (("poetiq", "E33_POETIQ_BUDGET"), ("direct", "E33_DIRECT_BUDGET")):
        t, a = chain.get(s + "_think"), chain.get(s + "_answer")
        if isinstance(t, int) and isinstance(a, int) and not isinstance(t, bool) and not isinstance(a, bool):
            env[name] = str(t + a)
    if "a1_order" in chain:        # FIX-A1ORDER
        env["E33_A1_ORDER"] = "e28" if str(chain.get("a1_order") or "").strip().lower() == "e28" else "direct_first"
    return env


def cell3(e12_dir, sched_rec, rec, chain=None):
    """The notebook's CELL 3 step (E33), never raises: only on the E29 scheduler (sched_rec = E28_SCHED_REC applied, path
    E29); the E33 env (server_env(chain)) only after apply + verify; any failure -> reverted to the E29 post file (no E33,
    every E33 env name removed), recorded."""
    rec = dict(rec)
    for k in ENV_NAMES:
        os.environ.pop(k, None)
    if not (sched_rec.get("applied") and sched_rec.get("path") == "E29"):
        rec["error"] = "not the E29 scheduler (path %s): no E33" % sched_rec.get("path")
        return rec
    try:
        rec.update({k: v for k, v in apply_e33(e12_dir).items() if k != "text"})
        ok, why = verify_e33(e12_dir)
        if not ok:
            raise RuntimeError("E33_VERIFY_FAILED: " + why)
        env = server_env(chain)
        os.environ.update(env)
        rec.update(ok=True, verify_why=why, env=env)
    except Exception as e:
        for k in ENV_NAMES:
            os.environ.pop(k, None)
        rec["error"] = f"{type(e).__name__}: {e}"[:600]
        try:
            rec["revert"] = {k: v for k, v in revert_e33(e12_dir).items() if k != "text"}
        except Exception as e2:
            rec["revert_error"] = f"{type(e2).__name__}: {e2}"[:300]
        print("!!!!!!!!!! E33 diff not applied -> the E29 scheduler without E33:", rec["error"], flush=True)
    return rec
'''


def make_sched(e29_post_text, e29_post_sha):
    """-> (post_text, patch_text, module_text). The diff is strict (n=3 context), checked forward and backward with the E28
    applier by the caller (regen)."""
    assert sha(e29_post_text) == e29_post_sha
    assert "E33" not in e29_post_text
    t = e29_post_text
    for lab, old, new in DUCK_EDITS:
        t = once(t, old, new, lab)
    assert t.endswith("    return out\n")
    post = t + DUCK_TAIL
    assert "\r" not in post and post.count(DUCK_MARKER) == 1
    compile(post, "arc2_duck.py", "exec")
    a, b = e29_post_text.splitlines(True), post.splitlines(True)
    body = "".join(difflib.unified_diff(a, b, "a/arc2_duck.py", "b/arc2_duck.py", n=3))
    patch = ("# E33 ATTEMPT CHAIN onto arc2_duck.py (the E29 SCHED post text).\n# pre-patch sha256  " + e29_post_sha
             + "\n# post-patch sha256 " + sha(post) + "\n" + body)
    assert "'''" not in patch and "\r" not in patch and not patch.rstrip("\n").endswith("\\")
    mod = (MODULE_TMPL.replace("__PRE__", e29_post_sha).replace("__POST__", sha(post))
           .replace("__MARKER__", repr(DUCK_MARKER)).replace("__TQ__", "'''" + patch + "'''"))
    compile(mod, "e33_sched_patch.py", "exec")
    return post, patch, mod


def e29_post_text():
    """The E29 post arc2_duck.py exactly as e29_direct_build.regen_sched() builds it: the pinned Duck payload ->
    e29_sched_patch.apply_e29_sched (import paths: the E28 Build dir, the E29 Build dir)."""
    import zipfile
    for p_ in (str(E28B_DIR), str(E29B_DIR)):
        if p_ not in sys.path:
            sys.path.insert(0, p_)
    import e29_sched_patch as E29S
    raw = DUCK_PAYLOAD.read_bytes()
    assert sha(raw) == DUCK_PAYLOAD_SHA, "the Duck payload is not 9e4b8928"
    with zipfile.ZipFile(DUCK_PAYLOAD) as z:
        duck = z.read("arc2_duck.py").decode("utf-8")
    p = E29S.apply_e29_sched(duck)
    assert p["ok"] and sha(p["text"]) == E29S.E29_SCHED_SHA["post"] == E29_POST_SHA
    return p["text"], E29S


def regen(write=True):
    """Build e33_sched_patch.py: the E29 post text -> make_sched, checked forward / backward with the E28 applier and
    through the generated module (text mode); written LF-only with write_bytes."""
    pre, E29S = e29_post_text()
    post, patch, mod = make_sched(pre, E29_POST_SHA)
    assert E29S.apply_unified(pre, patch) == post and E29S.apply_unified(post, E29S.reverse_unified(patch)) == pre
    ns = {"__name__": "e33_sched_patch_b"}
    exec(compile(mod, "e33_sched_patch.py", "exec"), ns)
    assert ns["apply_e33"](pre)["text"] == post and ns["revert_e33"](post)["text"] == pre
    assert ns["verify_e33"](post)[0] and ns["tree_state"](post) == "E33_PATCHED" and ns["tree_state"](pre) == "E29_POST"
    assert ns["BRANCH_MAP"] == {k: v for k, v in _branch_map_of(post).items()}
    data = mod.encode("utf-8")
    assert b"\r" not in data
    if write:
        (HERE / "e33_sched_patch.py").write_bytes(data)
        assert (HERE / "e33_sched_patch.py").read_bytes() == data
    heads = [ln for ln in patch.splitlines() if ln.startswith("@@")]
    return {"e33_sched_patch.py": sha(data), "pre": E29_POST_SHA, "post": sha(post), "patch": sha(patch),
            "applier": E29S.E28S.E28_SCHED_SHA["applier"], "hunks": len(heads),
            "lines_removed": sum(1 for ln in patch.splitlines() if ln.startswith("-") and not ln.startswith("---")),
            "lines_added": sum(1 for ln in patch.splitlines() if ln.startswith("+") and not ln.startswith("+++"))}


def _branch_map_of(post):
    """E33_SRC as the post text defines it (exec of the tail constants only, in a scratch namespace)."""
    tail = post[post.index("E33_ENV = \"E33_SCHED\""):post.index("def _e33_on()")]
    ns = {}
    exec(tail, ns)
    return ns["E33_SRC"]


if __name__ == "__main__":
    import json
    print(json.dumps(regen(write="--dry" not in sys.argv), indent=1))
